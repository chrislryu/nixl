/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Unit test for MpscRing, the post ring: the posting threads push and the progress thread pops.
 * The consumer keeps a copy of the producers' head and loads head again only once it has popped
 * every slot up to that copy. pushBatch() reserves the slots of several items at once.
 *
 * A single-threaded run checks the ring against a model queue through many wraps of a small,
 * odd capacity, and across destroy() and initialize(), with push() and with pushBatch(). Batches
 * of 1, 63 and 64 items, the capacity of a 64-slot ring, also start at slots that make them wrap.
 * The threaded runs have four producers and one consumer, each producer pushing numbered items,
 * alone or in batches, some larger than the ring. The consumer checks that every item arrives
 * intact, in its producer's order, and exactly once, and a batch's items one after another, while
 * another thread calls empty(). One run stops the consumer until the ring is full and checks that
 * the other pushes wait. Two more check that a batch larger than the free space, or than the
 * ring, waits for exactly the room it needs, while the consumer already pops its first items. The
 * ring is header-only, so nothing is mocked.
 */

#include "libfabric/libfabric_rail.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

struct Item {
    uint32_t producer = 0;
    uint32_t seq = 0;
    uint64_t check = 0; // derived from both, so that a slot read before it is written shows
};

uint64_t
checkOf(uint32_t producer, uint32_t seq) {
    return ((static_cast<uint64_t>(producer) << 32) | seq) * 0x9e3779b97f4a7c15ULL + 1;
}

Item
makeItem(uint32_t producer, uint32_t seq) {
    return Item{producer, seq, checkOf(producer, seq)};
}

std::string
describe(const Item &item) {
    return "item " + std::to_string(item.seq) + " of producer " + std::to_string(item.producer);
}

// --- Single thread ---

// Pushes and pops in bursts of pseudo-random length, checking every pop and empty() against a
// model queue, until about kLaps times the capacity has gone through the ring. With batch, a
// burst of pushes is one pushBatch(), of 0 items too.
int
runModel(MpscRing<Item> &ring, size_t capacity, uint32_t &next_seq, bool batch = false) {
    constexpr size_t kLaps = 200;
    std::deque<Item> model;
    uint32_t rng = 12345;
    auto draw = [&rng](size_t bound) {
        rng = rng * 1664525u + 1013904223u;
        return (rng >> 8) % bound;
    };
    const uint32_t end_seq = next_seq + static_cast<uint32_t>(kLaps * capacity);
    while (next_seq < end_seq || !model.empty()) {
        // Up to a full ring, and nothing past the end
        size_t pushes = next_seq < end_seq ? draw(capacity - model.size() + 1) : 0;
        if (batch) {
            std::vector<Item> items;
            for (; pushes > 0 && next_seq < end_seq; --pushes) {
                items.push_back(makeItem(0, next_seq++));
            }
            ring.pushBatch(items.data(), items.size());
            model.insert(model.end(), items.begin(), items.end());
            TEST_ASSERT(ring.empty() == model.empty(), "empty() after a batch");
        }
        for (; pushes > 0 && next_seq < end_seq; --pushes) {
            const Item item = makeItem(0, next_seq++);
            ring.push(item);
            model.push_back(item);
            TEST_ASSERT(!ring.empty(), "ring with items is not empty");
        }
        // Up to one more pop than there are items, so that pops on an empty ring happen too
        for (size_t pops = draw(model.size() + 2); pops > 0; --pops) {
            Item item;
            const bool popped = ring.pop(item);
            TEST_ASSERT(popped == !model.empty(),
                        popped ? "popped from an empty ring" : "pop failed with items in the ring");
            if (!popped) {
                break;
            }
            TEST_ASSERT(item.seq == model.front().seq && item.check == model.front().check,
                        "popped " + describe(item) + ", expected " + describe(model.front()));
            model.pop_front();
            TEST_ASSERT(ring.empty() == model.empty(), "empty() after a pop");
        }
    }
    Item item;
    TEST_ASSERT(ring.empty() && !ring.pop(item), "ring empty at the end");
    return 0;
}

int
testSingleThread() {
    NIXL_INFO << "  testSingleThread";
    MpscRing<Item> ring;
    Item item;
    TEST_ASSERT(!ring.initialize(0), "zero capacity refused");
    TEST_ASSERT(ring.initialize(3), "ring initialized");
    TEST_ASSERT(ring.empty() && !ring.pop(item), "new ring is empty");

    // The first pop copies head, and the item pushed after it comes out once the copy is used up
    ring.push(makeItem(0, 0));
    ring.push(makeItem(0, 1));
    TEST_ASSERT(ring.pop(item) && item.seq == 0, "first item");
    ring.push(makeItem(0, 2));
    TEST_ASSERT(ring.pop(item) && item.seq == 1, "second item, from the copy of head");
    TEST_ASSERT(ring.pop(item) && item.seq == 2, "third item, after loading head again");
    TEST_ASSERT(ring.empty() && !ring.pop(item), "ring empty again");

    uint32_t next_seq = 3;
    if (runModel(ring, 3, next_seq) != 0) {
        return 1;
    }

    // destroy() drops what is left and resets every index: the ring reads as empty, before and
    // after it is initialized again with another capacity
    ring.push(makeItem(0, next_seq++));
    ring.push(makeItem(0, next_seq++));
    TEST_ASSERT(ring.pop(item), "item popped before destroy()");
    ring.destroy();
    TEST_ASSERT(ring.empty() && !ring.pop(item), "destroyed ring is empty");
    TEST_ASSERT(ring.initialize(5), "ring initialized again");
    TEST_ASSERT(ring.empty() && !ring.pop(item), "initialized ring is empty");
    return runModel(ring, 5, next_seq);
}

int
testBatchSingleThread() {
    NIXL_INFO << "  testBatchSingleThread";
    constexpr size_t kCapacity = 64;
    MpscRing<Item> ring;
    TEST_ASSERT(ring.initialize(kCapacity), "ring initialized");
    Item item;
    uint32_t next_seq = 0;

    ring.pushBatch(nullptr, 0);
    TEST_ASSERT(ring.empty() && !ring.pop(item), "an empty batch pushes nothing");

    // Each batch starts at the given slot: 0, 1, and slots from which it wraps past the end
    for (size_t n : {size_t{1}, size_t{63}, kCapacity}) {
        for (size_t start : {size_t{0}, size_t{1}, size_t{40}, kCapacity - 1}) {
            while (next_seq % kCapacity != start) {
                ring.push(makeItem(0, next_seq++));
                TEST_ASSERT(ring.pop(item) && item.seq == next_seq - 1, "single item");
            }
            std::vector<Item> items;
            for (size_t i = 0; i < n; ++i) {
                items.push_back(makeItem(0, next_seq + static_cast<uint32_t>(i)));
            }
            ring.pushBatch(items.data(), items.size());
            const std::string what =
                "batch of " + std::to_string(n) + " from slot " + std::to_string(start);
            for (size_t i = 0; i < n; ++i) {
                TEST_ASSERT(ring.pop(item), what + ": item " + std::to_string(i) + " missing");
                TEST_ASSERT(item.seq == next_seq && item.check == checkOf(0, next_seq),
                            what + ": popped " + describe(item) + ", expected item " +
                                std::to_string(next_seq));
                ++next_seq;
            }
            TEST_ASSERT(ring.empty() && !ring.pop(item), what + ": ring empty after it");
        }
    }
    if (runModel(ring, kCapacity, next_seq, /*batch=*/true) != 0) {
        return 1;
    }

    // Small capacities, so that most batches wrap
    for (size_t capacity : {3, 7}) {
        ring.destroy();
        TEST_ASSERT(ring.initialize(capacity), "ring initialized again");
        if (runModel(ring, capacity, next_seq, /*batch=*/true) != 0) {
            return 1;
        }
    }
    return 0;
}

// --- Four producers and a consumer ---

constexpr uint32_t kProducers = 4;

class Threaded {
public:
    // batches: per producer, the items each pushBatch() takes, or 0 to push() one at a time.
    // Producers without an entry push() one at a time.
    Threaded(size_t capacity, uint32_t items_per_producer, std::vector<uint32_t> batches = {})
        : capacity_(capacity),
          items_per_producer_(items_per_producer),
          batches_(std::move(batches)),
          next_(kProducers, 0) {
        batches_.resize(kProducers, 0);
    }

    // stall_until_full: the consumer pops nothing until the ring is full, then checks that no
    // other push returns while it waits
    int
    run(bool stall_until_full) {
        TEST_ASSERT(ring_.initialize(capacity_), "ring initialized");

        std::vector<std::thread> producers;
        for (uint32_t p = 0; p < kProducers; ++p) {
            producers.emplace_back([this, p] { produce(p); });
        }
        // empty() reads the shared indices, so a thread other than the consumer may call it
        std::thread observer([this] { observe(); });
        start_.store(true, std::memory_order_release);

        if (stall_until_full) {
            stall();
        }
        consume();

        for (std::thread &producer : producers) {
            producer.join();
        }
        consumer_done_.store(true, std::memory_order_release);
        observer.join();

        TEST_ASSERT(error_.empty(), error_);
        for (uint32_t p = 0; p < kProducers; ++p) {
            TEST_ASSERT(next_[p] == items_per_producer_,
                        "producer " + std::to_string(p) + ": " + std::to_string(next_[p]) + " of " +
                            std::to_string(items_per_producer_) + " items arrived");
        }
        TEST_ASSERT(popped_ == uint64_t{kProducers} * items_per_producer_,
                    "every item popped once");
        Item item;
        TEST_ASSERT(ring_.empty() && !ring_.pop(item), "ring empty at the end");
        TEST_ASSERT(empty_calls_ > 0, "empty() called while the threads ran");

        std::string batches;
        for (uint32_t batch : batches_) {
            batches += " " + std::to_string(batch);
        }
        NIXL_INFO << "    capacity " << capacity_ << ", batches" << batches << ": " << popped_
                  << " items, " << empty_calls_ << " concurrent empty() calls";
        return 0;
    }

private:
    void
    fail(const std::string &msg) {
        if (error_.empty()) {
            error_ = msg;
        }
    }

    void
    produce(uint32_t p) {
        while (!start_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        const uint32_t batch = batches_[p];
        if (batch == 0) {
            for (uint32_t seq = 0; seq < items_per_producer_; ++seq) {
                ring_.push(makeItem(p, seq));
                pushed_.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            std::vector<Item> items;
            for (uint32_t seq = 0; seq < items_per_producer_; seq += batch) {
                // The last batch takes what is left
                const uint32_t n = std::min(batch, items_per_producer_ - seq);
                items.clear();
                for (uint32_t i = 0; i < n; ++i) {
                    items.push_back(makeItem(p, seq + i));
                }
                ring_.pushBatch(items.data(), items.size());
                pushed_.fetch_add(n, std::memory_order_relaxed);
            }
        }
        producers_done_.fetch_add(1, std::memory_order_release);
    }

    // Waits for the ring to fill, then gives a wrong push time to return
    void
    stall() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (pushed_.load(std::memory_order_relaxed) < capacity_) {
            if (std::chrono::steady_clock::now() > deadline) {
                fail("ring not full after 10 s: " + std::to_string(pushed_.load()) + " pushes of " +
                     std::to_string(capacity_) + " returned");
                return;
            }
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const uint64_t pushed = pushed_.load(std::memory_order_relaxed);
        if (pushed != capacity_) {
            fail(std::to_string(pushed) + " pushes returned into a ring of " +
                 std::to_string(capacity_) + " that nothing popped");
        }
        if (ring_.empty()) {
            fail("full ring reads as empty");
        }
    }

    // Pops until every producer is done and the ring is empty. It keeps popping after a wrong
    // item, so that no producer is left waiting for room.
    void
    consume() {
        for (;;) {
            // Every push returned before its producer counted itself done, so a pop that fails
            // after all were done means the ring is empty
            const bool all_done = producers_done_.load(std::memory_order_acquire) == kProducers;
            Item item;
            if (ring_.pop(item)) {
                check(item);
                continue;
            }
            if (all_done) {
                return;
            }
            std::this_thread::yield();
        }
    }

    void
    check(const Item &item) {
        ++popped_;
        if (item.producer >= kProducers || item.check != checkOf(item.producer, item.seq)) {
            fail("item " + std::to_string(item.seq) + " of producer " +
                 std::to_string(item.producer) + " corrupt: read before it was written");
            return;
        }
        uint32_t &next = next_[item.producer];
        if (item.seq != next) {
            fail("popped " + describe(item) + ", expected item " + std::to_string(next) +
                 (item.seq < next ? ": popped twice" : ": items lost or out of order"));
        }
        next = item.seq + 1;
        // A batch's slots are consecutive: no other producer's item comes between its items
        const uint32_t batch = batches_[item.producer];
        if (batch > 0 && item.seq % batch != 0 &&
            (popped_ == 1 || last_.producer != item.producer || last_.seq + 1 != item.seq)) {
            fail(describe(item) + " is not right after the previous item of its batch: " +
                 (popped_ == 1 ? std::string("nothing") : describe(last_)) + " came before it");
        }
        last_ = item;
    }

    void
    observe() {
        do {
            (void)ring_.empty();
            ++empty_calls_;
            std::this_thread::yield();
        } while (!consumer_done_.load(std::memory_order_acquire));
    }

    const size_t capacity_;
    const uint32_t items_per_producer_;
    std::vector<uint32_t> batches_; // per producer, read-only once the threads run
    MpscRing<Item> ring_;
    std::atomic<bool> start_{false};
    std::atomic<uint64_t> pushed_{0}; // pushes returned, over all producers
    std::atomic<uint32_t> producers_done_{0};
    std::atomic<bool> consumer_done_{false};
    // The consumer's, and the main thread's after the joins
    std::vector<uint32_t> next_; // per producer, the next item expected
    uint64_t popped_ = 0;
    Item last_; // the item popped before
    std::string error_;
    // The observer's, and the main thread's after the join
    uint64_t empty_calls_ = 0;
};

int
testFourProducers() {
    NIXL_INFO << "  testFourProducers";
    // Small capacities wrap the ring every few items and keep producers waiting for room. 7 is
    // not a power of two.
    for (size_t capacity : {1, 2, 7, 64}) {
        Threaded threaded(capacity, /*items_per_producer=*/50000);
        if (threaded.run(/*stall_until_full=*/false) != 0) {
            return 1;
        }
    }
    return 0;
}

int
testFullRing() {
    NIXL_INFO << "  testFullRing";
    // Three times the capacity: once the consumer stalls, the ring fills and the other pushes wait
    Threaded threaded(/*capacity=*/8, /*items_per_producer=*/6);
    return threaded.run(/*stall_until_full=*/true);
}

int
testFourProducersBatch() {
    NIXL_INFO << "  testFourProducersBatch";

    struct Run {
        size_t capacity;
        std::vector<uint32_t> batches;
    };

    // Batches of 1, 63, 64 and the capacity, and batches larger than the ring, which wrap it
    // while they are written. The last runs mix batches with single pushes.
    const std::vector<Run> runs = {
        {64, {1, 1, 1, 1}},
        {64, {63, 63, 63, 63}},
        {64, {64, 64, 64, 64}},
        {7, {7, 7, 7, 7}},
        {7, {63, 64, 63, 64}},
        {1, {64, 2, 64, 2}},
        {64, {0, 64, 0, 63}},
        {7, {0, 3, 0, 64}},
    };
    for (const Run &run : runs) {
        Threaded threaded(run.capacity, /*items_per_producer=*/50000, run.batches);
        if (threaded.run(/*stall_until_full=*/false) != 0) {
            return 1;
        }
    }
    return 0;
}

// Pops one item, waiting up to 10 s for the producer to write it
bool
popWithin10s(MpscRing<Item> &ring, Item &item) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ring.pop(item)) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

// One producer pushes a batch of n items, numbered from first_seq, into a ring holding
// `held` items. The consumer pops one at a time and checks that the batch returns only once
// every slot it needs is free: after n + held - capacity pops.
int
runBatchWaitsForRoom(size_t capacity, uint32_t held, uint32_t n) {
    const std::string what = "batch of " + std::to_string(n) + " into a ring of " +
        std::to_string(capacity) + " holding " + std::to_string(held);
    MpscRing<Item> ring;
    TEST_ASSERT(ring.initialize(capacity), "ring initialized");
    for (uint32_t seq = 0; seq < held; ++seq) {
        ring.push(makeItem(0, seq));
    }
    std::vector<Item> items;
    for (uint32_t i = 0; i < n; ++i) {
        items.push_back(makeItem(0, held + i));
    }
    std::atomic<bool> returned{false};
    std::thread producer([&] {
        ring.pushBatch(items.data(), items.size());
        returned.store(true, std::memory_order_release);
    });

    const uint32_t pops_needed = held + n - static_cast<uint32_t>(capacity);
    uint32_t next_seq = 0;
    Item item;
    int res = 0;
    for (uint32_t pops = 0; pops < held + n; ++pops) {
        if (pops < pops_needed) {
            // Give a wrong return time to happen
            if (pops == 0 || pops + 1 == pops_needed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (returned.load(std::memory_order_acquire)) {
                std::cerr << "FAIL: " << what << " returned after " << pops << " pops, needs "
                          << pops_needed << std::endl;
                res = 1;
            }
        }
        // The batch's first items come out while it still waits for room
        if (!popWithin10s(ring, item)) {
            std::cerr << "FAIL: " << what << ": item " << next_seq << " not there after 10 s"
                      << std::endl;
            res = 1;
            break;
        }
        if (item.seq != next_seq || item.check != checkOf(0, next_seq)) {
            std::cerr << "FAIL: " << what << ": popped " << describe(item) << ", expected item "
                      << next_seq << std::endl;
            res = 1;
        }
        ++next_seq;
    }
    producer.join();
    TEST_ASSERT(res == 0, what);
    TEST_ASSERT(returned.load() && ring.empty() && !ring.pop(item),
                what + ": ring empty at the end");
    return 0;
}

int
testBatchWaitsForRoom() {
    NIXL_INFO << "  testBatchWaitsForRoom";
    // Larger than the free space, as large as the ring, and larger than the ring
    if (runBatchWaitsForRoom(/*capacity=*/8, /*held=*/3, /*n=*/8) != 0) {
        return 1;
    }
    if (runBatchWaitsForRoom(/*capacity=*/8, /*held=*/5, /*n=*/6) != 0) {
        return 1;
    }
    return runBatchWaitsForRoom(/*capacity=*/8, /*held=*/0, /*n=*/20);
}

} // namespace

int
main() {
    NIXL_INFO << "=== MPSC Ring Test ===";

    int res;
    if ((res = testSingleThread()) != 0) {
        return res;
    }
    if ((res = testFourProducers()) != 0) {
        return res;
    }
    if ((res = testFullRing()) != 0) {
        return res;
    }
    if ((res = testBatchSingleThread()) != 0) {
        return res;
    }
    if ((res = testFourProducersBatch()) != 0) {
        return res;
    }
    if ((res = testBatchWaitsForRoom()) != 0) {
        return res;
    }

    NIXL_INFO << "=== All MPSC ring tests PASSED ===";
    return 0;
}
