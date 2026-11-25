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
 * every slot up to that copy.
 *
 * A single-threaded run checks the ring against a model queue through many wraps of a small,
 * odd capacity, and across destroy() and initialize(). The threaded runs have four producers
 * and one consumer, each producer pushing numbered items. The consumer checks that every item
 * arrives intact, in its producer's order, and exactly once, while another thread calls empty().
 * One run stops the consumer until the ring is full and checks that the other pushes wait. The
 * ring is header-only, so nothing is mocked.
 */

#include "libfabric/libfabric_rail.h"
#include "common/nixl_log.h"

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
// model queue, until about kLaps times the capacity has gone through the ring
int
runModel(MpscRing<Item> &ring, size_t capacity, uint32_t &next_seq) {
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

// --- Four producers and a consumer ---

constexpr uint32_t kProducers = 4;

class Threaded {
public:
    Threaded(size_t capacity, uint32_t items_per_producer)
        : capacity_(capacity),
          items_per_producer_(items_per_producer),
          next_(kProducers, 0) {}

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

        NIXL_INFO << "    capacity " << capacity_ << ": " << popped_ << " items, " << empty_calls_
                  << " concurrent empty() calls";
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
        for (uint32_t seq = 0; seq < items_per_producer_; ++seq) {
            ring_.push(makeItem(p, seq));
            pushed_.fetch_add(1, std::memory_order_relaxed);
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
    MpscRing<Item> ring_;
    std::atomic<bool> start_{false};
    std::atomic<uint64_t> pushed_{0}; // pushes returned, over all producers
    std::atomic<uint32_t> producers_done_{0};
    std::atomic<bool> consumer_done_{false};
    // The consumer's, and the main thread's after the joins
    std::vector<uint32_t> next_; // per producer, the next item expected
    uint64_t popped_ = 0;
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

    NIXL_INFO << "=== All MPSC ring tests PASSED ===";
    return 0;
}
