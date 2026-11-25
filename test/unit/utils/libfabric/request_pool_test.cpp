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
 * Unit test for the request pool's allocate and release sides. allocateReq() pops from the free
 * list, release() pushes to the released list under its own lock, and the allocator swaps the
 * released list in when the free list runs dry. It expands the pool only when both are empty.
 *
 * The stress test runs one thread that allocates data requests and hands them through the post
 * ring to a second thread that releases them, as the posting thread and the progress thread do,
 * while a third reads the pool's counts. No request may be handed out twice, and the active
 * count must be exact whenever the threads are stopped. A data pool calls no libfabric function,
 * so nothing is mocked.
 */

#include "libfabric/libfabric_rail.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
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

// --- Single-threaded cases ---

int
testSwapBeforeExpand() {
    NIXL_INFO << "  testSwapBeforeExpand";
    constexpr size_t kSize = 4;
    DataRequestPool pool(kSize, /*rail_id=*/0);
    TEST_ASSERT(pool.initialize() == NIXL_SUCCESS, "pool initialized");
    TEST_ASSERT(pool.getActiveRequestCount() == 0, "no request active at the start");
    TEST_ASSERT(pool.getPoolUtilization() == 0, "no utilization at the start");

    std::vector<nixlLibfabricReq *> reqs;
    for (uint32_t i = 0; i < kSize; ++i) {
        nixlLibfabricReq *req = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/i + 1);
        TEST_ASSERT(req != nullptr, "request allocated");
        TEST_ASSERT(req->in_use && req->xfer_id == i + 1, "request marked in use with its id");
        TEST_ASSERT(req->pool_index < kSize, "request from the initial pool");
        reqs.push_back(req);
    }
    TEST_ASSERT(std::set<nixlLibfabricReq *>(reqs.begin(), reqs.end()).size() == kSize,
                "every request handed out once");
    TEST_ASSERT(pool.getActiveRequestCount() == kSize, "every request active");
    TEST_ASSERT(pool.getPoolUtilization() == 100, "pool fully used");

    // Released requests count as free at once, and come back before the pool expands
    pool.release(reqs[1]);
    pool.release(reqs[3]);
    TEST_ASSERT(!reqs[1]->in_use && reqs[1]->xfer_id == 0, "released request reset");
    TEST_ASSERT(pool.getActiveRequestCount() == 2, "released requests not active");
    TEST_ASSERT(pool.getPoolUtilization() == 50, "utilization counts the released requests");

    std::set<nixlLibfabricReq *> swapped_in;
    for (uint32_t i = 0; i < 2; ++i) {
        nixlLibfabricReq *req = pool.allocate(nixlLibfabricReq::READ, /*req_id=*/10 + i);
        TEST_ASSERT(req != nullptr && req->operation_type == nixlLibfabricReq::READ,
                    "request allocated again");
        swapped_in.insert(req);
    }
    TEST_ASSERT(swapped_in == std::set<nixlLibfabricReq *>({reqs[1], reqs[3]}),
                "the released requests are swapped in instead of expanding the pool");
    TEST_ASSERT(pool.getActiveRequestCount() == kSize, "every request active again");

    // With both lists empty the pool expands, by its initial size
    nixlLibfabricReq *extra = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/20);
    TEST_ASSERT(extra != nullptr, "request allocated from the expanded pool");
    TEST_ASSERT(extra->pool_index >= kSize && extra->pool_index < 2 * kSize,
                "new request from the expansion");
    TEST_ASSERT(pool.getActiveRequestCount() == kSize + 1, "expansion counted");
    TEST_ASSERT(pool.getPoolUtilization() == (kSize + 1) * 100 / (2 * kSize),
                "utilization over the expanded pool");

    // The expansion leaves the requests already handed out where they are
    TEST_ASSERT(reqs[0]->in_use && reqs[0]->xfer_id == 1, "first request untouched");
    TEST_ASSERT(reqs[2]->in_use && reqs[2]->xfer_id == 3, "third request untouched");

    reqs.push_back(extra);
    for (nixlLibfabricReq *req : reqs) {
        pool.release(req);
    }
    TEST_ASSERT(pool.getActiveRequestCount() == 0, "every request released");
    TEST_ASSERT(pool.getPoolUtilization() == 0, "no utilization once released");
    return 0;
}

int
testDoubleRelease() {
    NIXL_INFO << "  testDoubleRelease";
    DataRequestPool pool(/*pool_size=*/1, /*rail_id=*/0);
    TEST_ASSERT(pool.initialize() == NIXL_SUCCESS, "pool initialized");

    // A second release of a request is ignored, so the request is not handed out twice
    nixlLibfabricReq *a = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/1);
    TEST_ASSERT(a != nullptr, "request allocated");
    pool.release(a);
    pool.release(a);
    TEST_ASSERT(pool.getActiveRequestCount() == 0, "a double release is counted once");
    nixlLibfabricReq *b = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/2);
    nixlLibfabricReq *c = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/3);
    TEST_ASSERT(b == a, "the released request is handed out again");
    TEST_ASSERT(c != nullptr && c != a, "and only once");
    TEST_ASSERT(pool.getActiveRequestCount() == 2, "both requests active");

    // An index outside the pool is refused, not added to the released list
    nixlLibfabricReq foreign;
    foreign.in_use = true;
    foreign.pool_index = 1000;
    pool.release(&foreign);
    TEST_ASSERT(pool.getActiveRequestCount() == 2, "out-of-range index not released");
    nixlLibfabricReq *d = pool.allocate(nixlLibfabricReq::WRITE, /*req_id=*/4);
    TEST_ASSERT(d != nullptr && d != &foreign, "foreign request never handed out");

    pool.release(b);
    pool.release(c);
    pool.release(d);
    TEST_ASSERT(pool.getActiveRequestCount() == 0, "every request released");
    return 0;
}

// --- Two threads ---

constexpr size_t kStressPoolSize = 16; // initial size, and the size of each expansion
constexpr size_t kStressRounds = 100;
// 100 rounds of 5000 allocations and as many releases: 10^6 pool operations
constexpr size_t kAllocationsPerRound = 5000;
// Most requests the allocator keeps in flight per round, cycled through. Windows above the pool
// size make it expand while the releaser runs.
constexpr size_t kWindows[] = {1, 2, 15, 16, 17, 64, 333, 1000};
constexpr size_t kMaxWindow = 1000;
constexpr size_t kRingSize = 1024; // holds a whole window, so a push never waits
// The releaser releases what it has popped in batches of up to this many, newest first, and
// whenever the ring is empty, as the progress thread releases per CQ read
constexpr size_t kReleaseBatch = 8;
// Fewer than kMaxWindow requests are in use whenever the allocator allocates, and the pool
// expands only when all of them are, so it never grows past this
constexpr size_t kMaxCapacity = kMaxWindow - 1 + kStressPoolSize;

class Stress {
public:
    Stress()
        : pool_(kStressPoolSize, /*rail_id=*/0),
          owned_(new std::atomic<uint8_t>[kMaxCapacity]()) {}

    int
    run() {
        TEST_ASSERT(pool_.initialize() == NIXL_SUCCESS, "pool initialized");
        TEST_ASSERT(ring_.initialize(kRingSize), "ring initialized");

        std::vector<nixlLibfabricReq *> held; // allocated and not released, between rounds
        for (size_t round = 0; round < kStressRounds; ++round) {
            const size_t window = kWindows[round % std::size(kWindows)];
            // Half a window, so that the kept requests never fill the allocator's window
            const size_t keep_limit = window / 2;
            // Requests the releaser keeps at the end of the round, up to a whole window
            const size_t hold = (round * 7) % (window + 1);
            const uint32_t first_id = next_id_;
            std::vector<nixlLibfabricReq *> held_next;
            std::atomic<bool> round_done{false};

            std::thread allocator([&] { allocate(window); });
            std::thread releaser([&] { release(first_id, keep_limit, hold, held, held_next); });
            std::thread observer([&] { observe(round_done); });
            allocator.join();
            releaser.join();
            round_done.store(true, std::memory_order_release);
            observer.join();

            TEST_ASSERT(!failed_.load(), first_error_);
            held.swap(held_next);
            const uint64_t released = released_.load();
            TEST_ASSERT(allocated_ - released == held.size(), "held requests accounted for");
            const size_t active = pool_.getActiveRequestCount();
            TEST_ASSERT(active == held.size(),
                        "round " + std::to_string(round) + ": " + std::to_string(active) +
                            " requests active, expected " + std::to_string(held.size()));
        }
        TEST_ASSERT(allocated_ == kStressRounds * kAllocationsPerRound, "every allocation made");
        TEST_ASSERT(observations_ > 0, "counts read while the threads ran");

        for (nixlLibfabricReq *req : held) {
            owned_[req->pool_index].store(0);
            pool_.release(req);
        }
        TEST_ASSERT(pool_.getActiveRequestCount() == 0, "every request released");
        TEST_ASSERT(pool_.getPoolUtilization() == 0, "no utilization once released");

        // Every index is back in exactly one list: allocating the whole pool yields each index
        // once, and only the allocation after that expands it
        std::vector<nixlLibfabricReq *> all;
        std::vector<bool> seen(kMaxCapacity + kStressPoolSize, false);
        do {
            nixlLibfabricReq *req = pool_.allocate(nixlLibfabricReq::WRITE, next_id_++);
            TEST_ASSERT(req != nullptr, "request allocated");
            TEST_ASSERT(req->pool_index < seen.size() && !seen[req->pool_index],
                        "index " + std::to_string(req->pool_index) + " handed out twice");
            seen[req->pool_index] = true;
            all.push_back(req);
        } while (pool_.getPoolUtilization() < 100);
        const size_t capacity = all.size();
        TEST_ASSERT(std::all_of(seen.begin(), seen.begin() + capacity, [](bool s) { return s; }),
                    "every index of the pool present");
        TEST_ASSERT(capacity > kStressPoolSize, "the pool expanded under load");
        TEST_ASSERT(capacity <= kMaxCapacity,
                    "pool grew to " + std::to_string(capacity) +
                        " requests: it expanded while released requests were available");
        nixlLibfabricReq *extra = pool_.allocate(nixlLibfabricReq::WRITE, next_id_++);
        TEST_ASSERT(extra != nullptr && extra->pool_index >= capacity,
                    "the next allocation expands the pool");
        all.push_back(extra);
        for (nixlLibfabricReq *req : all) {
            pool_.release(req);
        }
        TEST_ASSERT(pool_.getActiveRequestCount() == 0, "every request released at the end");

        NIXL_INFO << "    " << allocated_ << " allocations and releases, pool of " << capacity
                  << " requests, " << observations_ << " concurrent count reads";
        return 0;
    }

private:
    void
    fail(const std::string &msg) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!failed_.load()) {
            first_error_ = msg;
            failed_.store(true);
        }
    }

    // The posting thread: allocates kAllocationsPerRound requests, keeping fewer than window
    // in flight, and pushes each to the ring
    void
    allocate(size_t window) {
        for (size_t i = 0; i < kAllocationsPerRound; ++i) {
            while (allocated_ - released_.load(std::memory_order_acquire) >= window) {
                if (failed_.load(std::memory_order_relaxed)) {
                    return;
                }
                std::this_thread::yield();
            }
            const uint32_t id = next_id_++;
            const auto op = (id % 2) ? nixlLibfabricReq::READ : nixlLibfabricReq::WRITE;
            nixlLibfabricReq *req = pool_.allocate(op, id);
            if (req == nullptr) {
                fail("allocation failed");
                return;
            }
            if (!req->in_use || req->xfer_id != id || req->operation_type != op) {
                fail("allocated request not set up");
                return;
            }
            const size_t idx = req->pool_index;
            if (idx >= kMaxCapacity) {
                fail("pool index " + std::to_string(idx) +
                     ": the pool expanded while released requests were available");
                return;
            }
            if (owned_[idx].exchange(1, std::memory_order_acq_rel) != 0) {
                fail("request " + std::to_string(idx) + " handed out twice");
                return;
            }
            if (address_.size() <= idx) {
                address_.resize(idx + 1, nullptr);
            }
            if (address_[idx] == nullptr) {
                address_[idx] = req;
            } else if (address_[idx] != req) {
                fail("request " + std::to_string(idx) + " moved when the pool expanded");
                return;
            }
            ++allocated_;
            ring_.push(req);
        }
    }

    // The progress thread: releases the requests held over from the last round, then this
    // round's requests in batches. It keeps every other request back until it has keep_limit of
    // them, so that the number in use grows while it releases and the pool expands meanwhile.
    // It holds the round's last hold requests for the next round, in held_next.
    void
    release(uint32_t first_id,
            size_t keep_limit,
            size_t hold,
            const std::vector<nixlLibfabricReq *> &held,
            std::vector<nixlLibfabricReq *> &held_next) {
        for (auto it = held.rbegin(); it != held.rend(); ++it) {
            releaseOne(*it);
        }
        std::vector<nixlLibfabricReq *> batch;
        std::vector<nixlLibfabricReq *> kept;
        auto releaseAll = [this](std::vector<nixlLibfabricReq *> &reqs) {
            for (auto it = reqs.rbegin(); it != reqs.rend(); ++it) {
                releaseOne(*it);
            }
            reqs.clear();
        };
        uint32_t expected_id = first_id;
        for (size_t i = 0; i < kAllocationsPerRound; ++i) {
            nixlLibfabricReq *req = nullptr;
            while (!ring_.pop(req)) {
                releaseAll(batch);
                if (failed_.load(std::memory_order_relaxed)) {
                    return;
                }
                std::this_thread::yield();
            }
            // The ring is FIFO, so a request with another id was handed out again in flight
            if (!req->in_use || req->xfer_id != expected_id++) {
                fail("request changed while in flight");
                return;
            }
            if (i >= kAllocationsPerRound - hold) {
                releaseAll(kept); // so that the allocator can still allocate the held requests
                held_next.push_back(req);
            } else if (i % 2 == 1 && keep_limit > 0) {
                kept.push_back(req);
                if (kept.size() >= keep_limit) {
                    releaseAll(kept);
                }
            } else {
                batch.push_back(req);
                if (batch.size() == kReleaseBatch) {
                    releaseAll(batch);
                }
            }
        }
        releaseAll(batch);
        releaseAll(kept);
    }

    void
    releaseOne(nixlLibfabricReq *req) {
        // Before the release: once it is back in the pool, the allocator may take it again
        if (owned_[req->pool_index].exchange(0, std::memory_order_acq_rel) != 1) {
            fail("released a request that was not handed out");
        }
        pool_.release(req);
        released_.fetch_add(1, std::memory_order_release);
    }

    // Reads both counts, which take both locks, while the other two threads run
    void
    observe(const std::atomic<bool> &round_done) {
        while (!round_done.load(std::memory_order_acquire)) {
            const size_t active = pool_.getActiveRequestCount();
            const size_t utilization = pool_.getPoolUtilization();
            if (active > kMaxCapacity || utilization > 100) {
                fail("count out of range while running: " + std::to_string(active) + " active, " +
                     std::to_string(utilization) + "% used");
                return;
            }
            ++observations_;
            std::this_thread::yield();
        }
    }

    DataRequestPool pool_;
    MpscRing<nixlLibfabricReq *> ring_;
    // Per pool index, 1 while the request is handed out: the test's own record, set by the
    // allocator and cleared by the releaser
    std::unique_ptr<std::atomic<uint8_t>[]> owned_;
    std::vector<nixlLibfabricReq *> address_; // per pool index, when first allocated (allocator)
    uint64_t allocated_ = 0; // allocator, and the main thread between rounds
    uint32_t next_id_ = 1; // allocator, and the main thread between rounds
    std::atomic<uint64_t> released_{0}; // releaser, after each release
    size_t observations_ = 0; // observer, and the main thread between rounds
    std::atomic<bool> failed_{false};
    std::mutex error_mutex_;
    std::string first_error_;
};

int
testTwoThreads() {
    NIXL_INFO << "  testTwoThreads";
    Stress stress;
    return stress.run();
}

} // namespace

int
main() {
    NIXL_INFO << "=== Request Pool Test ===";

    int res;
    if ((res = testSwapBeforeExpand()) != 0) {
        return res;
    }
    if ((res = testDoubleRelease()) != 0) {
        return res;
    }
    if ((res = testTwoThreads()) != 0) {
        return res;
    }

    NIXL_INFO << "=== All request pool tests PASSED ===";
    return 0;
}
