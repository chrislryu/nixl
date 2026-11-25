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
 * Unit test for nixlLibfabricSpinLock: mutual exclusion under contention, try_lock(), and the
 * back-off of a long wait. A waiter on a lock held for tens of milliseconds must sleep rather
 * than keep its CPU busy, take the lock soon after the unlock, and count the wait. With the
 * holder and the waiter on one CPU, the waiter must let the holder run.
 */

#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define TEST_ASSERT(cond, msg)                                       \
    do {                                                             \
        if (!(cond)) {                                               \
            NIXL_ERROR << "FAIL: " << (msg) << " (" << #cond << ")"; \
            return 1;                                                \
        }                                                            \
    } while (0)

namespace {

using steady = std::chrono::steady_clock;

double
threadCpuMs() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

double
msSince(steady::time_point t) {
    return std::chrono::duration<double, std::milli>(steady::now() - t).count();
}

// Many threads add to a plain counter under the lock: no increment may be lost
int
testMutualExclusion() {
    nixlLibfabricSpinLock lock;
    constexpr int kThreads = 4;
    constexpr int kIters = 200000;
    uint64_t counter = 0;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < kIters; ++i) {
                std::lock_guard<nixlLibfabricSpinLock> guard(lock);
                ++counter;
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    TEST_ASSERT(counter == uint64_t(kThreads) * kIters, "every increment kept");
    NIXL_INFO << "mutual exclusion: " << counter << " increments, " << lock.waitStats().waits
              << " waits, " << lock.waitStats().sleeps << " sleeps";
    return 0;
}

int
testTryLock() {
    nixlLibfabricSpinLock lock;
    TEST_ASSERT(lock.try_lock(), "try_lock takes a free lock");
    bool other = true;
    std::thread([&] { other = lock.try_lock(); }).join();
    TEST_ASSERT(!other, "try_lock fails on a taken lock");
    lock.unlock();
    std::thread([&] {
        other = lock.try_lock();
        if (other) {
            lock.unlock();
        }
    }).join();
    TEST_ASSERT(other, "try_lock takes the lock once it is free");
    TEST_ASSERT(lock.waitStats().waits == 0, "try_lock never counts a wait");
    return 0;
}

// The holder keeps the lock for kHoldMs while the waiter blocks in lock(). The waiter must sleep
// through most of it (CPU time far below the wait), take the lock soon after the unlock, and count
// one wait with at least one sleep.
int
testLongHoldSleeps() {
    constexpr double kHoldMs = 50;
    nixlLibfabricSpinLock lock;
    lock.lock();
    double waiter_cpu_ms = 0;
    steady::time_point acquired;
    std::thread waiter([&] {
        const double cpu0 = threadCpuMs();
        lock.lock();
        acquired = steady::now();
        waiter_cpu_ms = threadCpuMs() - cpu0;
        lock.unlock();
    });
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(kHoldMs));
    const steady::time_point released = steady::now();
    lock.unlock();
    waiter.join();

    const double late_ms = std::chrono::duration<double, std::milli>(acquired - released).count();
    const nixlLibfabricSpinLock::WaitStats stats = lock.waitStats();
    NIXL_INFO << "long hold: waiter CPU " << waiter_cpu_ms << " ms over a " << kHoldMs
              << " ms hold, took the lock " << late_ms << " ms after the unlock, " << stats.waits
              << " waits, " << stats.sleeps << " sleeps";
    TEST_ASSERT(stats.waits == 1, "the waiter's lock() counted one wait");
    TEST_ASSERT(stats.sleeps >= 1, "the waiter slept");
    // Spinning or yielding through the hold would cost about kHoldMs of CPU
    TEST_ASSERT(waiter_cpu_ms < kHoldMs / 5, "the waiter slept through most of the hold");
    // At most one sleep of 1 ms after the unlock, plus scheduling slack on a loaded host
    TEST_ASSERT(late_ms < 25, "the waiter took the lock soon after the unlock");
    return 0;
}

// Holder and waiter pinned to one CPU. The holder busy-works with the lock held, so it needs the
// CPU the waiter is waiting on; the waiter must give it up, and both must finish.
int
testSameCpu() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        NIXL_INFO << "same CPU: skipped, sched_getaffinity failed";
        return 0;
    }
    int cpu = -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) {
            cpu = c;
            break;
        }
    }
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);

    nixlLibfabricSpinLock lock;
    std::atomic<bool> holding{false};
    std::atomic<bool> pinned_ok{true};
    const steady::time_point start = steady::now();
    std::thread holder([&] {
        if (pthread_setaffinity_np(pthread_self(), sizeof(one), &one) != 0) {
            pinned_ok = false;
        }
        lock.lock();
        holding = true;
        const steady::time_point t = steady::now();
        while (msSince(t) < 20) {
            // busy work with the lock held
        }
        lock.unlock();
    });
    std::thread waiter([&] {
        if (pthread_setaffinity_np(pthread_self(), sizeof(one), &one) != 0) {
            pinned_ok = false;
        }
        while (!holding) {
            std::this_thread::yield();
        }
        lock.lock();
        lock.unlock();
    });
    holder.join();
    waiter.join();
    const double total_ms = msSince(start);
    if (!pinned_ok) {
        NIXL_INFO << "same CPU: skipped, could not pin to CPU " << cpu;
        return 0;
    }
    NIXL_INFO << "same CPU (" << cpu << "): done in " << total_ms << " ms, "
              << lock.waitStats().waits << " waits, " << lock.waitStats().sleeps << " sleeps";
    TEST_ASSERT(total_ms < 2000, "holder and waiter on one CPU both finished");
    return 0;
}

} // namespace

int
main() {
    // A lock that never comes free must fail the test, not hang it
    alarm(120);
    NIXL_INFO << "=== Spin Lock Test ===";
    int res;
    if ((res = testMutualExclusion()) != 0) {
        return res;
    }
    if ((res = testTryLock()) != 0) {
        return res;
    }
    if ((res = testLongHoldSleeps()) != 0) {
        return res;
    }
    if ((res = testSameCpu()) != 0) {
        return res;
    }
    NIXL_INFO << "=== All spin lock tests PASSED ===";
    return 0;
}
