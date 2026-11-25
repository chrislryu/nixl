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
 * Unit test for how drainPostQueue() takes a rail's endpoint lock (setGroupEpLock()). With the
 * group lock on, the progress thread keeps the lock across an FI_MORE group of posts; with it
 * off, it takes the lock for every post. Either way it must release the lock before a CQ read
 * (pollForCompletions() takes it again), before it fails a request and when a post returns
 * FI_EAGAIN, and it must make every post once, in ring order, with its flags.
 *
 * A fake endpoint records every post attempt. It returns FI_EAGAIN while its send queue is full
 * and fails chosen posts; a fake CQ completes the queued posts. A probe thread takes the lock
 * through insertAddress() while a group is being posted, and a completion sink takes it when a
 * failed post is reported. Every fake call runs under the lock, so the record shows the order
 * in which the threads held it. A watchdog fails the test if a step hangs: a drain that still
 * holds the lock where it takes it again, or where another thread waits for it, never returns.
 * libfabric is mocked out.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Number of fake EFA devices to create. The tests post on rail 0.
static const size_t NUM_FAKE_RAILS = 2;
static const size_t RING_SIZE = 1024;
static const size_t SMALL = 16 * 1024;
static const size_t GROUP = NIXL_LIBFABRIC_FI_MORE_BATCH_SIZE;

// --- Unconditional __wrap_* functions ---

extern "C" int
__wrap_numa_max_node() {
    return 1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 2;
}

extern "C" int
__wrap_fi_getinfo(uint32_t /*version*/,
                  const char * /*node*/,
                  const char * /*service*/,
                  uint64_t /*flags*/,
                  const struct fi_info * /*hints*/,
                  struct fi_info **info) {
    *info = mock_fi_info_chain(NUM_FAKE_RAILS, 100ull * NIXL_LIBFABRIC_GIGA);
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr * /*attr*/, struct fid_fabric **fabric, void * /*context*/) {
    *fabric = mock_fabric_create();
    return 0;
}

// --- Watchdog ---

static std::atomic<const char *> watched_step{"setup"};
static std::atomic<bool> watched_group_lock{false};
static std::atomic<int64_t> watch_deadline_ns{INT64_MAX};
static std::atomic<bool> watchdog_stop{false};
static const int64_t STEP_TIMEOUT_NS = 30ll * 1000 * 1000 * 1000;

static int64_t
nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Starts the deadline of a step; step must be a string literal
static void
watch(const char *step, bool group_lock) {
    watched_step.store(step);
    watched_group_lock.store(group_lock);
    watch_deadline_ns.store(nowNs() + STEP_TIMEOUT_NS);
}

static void
watchdogMain() {
    while (!watchdog_stop.load()) {
        if (nowNs() > watch_deadline_ns.load()) {
            std::cerr << "FAIL: " << watched_step.load() << " (group lock "
                      << (watched_group_lock.load() ? "on" : "off")
                      << ") did not finish in 30 s: a thread waits for the endpoint lock that "
                         "drainPostQueue() still holds"
                      << std::endl;
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// --- Record of what ran under the endpoint lock ---

enum class Ev {
    POST, // fi_writemsg() or fi_readmsg()
    CQ_READ, // fi_cq_readfrom()
    PROBE, // fi_av_insert(), from insertAddress() on another thread or in a sink
    DRAIN_END, // drainPostQueue() returned (recorded by the test, without the lock)
};

struct Event {
    Ev kind;
    uintptr_t id = 0; // POST: the post's id, which the test puts in its local address
    bool more = false; // POST: FI_MORE
    bool write = false; // POST: fi_writemsg()
    ssize_t ret = 0; // POST: what the endpoint returned
};

static std::mutex events_mutex;
static std::vector<Event> events;

static void
record(const Event &ev) {
    const std::lock_guard<std::mutex> lock(events_mutex);
    events.push_back(ev);
}

static std::vector<Event>
recorded() {
    const std::lock_guard<std::mutex> lock(events_mutex);
    return events;
}

static size_t
postsDone() {
    const std::lock_guard<std::mutex> lock(events_mutex);
    size_t n = 0;
    for (const Event &ev : events) {
        n += ev.kind == Ev::POST && ev.ret != -FI_EAGAIN;
    }
    return n;
}

// --- Probe: takes the endpoint lock from another thread ---

static char probe_name[LF_EP_NAME_MAX_LEN] = {};
static std::atomic<uintptr_t> probe_trigger{0};
static std::atomic<bool> probe_go{false};
static std::atomic<bool> probe_waiting{false};

static int
fi_av_insert_probe_stub(struct fid_av *,
                        const void *,
                        size_t count,
                        fi_addr_t *fi_addr,
                        uint64_t,
                        void *) {
    record({Ev::PROBE});
    for (size_t i = 0; i < count; ++i) {
        fi_addr[i] = 0;
    }
    return static_cast<int>(count);
}

// Once the post with id trigger has started, takes the lock with insertAddress()
class Probe {
public:
    Probe(const nixlLibfabricRail &rail, uintptr_t trigger) : rail_(rail) {
        probe_go.store(false);
        probe_waiting.store(false);
        probe_trigger.store(trigger);
        thread_ = std::thread([this] {
            while (!probe_go.load()) {
                std::this_thread::yield();
            }
            probe_waiting.store(true);
            fi_addr_t addr;
            status_ = rail_.insertAddress(probe_name, 0, &addr);
        });
    }

    ~Probe() {
        join();
    }

    void
    join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    nixl_status_t
    status() const {
        return status_;
    }

private:
    const nixlLibfabricRail &rail_;
    std::thread thread_;
    nixl_status_t status_ = NIXL_ERR_UNKNOWN;
};

// Called by the post that starts the probe, under the lock: waits until the probe is about to
// take the lock (it needs no lock for that), then gives it time to start waiting for it
static void
startProbe(uintptr_t id) {
    if (probe_trigger.load() != id) {
        return;
    }
    probe_trigger.store(0);
    probe_go.store(true);
    while (!probe_waiting.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// --- Fake endpoint and CQ ---

// A post that went out, completed by the next CQ read
struct Queued {
    void *context;
    bool write;
};

// Used under the endpoint lock, or by the test while no drain runs
static size_t sq_depth = SIZE_MAX; // posts the send queue holds before FI_EAGAIN
static std::deque<Queued> sq;
static std::set<uintptr_t> fail_ids; // posts that fail with FI_EIO
static std::map<uintptr_t, size_t> extra_eagain; // FI_EAGAIN returns of a post, before the rest

static ssize_t
fakePost(const struct fi_msg_rma *msg, uint64_t flags, bool write) {
    const uintptr_t id = reinterpret_cast<uintptr_t>(msg->msg_iov[0].iov_base);
    ssize_t ret = 0;
    auto extra = extra_eagain.find(id);
    if (fail_ids.count(id) != 0) {
        ret = -FI_EIO;
    } else if (extra != extra_eagain.end() && extra->second > 0) {
        --extra->second;
        ret = -FI_EAGAIN;
    } else if (sq.size() >= sq_depth) {
        ret = -FI_EAGAIN;
    } else {
        sq.push_back({msg->context, write});
    }
    record({Ev::POST, id, (flags & FI_MORE) != 0, write, ret});
    startProbe(id);
    return ret;
}

static ssize_t
fi_writemsg_fake(struct fid_ep *, const struct fi_msg_rma *msg, uint64_t flags) {
    return fakePost(msg, flags, true);
}

static ssize_t
fi_readmsg_fake(struct fid_ep *, const struct fi_msg_rma *msg, uint64_t flags) {
    return fakePost(msg, flags, false);
}

static fi_ops_rma rma_ops_fake = {};

static ssize_t
fi_cq_readfrom_fake(struct fid_cq *, void *buf, size_t count, fi_addr_t *src_addr) {
    record({Ev::CQ_READ});
    if (sq.empty()) {
        return -FI_EAGAIN;
    }
    auto *entries = static_cast<struct fi_cq_data_entry *>(buf);
    size_t n = 0;
    while (n < count && !sq.empty()) {
        entries[n] = {};
        entries[n].op_context = sq.front().context;
        entries[n].flags = (sq.front().write ? FI_WRITE : FI_READ) | FI_RMA;
        entries[n].len = SMALL;
        if (src_addr != nullptr) {
            src_addr[n] = FI_ADDR_NOTAVAIL;
        }
        sq.pop_front();
        ++n;
    }
    return static_cast<ssize_t>(n);
}

// --- Test helpers ---

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// Counts a scenario's completions. A failed post is reported from drainPostQueue(), which
// must have released the lock by then: the sink takes it with insertAddress(), which would
// spin forever (and trip the watchdog) on a lock this thread still holds.
class CountingSink final : public nixlLibfabricCompletionSink {
public:
    explicit CountingSink(const nixlLibfabricRail &rail) : rail_(rail) {}

    void
    complete_requests(size_t count, nixl_status_t status) override {
        if (status == NIXL_SUCCESS) {
            completed += count;
            return;
        }
        failed += count;
        fi_addr_t addr;
        lock_taken_on_failure &= rail_.insertAddress(probe_name, 0, &addr) == NIXL_SUCCESS;
    }

    size_t completed = 0;
    size_t failed = 0;
    bool lock_taken_on_failure = true;

private:
    const nixlLibfabricRail &rail_;
};

// A post of a scenario, in ring order
struct Post {
    uintptr_t id;
    bool more;
    bool write = true;
};

// n WRITEs from first_id, all but the last with FI_MORE, as prepareAndSubmitTransfer() groups
// them
static std::vector<Post>
group(uintptr_t first_id, size_t n) {
    std::vector<Post> posts;
    for (size_t i = 0; i < n; ++i) {
        posts.push_back({first_id + i, i + 1 < n});
    }
    return posts;
}

static std::vector<Post>
reads(uintptr_t first_id, size_t n) {
    std::vector<Post> posts;
    for (size_t i = 0; i < n; ++i) {
        posts.push_back({first_id + i, false, false});
    }
    return posts;
}

static std::vector<Post>
concat(std::initializer_list<std::vector<Post>> parts) {
    std::vector<Post> posts;
    for (const auto &part : parts) {
        posts.insert(posts.end(), part.begin(), part.end());
    }
    return posts;
}

struct Scenario {
    const char *name; // a string literal
    std::vector<Post> posts;
    size_t sq_depth = SIZE_MAX;
    std::set<uintptr_t> fail_ids;
    std::map<uintptr_t, size_t> extra_eagain;
    uintptr_t probe_at; // the post that starts the probe
    size_t split = 0; // if not 0: the ring gets posts[0, split) first, drained before the rest
};

static std::string
idOf(const Event &ev) {
    return std::to_string(ev.id);
}

// Every event but the probes': what the draining thread did, in order
static std::vector<Event>
drainEvents(const std::vector<Event> &evs) {
    std::vector<Event> out;
    for (const Event &ev : evs) {
        if (ev.kind != Ev::PROBE) {
            out.push_back(ev);
        }
    }
    return out;
}

// Empty when every post was made once, in ring order, with its flags, and each FI_EAGAIN was
// followed by a CQ read and then by a retry of the same post; else what differs
static std::string
checkPosts(const std::vector<Event> &evs, const std::vector<Post> &expected) {
    const std::vector<Event> d = drainEvents(evs);
    size_t next = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        if (d[i].kind != Ev::POST) {
            continue;
        }
        if (next >= expected.size()) {
            return "unexpected post " + idOf(d[i]);
        }
        const Post &e = expected[next];
        if (d[i].id != e.id || d[i].more != e.more || d[i].write != e.write) {
            return "attempt " + std::to_string(i) + " posted id " + idOf(d[i]) +
                (d[i].more ? " with" : " without") + " FI_MORE, expected id " +
                std::to_string(e.id) + (e.more ? " with" : " without") + " FI_MORE";
        }
        if (d[i].ret == -FI_EAGAIN) {
            if (i + 1 >= d.size() || d[i + 1].kind != Ev::CQ_READ) {
                return "post " + idOf(d[i]) + " got FI_EAGAIN without a CQ read after it";
            }
            continue; // the next attempt must retry the same post
        }
        ++next;
    }
    if (next != expected.size()) {
        return std::to_string(next) + " of " + std::to_string(expected.size()) + " posts made";
    }
    return "";
}

// Empty when no probe took the lock inside a group: between a post with FI_MORE that went out
// and the drain's next post, with no CQ read and no drain end between them
static std::string
checkNoProbeInsideGroup(const std::vector<Event> &evs) {
    for (size_t p = 0; p < evs.size(); ++p) {
        if (evs[p].kind != Ev::PROBE) {
            continue;
        }
        const Event *prev = nullptr;
        for (size_t i = p; i-- > 0;) {
            if (evs[i].kind != Ev::PROBE) {
                prev = &evs[i];
                break;
            }
        }
        const Event *next = nullptr;
        for (size_t i = p + 1; i < evs.size(); ++i) {
            if (evs[i].kind != Ev::PROBE) {
                next = &evs[i];
                break;
            }
        }
        if (prev != nullptr && next != nullptr && prev->kind == Ev::POST && prev->ret == 0 &&
            prev->more && next->kind == Ev::POST) {
            return "another thread took the lock between post " + idOf(*prev) +
                " (FI_MORE) and post " + idOf(*next);
        }
    }
    return "";
}

static std::vector<nixlLibfabricReq *>
enqueue(nixlLibfabricRail &rail,
        const std::vector<Post> &posts,
        size_t from,
        size_t to,
        CountingSink &sink) {
    std::vector<nixlLibfabricReq *> reqs;
    for (size_t i = from; i < to; ++i) {
        const Post &p = posts[i];
        const auto op = p.write ? nixlLibfabricReq::WRITE : nixlLibfabricReq::READ;
        nixlLibfabricReq *req = rail.allocateDataRequest(op, /*req_id=*/1);
        if (req == nullptr) {
            return {};
        }
        req->completion_sink = &sink;
        nixlLibfabricPostRequest pr{};
        pr.type = p.write ? nixlLibfabricPostRequest::WRITE : nixlLibfabricPostRequest::READ;
        pr.local_addr = reinterpret_cast<void *>(p.id);
        pr.length = SMALL;
        pr.req = req;
        pr.fi_flags = p.more ? FI_MORE : 0;
        pr.device_id = -1;
        pr.is_cuda_vram = false;
        rail.enqueuePost(pr);
        reqs.push_back(req);
    }
    return reqs;
}

// Drains the ring as the progress thread does until done posts went out or failed
static void
drainUntil(nixlLibfabricRail &rail, size_t done) {
    for (size_t i = 0; i < 1000 && postsDone() < done; ++i) {
        rail.drainPostQueue();
        record({Ev::DRAIN_END});
    }
}

// Runs sc on rail 0 with the group lock on or off and checks what every mode must do. Returns
// the record in evs.
static int
runScenario(nixlLibfabricRailManager &mgr,
            const Scenario &sc,
            bool group_lock,
            std::vector<Event> &evs) {
    nixlLibfabricRail &rail = mgr.getRail(0);
    watch(sc.name, group_lock);
    {
        const std::lock_guard<std::mutex> lock(events_mutex);
        events.clear();
    }
    sq.clear();
    sq_depth = sc.sq_depth;
    fail_ids = sc.fail_ids;
    extra_eagain = sc.extra_eagain;
    rail.setGroupEpLock(group_lock);

    CountingSink sink(rail);
    const size_t n = sc.posts.size();
    const size_t split = sc.split != 0 ? sc.split : n;
    TEST_ASSERT(enqueue(rail, sc.posts, 0, split, sink).size() == split, "requests allocated");
    {
        Probe probe(rail, sc.probe_at);
        drainUntil(rail, split);
        if (split < n) {
            // The drain returned with the ring ending inside a group: the lock is free
            fi_addr_t addr;
            TEST_ASSERT(rail.insertAddress(probe_name, 0, &addr) == NIXL_SUCCESS,
                        std::string(sc.name) + ": lock taken after a drain ended in a group");
            TEST_ASSERT(enqueue(rail, sc.posts, split, n, sink).size() == n - split,
                        "requests allocated");
            drainUntil(rail, n);
        }
        probe.join();
        TEST_ASSERT(probe.status() == NIXL_SUCCESS, std::string(sc.name) + ": probe took the lock");
    }
    evs = recorded();

    const std::string posts_diff = checkPosts(evs, sc.posts);
    TEST_ASSERT(posts_diff.empty(), std::string(sc.name) + ": " + posts_diff);
    if (group_lock) {
        const std::string probe_diff = checkNoProbeInsideGroup(evs);
        TEST_ASSERT(probe_diff.empty(), std::string(sc.name) + ": " + probe_diff);
    }

    // Complete what went out, and check that every request was reported and released once
    for (size_t i = 0; i < 100 && !sq.empty(); ++i) {
        rail.progressCompletionQueue();
    }
    TEST_ASSERT(sq.empty(), std::string(sc.name) + ": every post completed");
    TEST_ASSERT(sink.failed == sc.fail_ids.size(), std::string(sc.name) + ": failures reported");
    TEST_ASSERT(sink.completed + sink.failed == n,
                std::string(sc.name) + ": every request reported once");
    TEST_ASSERT(sink.lock_taken_on_failure,
                std::string(sc.name) + ": lock taken while a failure is reported");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0,
                std::string(sc.name) + ": every request released");
    return 0;
}

// Runs sc with the group lock on and off: both must make the same calls in the same order
static int
runBothModes(nixlLibfabricRailManager &mgr, const Scenario &sc, std::vector<Event> &evs_on) {
    NIXL_INFO << "  " << sc.name;
    std::vector<Event> evs_off;
    if (runScenario(mgr, sc, true, evs_on) != 0 || runScenario(mgr, sc, false, evs_off) != 0) {
        return 1;
    }
    const std::vector<Event> on = drainEvents(evs_on);
    const std::vector<Event> off = drainEvents(evs_off);
    TEST_ASSERT(on.size() == off.size(),
                std::string(sc.name) + ": " + std::to_string(on.size()) + " calls with the " +
                    "group lock, " + std::to_string(off.size()) + " without");
    for (size_t i = 0; i < on.size(); ++i) {
        TEST_ASSERT(on[i].kind == off[i].kind && on[i].id == off[i].id &&
                        on[i].more == off[i].more && on[i].ret == off[i].ret,
                    std::string(sc.name) + ": call " + std::to_string(i) +
                        " differs between the modes");
    }
    return 0;
}

// --- Tests ---

static int
testGroups(nixlLibfabricRailManager &mgr) {
    // Two groups with READs between them; the probe waits from the first group's first post
    const Scenario sc{"groups and reads",
                      concat({group(100, GROUP), reads(150, 4), group(200, GROUP)}),
                      SIZE_MAX,
                      {},
                      {},
                      /*probe_at=*/100};
    std::vector<Event> evs;
    return runBothModes(mgr, sc, evs);
}

static int
testEagainInGroup(nixlLibfabricRailManager &mgr) {
    // A send queue of 5 returns FI_EAGAIN on posts with FI_MORE inside the group. Post 307 gets
    // it twice more, and the group's last post once.
    const uintptr_t last = 300 + GROUP - 1;
    const Scenario sc{"FI_EAGAIN inside a group",
                      group(300, GROUP),
                      /*sq_depth=*/5,
                      {},
                      {{307, 2}, {last, 1}},
                      /*probe_at=*/300};
    std::vector<Event> evs;
    if (runBothModes(mgr, sc, evs) != 0) {
        return 1;
    }
    size_t eagain_more = 0, eagain_307 = 0, eagain_last = 0;
    for (const Event &ev : evs) {
        if (ev.kind == Ev::POST && ev.ret == -FI_EAGAIN) {
            eagain_more += ev.more;
            eagain_307 += ev.id == 307;
            eagain_last += ev.id == last;
        }
    }
    TEST_ASSERT(eagain_more >= 3 && eagain_307 >= 2 && eagain_last == 1,
                "FI_EAGAIN on posts with FI_MORE and on the group's last post");
    return 0;
}

static int
testFailureInGroup(nixlLibfabricRailManager &mgr) {
    // Post 403 fails inside the group; the sink takes the lock as it is reported
    const Scenario sc{"failed post inside a group",
                      group(400, GROUP),
                      SIZE_MAX,
                      {403},
                      {},
                      /*probe_at=*/400};
    std::vector<Event> evs;
    return runBothModes(mgr, sc, evs);
}

static int
testRingEndsInGroup(nixlLibfabricRailManager &mgr) {
    // The ring holds the first 6 posts of a group, then the rest
    Scenario sc{"ring ends inside a group", group(500, GROUP), SIZE_MAX, {}, {}, 500};
    sc.split = 6;
    std::vector<Event> evs;
    return runBothModes(mgr, sc, evs);
}

static int
testLongRun(nixlLibfabricRailManager &mgr) {
    // A run of FI_MORE posts longer than a drain: the post/poll interleave reads the CQ, and the
    // drain limit ends a drain, both between two posts with FI_MORE. The limit counts pops and
    // the interleave counts posts that went out, so the failed post 1005 keeps the limit from
    // falling on an interleave read.
    const Scenario sc{
        "FI_MORE run longer than a drain", group(1000, 300), SIZE_MAX, {1005}, {}, 1000};
    std::vector<Event> evs;
    if (runBothModes(mgr, sc, evs) != 0) {
        return 1;
    }
    const std::vector<Event> d = drainEvents(evs);
    bool read_in_run = false, drain_end_in_run = false;
    for (size_t i = 1; i + 1 < d.size(); ++i) {
        const bool in_run = d[i - 1].kind == Ev::POST && d[i - 1].more && d[i - 1].ret == 0 &&
            d[i + 1].kind == Ev::POST;
        read_in_run |= in_run && d[i].kind == Ev::CQ_READ;
        drain_end_in_run |= in_run && d[i].kind == Ev::DRAIN_END;
    }
    TEST_ASSERT(read_in_run, "the post/poll interleave read the CQ between two FI_MORE posts");
    TEST_ASSERT(drain_end_in_run, "a drain ended between two FI_MORE posts");
    return 0;
}

static int
testSubmitFlags(nixlLibfabricRailManager &mgr) {
    // prepareAndSubmitTransfer() sets FI_MORE on WRITEs only: a READ keeps the per-post lock
    NIXL_INFO << "  submitted flags";
    watch("submitted flags", true);
    nixlLibfabricRail &rail = mgr.getRail(0);
    {
        const std::lock_guard<std::mutex> lock(events_mutex);
        events.clear();
    }
    sq.clear();
    sq_depth = SIZE_MAX;
    fail_ids.clear();
    extra_eagain.clear();
    rail.setGroupEpLock(true);

    CountingSink sink(rail);
    fid_mr mr{};
    const std::vector<size_t> rails = {0};
    const std::vector<fid_mr *> local_mrs(NUM_FAKE_RAILS, &mr);
    const std::vector<uint64_t> remote_keys(NUM_FAKE_RAILS, 7);
    const std::unordered_map<size_t, std::vector<fi_addr_t>> dest_addrs = {{0, {0}}};
    std::vector<char> buffer(SMALL);
    auto submit = [&](nixlLibfabricReq::OpType op, bool apply_fi_more) {
        size_t submitted = 0;
        const nixl_status_t status =
            mgr.prepareAndSubmitTransfer(op,
                                         buffer.data(),
                                         SMALL,
                                         /*remote_target_addr=*/0x10000,
                                         /*remote_registered_base=*/0x10000,
                                         rails,
                                         local_mrs,
                                         remote_keys,
                                         rails,
                                         dest_addrs,
                                         /*xfer_id=*/1,
                                         &sink,
                                         submitted,
                                         /*desc_idx=*/0,
                                         /*base_offset=*/0,
                                         apply_fi_more,
                                         /*device_id=*/-1,
                                         /*is_cuda_vram=*/false,
                                         /*stager=*/nullptr);
        return status == NIXL_SUCCESS && submitted == 1;
    };
    TEST_ASSERT(submit(nixlLibfabricReq::WRITE, true), "WRITE with FI_MORE submitted");
    TEST_ASSERT(submit(nixlLibfabricReq::READ, true), "READ asked for FI_MORE submitted");
    TEST_ASSERT(submit(nixlLibfabricReq::WRITE, false), "WRITE without FI_MORE submitted");
    drainUntil(rail, 3);
    const uintptr_t id = reinterpret_cast<uintptr_t>(buffer.data());
    const std::string diff =
        checkPosts(recorded(), {Post{id, true}, Post{id, false, false}, Post{id, false}});
    TEST_ASSERT(diff.empty(), "submitted flags: " + diff);

    for (size_t i = 0; i < 100 && !sq.empty(); ++i) {
        rail.progressCompletionQueue();
    }
    TEST_ASSERT(sink.completed == 3, "submitted posts completed");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "submitted requests released");
    return 0;
}

int
main() {
    NIXL_INFO << "=== Group EP Lock Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, fi_writemsg, fi_cq_readfrom, ...)";
    std::thread watchdog(watchdogMain);

    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_fake;
    av_ops_stub.insert = fi_av_insert_probe_stub;
    rma_ops_fake.size = sizeof(fi_ops_rma);
    rma_ops_fake.writemsg = fi_writemsg_fake;
    rma_ops_fake.readmsg = fi_readmsg_fake;

    int res = 0;
    {
        watch("setup", true);
        nixlLibfabricRailManager mgr(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
        if (mgr.getNumRails() != NUM_FAKE_RAILS) {
            std::cerr << "FAIL: expected " << NUM_FAKE_RAILS << " rails" << std::endl;
            res = 1;
        }
        for (size_t rail_id = 0; res == 0 && rail_id < NUM_FAKE_RAILS; ++rail_id) {
            nixlLibfabricRail &rail = mgr.getRail(rail_id);
            rail.setProgressThreadEnabled(true);
            if (!rail.initPostQueue(RING_SIZE)) {
                std::cerr << "FAIL: post queue initialized" << std::endl;
                res = 1;
            }
            rail.endpoint->rma = &rma_ops_fake;
        }
        for (auto test : {testGroups,
                          testEagainInGroup,
                          testFailureInGroup,
                          testRingEndsInGroup,
                          testLongRun,
                          testSubmitFlags}) {
            if (res == 0) {
                res = test(mgr);
            }
        }
        watch("teardown", true);
    }

    watchdog_stop.store(true);
    watchdog.join();
    if (res == 0) {
        NIXL_INFO << "=== All group EP lock tests PASSED ===";
    }
    return res;
}
