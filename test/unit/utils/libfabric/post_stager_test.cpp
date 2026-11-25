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
 * Unit test for nixlLibfabricPostStager, which stages a thread's progress-thread posts per rail
 * and pushes them to the rails' post rings in batches. Posts go through
 * deferTransferRequest() and prepareAndSubmitTransfer() as the engine sends them, and the test
 * drains the rings with drainPostQueue(), as the progress thread does, into a fake endpoint that
 * records every post. It checks that a rail's posts reach its ring in order, when the stage
 * fills, before and with a post of 64 KiB or more, and when the stager goes out of scope on a
 * return, an early return or an exception; that the ring never ends inside an FI_MORE group
 * whose last post is staged; that threads stage apart; and that a stage larger than its ring
 * still arrives whole. libfabric is mocked out.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

// Rail whose post ring is smaller than a stage
static const size_t SMALL_RING_RAIL = 2;
static const size_t SMALL_RING_SIZE = 8;
static const size_t RING_SIZE = 1024;

static const size_t SMALL = 16 * 1024;
static const size_t LARGE = nixlLibfabricPostStager::DIRECT_POST_MIN_LENGTH;
// A transfer at the default striping threshold is striped over the four rails in chunks
// below 64 KiB
static const size_t CHUNK = NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD / NUM_FAKE_RAILS;

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

// --- Fake CQ: always empty ---

static ssize_t
fi_cq_readfrom_empty_stub(struct fid_cq *, void *, size_t, fi_addr_t *) {
    return -FI_EAGAIN;
}

// --- Fake endpoint RMA ops: record every post ---

// One post, as the endpoint saw it. The test puts a post's id in its local address.
struct Posted {
    const fid_ep *ep;
    uintptr_t id;
    size_t length;
    bool write;
    bool more; // FI_MORE
};

// Written by the thread that drains, read by the test once that thread is done
static std::vector<Posted> posted;

static ssize_t
fi_writemsg_record_stub(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags) {
    posted.push_back({ep,
                      reinterpret_cast<uintptr_t>(msg->msg_iov[0].iov_base),
                      msg->msg_iov[0].iov_len,
                      true,
                      (flags & FI_MORE) != 0});
    return 0;
}

static ssize_t
fi_readmsg_record_stub(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags) {
    posted.push_back({ep,
                      reinterpret_cast<uintptr_t>(msg->msg_iov[0].iov_base),
                      msg->msg_iov[0].iov_len,
                      false,
                      (flags & FI_MORE) != 0});
    return 0;
}

static fi_ops_rma rma_ops_record_stub = {};

// --- Test helpers ---

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// What a test posts, and what it expects to see posted
struct Op {
    uintptr_t id;
    size_t length = SMALL;
    uint64_t fi_flags = 0;
    nixlLibfabricReq::OpType type = nixlLibfabricReq::WRITE;
};

// Builds the requests for the posts and keeps them alive for the whole test
class Poster {
public:
    explicit Poster(nixlLibfabricRailManager &mgr) : mgr_(mgr) {}

    // Posts op on rail_id through deferTransferRequest(), as prepareAndSubmitTransfer() does
    void
    post(size_t rail_id, const Op &op, nixlLibfabricPostStager *stager) {
        nixlLibfabricReq &req = reqs_.emplace_back();
        req.operation_type = op.type;
        req.local_addr = reinterpret_cast<void *>(op.id);
        req.chunk_size = op.length;
        req.local_mr = &mr_;
        req.rail_id = rail_id;
        mgr_.deferTransferRequest(op.type,
                                  /*xfer_id=*/1,
                                  op.fi_flags,
                                  /*dest_addr=*/0,
                                  /*device_id=*/-1,
                                  /*is_cuda_vram=*/false,
                                  rail_id,
                                  &req,
                                  stager);
    }

    // Posts n small writes without FI_MORE on rail_id, with ids from first_id on
    void
    postRun(size_t rail_id, uintptr_t first_id, size_t n, nixlLibfabricPostStager *stager) {
        for (size_t i = 0; i < n; ++i) {
            post(rail_id, Op{first_id + i}, stager);
        }
    }

private:
    nixlLibfabricRailManager &mgr_;
    std::deque<nixlLibfabricReq> reqs_; // a deque keeps their addresses
    fid_mr mr_{};
};

// Drains the rail's post ring as the progress thread does, until it is empty
static std::vector<Posted>
drain(nixlLibfabricRail &rail) {
    std::vector<Posted> out;
    for (;;) {
        posted.clear();
        rail.drainPostQueue();
        if (posted.empty()) {
            return out;
        }
        out.insert(out.end(), posted.begin(), posted.end());
    }
}

static std::vector<Op>
runOf(uintptr_t first_id, size_t n) {
    std::vector<Op> ops;
    for (size_t i = 0; i < n; ++i) {
        ops.push_back(Op{first_id + i});
    }
    return ops;
}

// Empty when got is exactly expected, on rail's endpoint; else what differs
static std::string
mismatch(const nixlLibfabricRail &rail,
         const std::vector<Posted> &got,
         const std::vector<Op> &expected) {
    if (got.size() != expected.size()) {
        return std::to_string(got.size()) + " posts, expected " + std::to_string(expected.size());
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const Posted &g = got[i];
        const Op &e = expected[i];
        if (g.ep != rail.endpoint || g.id != e.id || g.length != e.length ||
            g.write != (e.type == nixlLibfabricReq::WRITE) ||
            g.more != ((e.fi_flags & FI_MORE) != 0)) {
            return "post " + std::to_string(i) + " is id " + std::to_string(g.id) +
                (g.more ? " with" : " without") + " FI_MORE, expected id " + std::to_string(e.id) +
                ((e.fi_flags & FI_MORE) ? " with" : " without") + " FI_MORE";
        }
    }
    return "";
}

// Drains rail and checks that it posted the ops after what: a braced list or a vector of Op
#define EXPECT_POSTS(rail, what, ...)                                       \
    do {                                                                    \
        const std::vector<Op> expected_ = __VA_ARGS__;                      \
        const std::string diff_ = mismatch((rail), drain(rail), expected_); \
        TEST_ASSERT(diff_.empty(), std::string(what) + ": " + diff_);       \
    } while (0)

// --- Tests ---

static int
testStagedUntilScopeEnd(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testStagedUntilScopeEnd";
    Poster poster(mgr);
    nixlLibfabricRail &a = mgr.getRail(0);
    nixlLibfabricRail &b = mgr.getRail(1);

    std::vector<Op> on_a, on_b;
    {
        nixlLibfabricPostStager stager(mgr);
        for (uintptr_t i = 0; i < 10; ++i) {
            on_a.push_back(Op{100 + i});
            poster.post(0, on_a.back(), &stager);
            if (i % 2 == 0) {
                on_b.push_back(Op{200 + i, SMALL, 0, nixlLibfabricReq::READ});
                poster.post(1, on_b.back(), &stager);
            }
        }
        EXPECT_POSTS(a, "rail A while staged", {});
        EXPECT_POSTS(b, "rail B while staged", {});
    }
    EXPECT_POSTS(a, "rail A after the scope", on_a);
    EXPECT_POSTS(b, "rail B after the scope", on_b);

    // Without a stager, a post goes to the ring at once
    poster.postRun(0, 300, 3, nullptr);
    EXPECT_POSTS(a, "without a stager", runOf(300, 3));
    return 0;
}

static int
testFullStage(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testFullStage";
    Poster poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(0);
    const size_t stage = nixlLibfabricPostStager::STAGE_SIZE;
    {
        nixlLibfabricPostStager stager(mgr);
        poster.postRun(0, 1000, stage - 1, &stager);
        EXPECT_POSTS(rail, "a stage short of full", {});
        poster.postRun(0, 1000 + stage - 1, 1, &stager);
        EXPECT_POSTS(rail, "a full stage", runOf(1000, stage));
        poster.postRun(0, 2000, 2 * stage + 10, &stager);
        EXPECT_POSTS(rail, "two more full stages", runOf(2000, 2 * stage));
    }
    EXPECT_POSTS(rail, "the rest, after the scope", runOf(2000 + 2 * stage, 10));
    return 0;
}

static int
testLargeOps(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testLargeOps";
    Poster poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(1);
    {
        nixlLibfabricPostStager stager(mgr);
        std::vector<Op> ops = runOf(1, 5);
        ops.push_back(Op{6, LARGE - 1});
        for (const Op &op : ops) {
            poster.post(1, op, &stager);
        }
        EXPECT_POSTS(rail, "posts below 64 KiB are staged", {});

        // A large op pushes the stage first, then itself
        ops.push_back(Op{7, LARGE});
        poster.post(1, ops.back(), &stager);
        EXPECT_POSTS(rail, "a 64 KiB write", ops);

        ops = runOf(10, 2);
        ops.push_back(Op{12, 1024 * 1024, 0, nixlLibfabricReq::READ});
        for (const Op &op : ops) {
            poster.post(1, op, &stager);
        }
        EXPECT_POSTS(rail, "a 1 MiB read", ops);

        poster.post(1, Op{13, LARGE}, &stager);
        EXPECT_POSTS(rail, "a large op alone", {Op{13, LARGE}});
    }
    EXPECT_POSTS(rail, "nothing left after the scope", {});
    return 0;
}

static int
testFiMoreGroups(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testFiMoreGroups";
    Poster poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(0);
    auto post = [&](const Op &op, nixlLibfabricPostStager &stager) { poster.post(0, op, &stager); };
    {
        nixlLibfabricPostStager stager(mgr);
        // An open group is staged as long as the ring does not end inside it
        std::vector<Op> ops = {Op{1, SMALL, FI_MORE}, Op{2, SMALL, FI_MORE}, Op{3, SMALL, FI_MORE}};
        for (const Op &op : ops) {
            post(op, stager);
        }
        EXPECT_POSTS(rail, "an open group in the stage", {});

        // A large FI_MORE op leaves the ring inside the group: the group's next posts, up to
        // its last, go to the ring at once
        ops.push_back(Op{4, LARGE, FI_MORE});
        post(ops.back(), stager);
        EXPECT_POSTS(rail, "stage and large op, the ring ends inside the group", ops);
        post(Op{5, SMALL, FI_MORE}, stager);
        EXPECT_POSTS(rail, "the group goes on at once", {Op{5, SMALL, FI_MORE}});
        post(Op{6}, stager);
        EXPECT_POSTS(rail, "the group's last post goes at once", {Op{6}});

        // The group is closed, so posts are staged again
        post(Op{7, SMALL, FI_MORE}, stager);
        post(Op{8}, stager);
        EXPECT_POSTS(rail, "the next group is staged", {});
    }
    EXPECT_POSTS(rail, "after the scope", {Op{7, SMALL, FI_MORE}, Op{8}});

    {
        nixlLibfabricPostStager stager(mgr);
        // A full stage that ends inside a group
        std::vector<Op> ops = runOf(100, nixlLibfabricPostStager::STAGE_SIZE - 1);
        ops.push_back(Op{200, SMALL, FI_MORE});
        for (const Op &op : ops) {
            post(op, stager);
        }
        EXPECT_POSTS(rail, "a full stage ending on FI_MORE", ops);
        post(Op{201, SMALL, FI_MORE}, stager);
        EXPECT_POSTS(rail, "the group goes on at once", {Op{201, SMALL, FI_MORE}});
        post(Op{202}, stager);
        EXPECT_POSTS(rail, "the group's last post goes at once", {Op{202}});
        post(Op{203}, stager);
        EXPECT_POSTS(rail, "the next post is staged", {});
    }
    EXPECT_POSTS(rail, "after the scope", {Op{203}});

    {
        // A scope that ends inside a group, as a failed transfer does: the group's posts are
        // pushed, and the next scope stages again
        {
            nixlLibfabricPostStager stager(mgr);
            post(Op{300, LARGE, FI_MORE}, stager);
            EXPECT_POSTS(rail, "large op", {Op{300, LARGE, FI_MORE}});
            post(Op{301, SMALL, FI_MORE}, stager);
        }
        EXPECT_POSTS(rail, "the open group", {Op{301, SMALL, FI_MORE}});
        nixlLibfabricPostStager stager(mgr);
        post(Op{302}, stager);
        EXPECT_POSTS(rail, "a new scope stages", {});
    }
    EXPECT_POSTS(rail, "after the new scope", {Op{302}});
    return 0;
}

static nixl_status_t
postAndFail(Poster &poster, nixlLibfabricRailManager &mgr, size_t n) {
    nixlLibfabricPostStager stager(mgr);
    poster.postRun(3, 1000, n, &stager);
    return NIXL_ERR_BACKEND;
}

static void
postAndThrow(Poster &poster, nixlLibfabricRailManager &mgr, size_t n) {
    nixlLibfabricPostStager stager(mgr);
    poster.postRun(3, 2000, n, &stager);
    throw std::runtime_error("post failed");
}

static int
testEarlyExits(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testEarlyExits";
    Poster poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(3);
    const size_t n = nixlLibfabricPostStager::STAGE_SIZE + 6;

    TEST_ASSERT(postAndFail(poster, mgr, n) == NIXL_ERR_BACKEND, "error returned");
    EXPECT_POSTS(rail, "after an error return", runOf(1000, n));

    bool thrown = false;
    try {
        postAndThrow(poster, mgr, n);
    }
    catch (const std::runtime_error &) {
        thrown = true;
    }
    TEST_ASSERT(thrown, "exception thrown");
    EXPECT_POSTS(rail, "after an exception", runOf(2000, n));
    return 0;
}

static int
testThreads(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testThreads";
    Poster poster(mgr);
    Poster other_poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(3);

    // A stager on another thread keeps its own stage of the same rail
    std::mutex mutex;
    std::condition_variable cv;
    int step = 0;
    auto waitFor = [&](int s) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return step >= s; });
    };
    auto advance = [&](int s) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            step = s;
        }
        cv.notify_all();
    };
    std::thread other([&] {
        {
            nixlLibfabricPostStager stager(mgr);
            other_poster.postRun(3, 5000, 10, &stager);
            advance(1);
            waitFor(2);
        }
        advance(3);
    });
    waitFor(1);
    {
        nixlLibfabricPostStager stager(mgr);
        poster.postRun(3, 6000, 5, &stager);
        EXPECT_POSTS(rail, "staged while the other thread's stager is alive", {});
    }
    EXPECT_POSTS(rail, "this thread's posts, the other thread's still staged", runOf(6000, 5));
    advance(2);
    waitFor(3);
    other.join();
    EXPECT_POSTS(rail, "the other thread's posts, after its scope", runOf(5000, 10));

    // A second stager alive on the same thread posts at once
    {
        nixlLibfabricPostStager outer(mgr);
        poster.postRun(3, 7000, 3, &outer);
        {
            nixlLibfabricPostStager inner(mgr);
            poster.postRun(3, 8000, 2, &inner);
            EXPECT_POSTS(rail, "the inner stager posts at once", runOf(8000, 2));
        }
        EXPECT_POSTS(rail, "the outer stager still stages", {});
    }
    EXPECT_POSTS(rail, "the outer stager's posts", runOf(7000, 3));

    // After both, the thread stages again
    {
        nixlLibfabricPostStager stager(mgr);
        poster.postRun(3, 9000, 4, &stager);
        EXPECT_POSTS(rail, "staged again", {});
    }
    EXPECT_POSTS(rail, "after the scope", runOf(9000, 4));
    return 0;
}

static int
testStageLargerThanRing(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testStageLargerThanRing";
    Poster poster(mgr);
    nixlLibfabricRail &rail = mgr.getRail(SMALL_RING_RAIL);
    const size_t n = 2 * nixlLibfabricPostStager::STAGE_SIZE + 10;

    // The progress thread drains while the stager pushes stages larger than the ring
    std::atomic<bool> timed_out{false};
    std::vector<Posted> got;
    std::thread consumer([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (got.size() < n) {
            if (std::chrono::steady_clock::now() > deadline) {
                timed_out.store(true);
                return;
            }
            posted.clear();
            rail.drainPostQueue();
            got.insert(got.end(), posted.begin(), posted.end());
            std::this_thread::yield();
        }
    });
    {
        nixlLibfabricPostStager stager(mgr);
        poster.postRun(SMALL_RING_RAIL, 1, n, &stager);
    }
    consumer.join();
    TEST_ASSERT(!timed_out.load(),
                "only " + std::to_string(got.size()) + " of " + std::to_string(n) +
                    " posts drained after 10 s");
    const std::string diff = mismatch(rail, got, runOf(1, n));
    TEST_ASSERT(diff.empty(),
                "posts through a ring of " + std::to_string(SMALL_RING_SIZE) + ": " + diff);
    EXPECT_POSTS(rail, "ring empty at the end", {});
    return 0;
}

// A completion sink that nothing completes here: the fake endpoint reports no completions
class IdleSink final : public nixlLibfabricCompletionSink {
public:
    void
    complete_requests(size_t, nixl_status_t) override {}
};

static int
testThroughSubmit(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testThroughSubmit";
    IdleSink sink;
    fid_mr mr{};
    const std::vector<size_t> all_rails = {0, 1, 2, 3};
    const std::vector<fid_mr *> local_mrs(NUM_FAKE_RAILS, &mr);
    const std::vector<uint64_t> remote_keys(NUM_FAKE_RAILS, 7);
    std::unordered_map<size_t, std::vector<fi_addr_t>> dest_addrs;
    for (size_t rail_id : all_rails) {
        dest_addrs[rail_id] = std::vector<fi_addr_t>(NUM_FAKE_RAILS, 0);
    }
    std::vector<char> buffer(NUM_FAKE_RAILS * CHUNK);
    auto submit = [&](size_t size, int desc_idx, nixlLibfabricPostStager *stager) {
        size_t submitted = 0;
        const nixl_status_t status =
            mgr.prepareAndSubmitTransfer(nixlLibfabricReq::WRITE,
                                         buffer.data(),
                                         size,
                                         /*remote_target_addr=*/0x10000,
                                         /*remote_registered_base=*/0x10000,
                                         all_rails,
                                         local_mrs,
                                         remote_keys,
                                         all_rails,
                                         dest_addrs,
                                         /*xfer_id=*/1,
                                         &sink,
                                         submitted,
                                         desc_idx,
                                         /*base_offset=*/0,
                                         /*apply_fi_more=*/false,
                                         /*device_id=*/-1,
                                         /*is_cuda_vram=*/false,
                                         stager);
        return status == NIXL_SUCCESS ? submitted : 0;
    };
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.data());
    const size_t striped = NUM_FAKE_RAILS * CHUNK;
    {
        nixlLibfabricPostStager stager(mgr);
        // One rail, round-robin, then striped across all four in chunks below 64 KiB
        TEST_ASSERT(submit(SMALL, 0, &stager) == 1, "single-rail write submitted");
        TEST_ASSERT(submit(striped, 0, &stager) == NUM_FAKE_RAILS, "striped write submitted");
        for (size_t rail_id : all_rails) {
            EXPECT_POSTS(mgr.getRail(rail_id), "staged on rail " + std::to_string(rail_id), {});
        }
    }
    EXPECT_POSTS(mgr.getRail(0),
                 "rail 0: the single-rail write and the first chunk",
                 {Op{base}, Op{base, CHUNK}});
    for (size_t rail_id = 1; rail_id < NUM_FAKE_RAILS; ++rail_id) {
        EXPECT_POSTS(mgr.getRail(rail_id),
                     "rail " + std::to_string(rail_id) + ": its chunk",
                     {Op{base + rail_id * CHUNK, CHUNK}});
    }

    // Without a stager, both paths post at once
    TEST_ASSERT(submit(SMALL, 0, nullptr) == 1, "single-rail write submitted");
    EXPECT_POSTS(mgr.getRail(0), "single-rail write, no stager", {Op{base}});
    TEST_ASSERT(submit(striped, 0, nullptr) == NUM_FAKE_RAILS, "striped write submitted");
    for (size_t rail_id : all_rails) {
        EXPECT_POSTS(mgr.getRail(rail_id),
                     "striped write, no stager, rail " + std::to_string(rail_id),
                     {Op{base + rail_id * CHUNK, CHUNK}});
    }
    return 0;
}

int
main() {
    NIXL_INFO << "=== Post Stager Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, fi_writemsg, fi_readmsg, ...)";

    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_empty_stub;
    rma_ops_record_stub.size = sizeof(fi_ops_rma);
    rma_ops_record_stub.writemsg = fi_writemsg_record_stub;
    rma_ops_record_stub.readmsg = fi_readmsg_record_stub;

    nixlLibfabricRailManager mgr(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");
    for (size_t rail_id = 0; rail_id < NUM_FAKE_RAILS; ++rail_id) {
        nixlLibfabricRail &rail = mgr.getRail(rail_id);
        rail.setProgressThreadEnabled(true);
        TEST_ASSERT(rail.initPostQueue(rail_id == SMALL_RING_RAIL ? SMALL_RING_SIZE : RING_SIZE),
                    "post queue initialized");
        rail.endpoint->rma = &rma_ops_record_stub;
    }

    int res;
    if ((res = testStagedUntilScopeEnd(mgr)) != 0) {
        return res;
    }
    if ((res = testFullStage(mgr)) != 0) {
        return res;
    }
    if ((res = testLargeOps(mgr)) != 0) {
        return res;
    }
    if ((res = testFiMoreGroups(mgr)) != 0) {
        return res;
    }
    if ((res = testEarlyExits(mgr)) != 0) {
        return res;
    }
    if ((res = testThreads(mgr)) != 0) {
        return res;
    }
    if ((res = testStageLargerThanRing(mgr)) != 0) {
        return res;
    }
    if ((res = testThroughSubmit(mgr)) != 0) {
        return res;
    }

    NIXL_INFO << "=== All post stager tests PASSED ===";
    return 0;
}
