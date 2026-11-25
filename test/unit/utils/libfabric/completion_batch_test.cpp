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
 * Unit test for how a rail reports data completions to their completion sinks. The successful
 * completions of one CQ read are counted per run of the same sink and reported with one
 * complete_requests() call per run, after the run's requests are released. A failed request
 * (an error entry in the CQ, or a post that fails in drainPostQueue()) is reported on its own
 * with its error. libfabric is mocked out, with a fake CQ and endpoint.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <array>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

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

// --- Fake completion queue ---
//
// Like a provider's CQ, a read returns the successful entries in front of the first error
// entry, and -FI_EAVAIL while an error entry is at the head, which fi_cq_readerr() then takes.

struct FakeEntry {
    bool is_error;
    struct fi_cq_data_entry entry;
    struct fi_cq_err_entry err;
    fi_addr_t src_addr;
};

static std::deque<FakeEntry> cq_entries;

static ssize_t
fi_cq_readfrom_stub(struct fid_cq * /*cq*/, void *buf, size_t count, fi_addr_t *src_addr) {
    if (cq_entries.empty()) {
        return -FI_EAGAIN;
    }
    if (cq_entries.front().is_error) {
        return -FI_EAVAIL;
    }
    auto *entries = static_cast<struct fi_cq_data_entry *>(buf);
    size_t n = 0;
    while (n < count && !cq_entries.empty() && !cq_entries.front().is_error) {
        entries[n] = cq_entries.front().entry;
        if (src_addr != nullptr) {
            src_addr[n] = cq_entries.front().src_addr;
        }
        cq_entries.pop_front();
        ++n;
    }
    return static_cast<ssize_t>(n);
}

static ssize_t
fi_cq_readerr_stub(struct fid_cq * /*cq*/, struct fi_cq_err_entry *buf, uint64_t /*flags*/) {
    if (cq_entries.empty() || !cq_entries.front().is_error) {
        return -FI_EAGAIN;
    }
    *buf = cq_entries.front().err;
    cq_entries.pop_front();
    return 1;
}

// --- Fake endpoint RMA ops: every post fails ---

static size_t failed_posts = 0;

static ssize_t
fi_writemsg_fail_stub(struct fid_ep * /*ep*/, const struct fi_msg_rma * /*msg*/, uint64_t) {
    ++failed_posts;
    return -FI_EIO;
}

static ssize_t
fi_readmsg_fail_stub(struct fid_ep * /*ep*/, const struct fi_msg_rma * /*msg*/, uint64_t) {
    ++failed_posts;
    return -FI_EIO;
}

static fi_ops_rma rma_ops_fail_stub = {};

// --- Recording sink ---

// One complete_requests() call, as the sink saw it
struct SinkCall {
    std::string sink;
    size_t count;
    nixl_status_t status;
    size_t active_data_requests; // the rail's, when the call was made
};

// The calls of every sink, in order
static std::vector<SinkCall> sink_calls;

// Stands in for a transfer handle with a known number of requests. With
// delete_when_complete it frees itself as soon as its last request is counted, as an
// application may release a handle once is_completed() is true; any later use of it by the
// rail is then a use after free, which a sanitizer build reports.
class RecordingSink final : public nixlLibfabricCompletionSink {
public:
    RecordingSink(std::string name,
                  const nixlLibfabricRail &rail,
                  size_t requests,
                  bool delete_when_complete = false)
        : name_(std::move(name)),
          rail_(rail),
          requests_(requests),
          delete_when_complete_(delete_when_complete) {}

    void
    complete_requests(size_t count, nixl_status_t status) override {
        if (completed_ >= requests_) {
            ++calls_after_completion_;
        }
        sink_calls.push_back({name_, count, status, rail_.getActiveDataRequestCount()});
        completed_ += count;
        if (delete_when_complete_ && completed_ == requests_) {
            delete this;
        }
    }

    bool
    isComplete() const {
        return completed_ == requests_;
    }

    size_t
    callsAfterCompletion() const {
        return calls_after_completion_;
    }

private:
    const std::string name_;
    const nixlLibfabricRail &rail_;
    const size_t requests_;
    const bool delete_when_complete_;
    size_t completed_ = 0;
    size_t calls_after_completion_ = 0;
};

// --- Test helpers ---

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static void
reset() {
    cq_entries.clear();
    sink_calls.clear();
    failed_posts = 0;
}

// Allocates a data request for sink, as prepareAndSubmitTransfer() does
static nixlLibfabricReq *
allocateData(nixlLibfabricRail &rail,
             nixlLibfabricReq::OpType op_type,
             nixlLibfabricCompletionSink *sink) {
    nixlLibfabricReq *req = rail.allocateDataRequest(op_type, /*req_id=*/1);
    if (req != nullptr) {
        req->completion_sink = sink;
    }
    return req;
}

// The successful completion the provider reports for req
static FakeEntry
successEntry(const nixlLibfabricReq *req) {
    FakeEntry fe{};
    fe.is_error = false;
    fe.entry.op_context = const_cast<fi_context *>(&req->ctx);
    fe.entry.flags = (req->operation_type == nixlLibfabricReq::READ ? FI_READ : FI_WRITE) | FI_RMA;
    fe.entry.len = 16384;
    fe.src_addr = FI_ADDR_NOTAVAIL;
    return fe;
}

// The error completion the provider reports for req
static FakeEntry
errorEntry(const nixlLibfabricReq *req) {
    FakeEntry fe{};
    fe.is_error = true;
    fe.err.op_context = const_cast<fi_context *>(&req->ctx);
    fe.err.err = FI_EIO;
    fe.src_addr = FI_ADDR_NOTAVAIL;
    return fe;
}

static bool
callIs(const SinkCall &call,
       const std::string &sink,
       size_t count,
       nixl_status_t status,
       size_t active_data_requests) {
    return call.sink == sink && call.count == count && call.status == status &&
        call.active_data_requests == active_data_requests;
}

// --- Tests ---

static int
testOneSinkOneCall(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testOneSinkOneCall";
    reset();

    nixlLibfabricRail &rail = mgr.getRail(0);
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request active at the start");

    const size_t n = NIXL_LIBFABRIC_CQ_BATCH_SIZE / 2 + 8;
    RecordingSink sink("A", rail, n);
    for (size_t i = 0; i < n; ++i) {
        nixlLibfabricReq *req = allocateData(rail, nixlLibfabricReq::WRITE, &sink);
        TEST_ASSERT(req != nullptr, "data request allocated");
        cq_entries.push_back(successEntry(req));
    }
    TEST_ASSERT(rail.getActiveDataRequestCount() == n, "all requests active before progress");

    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "completions processed");
    TEST_ASSERT(sink_calls.size() == 1, "N completions of one sink give one call");
    TEST_ASSERT(callIs(sink_calls[0], "A", n, NIXL_SUCCESS, 0),
                "the call counts all N, after all N requests are released");
    TEST_ASSERT(sink.isComplete(), "sink complete");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request left active");
    TEST_ASSERT(cq_entries.empty(), "CQ drained");
    return 0;
}

static int
testOneCallPerRead(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testOneCallPerRead";
    reset();

    nixlLibfabricRail &rail = mgr.getRail(0);
    const size_t first = NIXL_LIBFABRIC_CQ_BATCH_SIZE;
    const size_t second = NIXL_LIBFABRIC_CQ_BATCH_SIZE / 2;
    RecordingSink sink("A", rail, first + second);
    for (size_t i = 0; i < first + second; ++i) {
        nixlLibfabricReq *req = allocateData(rail, nixlLibfabricReq::READ, &sink);
        TEST_ASSERT(req != nullptr, "data request allocated");
        cq_entries.push_back(successEntry(req));
    }

    // A read takes at most a batch, and every read reports its own count
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "first read processed");
    TEST_ASSERT(sink_calls.size() == 1, "one call for the first read");
    TEST_ASSERT(callIs(sink_calls[0], "A", first, NIXL_SUCCESS, second),
                "first call counts a full batch, the rest still active");
    TEST_ASSERT(!sink.isComplete(), "sink not complete after the first read");

    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "second read processed");
    TEST_ASSERT(sink_calls.size() == 2, "one call for the second read");
    TEST_ASSERT(callIs(sink_calls[1], "A", second, NIXL_SUCCESS, 0), "second call counts the rest");
    TEST_ASSERT(sink.isComplete(), "sink complete");

    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_IN_PROG, "empty CQ reports no progress");
    TEST_ASSERT(sink_calls.size() == 2, "an empty read makes no call");
    TEST_ASSERT(sink.callsAfterCompletion() == 0, "no call after the sink completed");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request left active");
    return 0;
}

static int
testInterleavedSinks(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testInterleavedSinks";
    reset();
    mock_av_reset();

    nixlLibfabricRail &rail = mgr.getRail(1);
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request active at the start");

    // A remote write from a known peer, between the data completions
    std::array<char, LF_EP_NAME_MAX_LEN> peer_ep{};
    std::memcpy(peer_ep.data(), "peerP", 5);
    fi_addr_t peer_addr = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(peer_ep.data(), 7, &peer_addr) == NIXL_SUCCESS, "insert peer");
    size_t remote_writes = 0;
    rail.setXferIdCallback([&remote_writes](uint64_t, uint32_t) { ++remote_writes; });

    // Both sinks free themselves on completion. Order: A A B B B <remote write> B A
    auto *a = new RecordingSink("A", rail, 3, /*delete_when_complete=*/true);
    auto *b = new RecordingSink("B", rail, 4, /*delete_when_complete=*/true);
    const std::vector<RecordingSink *> order = {a, a, b, b, b, nullptr, b, a};
    for (RecordingSink *sink : order) {
        if (sink == nullptr) {
            FakeEntry fe{};
            fe.entry.flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA | FI_RMA;
            fe.entry.len = 16384;
            fe.entry.data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, 9, 0);
            fe.src_addr = peer_addr;
            cq_entries.push_back(fe);
            continue;
        }
        nixlLibfabricReq *req = allocateData(rail, nixlLibfabricReq::WRITE, sink);
        TEST_ASSERT(req != nullptr, "data request allocated");
        cq_entries.push_back(successEntry(req));
    }

    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "completions processed");
    TEST_ASSERT(remote_writes == 1, "remote write delivered");
    TEST_ASSERT(sink_calls.size() == 3, "one call per run of the same sink");
    // A run is reported when the next sink's first completion arrives, after that request
    // is released too, so 7 - 3 requests are still active at the first call.
    TEST_ASSERT(callIs(sink_calls[0], "A", 2, NIXL_SUCCESS, 4), "A's first run counts 2");
    TEST_ASSERT(callIs(sink_calls[1], "B", 4, NIXL_SUCCESS, 0),
                "B's run counts 4, not split by the remote write");
    TEST_ASSERT(callIs(sink_calls[2], "A", 1, NIXL_SUCCESS, 0), "A's second run counts 1");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request left active");

    rail.setXferIdCallback(nullptr);
    return 0;
}

static int
testErrorEntry(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testErrorEntry";
    reset();

    nixlLibfabricRail &rail = mgr.getRail(2);
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request active at the start");

    // Two successes, then an error, for the same sink
    RecordingSink sink("A", rail, 3);
    std::vector<nixlLibfabricReq *> reqs;
    for (size_t i = 0; i < 3; ++i) {
        nixlLibfabricReq *req = allocateData(rail, nixlLibfabricReq::WRITE, &sink);
        TEST_ASSERT(req != nullptr, "data request allocated");
        reqs.push_back(req);
    }
    cq_entries.push_back(successEntry(reqs[0]));
    cq_entries.push_back(successEntry(reqs[1]));
    cq_entries.push_back(errorEntry(reqs[2]));

    // The read stops in front of the error, so the successes are reported first
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "successes processed");
    TEST_ASSERT(sink_calls.size() == 1, "one call for the successes");
    TEST_ASSERT(callIs(sink_calls[0], "A", 2, NIXL_SUCCESS, 1), "successes counted first");

    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_ERR_BACKEND, "error entry reported");
    TEST_ASSERT(sink_calls.size() == 2, "one call for the error");
    TEST_ASSERT(callIs(sink_calls[1], "A", 1, NIXL_ERR_BACKEND, 0),
                "error counted on its own, with the error status, after its release");
    TEST_ASSERT(sink.isComplete() && sink.callsAfterCompletion() == 0, "sink complete");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "errored request released");
    TEST_ASSERT(cq_entries.empty(), "CQ drained");

    // A control request keeps its callback, which sees the error
    nixl_status_t send_status = NIXL_SUCCESS;
    size_t send_calls = 0;
    nixlLibfabricReq *send_req =
        rail.allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE, /*req_id=*/2);
    TEST_ASSERT(send_req != nullptr, "control request allocated");
    send_req->completion_callback = [&send_status, &send_calls](nixl_status_t status) {
        send_status = status;
        ++send_calls;
    };
    cq_entries.push_back(errorEntry(send_req));
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_ERR_BACKEND, "send error reported");
    TEST_ASSERT(send_calls == 1 && send_status == NIXL_ERR_BACKEND,
                "control callback ran once with the error");
    TEST_ASSERT(sink_calls.size() == 2, "no sink call for a control request");
    TEST_ASSERT(!send_req->in_use, "control request released");
    return 0;
}

static int
testPostFailure(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testPostFailure";
    reset();

    nixlLibfabricRail &rail = mgr.getRail(3);
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "no data request active at the start");
    TEST_ASSERT(rail.initPostQueue(64), "post queue initialized");

    rma_ops_fail_stub.size = sizeof(fi_ops_rma);
    rma_ops_fail_stub.writemsg = fi_writemsg_fail_stub;
    rma_ops_fail_stub.readmsg = fi_readmsg_fail_stub;
    fi_ops_rma *saved_rma = rail.endpoint->rma;
    rail.endpoint->rma = &rma_ops_fail_stub;

    // Every post fails: each is reported on its own, with the error, after its release
    RecordingSink sink("A", rail, 3);
    const nixlLibfabricReq::OpType ops[] = {
        nixlLibfabricReq::WRITE, nixlLibfabricReq::READ, nixlLibfabricReq::WRITE};
    for (nixlLibfabricReq::OpType op : ops) {
        nixlLibfabricReq *req = allocateData(rail, op, &sink);
        TEST_ASSERT(req != nullptr, "data request allocated");
        nixlLibfabricPostRequest pr{};
        pr.type = (op == nixlLibfabricReq::WRITE) ? nixlLibfabricPostRequest::WRITE :
                                                    nixlLibfabricPostRequest::READ;
        pr.length = 16384;
        pr.req = req;
        pr.device_id = -1;
        pr.is_cuda_vram = false;
        rail.enqueuePost(pr);
    }

    const nixl_status_t status = rail.drainPostQueue();
    rail.endpoint->rma = saved_rma;

    TEST_ASSERT(status == NIXL_SUCCESS, "drain reports failed posts through their sinks");
    TEST_ASSERT(failed_posts == 3, "every post attempted once");
    TEST_ASSERT(sink_calls.size() == 3, "one call per failed post");
    TEST_ASSERT(callIs(sink_calls[0], "A", 1, NIXL_ERR_BACKEND, 2), "first failure");
    TEST_ASSERT(callIs(sink_calls[1], "A", 1, NIXL_ERR_BACKEND, 1), "second failure");
    TEST_ASSERT(callIs(sink_calls[2], "A", 1, NIXL_ERR_BACKEND, 0), "third failure");
    TEST_ASSERT(sink.isComplete() && sink.callsAfterCompletion() == 0, "sink complete");
    TEST_ASSERT(rail.getActiveDataRequestCount() == 0, "failed requests released");
    return 0;
}

int
main() {
    NIXL_INFO << "=== Completion Batch Test ===";
    NIXL_INFO << "Using mock stubs (__wrap_fi_getinfo, fi_cq_readfrom, fi_cq_readerr, ...)";

    // Route CQ reads to the fake queue before the manager builds its rails
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_stub;
    cq_ops_stub.readerr = fi_cq_readerr_stub;

    nixlLibfabricRailManager mgr(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");

    int res;
    if ((res = testOneSinkOneCall(mgr)) != 0) {
        return res;
    }
    if ((res = testOneCallPerRead(mgr)) != 0) {
        return res;
    }
    if ((res = testInterleavedSinks(mgr)) != 0) {
        return res;
    }
    if ((res = testErrorEntry(mgr)) != 0) {
        return res;
    }
    if ((res = testPostFailure(mgr)) != 0) {
        return res;
    }

    NIXL_INFO << "=== All completion batch tests PASSED ===";
    return 0;
}
