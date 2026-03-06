// Standalone correctness test for flashinfer/comm/sycl_allreduce_fusion.h
//
// Tests all three kernels on a single Intel GPU (SYCL device):
//   - OneShotAllReduceKernel        (1-rank: AR = identity)
//   - OneShotAllReduceRMSNormKernel (1-rank: AR + residual + RMSNorm)
//   - TwoShotAllReduceKernel        (1-rank: AR = identity via scatter+gather)
// A 2-rank simulation (shared USM on one device, two concurrent queues) is
// also included for deeper barrier / reduction path coverage.
//
// Build (Intel oneAPI 2024+) — run from the repo root:
//   icpx -fsycl -std=c++17 \
//        -Iinclude \
//        -o test_sycl_allreduce_fusion \
//        tests/comm/test_sycl_allreduce_fusion.cpp
//
// Run:
//   ./test_sycl_allreduce_fusion
//
// To skip the 2-rank concurrent-kernel test (in case of deadlock risk):
//   SYCL_AR_SKIP_MULTIRANK=1 ./test_sycl_allreduce_fusion
//
// Exit code: 0 = all tests passed, 1 = at least one test failed.

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "flashinfer/comm/sycl_allreduce_fusion.h"

namespace fi = flashinfer::sycl_allreduce;

// ============================================================================
// Utility: convert to/from float for generic T (sycl::half or float)
// ============================================================================
template <typename T>
static float to_f(T v) {
  return static_cast<float>(v);
}
template <typename T>
static T from_f(float v) {
  return static_cast<T>(v);
}

// ============================================================================
// Reference implementations (scalar CPU)
// ============================================================================

// AllReduce over N_ranks inputs → element-wise sum.
template <typename T>
static void ref_allreduce(const std::vector<std::vector<T>>& inputs,
                          std::vector<T>& out) {
  size_t n = inputs[0].size();
  out.assign(n, T(0));
  for (size_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (auto& inp : inputs) s += to_f(inp[i]);
    out[i] = from_f<T>(s);
  }
}

// prenorm[i] = ar[i] + residual[i]
template <typename T>
static void ref_prenorm(const std::vector<T>& ar, const std::vector<T>& res,
                        std::vector<T>& out) {
  out.resize(ar.size());
  for (size_t i = 0; i < ar.size(); ++i)
    out[i] = from_f<T>(to_f(ar[i]) + to_f(res[i]));
}

// Per-token RMSNorm: norm[t,h] = x[t,h] / sqrt(mean(x[t,:]^2)+eps) * gamma[h]
template <typename T>
static void ref_rmsnorm(const std::vector<T>& x, const std::vector<T>& gamma,
                        std::vector<T>& out, int tokens, int hidden, float eps) {
  out.resize(tokens * hidden);
  for (int t = 0; t < tokens; ++t) {
    float acc = 0.f;
    for (int h = 0; h < hidden; ++h) {
      float v = to_f(x[t * hidden + h]);
      acc += v * v;
    }
    float denom = 1.f / std::sqrt(acc / static_cast<float>(hidden) + eps);
    for (int h = 0; h < hidden; ++h) {
      out[t * hidden + h] =
          from_f<T>(to_f(x[t * hidden + h]) * denom * to_f(gamma[h]));
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

// Minimum flag array size needed by block_barrier for given (grid_size, world_size).
// Layout: [world_size | (grid_size+1)*world_size*2 ]
static size_t flags_needed(int gs, int ws) {
  return static_cast<size_t>(ws + (gs + 1) * ws * 2);
}

static bool approx_eq(float a, float b, float atol, float rtol) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

template <typename T>
static bool check_close(const std::vector<T>& got, const std::vector<T>& ref,
                        const std::string& name, float atol = 2e-2f,
                        float rtol = 2e-2f) {
  int fails = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (!approx_eq(to_f(got[i]), to_f(ref[i]), atol, rtol)) {
      if (fails++ < 5)
        std::printf("  MISMATCH[%zu]: got=%.6f ref=%.6f\n",
                    i, (double)to_f(got[i]), (double)to_f(ref[i]));
    }
  if (fails == 0) {
    std::printf("  PASS: %s\n", name.c_str());
    return true;
  }
  std::printf("  FAIL: %s  (%d/%zu mismatches)\n", name.c_str(), fails,
              got.size());
  return false;
}



// ============================================================================
// Test A: OneShotAllReduceKernel  –  1 rank  (AR = identity)
// ============================================================================
template <typename T>
static bool test_oneshot_ar_1rank(sycl::queue& q, int tokens, int hidden) {
  constexpr int RANKS = 1;
  const size_t N = static_cast<size_t>(tokens * hidden);

  std::vector<T> h_in(N), h_out(N, T(0));
  for (size_t i = 0; i < N; ++i)
    h_in[i] = from_f<T>((static_cast<float>(i % 64) / 64.f - 0.5f) * 2.f);

  T* d_in   = sycl::malloc_device<T>(N, q);
  T* d_out  = sycl::malloc_device<T>(N, q);
  T* d_peer = sycl::malloc_device<T>(N, q);

  size_t epb = 0;
  auto [gs, bs] = fi::compute_launch_config<T>(N, RANKS, false, epb);
  size_t fn = flags_needed(gs, RANKS);
  uint32_t* d_flags = sycl::malloc_device<uint32_t>(fn, q);

  q.memcpy(d_in, h_in.data(), N * sizeof(T));
  q.memset(d_out, 0, N * sizeof(T));
  q.memset(d_flags, 0, fn * sizeof(uint32_t));
  q.wait();

  fi::AllReduceParamsSYCL<T> p{};
  p.elts_total              = N;
  p.elts_per_block          = epb;
  p.local_rank              = 0;
  p.ranks_per_node          = RANKS;
  p.barrier_flag            = 1;
  p.peer_barrier_ptrs_in[0]    = d_flags;
  p.peer_comm_buffer_ptrs[0]   = d_peer;
  p.local_output_buffer_ptr    = d_out;
  p.local_input_buffer_ptr     = d_in;

  fi::launch_oneshot_allreduce<T, RANKS>(q, p);
  q.wait();

  q.memcpy(h_out.data(), d_out, N * sizeof(T)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q);
  sycl::free(d_peer, q); sycl::free(d_flags, q);

  const char* dt = sizeof(T) == 2 ? "half" : "float";
  auto name = std::string("oneshot_ar_1rank<") + dt + "> tok=" +
              std::to_string(tokens) + " hid=" + std::to_string(hidden);
  return check_close(h_out, h_in, name, 1e-3f, 1e-3f);
}

// ============================================================================
// Test B: OneShotAllReduceRMSNormKernel  –  1 rank
//         Expected: prenorm = input + residual
//                   norm_out = RMSNorm(prenorm, gamma)
// ============================================================================
template <typename T>
static bool test_oneshot_ar_rmsnorm_1rank(sycl::queue& q, int tokens,
                                          int hidden) {
  constexpr int RANKS = 1;
  const size_t N = static_cast<size_t>(tokens * hidden);
  const float EPS = 1e-5f;

  std::vector<T> h_in(N), h_res(N), h_gamma(hidden);
  std::vector<T> h_prenorm(N, T(0)), h_norm(N, T(0));
  for (size_t i = 0; i < N; ++i)
    h_in[i] = from_f<T>((static_cast<float>(i % 64) / 64.f - 0.5f) * 0.4f);
  for (size_t i = 0; i < N; ++i)
    h_res[i] = from_f<T>((static_cast<float>((i + 13) % 64) / 64.f - 0.5f) * 0.2f);
  for (int h = 0; h < hidden; ++h)
    h_gamma[h] = from_f<T>(0.5f + static_cast<float>(h % 8) / 8.f);

  T* d_in    = sycl::malloc_device<T>(N, q);
  T* d_peer  = sycl::malloc_device<T>(N, q);
  T* d_res   = sycl::malloc_device<T>(N, q);
  T* d_gamma = sycl::malloc_device<T>(hidden, q);
  T* d_pre   = sycl::malloc_device<T>(N, q);
  T* d_norm  = sycl::malloc_device<T>(N, q);

  size_t epb = 0;
  auto [gs, bs] = fi::compute_launch_config<T>(N, RANKS, false, epb);
  size_t fn = flags_needed(gs, RANKS);
  uint32_t* d_flags = sycl::malloc_device<uint32_t>(fn, q);

  q.memcpy(d_in,    h_in.data(),    N * sizeof(T));
  q.memcpy(d_res,   h_res.data(),   N * sizeof(T));
  q.memcpy(d_gamma, h_gamma.data(), hidden * sizeof(T));
  q.memset(d_pre,   0, N * sizeof(T));
  q.memset(d_norm,  0, N * sizeof(T));
  q.memset(d_flags, 0, fn * sizeof(uint32_t));
  q.wait();

  fi::AllReduceParamsSYCL<T> p{};
  p.elts_total              = N;
  p.elts_per_block          = epb;
  p.local_rank              = 0;
  p.ranks_per_node          = RANKS;
  p.barrier_flag            = 1;
  p.peer_barrier_ptrs_in[0]            = d_flags;
  p.peer_comm_buffer_ptrs[0]           = d_peer;
  p.local_output_buffer_ptr            = d_norm;
  p.local_input_buffer_ptr             = d_in;
  p.fusion_params.residual_buffer      = d_res;
  p.fusion_params.weight_buffer        = d_gamma;
  p.fusion_params.intermediate_buffer  = d_pre;
  p.fusion_params.hidden_size          = hidden;
  p.fusion_params.eps                  = EPS;

  fi::launch_oneshot_allreduce_rmsnorm<T, RANKS>(q, p);
  q.wait();

  q.memcpy(h_prenorm.data(), d_pre,  N * sizeof(T));
  q.memcpy(h_norm.data(),    d_norm, N * sizeof(T));
  q.wait();

  sycl::free(d_in, q); sycl::free(d_peer, q); sycl::free(d_res, q);
  sycl::free(d_gamma, q); sycl::free(d_pre, q); sycl::free(d_norm, q);
  sycl::free(d_flags, q);

  // Reference: AR with 1 rank = identity, then prenorm = ar + residual
  std::vector<T> ref_pre, ref_norm;
  ref_prenorm(h_in, h_res, ref_pre);
  ref_rmsnorm(ref_pre, h_gamma, ref_norm, tokens, hidden, EPS);

  const char* dt = sizeof(T) == 2 ? "half" : "float";
  std::string tag = std::string(dt) + " tok=" + std::to_string(tokens) +
                    " hid=" + std::to_string(hidden);
  bool ok = check_close(h_prenorm, ref_pre,  "oneshot_rmsnorm_1rank prenorm " + tag);
  ok     &= check_close(h_norm,    ref_norm, "oneshot_rmsnorm_1rank norm_out " + tag);
  return ok;
}

// ============================================================================
// Test C: TwoShotAllReduceKernel  –  1 rank  (AR = identity)
//   elts_total must be divisible by VEC_SIZE * RANKS (= VEC_SIZE for 1 rank).
// ============================================================================
template <typename T>
static bool test_twoshot_ar_1rank(sycl::queue& q, int tokens, int hidden) {
  constexpr int RANKS = 1;
  const size_t N = static_cast<size_t>(tokens * hidden);

  std::vector<T> h_in(N), h_out(N, T(0));
  for (size_t i = 0; i < N; ++i)
    h_in[i] = from_f<T>((static_cast<float>(i % 64) / 64.f - 0.5f) * 2.f);

  T* d_in   = sycl::malloc_device<T>(N, q);
  T* d_out  = sycl::malloc_device<T>(N, q);
  T* d_peer = sycl::malloc_device<T>(N, q);

  size_t epb = 0;
  auto [gs, bs] = fi::compute_launch_config<T>(N, RANKS, /*twoshot=*/true, epb);
  size_t fn = flags_needed(gs, RANKS);
  // Two barrier arrays: one for ReduceScatter phase, one for AllGather phase.
  uint32_t* d_flags_in  = sycl::malloc_device<uint32_t>(fn, q);
  uint32_t* d_flags_out = sycl::malloc_device<uint32_t>(fn, q);

  q.memcpy(d_in, h_in.data(), N * sizeof(T));
  q.memset(d_out,       0, N * sizeof(T));
  q.memset(d_flags_in,  0, fn * sizeof(uint32_t));
  q.memset(d_flags_out, 0, fn * sizeof(uint32_t));
  q.wait();

  fi::AllReduceParamsSYCL<T> p{};
  p.elts_total              = N;
  p.elts_per_block          = epb;
  p.local_rank              = 0;
  p.ranks_per_node          = RANKS;
  p.barrier_flag            = 1;
  // peer_barrier_ptrs_in: used in ReduceScatter barrier (phase B).
  p.peer_barrier_ptrs_in[0]  = d_flags_in;
  // peer_barrier_ptrs_out: used in AllGather barrier (phase D).
  // Embedded in params so the GPU kernel can safely dereference it.
  p.peer_barrier_ptrs_out[0] = d_flags_out;
  p.peer_comm_buffer_ptrs[0]   = d_peer;
  p.local_output_buffer_ptr    = d_out;
  p.local_input_buffer_ptr     = d_in;

  fi::launch_twoshot_allreduce<T, RANKS>(q, p);
  q.wait();

  q.memcpy(h_out.data(), d_out, N * sizeof(T)).wait();
  sycl::free(d_in, q); sycl::free(d_out, q); sycl::free(d_peer, q);
  sycl::free(d_flags_in, q); sycl::free(d_flags_out, q);

  const char* dt = sizeof(T) == 2 ? "half" : "float";
  auto name = std::string("twoshot_ar_1rank<") + dt + "> tok=" +
              std::to_string(tokens) + " hid=" + std::to_string(hidden);
  return check_close(h_out, h_in, name, 1e-3f, 1e-3f);
}

// ============================================================================
// Test D: OneShotAllReduceKernel  –  N ranks, single device
//
//   All RANKS queues target the same GPU.  USM device memory allocated from
//   any one queue is visible to all queues on the same device, simulating
//   XeLink P2P without requiring multiple physical GPUs.
//
//   Each rank runs in its own CPU thread so the barrier spinlocks in the
//   kernel can be satisfied concurrently.
//
//   NOTE: requires the GPU to run RANKS work-groups simultaneously.  If the
//   device cannot sustain concurrent dispatch the barrier will deadlock; set
//   SYCL_AR_SKIP_MULTIRANK=1 to skip all multi-rank tests.
// ============================================================================
template <typename T, int RANKS>
static bool test_oneshot_ar_Nrank(std::vector<sycl::queue*>& qs, int tokens,
                                  int hidden) {
  static_assert(RANKS >= 2 && RANKS <= fi::kMaxRanksPerNode, "unsupported rank count");
  const size_t N = static_cast<size_t>(tokens * hidden);

  // Build RANKS distinct host inputs (different offsets avoid cancellation).
  std::vector<std::vector<T>> h_in(RANKS, std::vector<T>(N));
  for (int r = 0; r < RANKS; ++r)
    for (size_t i = 0; i < N; ++i)
      h_in[r][i] = from_f<T>(
          (static_cast<float>((i + static_cast<size_t>(r) * 16) % 64) / 64.f - 0.5f) * 2.f);

  std::vector<T> ref_out;
  ref_allreduce(h_in, ref_out);

  // Allocate all USM from qs[0]; it is visible to every queue on the same device.
  sycl::queue& q0 = *qs[0];
  std::array<T*,        RANKS> d_in{}, d_out{}, d_peer{};
  std::array<uint32_t*, RANKS> d_flags{};

  size_t epb = 0;
  auto [gs, bs] = fi::compute_launch_config<T>(N, RANKS, /*twoshot=*/false, epb);
  const size_t fn = flags_needed(gs, RANKS);

  for (int r = 0; r < RANKS; ++r) {
    d_in[r]    = sycl::malloc_device<T>(N, q0);
    d_out[r]   = sycl::malloc_device<T>(N, q0);
    d_peer[r]  = sycl::malloc_device<T>(N, q0);
    d_flags[r] = sycl::malloc_device<uint32_t>(fn, q0);
    q0.memcpy(d_in[r], h_in[r].data(), N * sizeof(T));
    q0.memset(d_out[r],   0, N  * sizeof(T));
    q0.memset(d_flags[r], 0, fn * sizeof(uint32_t));
  }
  q0.wait();

  // Build per-rank params; all ranks see the same peer arrays.
  std::array<fi::AllReduceParamsSYCL<T>, RANKS> params{};
  for (int r = 0; r < RANKS; ++r) {
    auto& p          = params[r];
    p.elts_total     = N;
    p.elts_per_block = epb;
    p.ranks_per_node = RANKS;
    p.barrier_flag   = 1;
    p.local_rank     = static_cast<size_t>(r);
    p.local_input_buffer_ptr  = d_in[r];
    p.local_output_buffer_ptr = d_out[r];
    for (int j = 0; j < RANKS; ++j) {
      p.peer_barrier_ptrs_in[j] = d_flags[j];
      p.peer_comm_buffer_ptrs[j] = d_peer[j];
    }
  }

  // Launch all RANKS kernels concurrently from separate CPU threads.
  std::vector<std::future<void>> futures;
  futures.reserve(RANKS);
  for (int r = 0; r < RANKS; ++r) {
    futures.push_back(std::async(std::launch::async, [&qs, &params, r] {
      fi::launch_oneshot_allreduce<T, RANKS>(*qs[r], params[r]);
      qs[r]->wait();
    }));
  }
  for (auto& f : futures) f.get();

  // Verify: every rank must have the same all-reduced result.
  const char* dt = sizeof(T) == 2 ? "half" : "float";
  const std::string tag = std::string(dt) + " tok=" + std::to_string(tokens) +
                          " hid=" + std::to_string(hidden);
  bool ok = true;
  for (int r = 0; r < RANKS; ++r) {
    std::vector<T> h_out(N);
    q0.memcpy(h_out.data(), d_out[r], N * sizeof(T)).wait();
    const std::string name = "oneshot_ar_" + std::to_string(RANKS) + "rank rank" +
                             std::to_string(r) + " " + tag;
    ok &= check_close(h_out, ref_out, name);
  }

  for (int r = 0; r < RANKS; ++r) {
    sycl::free(d_in[r],    q0);
    sycl::free(d_out[r],   q0);
    sycl::free(d_peer[r],  q0);
    sycl::free(d_flags[r], q0);
  }
  return ok;
}

// ============================================================================
// main: run all tests
// ============================================================================
int main() {
  sycl::device dev;
  try {
    dev = sycl::device(sycl::gpu_selector_v);
  } catch (...) {
    std::fprintf(stderr, "No GPU found; falling back to default.\n");
    dev = sycl::device(sycl::default_selector_v);
  }
  std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

  // Create 8 independent in-order queues on the same device.
  // Kernels submitted to different queues can run concurrently (device permitting),
  // which is required for the barrier spinlocks to make progress.
  constexpr int kMaxQueues = 8;
  std::vector<sycl::queue> queues;
  queues.reserve(kMaxQueues);
  for (int i = 0; i < kMaxQueues; ++i)
    queues.emplace_back(dev, sycl::property::queue::in_order{});

  // Build pointer-vector slices for each world size.
  auto make_qptrs = [&](int n) {
    std::vector<sycl::queue*> v;
    for (int i = 0; i < n; ++i) v.push_back(&queues[i]);
    return v;
  };
  auto q2 = make_qptrs(2);
  auto q4 = make_qptrs(4);
  auto q8 = make_qptrs(8);

  const bool skip_multirank = (std::getenv("SYCL_AR_SKIP_MULTIRANK") != nullptr);
  sycl::queue& q0 = queues[0];
  int total = 0, passed = 0;

  // ── Test A: OneShotAllReduceKernel, 1 rank (smoke-test: AR = identity) ────
  std::printf("\n=== Test A: oneshot_ar_1rank ===\n");
  for (auto [tok, hid] : std::vector<std::pair<int,int>>{
           {1, 128}, {4, 512}, {16, 1024}, {32, 2048}}) {
    ++total; if (test_oneshot_ar_1rank<sycl::half>(q0, tok, hid)) ++passed;
    ++total; if (test_oneshot_ar_1rank<float>      (q0, tok, hid)) ++passed;
  }

  // ── Test B: OneShotAllReduceRMSNormKernel, 1 rank ─────────────────────────
  std::printf("\n=== Test B: oneshot_ar_rmsnorm_1rank ===\n");
  for (auto [tok, hid] : std::vector<std::pair<int,int>>{
           {1, 128}, {4, 512}, {16, 1024}, {32, 2048}}) {
    ++total; if (test_oneshot_ar_rmsnorm_1rank<sycl::half>(q0, tok, hid)) ++passed;
    ++total; if (test_oneshot_ar_rmsnorm_1rank<float>      (q0, tok, hid)) ++passed;
  }

  // ── Test C: TwoShotAllReduceKernel, 1 rank ─────────────────────────────────
  std::printf("\n=== Test C: twoshot_ar_1rank ===\n");
  for (auto [tok, hid] : std::vector<std::pair<int,int>>{
           {1, 128}, {4, 512}, {16, 1024}}) {
    ++total; if (test_twoshot_ar_1rank<sycl::half>(q0, tok, hid)) ++passed;
    ++total; if (test_twoshot_ar_1rank<float>      (q0, tok, hid)) ++passed;
  }

  // ── Tests D/E/F: OneShotAllReduceKernel, 2 / 4 / 8 ranks ─────────────────
  // These tests require concurrent GPU kernel execution across RANKS queues.
  // Set SYCL_AR_SKIP_MULTIRANK=1 to skip all of them.
  const std::vector<std::pair<int,int>> mr_shapes = {{1,128},{4,512},{16,1024}};

  auto run_multirank = [&](auto& qptrs, int nranks, char label, auto run_fn) {
    if (skip_multirank) {
      std::printf("\n=== Test %c: oneshot_ar_%drank  [SKIPPED] ===\n", label, nranks);
      return;
    }
    std::printf("\n=== Test %c: oneshot_ar_%drank (concurrent queues) ===\n",
                label, nranks);
    for (auto [tok, hid] : mr_shapes) {
      ++total; if (run_fn(qptrs, tok, hid, (sycl::half*)nullptr)) ++passed;
      ++total; if (run_fn(qptrs, tok, hid, (float*)nullptr))       ++passed;
    }
  };

  run_multirank(q2, 2, 'D', [](auto& q, int tok, int hid, auto* tag) {
    return test_oneshot_ar_Nrank<std::remove_pointer_t<decltype(tag)>, 2>(q, tok, hid);
  });
  run_multirank(q4, 4, 'E', [](auto& q, int tok, int hid, auto* tag) {
    return test_oneshot_ar_Nrank<std::remove_pointer_t<decltype(tag)>, 4>(q, tok, hid);
  });
  run_multirank(q8, 8, 'F', [](auto& q, int tok, int hid, auto* tag) {
    return test_oneshot_ar_Nrank<std::remove_pointer_t<decltype(tag)>, 8>(q, tok, hid);
  });

  std::printf("\n=== Results: %d / %d passed ===\n", passed, total);
  return (passed == total) ? 0 : 1;
}
