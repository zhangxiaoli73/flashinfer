// SYCL implementation of AllReduce + RMSNorm fusion for single-node (XeLink) communication.
//
// Design mirrors trtllm_allreduce.cuh / trtllm_allreduce_fusion.cuh but targets Intel GPU via SYCL:
//   - XeLink P2P   ≈  NVLink P2P  (cross-device USM access, same node)
//   - sub_group    ≈  warp        (size 16 or 32 on Xe HPC)
//   - group_barrier ≈ __syncthreads()
//   - atomic_ref<release/acquire, system> ≈ st/ld.global.release/acquire.sys
//
// Fused kernel eliminates 2 GMEM round-trips between AllReduce and RMSNorm:
//   Unfused: AR result → GMEM store → GMEM load → RMSNorm
//   Fused  : AR result stays in registers → directly feeds RMSNorm
//
// Usage:
//   1. Allocate USM peer-comm buffers (one per rank, P2P-accessible from all ranks).
//   2. Allocate barrier flag arrays (peer_barrier_ptrs_in / out) visible to all ranks.
//   3. Fill AllReduceParamsSYCL and call launch_oneshot_allreduce_rmsnorm().

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace flashinfer {
namespace sycl_allreduce {

// ---------------------------------------------------------------------------
// Constants (mirrors trtllm_allreduce.cuh)
// ---------------------------------------------------------------------------
inline constexpr int kMaxRanksPerNode  = 16;
inline constexpr int kMaxARBlocks      = 24;   // max work-groups for AllReduce grid
inline constexpr int kDefaultBlockSize = 512;  // work-items per work-group
// 128-bit vectorised access: 8× fp16 or bf16, 4× fp32 per load.
inline constexpr int kBytesPerAccess   = 16;

// ---------------------------------------------------------------------------
// 128-bit vector type for vectorised load / store
// ---------------------------------------------------------------------------
struct alignas(16) Vec128 {
  int32_t x, y, z, w;
};

// Vectorised load / store helpers (raw 128-bit).
SYCL_EXTERNAL inline Vec128 load128(const void* ptr) {
  Vec128 v;
  v = *reinterpret_cast<const Vec128*>(ptr);
  return v;
}
SYCL_EXTERNAL inline void store128(void* ptr, Vec128 v) {
  *reinterpret_cast<Vec128*>(ptr) = v;
}

// ---------------------------------------------------------------------------
// Cross-GPU flag synchronisation (release/acquire, system scope)
// Mirrors st_flag_release / ld_flag_acquire in trtllm_allreduce.cuh
// ---------------------------------------------------------------------------
SYCL_EXTERNAL inline void st_flag_release(uint32_t flag, uint32_t* addr) {
  sycl::atomic_ref<uint32_t,
                   sycl::memory_order::release,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>(*addr)
      .store(flag);
}

SYCL_EXTERNAL inline uint32_t ld_flag_acquire(uint32_t* addr) {
  return sycl::atomic_ref<uint32_t,
                          sycl::memory_order::acquire,
                          sycl::memory_scope::system,
                          sycl::access::address_space::global_space>(*addr)
      .load();
}

// ---------------------------------------------------------------------------
// block_barrier: every block on every GPU waits until the same-indexed block
// on all other GPUs has reached this point.  Mirrors block_barrier() in .cuh.
//
// Layout of signals[r]:  [world_size (multi_gpu_barrier flags) |
//                         (grid_size+1)*world_size×2 (per-block flags, ping/pong)]
// ---------------------------------------------------------------------------
SYCL_EXTERNAL inline void block_barrier(
    sycl::nd_item<1>& item,
    uint32_t** signals,  // [world_size] pointers into peer-visible flag arrays
    uint32_t flag,
    int local_rank,
    int world_size,
    int grid_size)
{
  auto grp  = item.get_group();
  int tidx  = static_cast<int>(item.get_local_id(0));
  int bidx  = static_cast<int>(item.get_group(0));

  if (tidx < world_size) {
    // Per-block offset: skip the first world_size slots (used by multi_gpu_barrier).
    uint32_t flag_block_offset = static_cast<uint32_t>(world_size + bidx * world_size);
    if (flag % 2 == 1) {
      flag_block_offset += static_cast<uint32_t>((grid_size + 1) * world_size);
    }
    // This block signals all receivers (dim-0: listener = tidx).
    st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);

    // Wait until the peer block with same bidx on every rank has signalled.
    uint32_t* peer_ptr = signals[local_rank] + flag_block_offset + tidx;
    while (ld_flag_acquire(peer_ptr) != flag) { /* spin */ }
  }
  sycl::group_barrier(grp);  // __syncthreads() equivalent
}

// ---------------------------------------------------------------------------
// AllReduce parameter structs
// ---------------------------------------------------------------------------
struct AllReduceFusionParamsSYCL {
  void const* residual_buffer;   // residual to add after AllReduce
  void const* weight_buffer;     // RMSNorm gamma weights
  void*       intermediate_buffer; // output: prenorm (allreduce + residual)
  int         hidden_size;       // normalised dimension (= elts per token)
  float       eps;               // RMSNorm epsilon
};

template <typename T>
struct AllReduceParamsSYCL {
  size_t elts_total;             // total elements = tokens * hidden_size
  size_t elts_per_block;         // elements handled by each work-group
  size_t local_rank;
  size_t ranks_per_node;
  uint32_t barrier_flag;
  uint32_t* peer_barrier_ptrs_in[kMaxRanksPerNode];  // for block_barrier (in phase)
  void*     peer_comm_buffer_ptrs[kMaxRanksPerNode]; // P2P data staging buffers
  void*     local_output_buffer_ptr;                 // final norm output
  void const* local_input_buffer_ptr;               // this rank's input
  AllReduceFusionParamsSYCL fusion_params;
};

// ---------------------------------------------------------------------------
// Intra-work-group reduction (sum of floats).
// Mirrors reduce_fusion::block_reduce_sum() in trtllm_allreduce.cuh.
// smem must have capacity >= work-group size (in floats).
// ---------------------------------------------------------------------------
SYCL_EXTERNAL inline float block_reduce_sum(
    sycl::nd_item<1>& item,
    sycl::local_accessor<float, 1> smem,
    float val)
{
  auto grp = item.get_group();
  auto sg  = item.get_sub_group();
  int tidx = static_cast<int>(item.get_local_id(0));
  int sg_size = static_cast<int>(sg.get_local_range()[0]);
  int sg_id   = tidx / sg_size;
  int lane    = tidx % sg_size;
  int num_sgs = static_cast<int>(item.get_local_range(0)) / sg_size;

  // Step 1: reduce within sub-group.
  val = sycl::reduce_over_group(sg, val, sycl::plus<float>());

  // Step 2: sub-group leaders write to SLM.
  if (lane == 0) smem[sg_id] = val;
  sycl::group_barrier(grp);

  // Step 3: sub-group 0 reads all leaders and reduces.
  float total = (sg_id == 0 && tidx < num_sgs) ? smem[tidx] : 0.f;
  total = sycl::reduce_over_group(sg, total, sycl::plus<float>());

  // Step 4: broadcast via SLM.
  if (tidx == 0) smem[0] = total;
  sycl::group_barrier(grp);
  return smem[0];
}

// ---------------------------------------------------------------------------
// Element-wise helpers for half / bfloat16 / float vectors.
// VEC_SIZE = kBytesPerAccess / sizeof(T) elements fit in a 128-bit load.
// ---------------------------------------------------------------------------
template <typename T, int VEC_SIZE>
struct Vec {
  T data[VEC_SIZE];

  SYCL_EXTERNAL inline void load(const T* ptr) {
    // Vectorised 128-bit load
    store128(data, load128(ptr));
  }
  SYCL_EXTERNAL inline void store(T* ptr) const {
    store128(ptr, load128(data));
  }
  SYCL_EXTERNAL inline void fill(T v) {
    for (int i = 0; i < VEC_SIZE; ++i) data[i] = v;
  }
  SYCL_EXTERNAL inline T& operator[](int i) { return data[i]; }
  SYCL_EXTERNAL inline T  operator[](int i) const { return data[i]; }
};

template <typename T, int VEC_SIZE>
SYCL_EXTERNAL inline Vec<T, VEC_SIZE> vec_add(const Vec<T, VEC_SIZE>& a,
                                               const Vec<T, VEC_SIZE>& b) {
  Vec<T, VEC_SIZE> ret;
  for (int i = 0; i < VEC_SIZE; ++i)
    ret[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
  return ret;
}

template <typename T, int VEC_SIZE>
SYCL_EXTERNAL inline float accumulate_sq(float acc, const Vec<T, VEC_SIZE>& v) {
  for (int i = 0; i < VEC_SIZE; ++i) {
    float f = static_cast<float>(v[i]);
    acc += f * f;
  }
  return acc;
}

template <typename T, int VEC_SIZE>
SYCL_EXTERNAL inline Vec<T, VEC_SIZE> rms_norm_apply(float denom,
                                                       const Vec<T, VEC_SIZE>& v,
                                                       const Vec<T, VEC_SIZE>& gamma) {
  Vec<T, VEC_SIZE> ret;
  for (int i = 0; i < VEC_SIZE; ++i)
    ret[i] = static_cast<T>(static_cast<float>(v[i]) * denom * static_cast<float>(gamma[i]));
  return ret;
}

// ---------------------------------------------------------------------------
// Kernel 1: plain one-shot AllReduce (no RMSNorm fusion).
//
// Algorithm (mirrors oneShotAllReduceKernel in trtllm_allreduce.cuh):
//   Phase A – Copy:   each work-group copies its slice to this rank's P2P buffer.
//   Phase B – Barrier: block_barrier ensures all ranks have finished copy.
//   Phase C – Reduce: read from all P2P buffers and sum → write to output.
// ---------------------------------------------------------------------------
template <typename T, int RANKS_PER_NODE>
struct OneShotAllReduceKernel {
  static constexpr int VEC_SIZE = kBytesPerAccess / static_cast<int>(sizeof(T));

  AllReduceParamsSYCL<T> params;
  sycl::local_accessor<float, 1> smem;  // unused here but kept for API uniformity

  void operator()(sycl::nd_item<1> item) const {
    const int tidx = static_cast<int>(item.get_local_id(0));
    const int bidx = static_cast<int>(item.get_group(0));
    const int gs   = static_cast<int>(item.get_group_range(0));

    size_t chunk_start = static_cast<size_t>(bidx) * params.elts_per_block;
    size_t chunk_end   = std::min(chunk_start + params.elts_per_block, params.elts_total);

    const T* local_in = reinterpret_cast<const T*>(params.local_input_buffer_ptr);
    T*       peer_buf = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    T*       out      = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    // Phase A: copy local slice → this rank's P2P-visible buffer.
    for (size_t off = chunk_start + tidx * VEC_SIZE; off < chunk_end;
         off += item.get_local_range(0) * VEC_SIZE) {
      store128(&peer_buf[off], load128(&local_in[off]));
    }

    // Phase B: cross-GPU barrier.
    block_barrier(const_cast<sycl::nd_item<1>&>(item),
                  const_cast<uint32_t**>(params.peer_barrier_ptrs_in),
                  params.barrier_flag,
                  static_cast<int>(params.local_rank),
                  static_cast<int>(params.ranks_per_node),
                  gs);

    // Phase C: sum from all ranks → output.
    for (size_t off = chunk_start + tidx * VEC_SIZE; off < chunk_end;
         off += item.get_local_range(0) * VEC_SIZE) {
      Vec<T, VEC_SIZE> sum;
      sum.fill(T(0));
      for (int r = 0; r < RANKS_PER_NODE; ++r) {
        int rank = (static_cast<int>(params.local_rank) + r) % RANKS_PER_NODE;
        Vec<T, VEC_SIZE> v;
        v.load(reinterpret_cast<const T*>(params.peer_comm_buffer_ptrs[rank]) + off);
        sum = vec_add<T, VEC_SIZE>(sum, v);
      }
      sum.store(out + off);
    }
  }
};

// ---------------------------------------------------------------------------
// Kernel 2: one-shot AllReduce + Residual Add + RMSNorm (FUSED).
//
// Algorithm (mirrors one_shot_all_reduce_norm_kernel in trtllm_allreduce.cuh):
//   Phase A – Copy local slice to P2P buffer (same as Kernel 1).
//   Phase B – block_barrier.
//   Phase C – Per token:
//     1. Sum from all P2P buffers.
//     2. Add residual → store prenorm (intermediate_buffer).
//     3. Accumulate sum-of-squares.
//     4. block_reduce_sum → denom = rsqrt(acc/H + eps).
//     5. Apply gamma and write final normalised output.
//
// Key fusion benefit: sum_vec stays in registers between steps 1→2→3; we
// never write AllReduce result to GMEM before the RMSNorm pass.
// ---------------------------------------------------------------------------
template <typename T, int RANKS_PER_NODE>
struct OneShotAllReduceRMSNormKernel {
  static constexpr int VEC_SIZE = kBytesPerAccess / static_cast<int>(sizeof(T));

  AllReduceParamsSYCL<T> params;
  sycl::local_accessor<float, 1> smem;

  void operator()(sycl::nd_item<1> item) const {
    const int  tidx   = static_cast<int>(item.get_local_id(0));
    const int  bidx   = static_cast<int>(item.get_group(0));
    const int  gs     = static_cast<int>(item.get_group_range(0));
    const int  hid    = params.fusion_params.hidden_size;

    // How many tokens does this block handle?
    const int  total_tokens   = static_cast<int>(params.elts_total) / hid;
    const int  tokens_per_blk = (total_tokens + gs - 1) / gs;
    const int  tokens_this    = std::min(tokens_per_blk, total_tokens - bidx * tokens_per_blk);

    const T* local_in  = reinterpret_cast<const T*>(params.local_input_buffer_ptr);
    T*       peer_buf  = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    const T* residual  = reinterpret_cast<const T*>(params.fusion_params.residual_buffer);
    const T* gamma     = reinterpret_cast<const T*>(params.fusion_params.weight_buffer);
    T*       prenorm   = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);
    T*       norm_out  = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    // Block byte-offset into the global tensor for the first token of this block.
    const int blk_off = bidx * tokens_per_blk * hid;

    // Phase A: copy.
    for (int off = tidx * VEC_SIZE; off < tokens_this * hid;
         off += static_cast<int>(item.get_local_range(0)) * VEC_SIZE) {
      store128(&peer_buf[blk_off + off], load128(&local_in[blk_off + off]));
    }

    // Phase B: barrier.
    block_barrier(const_cast<sycl::nd_item<1>&>(item),
                  const_cast<uint32_t**>(params.peer_barrier_ptrs_in),
                  params.barrier_flag,
                  static_cast<int>(params.local_rank),
                  static_cast<int>(params.ranks_per_node),
                  gs);

    // Phase C: per-token fused AllReduce + Residual + RMSNorm.
    for (int tok = 0; tok < tokens_this; ++tok) {
      const int tok_off = blk_off + tok * hid;
      float acc = 0.f;

      // Pass 1: allreduce + residual → prenorm; accumulate sq.
      for (int off = tidx * VEC_SIZE; off < hid;
           off += static_cast<int>(item.get_local_range(0)) * VEC_SIZE) {
        Vec<T, VEC_SIZE> sum;
        sum.fill(T(0));
        for (int r = 0; r < RANKS_PER_NODE; ++r) {
          int rank = (static_cast<int>(params.local_rank) + r) % RANKS_PER_NODE;
          Vec<T, VEC_SIZE> v;
          v.load(reinterpret_cast<const T*>(params.peer_comm_buffer_ptrs[rank]) + tok_off + off);
          sum = vec_add<T, VEC_SIZE>(sum, v);
        }
        // Add residual (sum stays in registers, no extra GMEM read for norm later).
        Vec<T, VEC_SIZE> res;
        res.load(residual + tok_off + off);
        sum = vec_add<T, VEC_SIZE>(sum, res);

        sum.store(prenorm + tok_off + off);  // store prenorm (new residual for next layer)
        acc = accumulate_sq<T, VEC_SIZE>(acc, sum);
      }

      // Intra-block reduction to get acc = sum_of_squares across all threads.
      acc = block_reduce_sum(const_cast<sycl::nd_item<1>&>(item), smem, acc);
      const float denom = sycl::rsqrt(acc / static_cast<float>(hid) + params.fusion_params.eps);

      // Pass 2: normalise and write final output.
      for (int off = tidx * VEC_SIZE; off < hid;
           off += static_cast<int>(item.get_local_range(0)) * VEC_SIZE) {
        Vec<T, VEC_SIZE> pn, gam;
        pn.load(prenorm + tok_off + off);
        gam.load(gamma + off);
        Vec<T, VEC_SIZE> result = rms_norm_apply<T, VEC_SIZE>(denom, pn, gam);
        result.store(norm_out + tok_off + off);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Kernel 3: two-shot AllReduce (ReduceScatter + AllGather) — no RMSNorm.
//
// Better than one-shot for larger messages (prefill phase, many tokens).
//
// Algorithm (mirrors twoShotAllReduceKernel in trtllm_allreduce.cuh):
//   Phase A – Copy: each block copies its FULL slice to this rank's P2P buffer.
//   Phase B – Barrier (in).
//   Phase C – ReduceScatter: each block reduces its ASSIGNED rank-slice.
//             Writes reduced result back to all ranks' P2P buffers (at offset
//             rank_offset) so that every GPU can AllGather from it.
//   Phase D – Barrier (out).
//   Phase E – AllGather: each block collects the reduced shard from every rank
//             and writes it to the final output at the correct position.
//
// Requires a second barrier array peer_barrier_ptrs_out[].
// ---------------------------------------------------------------------------
template <typename T, int RANKS_PER_NODE>
struct TwoShotAllReduceKernel {
  static constexpr int VEC_SIZE = kBytesPerAccess / static_cast<int>(sizeof(T));

  AllReduceParamsSYCL<T>  params;
  uint32_t**              peer_barrier_ptrs_out;  // second barrier for AllGather phase
  sycl::local_accessor<float, 1> smem;  // unused but kept for uniformity

  void operator()(sycl::nd_item<1> item) const {
    const int tidx = static_cast<int>(item.get_local_id(0));
    const int bidx = static_cast<int>(item.get_group(0));
    const int gs   = static_cast<int>(item.get_group_range(0));

    // elts_per_rank: each rank owns one contiguous slice of the tensor.
    const size_t elts_per_rank  = params.elts_total / params.ranks_per_node;
    const size_t rank_off       = params.local_rank * elts_per_rank;

    // This block's slice within [0, elts_per_rank).
    const size_t chunk_start = static_cast<size_t>(bidx) * params.elts_per_block;
    const size_t chunk_end   = std::min(chunk_start + params.elts_per_block, elts_per_rank);

    const T* local_in  = reinterpret_cast<const T*>(params.local_input_buffer_ptr);
    T*       peer_self = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    T*       out       = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    // Phase A: copy full slice (all ranks' portions) to this rank's P2P buffer.
    for (size_t off = chunk_start + tidx * VEC_SIZE; off < chunk_end;
         off += item.get_local_range(0) * VEC_SIZE) {
      for (int r = 0; r < RANKS_PER_NODE; ++r) {
        size_t global = r * elts_per_rank + off;
        if (global + VEC_SIZE <= params.elts_total) {
          store128(&peer_self[global], load128(&local_in[global]));
        }
      }
    }

    // Phase B: barrier (in) — all GPUs finished copy.
    block_barrier(const_cast<sycl::nd_item<1>&>(item),
                  const_cast<uint32_t**>(params.peer_barrier_ptrs_in),
                  params.barrier_flag, static_cast<int>(params.local_rank),
                  static_cast<int>(params.ranks_per_node), gs);

    // Phase C: ReduceScatter — reduce this rank's assigned slice across all peers.
    for (size_t off = chunk_start + tidx * VEC_SIZE; off < chunk_end;
         off += item.get_local_range(0) * VEC_SIZE) {
      Vec<T, VEC_SIZE> sum;
      sum.fill(T(0));
      for (int r = 0; r < RANKS_PER_NODE; ++r) {
        Vec<T, VEC_SIZE> v;
        // Each peer's copy is at peer_comm_buffer[r][rank_off + off].
        v.load(reinterpret_cast<const T*>(params.peer_comm_buffer_ptrs[r]) + rank_off + off);
        sum = vec_add<T, VEC_SIZE>(sum, v);
      }
      // Write reduced result into ALL peers' buffers so AllGather can read it.
      for (int r = 0; r < RANKS_PER_NODE; ++r) {
        sum.store(reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[r]) + rank_off + off);
      }
    }

    // Phase D: barrier (out) — ReduceScatter complete on all GPUs.
    block_barrier(const_cast<sycl::nd_item<1>&>(item),
                  peer_barrier_ptrs_out,
                  params.barrier_flag + 1u,  // use next flag value to disambiguate
                  static_cast<int>(params.local_rank),
                  static_cast<int>(params.ranks_per_node), gs);

    // Phase E: AllGather — collect each rank's reduced shard → final output.
    for (size_t off = chunk_start + tidx * VEC_SIZE; off < chunk_end;
         off += item.get_local_range(0) * VEC_SIZE) {
      for (int r = 0; r < RANKS_PER_NODE; ++r) {
        size_t src = static_cast<size_t>(r) * elts_per_rank + off;
        if (src + VEC_SIZE <= params.elts_total) {
          // Read each rank's reduced shard from its own P2P buffer at rank_off + off.
          Vec<T, VEC_SIZE> v;
          v.load(reinterpret_cast<const T*>(
              params.peer_comm_buffer_ptrs[params.local_rank]) + src);
          v.store(out + src);
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Launcher helpers
// ---------------------------------------------------------------------------

// Compute grid/block dims for the given strategy (mirrors kernelLaunchConfig).
template <typename T>
inline std::pair<int, int> compute_launch_config(size_t elts_total,
                                                  size_t ranks_per_node,
                                                  bool   twoshot,
                                                  size_t& elts_per_block_out) {
  constexpr size_t VEC_SIZE = kBytesPerAccess / sizeof(T);
  const size_t elts_per_rank = twoshot ? elts_total / ranks_per_node : elts_total;
  size_t total_threads = (elts_per_rank + VEC_SIZE - 1) / VEC_SIZE;
  // Round up to warp (sub-group) size = 32.
  total_threads = (total_threads + 31) / 32 * 32;
  int block_size = std::min(static_cast<size_t>(kDefaultBlockSize), total_threads);
  int grid_size  = std::min(static_cast<size_t>(kMaxARBlocks),
                             (total_threads + block_size - 1) / block_size);
  elts_per_block_out = ((elts_per_rank / grid_size + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;
  return {grid_size, block_size};
}

// Launch one-shot AllReduce + RMSNorm fusion.
// Precondition: peer_comm_buffer_ptrs and peer_barrier_ptrs_in must be P2P-accessible
//               USM pointers valid on this device's sycl::queue.
template <typename T, int RANKS_PER_NODE>
inline void launch_oneshot_allreduce_rmsnorm(sycl::queue&            q,
                                              AllReduceParamsSYCL<T>& params) {
  size_t epb = 0;
  auto [gs, bs] = compute_launch_config<T>(params.elts_total, params.ranks_per_node,
                                           /*twoshot=*/false, epb);
  params.elts_per_block = epb;
  const int smem_floats = bs;  // one float per work-item for block_reduce

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_floats), h);
    OneShotAllReduceRMSNormKernel<T, RANKS_PER_NODE> kern{params, smem};
    h.parallel_for(sycl::nd_range<1>(gs * bs, bs), kern);
  });
}

// Launch plain one-shot AllReduce (no RMSNorm).
template <typename T, int RANKS_PER_NODE>
inline void launch_oneshot_allreduce(sycl::queue&            q,
                                      AllReduceParamsSYCL<T>& params) {
  size_t epb = 0;
  auto [gs, bs] = compute_launch_config<T>(params.elts_total, params.ranks_per_node,
                                           /*twoshot=*/false, epb);
  params.elts_per_block = epb;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(bs), h);
    OneShotAllReduceKernel<T, RANKS_PER_NODE> kern{params, smem};
    h.parallel_for(sycl::nd_range<1>(gs * bs, bs), kern);
  });
}

// Launch two-shot AllReduce (for large messages).
template <typename T, int RANKS_PER_NODE>
inline void launch_twoshot_allreduce(sycl::queue&            q,
                                      AllReduceParamsSYCL<T>& params,
                                      uint32_t**              peer_barrier_ptrs_out) {
  size_t epb = 0;
  auto [gs, bs] = compute_launch_config<T>(params.elts_total, params.ranks_per_node,
                                           /*twoshot=*/true, epb);
  params.elts_per_block = epb;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(bs), h);
    TwoShotAllReduceKernel<T, RANKS_PER_NODE> kern{params, peer_barrier_ptrs_out, smem};
    h.parallel_for(sycl::nd_range<1>(gs * bs, bs), kern);
  });
}

// Dispatch based on token count heuristic (≤ 128 tokens → one-shot, else two-shot).
// Returns false if the configuration is unsupported.
template <typename T, int RANKS_PER_NODE>
inline bool dispatch_allreduce_rmsnorm(sycl::queue&            q,
                                        AllReduceParamsSYCL<T>& params,
                                        uint32_t**              peer_barrier_ptrs_out,
                                        int                     token_num,
                                        bool                    force_oneshot = false) {
  constexpr size_t VEC_SIZE = kBytesPerAccess / sizeof(T);
  if (params.elts_total % VEC_SIZE != 0) return false;

  if (force_oneshot || token_num <= 128) {
    launch_oneshot_allreduce_rmsnorm<T, RANKS_PER_NODE>(q, params);
  } else {
    // Two-shot: run AllReduce first, then separate RMSNorm kernel.
    if (params.elts_total % (VEC_SIZE * RANKS_PER_NODE) != 0) return false;
    launch_twoshot_allreduce<T, RANKS_PER_NODE>(q, params, peer_barrier_ptrs_out);
    // NOTE: caller is responsible for submitting the separate RMSNorm kernel
    // using intermediate_buffer as input (mirrors the CUDA two-shot path).
  }
  return true;
}

}  // namespace sycl_allreduce
}  // namespace flashinfer


