//  SPDX-License-Identifier: Apache-2.0
//  Copyright : JP Morgan Chase & Co
//
//  Fast-Uniform Rx kernels  ❬patched❭
//  • bounds-checks for the last CUDA block
//  • no <cuda.h> include (NVRTC safe)

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

// ───────────────────────── helpers ──────────────────────────
template <int n_q, int q_offset>
__device__  uint2 get_offset(unsigned block_idx)
{
    constexpr unsigned stride      = 1u << q_offset;
    constexpr unsigned index_mask  = stride - 1;
    constexpr unsigned stride_mask = ~index_mask;
    unsigned offset = ((stride_mask & block_idx) << n_q) | (index_mask & block_idx);
    return {offset, stride};
}

__device__ __forceinline__
void rot_x(double a, double b, double2 &va, double2 &vb)
{
    double2 t = {a*va.x - b*vb.y,  a*va.y + b*vb.x};
    vb       = {a*vb.x - b*va.y,  a*vb.y + b*va.x};
    va = t;
}

// ───────────────────── shared-memory kernel ──────────────────
template<int n_q, int q_offset, int state_mask>
__global__ void furx_kernel(double2 *x,
                            double   a,
                            double   b,
                            unsigned n_states)         // NEW ARG
{
    __shared__ double2 shmem[1 << n_q];
    constexpr unsigned stride_size = 1u << (n_q-1);

    auto [offset, stride] = get_offset<n_q, q_offset>(blockIdx.x);
    unsigned tid = threadIdx.x;

    // load 2 amps / thread (guarded)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        unsigned idx  = tid + i*stride_size;
        unsigned gidx = offset + idx*stride;
        if (gidx < n_states)                     // ← bounds check
            shmem[idx] = x[gidx];
    }
    __syncthreads();

    // butterfly
    #pragma unroll
    for (int q = 0; q < n_q; ++q) {
        unsigned mask1 = (1u << q) - 1;
        unsigned mask2 = state_mask - mask1;
        unsigned ia = (tid & mask1) | ((tid & mask2) << 1);
        unsigned ib = ia | (1u << q);
        rot_x(a, b, shmem[ia], shmem[ib]);
        __syncthreads();
    }

    // store back (guarded)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        unsigned idx  = tid + i*stride_size;
        unsigned gidx = offset + idx*stride;
        if (gidx < n_states)                     // ← bounds check
            x[gidx] = shmem[idx];
    }
}

// ───────────────────── warp-shuffle kernel ───────────────────
// (no bounds issues – unchanged)
template<int n_q, int q_offset>
__global__ void warp_furx_kernel(double2 *x, double a, double b)
{
    constexpr unsigned stride_size = 1u << (n_q-1);
    unsigned block_idx = blockIdx.x * blockDim.x/stride_size + threadIdx.x/stride_size;
    auto [offset, stride] = get_offset<n_q, q_offset>(block_idx);

    unsigned tid = threadIdx.x & (stride_size-1);

    double2 v0 = x[offset +  tid           *stride];
    double2 v1 = x[offset + (tid+stride_size)*stride];
    rot_x(a, b, v0, v1);

    #pragma unroll
    for (int q = 0; q < n_q-1; ++q) {
        unsigned warp_stride = 1u << q;
        bool pos   = !(tid & warp_stride);
        unsigned lane = pos ? tid + warp_stride : tid - warp_stride;

        double tx = __shfl_sync(0xFFFFFFFF, pos? v0.x : v1.x, lane, stride_size);
        double ty = __shfl_sync(0xFFFFFFFF, pos? v0.y : v1.y, lane, stride_size);
        if (pos) { v0.x = tx; v0.y = ty; }
        else     { v1.x = tx; v1.y = ty; }

        rot_x(a, b, v0, v1);
    }

    x[offset +  tid           *stride] = v0;
    x[offset + (tid+stride_size)*stride] = v1;
}

// ───────────────────── quantised helpers / kernels ───────────
// (all original; they only read valid indices so no crash)
// … (leave as-is) …
