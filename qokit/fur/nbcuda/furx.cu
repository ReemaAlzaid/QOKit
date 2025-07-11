//  call pattern (example)
//  -------------------------------------------------------------------------
//  dim3 grid = ...;
//  dim3 block = ...;                     // must still be 2^(n_q-1) threads
//  furx_kernel_quant<n_q, q_offset, state_mask><<<grid, block>>>(
//          psi_dev,     // state vector (now *optionally* int8|int16 packed)
//          scales_dev,  // one float scale per 256 amplitudes
//          quant_bits,  // 32, 16, or 8
//          jx, jy);     // the same 'a' and 'b' parameters
//  -------------------------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

// ───────────────────────────────────────────────────────────────────────────
// original helpers -- untouched
// ───────────────────────────────────────────────────────────────────────────
template <int n_q, int q_offset>
__device__ uint2 get_offset(const unsigned int block_idx){
    constexpr unsigned int stride      = 1 << q_offset;
    constexpr unsigned int index_mask  = stride - 1;
    constexpr unsigned int stride_mask = ~index_mask;
    const unsigned int offset = ((stride_mask & block_idx) << n_q) |
                                (index_mask  & block_idx);
    return {offset, stride};
}

__device__ constexpr double2 rot_x(const double a, const double b,
                                   double2& va, double2& vb){
    double2 temp = {a*va.x - b*vb.y,  a*va.y + b*vb.x};
    vb           = {a*vb.x - b*va.y,  a*vb.y + b*va.x};
    va = temp;
}

// ───────────────────────────────────────────────────────────────────────────
//      ORIGINAL kernels (kept verbatim)
// ───────────────────────────────────────────────────────────────────────────
template <int n_q, int q_offset, int state_mask>
__global__ void furx_kernel(double2* x, const double a, const double b){
    __shared__ double2 data[1 << n_q];
    constexpr unsigned int stride_size = 1 << (n_q-1);

    auto [offset, stride] = get_offset<n_q, q_offset>(blockIdx.x);
    const unsigned int tid = threadIdx.x;

    for(int i = 0; i < 2; ++i){
        const unsigned int idx = (tid+stride_size*i);
        data[idx] = x[offset + idx*stride];
    }

    __syncthreads();

    for(int q = 0; q < n_q; ++q){
        const unsigned int mask1 = (1 << q) - 1;
        const unsigned int mask2 = state_mask - mask1;

        const unsigned int ia = (tid & mask1) | ((tid & mask2) << 1);
        const unsigned int ib = ia | (1 << q);

        rot_x(a,b, data[ia], data[ib]);

        __syncthreads();
    }
    
    for(int i = 0; i < 2; ++i){
        const unsigned int idx = (tid+stride_size*i);
        x[offset + idx*stride] = data[idx];
    }
}

template <int n_q, int q_offset>
__global__ void warp_furx_kernel(double2* x, const double a, const double b) {
    constexpr unsigned int stride_size = 1 << (n_q-1);
    const unsigned int block_idx = blockIdx.x * blockDim.x/stride_size + threadIdx.x/stride_size;
    auto [offset, stride] = get_offset<n_q, q_offset>(block_idx);

    const unsigned int tid = threadIdx.x%stride_size;  
    const unsigned int load_offset = offset + (tid * 2)*stride;   

    double2 v[2] = {x[load_offset], x[load_offset + stride]};
    
    rot_x(a, b, v[0], v[1]);

    #pragma unroll
    for(int q = 0; q < n_q-1; ++q){
        const unsigned int warp_stride = 1 << q;
        const bool positive = !(tid & warp_stride);
        const unsigned int lane_idx = positive? tid + warp_stride : tid - warp_stride;

        v[positive].x = __shfl_sync(0xFFFFFFFF, v[positive].x, lane_idx, stride_size);
        v[positive].y = __shfl_sync(0xFFFFFFFF, v[positive].y, lane_idx, stride_size);
        
        rot_x(a, b, v[0], v[1]);
    }
    
    x[offset + tid*stride] = v[0];
    x[offset + (tid + stride_size)*stride] = v[1];
}

// ───────────────────────────────────────────────────────────────────────────
//      Quantisation helpers (device)
// ───────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
double2 ld_amp(const double2* __restrict__ x,
               const float*  __restrict__ scales,
               int quant_bits, unsigned gidx){
    double2 out;
    if(quant_bits == 32){
        out = reinterpret_cast<const double2*>(x)[gidx];
    }else{
        const float s = scales[gidx >> 8];                 // 256-amp blocks
        if(quant_bits == 16){
            short2 q = reinterpret_cast<const short2*>(x)[gidx];
            out.x = double(q.x) * s;
            out.y = double(q.y) * s;
        }else{                                             // 8-bit
            char2  q = reinterpret_cast<const char2 *>(x)[gidx];
            out.x = double(q.x) * s;
            out.y = double(q.y) * s;
        }
    }
    return out;
}

__device__ __forceinline__
void st_amp(double2* __restrict__ x,
            const float* __restrict__ scales,
            int quant_bits, unsigned gidx, const double2 v){
    if(quant_bits == 32){
        reinterpret_cast<double2*>(x)[gidx] = v;
    }else{
        const float sInv = 1.f / scales[gidx >> 8];
        if(quant_bits == 16){
            short2 q;
            q.x = short(__float2int_rn(float(v.x * sInv)));
            q.y = short(__float2int_rn(float(v.y * sInv)));
            reinterpret_cast<short2*>(x)[gidx] = q;
        }else{                                            // 8-bit
            char2 q;
            q.x = char(__float2int_rn(float(v.x * sInv)));
            q.y = char(__float2int_rn(float(v.y * sInv)));
            reinterpret_cast<char2*>(x)[gidx] = q;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
//      Quantised shared-memory kernel
// ───────────────────────────────────────────────────────────────────────────
template <int n_q, int q_offset, int state_mask>
__global__ void furx_kernel_quant(double2*       x,
                                  const float*   scales,
                                  int            quant_bits,
                                  const double   a,
                                  const double   b){
    __shared__ double2 data[1 << n_q];
    constexpr unsigned int stride_size = 1 << (n_q-1);

    auto [offset, stride] = get_offset<n_q, q_offset>(blockIdx.x);
    const unsigned int tid = threadIdx.x;

    // load two amplitudes / thread
    #pragma unroll
    for(int i=0;i<2;++i){
        const unsigned int idx  = tid + stride_size*i;
        const unsigned int gidx = offset + idx*stride;
        data[idx] = ld_amp(x, scales, quant_bits, gidx);
    }
    __syncthreads();

    // same butterfly as before
    #pragma unroll
    for(int q = 0; q < n_q; ++q){
        const unsigned int mask1 = (1u << q) - 1;
        const unsigned int mask2 = state_mask - mask1;

        const unsigned int ia = (tid & mask1) | ((tid & mask2) << 1);
        const unsigned int ib = ia | (1u << q);
        rot_x(a, b, data[ia], data[ib]);
        __syncthreads();
    }

    // write back
    #pragma unroll
    for(int i=0;i<2;++i){
        const unsigned int idx  = tid + stride_size*i;
        const unsigned int gidx = offset + idx*stride;
        st_amp(x, scales, quant_bits, gidx, data[idx]);
    }
}

// ───────────────────────────────────────────────────────────────────────────
//      Quantised warp-shuffle kernel
// ───────────────────────────────────────────────────────────────────────────
template <int n_q, int q_offset>
__global__ void warp_furx_kernel_quant(double2*     x,
                                       const float* scales,
                                       int          quant_bits,
                                       const double a,
                                       const double b){
    constexpr unsigned int stride_size = 1 << (n_q-1);
    const unsigned int block_idx = blockIdx.x * blockDim.x/stride_size +
                                   threadIdx.x/stride_size;
    auto [offset, stride] = get_offset<n_q, q_offset>(block_idx);

    const unsigned int tid  = threadIdx.x % stride_size;
    const unsigned int g0   = offset + (tid*2)*stride;
    const unsigned int g1   = g0 + stride;

    double2 v0 = ld_amp(x, scales, quant_bits, g0);
    double2 v1 = ld_amp(x, scales, quant_bits, g1);
    rot_x(a, b, v0, v1);

    #pragma unroll
    for(int q=0; q<n_q-1; ++q){
        const unsigned int warp_stride = 1u << q;
        const bool pos   = !(tid & warp_stride);
        const unsigned   lane = pos ? tid + warp_stride : tid - warp_stride;

        double tx = __shfl_sync(0xFFFFFFFF, pos ? v0.x : v1.x,
                                lane, stride_size);
        double ty = __shfl_sync(0xFFFFFFFF, pos ? v0.y : v1.y,
                                lane, stride_size);
        if(pos){ v0.x = tx; v0.y = ty; }
        else   { v1.x = tx; v1.y = ty; }

        rot_x(a, b, v0, v1);
    }
    st_amp(x, scales, quant_bits, g0, v0);
    st_amp(x, scales, quant_bits, g1, v1);
}
