#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace transformer_engine {
namespace kf_rowwise {

// Launch the KF campaign rowwise MXFP8 quantization kernel.
// Requires SM90+ (uses bf16x2 PTX instructions: max.xorsign.abs.bf16x2,
// cvt.rn.satfinite.e4m3x2.bf16x2, cvt.rp.satfinite.ue8m0x2.f32).
//
//   x       : BF16 input   [M, K]
//   q       : FP8E4M3 output [M, K]
//   s       : E8M0 scale output [M, K/32]  (one byte per 32-element MX block)
//   M, K    : matrix dimensions (K must be divisible by 32)
//   sstride : number of scale elements per row (may be >= K/32 for padded layouts)
//   stream  : CUDA stream
void launch_mxfp8_kf(const void* x, void* q, void* s,
                     int M, int K, int sstride,
                     cudaStream_t stream);

}  // namespace kf_rowwise
}  // namespace transformer_engine
