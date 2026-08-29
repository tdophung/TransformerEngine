#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace transformer_engine {
namespace kf_bidim {

// Launch the KF campaign bidimensional MXFP8 quantization kernel.
// Requires SM90+ (uses bf16x2 PTX: max.xorsign.abs.bf16x2,
// cvt.rn.satfinite.e4m3x2.bf16x2, cudaLaunchKernelEx for clusters).
//
//   x            : BF16 input         [M, K]
//   qrow         : FP8E4M3 rowwise output [M, K]  (row-major, same layout as input)
//   srow         : E8M0 rowwise scales  [M, K/32]  (one byte per 32-col MX block)
//   qcol         : FP8E4M3 colwise output [M, K]   (row-major, NOT transposed)
//   scol         : E8M0 colwise scales  [M/32, K]  (one byte per 32-row MX block)
//   M, K         : matrix dimensions (must be multiples of 32)
//   srow_stride  : number of scale elements per row in srow (>= K/32)
//   scol_stride  : number of scale elements per row in scol (>= K)
//   stream       : CUDA stream
void launch_mxfp8_kf_bidim(const void* x, void* qrow, void* srow,
                            void* qcol, void* scol,
                            int M, int K, int srow_stride, int scol_stride,
                            cudaStream_t stream);

}  // namespace kf_bidim
}  // namespace transformer_engine
