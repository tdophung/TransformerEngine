/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file kernel.h
 *  \brief Entry point for the register-resident rowwise MXFP8 cast kernel.
 */

#ifndef TRANSFORMER_ENGINE_MXFP8_KF_ROWWISE_KERNEL_H_
#define TRANSFORMER_ENGINE_MXFP8_KF_ROWWISE_KERNEL_H_

#include <cuda_runtime.h>

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace kf_rowwise {

/*! \brief Cast BF16 to rowwise-scaled MXFP8.
 *
 * Every run of 32 consecutive elements in a row forms one MX block sharing a
 * single E8M0 scale.  Unlike the TMA kernel in mxfp8/specialized this one is
 * register-resident, which wins on the cast-only path; see cast_kf_rowwise.cu
 * for the layout and tuning rationale.
 *
 * Requires SM 10.0+ (Blackwell), matching MXFP8 support in the rest of TE.
 *
 *  \param[in]  input         BF16 input, [rows, cols], row-major.
 *  \param[out] output        FP8E4M3 output, [rows, cols], row-major.
 *  \param[out] scales        E8M0 scales, one byte per MX block.
 *  \param[in]  rows          Number of rows.
 *  \param[in]  cols          Number of columns; must be a multiple of 32.
 *  \param[in]  scale_stride  Scale elements per row.  Equals cols/32 for a
 *                            packed scale array, or more when it is padded.
 *  \param[in]  stream        CUDA stream.
 */
void launch_mxfp8_kf(const void *input, void *output, void *scales, int rows, int cols,
                     int scale_stride, cudaStream_t stream);

}  // namespace kf_rowwise
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_MXFP8_KF_ROWWISE_KERNEL_H_
