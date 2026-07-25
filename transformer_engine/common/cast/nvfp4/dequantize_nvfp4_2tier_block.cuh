/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_nvfp4_2tier_block.cuh
 *  \brief Rowwise NVFP4 2-tier block dequantization.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_2TIER_BLOCK_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_2TIER_BLOCK_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4_2tier_block {

#if FP4_TYPE_SUPPORTED

namespace dequantize_detail {

using namespace transformer_engine::ptx;
using namespace nvfp4::core::two_tier_block_scaling;

template <typename OType>
__global__ void __launch_bounds__(256) dequantize_kernel(
    const uint8_t *const input, OType *const output, const fp8e4m3 *const scale_inv_1,
    const fp8e4m3 *const scale_inv_2, const size_t rows, const size_t cols,
    const size_t scale_inv_1_stride, const size_t scale_inv_2_stride) {
  constexpr int kThreadsPerWarp = 32;

  const int warp = threadIdx.x / kThreadsPerWarp;
  const int lane = threadIdx.x % kThreadsPerWarp;
  const int half_warp = lane / static_cast<int>(kInnerBlockSize);
  const int inner = lane % static_cast<int>(kInnerBlockSize);
  const size_t row_local = 2 * warp + half_warp;
  const size_t row = blockIdx.y * kRowsPerCTA + row_local;
  if (row >= rows) {
    return;
  }

  const size_t outer = blockIdx.x;
  const size_t col = outer * kOuterBlockSize + inner * kInnerBlockSize;
  const size_t packed_col = outer * (kOuterBlockSize / 2) + inner * (kInnerBlockSize / 2);

  union PackedBlock {
    uint64_t bytes;
    fp4e2m1x4 values[4];
  } packed;
  packed.bytes =
      *reinterpret_cast<const uint64_t *>(input + row * (cols / 2) + packed_col);

  const float inner_scale =
      static_cast<float>(scale_inv_1[row * scale_inv_1_stride + outer * 16 + inner]);
  const float outer_scale =
      static_cast<float>(scale_inv_2[row * scale_inv_2_stride + outer]);
  const float final_scale = inner_scale * outer_scale;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 values = static_cast<float4>(packed.values[i]);
    Vec<OType, 4> out;
    out.data.elt[0] = static_cast<OType>(values.x * final_scale);
    out.data.elt[1] = static_cast<OType>(values.y * final_scale);
    out.data.elt[2] = static_cast<OType>(values.z * final_scale);
    out.data.elt[3] = static_cast<OType>(values.w * final_scale);
    out.store_to_elts(output + row * cols + col + 4 * i);
  }
}

}  // namespace dequantize_detail

#endif  // FP4_TYPE_SUPPORTED

inline void dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output");
  NVTE_CHECK(input.scaling_mode == NVTE_NVFP4_2TIER_BLOCK_SCALING,
             "Input must use NVTE_NVFP4_2TIER_BLOCK_SCALING.");
  NVTE_CHECK(input.has_data() && !input.has_columnwise_data(),
             "NVFP4 2-tier block dequantization requires rowwise-only input.");
  NVTE_CHECK(output->has_data() && !output->has_columnwise_data(),
             "NVFP4 2-tier block dequantization produces rowwise-only output.");
  NVTE_CHECK(input.data.dtype == DType::kFloat4E2M1,
             "NVFP4 2-tier block dequantization input must be FP4.");
  NVTE_CHECK(is_high_precision_dtype(output->data.dtype),
             "NVFP4 2-tier block dequantization output must be BF16, FP16, or FP32.");
  NVTE_CHECK(input.data.shape == output->data.shape,
             "NVFP4 2-tier block dequantization input and output shapes must match.");
  NVTE_CHECK(!input.with_gemm_swizzled_scales,
             "NVFP4 2-tier block dequantization v0 requires unswizzled scales.");
  NVTE_CHECK(input.nvfp4_e4m3_max == dequantize_detail::kE4M3Max,
             "NVFP4 2-tier block scaling v0 requires E4M3 max 448.");

  const auto [rows, cols] = input.flat_2d_dims();
  NVTE_CHECK(cols % dequantize_detail::kOuterBlockSize == 0,
             "NVFP4 2-tier block dequantization requires K % 256 == 0 (got K=", cols, ").");
  if (rows == 0 || cols == 0) {
    return;
  }

  NVTE_CHECK(is_supported_by_CC_100(),
             "NVFP4 2-tier block dequantization requires compute capability 10.0 or newer.");

  constexpr int threads = 256;
  const dim3 blocks(
      cols / dequantize_detail::kOuterBlockSize,
      DIVUP(rows, static_cast<size_t>(dequantize_detail::kRowsPerCTA)));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,
      dequantize_detail::dequantize_kernel<OType><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const uint8_t *>(input.data.dptr),
          reinterpret_cast<OType *>(output->data.dptr),
          reinterpret_cast<const fp8e4m3 *>(input.scale_inv.dptr),
          reinterpret_cast<const fp8e4m3 *>(input.scale_inv_2.dptr), rows, cols,
          input.scale_inv.shape.back(), input.scale_inv_2.shape.back()););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4_2tier_block
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_2TIER_BLOCK_CUH_
