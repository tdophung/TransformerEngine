/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4_2tier_block.cuh
 *  \brief Correctness-first rowwise NVFP4 2-tier block quantization.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_NVFP4_2TIER_BLOCK_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_NVFP4_2TIER_BLOCK_CUH_

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

namespace detail {

using namespace transformer_engine::detail;
using namespace transformer_engine::ptx;
using namespace nvfp4::core::two_tier_block_scaling;

template <typename IType>
__global__ void __launch_bounds__(256) quantize_kernel(
    const IType *const input, uint8_t *const output, fp8e4m3 *const scale_inv_1,
    fp8e4m3 *const scale_inv_2, const size_t rows, const size_t cols,
    const size_t scale_inv_1_stride, const size_t scale_inv_2_stride) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kThreadsPerCTA = 256;
  static_assert(kThreadsPerCTA / kThreadsPerWarp * 2 == kRowsPerCTA);

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
  const IType *const input_block = input + row * cols + col;

  Vec<IType, 8> input_lo;
  Vec<IType, 8> input_hi;
  // load_from_elts uses a vector load when the address has the required
  // alignment and safely scalarizes otherwise.
  input_lo.load_from_elts(input_block);
  input_hi.load_from_elts(input_block + 8);

  float values[kInnerBlockSize];
  float inner_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = static_cast<float>(input_lo.data.elt[i]);
    values[i + 8] = static_cast<float>(input_hi.data.elt[i]);
    inner_amax = fmaxf(inner_amax, fabsf(values[i]));
    inner_amax = fmaxf(inner_amax, fabsf(values[i + 8]));
  }

  const unsigned int half_warp_mask =
      half_warp == 0 ? static_cast<unsigned int>(0x0000ffff)
                     : static_cast<unsigned int>(0xffff0000);
  float outer_amax = inner_amax;
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    outer_amax =
        fmaxf(outer_amax, __shfl_down_sync(half_warp_mask, outer_amax, offset, 16));
  }
  outer_amax = __shfl_sync(half_warp_mask, outer_amax, 0, 16);

  float outer_scale = 0.0f;
  if (inner == 0) {
    const fp8e4m3 stored_outer_scale = compute_outer_decode_scale(outer_amax);
    scale_inv_2[row * scale_inv_2_stride + outer] = stored_outer_scale;
    outer_scale = static_cast<float>(stored_outer_scale);
  }
  // Broadcast the post-cast S2 value. S1 must be derived from exactly
  // the same value that dequantization will load.
  // TODO(nvfp4-2tier): Re-examine this numerical choice only with
  // bit-exact roundtrip evidence.
  outer_scale = __shfl_sync(half_warp_mask, outer_scale, 0, 16);

  const fp8e4m3 stored_inner_scale =
      compute_inner_decode_scale(inner_amax, outer_scale);
  scale_inv_1[row * scale_inv_1_stride + outer * 16 + inner] = stored_inner_scale;
  const float encode_scale =
      compute_encode_scale(static_cast<float>(stored_inner_scale), outer_scale);

  Vec<fp4e2m1x4, 4> packed;
  const float2 scale = make_float2(encode_scale, encode_scale);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int j = 4 * i;
    const float2 in01 = make_float2(values[j], values[j + 1]);
    const float2 in23 = make_float2(values[j + 2], values[j + 3]);
    packed.data.elt[i] =
        mul_cvt_fp32_to_fp4_4x</*USE_STOCHASTIC_ROUNDING=*/false>(in01, in23, scale, 0);
  }
  const size_t packed_col = outer * (kOuterBlockSize / 2) + inner * (kInnerBlockSize / 2);
  packed.store_to(output + row * (cols / 2) + packed_col);
}

}  // namespace detail

#endif  // FP4_TYPE_SUPPORTED

inline void quantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output");
  NVTE_CHECK(output->scaling_mode == NVTE_NVFP4_2TIER_BLOCK_SCALING,
             "Output must use NVTE_NVFP4_2TIER_BLOCK_SCALING.");
  NVTE_CHECK(input.has_data() && !input.has_columnwise_data(),
             "NVFP4 2-tier block quantization requires rowwise-only input.");
  NVTE_CHECK(output->has_data() && !output->has_columnwise_data(),
             "NVFP4 2-tier block quantization produces rowwise-only output.");
  NVTE_CHECK(is_high_precision_dtype(input.data.dtype),
             "NVFP4 2-tier block quantization input must be BF16, FP16, or FP32.");
  NVTE_CHECK(output->data.dtype == DType::kFloat4E2M1,
             "NVFP4 2-tier block quantization output must be FP4.");
  NVTE_CHECK(input.data.shape == output->data.shape,
             "NVFP4 2-tier block quantization input and output shapes must match.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "NVFP4 2-tier block quantization v0 requires unswizzled scales.");
  NVTE_CHECK(!output->row_scaled_nvfp4,
             "NVFP4 row-scaled metadata is not valid for 2-tier block scaling.");
  NVTE_CHECK(output->nvfp4_e4m3_max == detail::kE4M3Max,
             "NVFP4 2-tier block scaling v0 requires E4M3 max 448.");

  const auto [rows, cols] = input.flat_2d_dims();
  NVTE_CHECK(cols % detail::kOuterBlockSize == 0,
             "NVFP4 2-tier block quantization requires K % 256 == 0 (got K=", cols, ").");
  if (rows == 0 || cols == 0) {
    return;
  }

  NVTE_CHECK(is_supported_by_CC_100(),
             "NVFP4 2-tier block quantization requires compute capability 10.0 or newer.");

  // Padding is part of the v0 physical contract and is kept deterministic.
  NVTE_CHECK_CUDA(cudaMemsetAsync(output->scale_inv.dptr, 0,
                                  output->scale_inv.buffer_size_bytes(), stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(output->scale_inv_2.dptr, 0,
                                  output->scale_inv_2.buffer_size_bytes(), stream));

  constexpr int threads = 256;
  const dim3 blocks(cols / detail::kOuterBlockSize,
                    DIVUP(rows, static_cast<size_t>(detail::kRowsPerCTA)));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.data.dtype, IType,
      detail::quantize_kernel<IType><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const IType *>(input.data.dptr),
          reinterpret_cast<uint8_t *>(output->data.dptr),
          reinterpret_cast<fp8e4m3 *>(output->scale_inv.dptr),
          reinterpret_cast<fp8e4m3 *>(output->scale_inv_2.dptr), rows, cols,
          output->scale_inv.shape.back(), output->scale_inv_2.shape.back()););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4_2tier_block
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_NVFP4_2TIER_BLOCK_CUH_
