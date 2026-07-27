/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_fp8.h>
#include <gtest/gtest.h>

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <transformer_engine/cast.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

#if FP4_TYPE_SUPPORTED

namespace {

float2 unpack_pair(const uint8_t byte) {
  fp4e2m1x2 pair;
  std::memcpy(&pair, &byte, 1);
  const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
      *reinterpret_cast<const __nv_fp4x2_storage_t *>(&pair), __NV_E2M1);
  const __half2 values(raw);
  return {static_cast<float>(values.x), static_cast<float>(values.y)};
}

template <typename OType>
void run_dequantize_case(size_t rows, size_t cols) {
  Tensor input("input_2tier_dequant", std::vector<size_t>{rows, cols}, DType::kFloat32);
  Tensor quantized("quantized_2tier_dequant", std::vector<size_t>{rows, cols},
                   DType::kFloat4E2M1, true, false,
                   NVTE_NVFP4_2TIER_BLOCK_SCALING);
  Tensor output("output_2tier_dequant", std::vector<size_t>{rows, cols},
                TypeInfo<OType>::dtype);

  float *input_cpu = input.rowwise_cpu_dptr<float>();
  for (size_t i = 0; i < rows * cols; ++i) {
    input_cpu[i] = std::sin(static_cast<float>(i) * 0.03125f) *
                   static_cast<float>((i % 101) + 1);
  }
  input.from_cpu();
  nvte_quantize(input.data(), quantized.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  nvte_nvfp4_2tier_block_dequantize(quantized.data(), output.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  quantized.to_cpu();
  const auto *packed = reinterpret_cast<const uint8_t *>(
      quantized.rowwise_cpu_dptr<fp4e2m1>());
  const auto *s1 = quantized.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
  const auto *s2 = quantized.cpu_nvfp4_scale_inv_2_ptr<fp8e4m3>();
  const size_t s1_stride = quantized.rowwise_scale_inv_shape().data[1];
  const size_t s2_stride = quantized.nvfp4_scale_inv_2_shape().data[1];

  std::vector<OType> reference(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; col += 2) {
      const float2 values = unpack_pair(packed[row * (cols / 2) + col / 2]);
      const float final_scale =
          static_cast<float>(s1[row * s1_stride + col / 16]) *
          static_cast<float>(s2[row * s2_stride + col / 256]);
      reference[row * cols + col] = static_cast<OType>(values.x * final_scale);
      reference[row * cols + col + 1] = static_cast<OType>(values.y * final_scale);
    }
  }

  auto [atol, rtol] = getTolerances(TypeInfo<OType>::dtype);
  compareResults("nvfp4_2tier_dequant", output, reference.data(), true, atol, rtol);
}

TEST(NVFP4TwoTierBlockDequantize, FP32PartialMTileAndMultipleOuterBlocks) {
  run_dequantize_case<float>(17, 512);
}

TEST(NVFP4TwoTierBlockDequantize, BF16PartialMTileAndMultipleOuterBlocks) {
  run_dequantize_case<bf16>(19, 512);
}

// Match the large-M quantization coverage so a future 64x256 CTA tile with
// four-way M looping is checked in both directions.
TEST(NVFP4TwoTierBlockDequantize, FP32Multiple64RowTiles) {
  run_dequantize_case<float>(256, 512);
}

TEST(NVFP4TwoTierBlockDequantize, BF16LargeMultiple64RowTilesAndOuterBlocks) {
  run_dequantize_case<bf16>(1024, 1024);
}

}  // namespace

#endif  // FP4_TYPE_SUPPORTED
