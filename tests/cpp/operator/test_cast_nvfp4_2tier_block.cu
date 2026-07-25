/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
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

constexpr size_t kInnerBlockSize = 16;
constexpr size_t kOuterBlockSize = 256;
constexpr float kE4M3Max = 448.0f;
constexpr float kE2M1Max = 6.0f;

fp8e4m3 ref_outer_scale(float amax) {
  if (amax == 0.0f) {
    return static_cast<fp8e4m3>(1.0f);
  }
  return static_cast<fp8e4m3>(std::fmin(amax / (kE4M3Max * kE2M1Max), kE4M3Max));
}

fp8e4m3 ref_inner_scale(float amax, float outer_scale) {
  if (outer_scale == 0.0f) {
    return amax == 0.0f ? static_cast<fp8e4m3>(0.0f)
                        : static_cast<fp8e4m3>(kE4M3Max);
  }
  return static_cast<fp8e4m3>(
      std::fmin((amax / kE2M1Max) / outer_scale, kE4M3Max));
}

float ref_encode_scale(float inner_scale, float outer_scale) {
  const float decode_scale = inner_scale * outer_scale;
  return decode_scale == 0.0f ? FLT_MAX : std::fmin(1.0f / decode_scale, FLT_MAX);
}

template <typename IType>
void compute_reference(const IType *input, size_t rows, size_t cols, size_t s1_stride,
                       size_t s2_stride, std::vector<uint8_t> *packed,
                       std::vector<fp8e4m3> *s1, std::vector<fp8e4m3> *s2) {
  for (size_t row = 0; row < rows; ++row) {
    for (size_t outer = 0; outer < cols / kOuterBlockSize; ++outer) {
      float outer_amax = 0.0f;
      for (size_t i = 0; i < kOuterBlockSize; ++i) {
        outer_amax = std::fmax(
            outer_amax,
            std::fabs(static_cast<float>(input[row * cols + outer * kOuterBlockSize + i])));
      }
      const fp8e4m3 outer_stored = ref_outer_scale(outer_amax);
      (*s2)[row * s2_stride + outer] = outer_stored;
      const float outer_scale = static_cast<float>(outer_stored);

      for (size_t inner = 0; inner < kOuterBlockSize / kInnerBlockSize; ++inner) {
        const size_t col = outer * kOuterBlockSize + inner * kInnerBlockSize;
        float inner_amax = 0.0f;
        for (size_t i = 0; i < kInnerBlockSize; ++i) {
          inner_amax =
              std::fmax(inner_amax, std::fabs(static_cast<float>(input[row * cols + col + i])));
        }
        const fp8e4m3 inner_stored = ref_inner_scale(inner_amax, outer_scale);
        (*s1)[row * s1_stride + outer * 16 + inner] = inner_stored;
        const float encode_scale =
            ref_encode_scale(static_cast<float>(inner_stored), outer_scale);

        for (size_t pair = 0; pair < kInnerBlockSize / 2; ++pair) {
          const float2 values = {
              static_cast<float>(input[row * cols + col + 2 * pair]) * encode_scale,
              static_cast<float>(input[row * cols + col + 2 * pair + 1]) * encode_scale};
          const fp4e2m1x2 fp4_pair(values);
          const size_t byte =
              row * (cols / 2) + outer * (kOuterBlockSize / 2) +
              inner * (kInnerBlockSize / 2) + pair;
          std::memcpy(packed->data() + byte, &fp4_pair, 1);
        }
      }
    }
  }
}

template <typename IType>
void run_quantize_case(size_t rows, size_t cols, bool all_zero) {
  const DType dtype = TypeInfo<IType>::dtype;
  Tensor input("input_2tier", std::vector<size_t>{rows, cols}, dtype);
  Tensor quantized("quantized_2tier", std::vector<size_t>{rows, cols}, DType::kFloat4E2M1,
                   true, false, NVTE_NVFP4_2TIER_BLOCK_SCALING);

  IType *input_cpu = input.rowwise_cpu_dptr<IType>();
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      float value = 0.0f;
      if (!all_zero) {
        const float sign = (row + col) % 2 == 0 ? 1.0f : -1.0f;
        value = sign * static_cast<float>((col % 37) + 1) / 9.0f;
        if (col % kOuterBlockSize == 0) {
          value = (col == 0 ? 2.0e7f : 1.0e-8f) * sign;
        }
        if (col % 53 == 0) {
          value = 0.0f;
        }
        if (col == 127) {
          value = std::numeric_limits<float>::infinity();
        }
      }
      input_cpu[row * cols + col] = static_cast<IType>(value);
    }
  }
  input.from_cpu();

  // Make padding observably nonzero before the API call. Quantization
  // must zero it as part of the deterministic padded-layout contract.
  const NVTEShape s1_shape = quantized.rowwise_scale_inv_shape();
  const NVTEShape s2_shape = quantized.nvfp4_scale_inv_2_shape();
  const size_t s1_numel = product(s1_shape);
  const size_t s2_numel = product(s2_shape);
  fp8e4m3 *s1_pre = quantized.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
  fp8e4m3 *s2_pre = quantized.cpu_nvfp4_scale_inv_2_ptr<fp8e4m3>();
  std::fill(s1_pre, s1_pre + s1_numel, static_cast<fp8e4m3>(kE4M3Max));
  std::fill(s2_pre, s2_pre + s2_numel, static_cast<fp8e4m3>(kE4M3Max));
  quantized.from_cpu();

  nvte_nvfp4_2tier_block_quantize(input.data(), quantized.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  quantized.to_cpu();

  std::vector<uint8_t> ref_packed(rows * cols / 2, 0);
  std::vector<fp8e4m3> ref_s1(s1_numel, static_cast<fp8e4m3>(0.0f));
  std::vector<fp8e4m3> ref_s2(s2_numel, static_cast<fp8e4m3>(0.0f));
  compute_reference(input_cpu, rows, cols, s1_shape.data[1], s2_shape.data[1], &ref_packed,
                    &ref_s1, &ref_s2);

  const auto *packed = reinterpret_cast<const uint8_t *>(
      quantized.rowwise_cpu_dptr<fp4e2m1>());
  const auto *s1 = quantized.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
  const auto *s2 = quantized.cpu_nvfp4_scale_inv_2_ptr<fp8e4m3>();
  EXPECT_EQ(std::memcmp(packed, ref_packed.data(), ref_packed.size()), 0);
  EXPECT_EQ(std::memcmp(s1, ref_s1.data(), s1_numel * sizeof(fp8e4m3)), 0);
  EXPECT_EQ(std::memcmp(s2, ref_s2.data(), s2_numel * sizeof(fp8e4m3)), 0);
}

TEST(NVFP4TwoTierBlockQuantize, FP32PartialMTileAndMultipleOuterBlocks) {
  run_quantize_case<float>(17, 512, false);
}

TEST(NVFP4TwoTierBlockQuantize, BF16PartialMTileAndMultipleOuterBlocks) {
  run_quantize_case<bf16>(19, 512, false);
}

TEST(NVFP4TwoTierBlockQuantize, ZeroAmaxUsesBenignOuterScale) {
  run_quantize_case<float>(3, 256, true);
}

}  // namespace

#endif  // FP4_TYPE_SUPPORTED
