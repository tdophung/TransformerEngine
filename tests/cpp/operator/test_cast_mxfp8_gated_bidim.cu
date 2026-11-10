/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// gated_bidim (cast/mxfp8/specialized/gated_bidim.cuh) takes BF16 SwiGLU and dSwiGLU quantized in
// both directions in place of the generic gated kernel, and must write its bytes. The generic
// kernel's output comes from the same request with the column-wise scales one byte past an
// aligned address, which gated_bidim does not take. Both outputs start filled with a marker byte,
// so the two kernels must also write the same set of bytes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

size_t count_byte_mismatches(const uint8_t *a, const uint8_t *b, const size_t n) {
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) m += (a[i] != b[i]);
  return m;
}

enum class GatedFill { Normal, Wide, Specials, AllBF16 };

// Normal: N(0, 3). Wide: each 32x32 block scaled by its own power of two. Specials: a few
// infinities, NaNs, zeros and extreme values. AllBF16: the activated half (the first cols / 2
// columns) walks through every BF16 bit pattern.
void fill_gated(Tensor *t, const size_t rows, const size_t cols, const GatedFill kind,
                const int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  const float specials[] = {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),
                            0.0f,
                            -0.0f,
                            1e-30f,
                            -1e30f};
  bf16 *p = t->rowwise_cpu_dptr<bf16>();
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      if (kind == GatedFill::AllBF16 && c < cols / 2) {
        const uint16_t bits = static_cast<uint16_t>(r * (cols / 2) + c);
        std::memcpy(&p[r * cols + c], &bits, sizeof(bits));
        continue;
      }
      float v = normal(gen);
      if (kind == GatedFill::Wide) {
        const size_t block = (r / 32) * 104729 + c / 32 + seed;
        v = std::ldexp(v, static_cast<int>(block % 20) - 10);
      } else if (kind == GatedFill::Specials && gen() % 61 == 0) {
        v = specials[gen() % 7];
      }
      p[r * cols + c] = static_cast<bf16>(v);
    }
  }
  t->from_cpu();
}

void run_gated_case(const bool bwd, const GatedFill kind, const size_t rows, const size_t cols,
                    const bool swizzled, const DType otype) {
  Tensor input("input", std::vector<size_t>{rows, 2 * cols}, DType::kBFloat16);
  Tensor grad("grad", std::vector<size_t>{rows, cols}, DType::kBFloat16);
  fill_gated(&input, rows, 2 * cols, kind, 1);
  if (bwd) {
    fill_gated(&grad, rows, cols, kind == GatedFill::Wide ? kind : GatedFill::Normal, 2);
  }
  const std::vector<size_t> out_shape{rows, bwd ? 2 * cols : cols};
  Tensor out("out", out_shape, otype, true, true, NVTE_MXFP8_1D_SCALING);
  Tensor ref("ref", out_shape, otype, true, true, NVTE_MXFP8_1D_SCALING);
  const NVTEShape colwise_scale_shape = ref.columnwise_scale_inv_shape();
  const size_t colwise_scale_bytes = product(colwise_scale_shape);
  auto ref_colwise_scales = cuda_alloc<uint8_t>(colwise_scale_bytes + 1);
  const NVTEBasicTensor odd_scales = {ref_colwise_scales.get() + 1, kNVTEFloat8E8M0,
                                      colwise_scale_shape};
  nvte_set_tensor_param_v2(ref.data(), kNVTEColumnwiseScaleInv, &odd_scales, sizeof(odd_scales));
  for (Tensor *t : {&out, &ref}) {
    t->set_with_gemm_swizzled_scales(swizzled);
    for (const auto param :
         {kNVTERowwiseData, kNVTEColumnwiseData, kNVTERowwiseScaleInv, kNVTEColumnwiseScaleInv}) {
      NVTEBasicTensor buffer;
      nvte_get_tensor_param_v2(t->data(), param, &buffer, sizeof(buffer), nullptr);
      ASSERT_EQ(cudaMemset(buffer.data_ptr, 0xA5, product(buffer.shape)), cudaSuccess);
    }
    if (bwd) {
      nvte_dswiglu(grad.data(), input.data(), t->data(), 0);
    } else {
      nvte_swiglu(input.data(), t->data(), 0);
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  out.to_cpu();
  ref.to_cpu();
  std::vector<uint8_t> ref_colwise_scales_cpu(colwise_scale_bytes);
  ASSERT_EQ(cudaMemcpy(ref_colwise_scales_cpu.data(), ref_colwise_scales.get() + 1,
                       colwise_scale_bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  // The FP8 data as raw bytes, whichever FP8 type the output holds.
  const auto rowwise_bytes = [&](Tensor &t) {
    return otype == DType::kFloat8E5M2
               ? reinterpret_cast<const uint8_t *>(t.rowwise_cpu_dptr<fp8e5m2>())
               : reinterpret_cast<const uint8_t *>(t.rowwise_cpu_dptr<fp8e4m3>());
  };
  const auto columnwise_bytes = [&](Tensor &t) {
    return otype == DType::kFloat8E5M2
               ? reinterpret_cast<const uint8_t *>(t.columnwise_cpu_dptr<fp8e5m2>())
               : reinterpret_cast<const uint8_t *>(t.columnwise_cpu_dptr<fp8e4m3>());
  };
  const size_t n = product(out_shape);
  EXPECT_EQ(count_byte_mismatches(rowwise_bytes(out), rowwise_bytes(ref), n), 0u) << "rowwise data";
  EXPECT_EQ(count_byte_mismatches(columnwise_bytes(out), columnwise_bytes(ref), n), 0u)
      << "colwise data";
  EXPECT_EQ(count_byte_mismatches(out.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                  ref.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                  product(out.rowwise_scale_inv_shape())),
            0u)
      << "rowwise scales";
  EXPECT_EQ(count_byte_mismatches(out.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                                  ref_colwise_scales_cpu.data(), colwise_scale_bytes),
            0u)
      << "colwise scales";
}

using GatedParams = std::tuple<bool /*bwd*/, GatedFill, bool /*swizzled*/,
                               std::pair<size_t, size_t>, DType /*output*/>;

class CastMXFP8GatedBidimTestSuite : public ::testing::TestWithParam<GatedParams> {};

}  // namespace

TEST_P(CastMXFP8GatedBidimTestSuite, MatchesGenericKernel) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "This MXFP8 quantize kernel requires compute capability 10.0 or newer";
  }
  // The CuTeDSL backend takes these requests in place of the CUDA kernels compared here;
  // L1_cppunittest_cutedsl_off runs this suite with it off.
  const char *cutedsl = std::getenv("NVTE_ENABLE_CUTEDSL_BACKEND");
  if (cutedsl != nullptr && std::string(cutedsl) == "1") {
    GTEST_SKIP() << "Compares two CUDA kernels, and the CuTeDSL backend is on";
  }
  const auto [bwd, kind, swizzled, dims, otype] = GetParam();
  run_gated_case(bwd, kind, dims.first, dims.second, swizzled, otype);
}

namespace {

std::string gated_case_name(const testing::TestParamInfo<GatedParams> &info) {
  const auto [bwd, kind, swizzled, dims, otype] = info.param;
  const char *fills[] = {"Normal", "Wide", "Specials", "AllBF16"};
  return std::string(bwd ? "DSwiGLU" : "SwiGLU") + "X" + fills[static_cast<int>(kind)] + "X" +
         (swizzled ? "Swizzled" : "Compact") + "X" + std::to_string(dims.first) + "X" +
         std::to_string(dims.second) + (otype == DType::kFloat8E5M2 ? "XE5M2" : "XE4M3");
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastMXFP8GatedBidimTestSuite,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(GatedFill::Normal, GatedFill::Wide, GatedFill::Specials),
                       ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{1024, 2048},
                                         // Two row bands per CTA.
                                         std::pair<size_t, size_t>{4160, 2048},
                                         std::pair<size_t, size_t>{128, 256},
                                         // Row and column tails (K % 32 == 0); two row bands
                                         // per CTA.
                                         std::pair<size_t, size_t>{993, 7200},
                                         std::pair<size_t, size_t>{1, 32},
                                         std::pair<size_t, size_t>{33, 288},
                                         std::pair<size_t, size_t>{100, 2080}),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    gated_case_name);

// Every BF16 value in the activated half: 64 rows x 1024 columns.
INSTANTIATE_TEST_SUITE_P(
    AllBF16, CastMXFP8GatedBidimTestSuite,
    ::testing::Combine(::testing::Bool(), ::testing::Values(GatedFill::AllBF16), ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{64, 1024}),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    gated_case_name);

// 65537 row bands, more than grid.y holds with one band per CTA: gated_bidim leaves the request
// to the generic kernel.
INSTANTIATE_TEST_SUITE_P(
    LargeM, CastMXFP8GatedBidimTestSuite,
    ::testing::Combine(::testing::Values(false), ::testing::Values(GatedFill::Normal),
                       ::testing::Values(false),
                       ::testing::Values(std::pair<size_t, size_t>{65537 * 32, 32}),
                       ::testing::Values(DType::kFloat8E4M3)),
    gated_case_name);
