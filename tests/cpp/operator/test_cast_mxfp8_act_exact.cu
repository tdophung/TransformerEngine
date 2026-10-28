/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The fused MXFP8 activation quantize must round exactly as if the activation were computed per
// element, truncated to the input type, and then quantized on its own. This compares the fused
// kernels (CUDA, or CuTeDSL with NVTE_ENABLE_CUTEDSL_BACKEND=1) bit for bit against that
// composition: the activation is computed on the GPU, and the result is quantized by the generic
// cast-only kernel (a noop tensor keeps the call off the specialized cast kernels).
// dbias is compared against a host sum in the order the kernels accumulate it, and the tensor amax
// against the largest magnitude of the activation.
//
// Without the tables (NVTE_ACTIVATION_TABLE=0, the default) the activation is the scalar util/math.h
// function. With NVTE_ACTIVATION_TABLE=1, BF16 inputs follow the table specification (see util/activation_table.cuh): -inf
// gives -0 in the forward functions, and the derivatives below the table window are the linear
// Taylor term. Other input types do not use the tables.
//
// gated_bidim (cast/mxfp8/specialized/gated_bidim.cuh) takes BF16 SwiGLU and dSwiGLU quantized in
// both directions in place of the generic gated kernel, and must write its bytes. The generic
// kernel's output comes from the same request with the column-wise scales one byte past an
// aligned address, which gated_bidim does not take. Both outputs start filled with a marker byte,
// so the two kernels must also write the same set of bytes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"
#include "util/math.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class Mode { GeLU, DGeLU, DBiasDGeLU, SiLU, DSiLU, DBiasDSiLU };

bool is_dbias(const Mode m) { return m == Mode::DBiasDGeLU || m == Mode::DBiasDSiLU; }

const char *mode_name(const Mode m) {
  switch (m) {
    case Mode::GeLU:
      return "GeLU";
    case Mode::DGeLU:
      return "DGeLU";
    case Mode::DBiasDGeLU:
      return "DBiasDGeLU";
    case Mode::SiLU:
      return "SiLU";
    case Mode::DSiLU:
      return "DSiLU";
    default:
      return "DBiasDSiLU";
  }
}

// act: y = OP(x); dact: y = grad * dOP(x), as the fused kernels form it in FP32. z keeps the FP32
// value (dbias sums it before truncation), y the value truncated to the input type.
template <typename IType>
__global__ void scalar_activation_kernel(const Mode mode, const bool table_spec,
                                         const float dsilu_window_low, const IType *x,
                                         const IType *grad, IType *y, float *z, const size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float xv = static_cast<float>(x[i]);
  const Empty e{};
  // With a table, -inf gives -0 in the forward functions, and the derivatives below
  // 2^-12 (dGeLU) and dsilu_window_low (dSiLU) are their linear Taylor terms.
  const bool negative_inf = table_spec && xv == -INFINITY;
  const float magnitude = fabsf(xv);
  float v;
  switch (mode) {
    case Mode::GeLU:
      v = negative_inf ? -0.0f : gelu<float, float>(xv, e);
      break;
    case Mode::SiLU:
      v = negative_inf ? -0.0f : silu<float, float>(xv, e);
      break;
    case Mode::DGeLU:
    case Mode::DBiasDGeLU:
      v = static_cast<float>(grad[i]);
      v *= table_spec && magnitude < 0x1p-12f ? fmaf(0.79788456f, xv, 0.5f)
                                              : dgelu<float, float>(xv, e);
      break;
    default:
      v = static_cast<float>(grad[i]);
      v *= table_spec && magnitude < dsilu_window_low ? fmaf(0.5f, xv, 0.5f)
                                                       : dsilu<float, float>(xv, e);
      break;
  }
  z[i] = v;
  y[i] = static_cast<IType>(v);
}

enum class Fill { Typical, OrderedInputs, Specials, TinyGradients };

template <typename T>
T from_bits(const uint32_t bits) {
  if constexpr (std::is_same_v<T, float>) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  } else {
    const uint16_t half_bits = static_cast<uint16_t>(bits);
    T value;
    std::memcpy(&value, &half_bits, sizeof(value));
    return value;
  }
}

// Enumerate the entire 16-bit format in numerical order: -Inf, negative finite values, -0,
// +0, positive finite values, +Inf, then both signs of NaN (which have no numerical order).
// For FP32, enumerate the BF16 exponent/mantissa prefixes and vary the low mantissa bits.
template <typename T>
uint32_t ordered_bits(const size_t i, const size_t n) {
  // The high 16 bits select a format encoding; the low 16 sample FP32 mantissas.
  const uint32_t position = static_cast<uint32_t>((static_cast<uint64_t>(i) << 32) / n);
  const uint32_t rank = position >> 16;
  const uint32_t pos_inf = std::is_same_v<T, fp16> ? 0x7c00u : 0x7f80u;
  const uint32_t non_nan_count = 2 * (pos_inf + 1);
  uint32_t high;
  if (rank <= pos_inf) {
    high = 0x8000u | (pos_inf - rank);
  } else if (rank < non_nan_count) {
    high = rank - pos_inf - 1;
  } else {
    const uint32_t nan_rank = rank - non_nan_count;
    const uint32_t nan_count_per_sign = 0x7fffu - pos_inf;
    high = nan_rank < nan_count_per_sign ? 0x8000u | (pos_inf + 1 + nan_rank)
                                         : pos_inf + 1 + nan_rank - nan_count_per_sign;
  }
  if constexpr (std::is_same_v<T, float>) {
    const uint32_t fraction = position & 0xffffu;
    const uint32_t low = (high & 0x8000u) ? 0xffffu - fraction : fraction;
    return (high << 16) | ((high & 0x7fffu) == pos_inf ? 0u : low);
  } else {
    return high;
  }
}

// Exactly representable edge cases in each input format, ordered except for the final NaNs.
template <typename T>
std::array<T, 14> special_values() {
  const float inf = Numeric_Traits<T>::artifInf;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float max_normal = Numeric_Traits<T>::maxNorm;
  const float min_normal = Numeric_Traits<T>::minNorm;
  const float min_subnormal = Numeric_Traits<T>::minSubnorm;
  return {static_cast<T>(-inf),           static_cast<T>(-max_normal),
          static_cast<T>(-1.0f),          static_cast<T>(-min_normal),
          static_cast<T>(-min_subnormal), static_cast<T>(-0.0f),
          static_cast<T>(0.0f),           static_cast<T>(min_subnormal),
          static_cast<T>(min_normal),     static_cast<T>(1.0f),
          static_cast<T>(max_normal),     static_cast<T>(inf),
          static_cast<T>(-nan),           static_cast<T>(nan)};
}

// Typical activations range over [-16, 16] and gradients over [-4, 4]. The ordered sweep
// covers every BF16/FP16 bit pattern, including subnormals, infinities, and NaNs. The special
// case set guarantees exact FP32 boundary values that a sampled mantissa sweep may miss.
template <typename T>
void fill_inputs(const Fill fill, Tensor *x, Tensor *grad) {
  const size_t n = product(x->rowwise_shape());
  T *xp = x->rowwise_cpu_dptr<T>();
  T *gp = grad->rowwise_cpu_dptr<T>();
  const auto specials = special_values<T>();
  for (size_t i = 0; i < n; ++i) {
    switch (fill) {
      case Fill::Typical:
        xp[i] = static_cast<T>(-16.0f + 32.0f * static_cast<float>(i) / (n - 1));
        gp[i] = static_cast<T>(-4.0f + 8.0f * static_cast<float>(i % 257) / 256.0f);
        break;
      case Fill::OrderedInputs:
        xp[i] = from_bits<T>(ordered_bits<T>(i, n));
        gp[i] = static_cast<T>(-4.0f + 8.0f * static_cast<float>(i % 257) / 256.0f);
        break;
      case Fill::TinyGradients: {
        // At x=0 both derivatives are 0.5. Cover subnormal gradients and a normal
        // gradient whose product is subnormal, including both signs.
        const float tiny_gradients[] = {static_cast<float>(2 * Numeric_Traits<T>::minSubnorm),
                                        static_cast<float>(Numeric_Traits<T>::maxSubnorm),
                                        static_cast<float>(Numeric_Traits<T>::minNorm)};
        xp[i] = static_cast<T>(0.0f);
        const float magnitude = tiny_gradients[(i / 2) % 3];
        gp[i] = static_cast<T>(i % 2 == 0 ? magnitude : -magnitude);
        break;
      }
      case Fill::Specials: {
        const size_t index = i * specials.size() / n;
        xp[i] = specials[index];
        gp[i] = specials[(index + 4) % specials.size()];
        break;
      }
    }
  }
  x->from_cpu();
  grad->from_cpu();
}

// dbias in the order the quantize kernel accumulates it: per 64-row tile, then over the tiles.
// With rowwise_reduction (CUDA, for FP32 input, rowwise-only scaling and a fused dact), each of the
// tile's 32 threads first adds its two rows (r and r + 32), and the 32 thread sums are then added
// in order.
std::vector<float> reference_dbias(const std::vector<float> &z, const size_t rows,
                                   const size_t cols, const bool rowwise_reduction) {
  constexpr size_t kTileRows = 64;
  constexpr size_t kThreadsY = 32;
  const size_t tiles = (rows + kTileRows - 1) / kTileRows;
  auto at = [&](const size_t r, const size_t c) { return r < rows ? z[r * cols + c] : 0.0f; };
  std::vector<float> dbias(cols);
  for (size_t c = 0; c < cols; ++c) {
    float total = 0.0f;
    for (size_t t = 0; t < tiles; ++t) {
      const size_t r0 = t * kTileRows;
      float partial = 0.0f;
      if (rowwise_reduction) {
        for (size_t i = 0; i < kThreadsY; ++i) {
          float thread_sum = 0.0f;
          thread_sum += at(r0 + i, c);
          thread_sum += at(r0 + kThreadsY + i, c);
          partial += thread_sum;
        }
      } else {
        for (size_t r = r0; r < r0 + kTileRows; ++r) {
          partial += at(r, c);
        }
      }
      total += partial;
    }
    dbias[c] = total;
  }
  return dbias;
}

// Sets an environment variable for one scope, and restores its previous value afterwards.
class ScopedEnv {
 public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    if (old != nullptr) old_ = old;
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char *name_;
  std::optional<std::string> old_;
};

size_t count_byte_mismatches(const uint8_t *a, const uint8_t *b, const size_t n) {
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) m += (a[i] != b[i]);
  return m;
}

template <typename IType>
void run_case(const Mode mode, const Fill fill, const size_t rows, const size_t cols,
              const bool rowwise, const bool colwise, const bool swizzled, const bool use_table,
              const DType otype, const bool with_amax) {
  const DType itype = TypeInfo<IType>::dtype;
  const std::vector<size_t> shape{rows, cols};
  const size_t n = rows * cols;

  Tensor x("x", shape, itype);
  Tensor grad("grad", shape, itype);
  fill_inputs<IType>(fill, &x, &grad);

  // Reference: scalar activation on the GPU, then the generic cast-only quantize.
  Tensor y("y", shape, itype);
  float *z_dev = nullptr;
  ASSERT_EQ(cudaMalloc(&z_dev, n * sizeof(float)), cudaSuccess);
  constexpr int block_size = 256;
  const size_t grid_size = divide_round_up(n, block_size);
  const bool table_spec = use_table && std::is_same_v<IType, __nv_bfloat16>;
  // NVTE_ACTIVATION_TABLE_EXACT_DSILU=1 extends the dSiLU table down to 2^-25.
  const char *exact_env = std::getenv("NVTE_ACTIVATION_TABLE_EXACT_DSILU");
  const float dsilu_window_low =
      exact_env != nullptr && std::string(exact_env) == "1" ? 0x1p-25f : 0x1p-9f;
  scalar_activation_kernel<IType>
      <<<grid_size, block_size>>>(mode, table_spec, dsilu_window_low,
                                  static_cast<const IType *>(x.rowwise_dptr()),
                                  static_cast<const IType *>(grad.rowwise_dptr()),
                                  static_cast<IType *>(y.rowwise_dptr()), z_dev, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<float> z(n);
  ASSERT_EQ(cudaMemcpy(z.data(), z_dev, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(z_dev);

  Tensor ref("ref", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
  ref.set_with_gemm_swizzled_scales(swizzled);
  Tensor noop("noop", std::vector<size_t>{1}, DType::kFloat32);
  nvte_quantize_noop(y.data(), ref.data(), noop.data(), 0);

  // Fused.
  Tensor out("out", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
  out.set_with_gemm_swizzled_scales(swizzled);
  float *amax_dev = nullptr;
  if (with_amax) {
    ASSERT_EQ(cudaMalloc(&amax_dev, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(amax_dev, 0, sizeof(float)), cudaSuccess);
    const size_t one = 1;
    const NVTEBasicTensor amax = {amax_dev, kNVTEFloat32, nvte_make_shape(&one, 1)};
    nvte_set_tensor_param_v2(out.data(), kNVTEAmax, &amax, sizeof(amax));
  }
  Tensor dbias("dbias", std::vector<size_t>{cols}, itype);
  Tensor workspace;
  {
    const ScopedEnv tables("NVTE_ACTIVATION_TABLE", use_table ? "1" : "0");
    switch (mode) {
      case Mode::GeLU:
        nvte_gelu(x.data(), out.data(), 0);
        break;
      case Mode::SiLU:
        nvte_silu(x.data(), out.data(), 0);
        break;
      case Mode::DGeLU:
        nvte_dgelu(grad.data(), x.data(), out.data(), 0);
        break;
      case Mode::DSiLU:
        nvte_dsilu(grad.data(), x.data(), out.data(), 0);
        break;
      default: {
        auto fn =
            mode == Mode::DBiasDGeLU ? &nvte_quantize_dbias_dgelu : &nvte_quantize_dbias_dsilu;
        fn(grad.data(), x.data(), out.data(), dbias.data(), workspace.data(), 0);
        workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
        fn(grad.data(), x.data(), out.data(), dbias.data(), workspace.data(), 0);
      }
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  out.to_cpu();
  ref.to_cpu();
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
  if (with_amax) {
    // The largest magnitude of the activation as quantized (rounded to IType), ignoring NaN.
    y.to_cpu();
    const IType *act = y.rowwise_cpu_dptr<IType>();
    float expected = 0.0f;
    for (size_t i = 0; i < n; ++i) expected = std::fmax(expected, std::fabs(static_cast<float>(act[i])));
    float got;
    ASSERT_EQ(cudaMemcpy(&got, amax_dev, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(amax_dev);
    EXPECT_EQ(std::memcmp(&got, &expected, sizeof(float)), 0) << "amax " << got << " vs " << expected;
  }
  if (rowwise) {
    EXPECT_EQ(count_byte_mismatches(rowwise_bytes(out), rowwise_bytes(ref), n), 0u)
        << "rowwise data";
    const size_t s = product(out.rowwise_scale_inv_shape());
    EXPECT_EQ(count_byte_mismatches(out.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                    ref.rowwise_cpu_scale_inv_ptr<uint8_t>(), s),
              0u)
        << "rowwise scales";
  }
  if (colwise) {
    EXPECT_EQ(count_byte_mismatches(columnwise_bytes(out), columnwise_bytes(ref), n), 0u)
        << "colwise data";
    const size_t s = product(out.columnwise_scale_inv_shape());
    EXPECT_EQ(count_byte_mismatches(out.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                                    ref.columnwise_cpu_scale_inv_ptr<uint8_t>(), s),
              0u)
        << "colwise scales";
  }
  if (is_dbias(mode)) {
    // CuTeDSL's general kernel adds FP32 rows in row order too (no rowwise dbias pass).
    const char *cutedsl = std::getenv("NVTE_ENABLE_CUTEDSL_BACKEND");
    const bool cutedsl_on = cutedsl != nullptr && std::string(cutedsl) == "1";
    const bool rowwise_reduction = std::is_same_v<IType, float> && !colwise && !cutedsl_on;
    const std::vector<float> expected = reference_dbias(z, rows, cols, rowwise_reduction);
    dbias.to_cpu();
    const IType *got = dbias.rowwise_cpu_dptr<IType>();
    size_t mismatches = 0;
    for (size_t c = 0; c < cols; ++c) {
      const IType e = static_cast<IType>(expected[c]);
      mismatches += std::memcmp(&e, &got[c], sizeof(IType)) != 0 &&
                    !(std::isnan(static_cast<float>(e)) && std::isnan(static_cast<float>(got[c])));
    }
    EXPECT_EQ(mismatches, 0u) << "dbias";
  }
}

using Params =
    std::tuple<Mode, Fill, DType, int /*layout: 0 row, 1 col, 2 both*/, bool /*swizzled*/,
               std::pair<size_t, size_t>, bool /*use_table*/, DType /*output*/, bool /*amax*/>;

class CastMXFP8ActExactTestSuite : public ::testing::TestWithParam<Params> {};

}  // namespace

TEST_P(CastMXFP8ActExactTestSuite, MatchesScalarActivationThenCast) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 10) {
    GTEST_SKIP() << "This MXFP8 quantize kernel requires compute capability 10.0 or newer";
  }
  const auto [mode, fill, itype, layout, swizzled, dims, use_table, otype, with_amax] = GetParam();
  const bool rowwise = layout != 1;
  const bool colwise = layout != 0;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      itype, IType,
      run_case<IType>(mode, fill, dims.first, dims.second, rowwise, colwise, swizzled, use_table,
                      otype, with_amax););
}

namespace {

std::string case_name(const testing::TestParamInfo<Params> &info) {
  const char *layouts[] = {"Rowwise", "Colwise", "Both"};
  const auto [mode, fill, itype, layout, swizzled, dims, use_table, otype, with_amax] = info.param;
  const char *fills[] = {"XTypical", "XOrderedInputs", "XSpecials", "XTinyGradients"};
  return std::string(mode_name(mode)) + fills[static_cast<int>(fill)] + "X" +
         test::typeName(itype) + "X" + layouts[layout] + (swizzled ? "XSwizzled" : "XCompact") +
         "X" + std::to_string(dims.first) + "X" + std::to_string(dims.second) +
         (use_table ? "XTable" : "XNoTable") + (otype == DType::kFloat8E5M2 ? "XE5M2" : "") +
         (with_amax ? "XAmax" : "");
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastMXFP8ActExactTestSuite,
    ::testing::Combine(::testing::Values(Mode::GeLU, Mode::DGeLU, Mode::DBiasDGeLU, Mode::SiLU,
                                         Mode::DSiLU, Mode::DBiasDSiLU),
                       ::testing::Values(Fill::Typical, Fill::OrderedInputs),
                       ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
                       ::testing::Values(0, 1, 2), ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{1024, 2048},
                                         // Enough row tiles that the CuTeDSL register-resident
                                         // kernel walks several per CTA.
                                         std::pair<size_t, size_t>{4096, 2048},
                                         // A row tail on 256-column tiles: 31 row tiles, the
                                         // last of 16 rows, and an odd number for dbias.
                                         std::pair<size_t, size_t>{976, 2048},
                                         std::pair<size_t, size_t>{544, 2080},
                                         // A row tail and a column tail ending inside an MX
                                         // block.
                                         std::pair<size_t, size_t>{4127, 2064},
                                         // A dbias workspace of 130 rows, past the dbias
                                         // reduction's 128-row pipeline.
                                         std::pair<size_t, size_t>{8257, 544}),
                       ::testing::Bool(), ::testing::Values(DType::kFloat8E4M3),
                       ::testing::Values(false)),
    case_name);

// E5M2 output and the tensor amax, both directions.
INSTANTIATE_TEST_SUITE_P(
    OutputTypeAndAmax, CastMXFP8ActExactTestSuite,
    ::testing::Combine(::testing::Values(Mode::GeLU, Mode::DGeLU, Mode::DBiasDGeLU, Mode::SiLU,
                                         Mode::DSiLU, Mode::DBiasDSiLU),
                       ::testing::Values(Fill::Typical, Fill::Specials),
                       ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
                       ::testing::Values(2), ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{1024, 2048},
                                         std::pair<size_t, size_t>{4127, 2064}),
                       ::testing::Bool(),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
                       ::testing::Values(true)),
    case_name);

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, CastMXFP8ActExactTestSuite,
    ::testing::Combine(::testing::Values(Mode::GeLU, Mode::DGeLU, Mode::DBiasDGeLU, Mode::SiLU,
                                         Mode::DSiLU, Mode::DBiasDSiLU),
                       ::testing::Values(Fill::Specials),
                       ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
                       ::testing::Values(0, 1, 2), ::testing::Bool(),
                       ::testing::Values(std::pair<size_t, size_t>{64, 128}), ::testing::Bool(),
                       ::testing::Values(DType::kFloat8E4M3), ::testing::Values(false)),
    case_name);

INSTANTIATE_TEST_SUITE_P(TinyGradients, CastMXFP8ActExactTestSuite,
                         ::testing::Combine(::testing::Values(Mode::DGeLU, Mode::DBiasDGeLU,
                                                              Mode::DSiLU, Mode::DBiasDSiLU),
                                            ::testing::Values(Fill::TinyGradients),
                                            ::testing::Values(DType::kBFloat16, DType::kFloat16,
                                                              DType::kFloat32),
                                            ::testing::Values(0, 1, 2), ::testing::Bool(),
                                            ::testing::Values(std::pair<size_t, size_t>{64, 128}),
                                            ::testing::Bool(), ::testing::Values(DType::kFloat8E4M3),
                                            ::testing::Values(false)),
                         case_name);

namespace {

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
