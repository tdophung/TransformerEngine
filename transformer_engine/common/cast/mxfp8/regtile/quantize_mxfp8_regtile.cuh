/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_regtile.cuh
 *  \brief Register-resident MXFP8 bidimensional fused quantize kernel.
 *
 *  Bidimensional MXFP8 produces two quantizations of one BF16 tensor: a rowwise
 *  one, where 32 consecutive elements of a row share an E8M0 scale, and a
 *  colwise one, where 32 consecutive elements of a column do.  Both need an
 *  amax over the same data in two different directions, and that conflict is
 *  what the tiling here is built around.
 *
 *  The generic staged kernel in ../quantize_mxfp8.cuh resolves it by
 *  materializing: it TMAs a tile into shared memory and walks it twice, once per
 *  direction, using shared as the pivot that lets the same bytes be read in two
 *  orientations.  When an activation is fused it goes further and caches the
 *  activated values back through shared so the second pass does not recompute
 *  them.
 *
 *  This kernel resolves it by geometry instead.  A CTA covers whole 32-row
 *  bands, which is exactly the colwise MX block height, so the colwise reduction
 *  closes inside the CTA and nothing has to be staged for a second pass.  The
 *  tile is read from global memory once into registers (dv[]), and BOTH
 *  directions are then driven from those same registers.  Shared memory carries
 *  only the cross-warp columnwise partials, the row-scale bytes, and the dGeLU
 *  table -- never the tile.  Data path: LDG.128 in, STG.64 out.
 *
 *  Two tilings live here.  Four of the five fusion modes take the
 *  register-resident one (quantize_regtile); dbias without dGeLU keeps a
 *  column-owned shared-staged tile (quantize_streaming) because its much tighter
 *  dbias tolerance demands TE's exact row-order FP32 accumulation chain.
 *  Cfg::kRegisterResident picks between them and Cfg holds every other shape
 *  constant the two bodies need.
 *
 *  Numerics: the quantized output must match the staged kernel bit for bit.
 *  Everything here is arranged to preserve that -- the activations replicate
 *  util/math.h's operation order with non-contracted intrinsics, tanh_f32x2 is
 *  an instruction-for-instruction replica of libdevice tanhf, and the dGeLU
 *  lookup table stores the very same FP32 values the arithmetic body produces.
 *  The speedup is structural; none of it comes from cheaper math.
 *
 *  Derived from the winning candidate of Kernel Factory campaign
 *  rv390dmap97kd7jaxfef2kjcmw (mxfp8-quantize-b200-bidim-fused-te-unified),
 *  candidate 12920c25eba5de8346062b46c75d5e041fdf2fc2d2a7e97f43b98be44ad59ba5.
 *  The campaign's search-space knobs have been resolved to their shipped values
 *  and the unreachable variants removed, so this no longer diffs against the
 *  campaign export.
 *
 *  Envelope this kernel was validated on -- see RegtileOpSupported and
 *  regtile_shape_supported(), which gate it together with the dtype and
 *  scaling-type checks at the call site in ../quantize_mxfp8.cuh:
 *    bf16 -> e4m3, BIDIMENSIONAL scaling, rows % 64 == 0, cols % 256 == 0,
 *    no GEMM-swizzled scales, no amax output, no noop tensor, gelu/dgelu only.
 *  Anything outside that falls through to the generic TMA kernel.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "../../../common.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace regtile {

namespace ptx = transformer_engine::ptx;

namespace {

// ---------------------------------------------------------------------------
// Packed BF16 helpers.
//
// Every value this kernel moves is BF16 or FP8, so the whole datapath is
// expressed on 32-bit words holding two BF16 elements, and the arithmetic below
// is the packed PTX that operates on them directly.  A word is the natural
// granularity: it is what a column pair occupies, what the FP32x2 pipe consumes
// and what the E8M0 and E4M3 converters emit two of at a time.
// ---------------------------------------------------------------------------

/*! \brief Elementwise max(|a|, |b|) on a packed BF16 pair, as a raw word.
 *
 * ptx::abs_max_2x spells this with typed operands and an out-parameter; the
 * datapath here is raw 32-bit words and the calls nest, so it gets a return
 * value instead.
 */
__device__ __forceinline__ unsigned abs_max_bf16x2(unsigned a, unsigned b) {
  ptx::bf16x2 d;
  ptx::abs_max_2x(d, reinterpret_cast<const ptx::bf16x2&>(a),
                  reinterpret_cast<const ptx::bf16x2&>(b));
  return reinterpret_cast<const unsigned&>(d);
}

/*! \brief Widen the low / high BF16 of a word to FP32. */
__device__ __forceinline__ float bf16_lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

/*! \brief The reciprocal of an E8M0 scale, 2^(127-e), broadcast to a BF16 pair.
 *
 * ptx::float_to_e8m0_2x produces the scale byte; this turns it back into the
 * BF16 multiplier the quantized conversion consumes.  (254 - e) <= 254, so
 * (254 - e) << 7 < 2^15 and the multiply by 0x00800080 cannot carry across the
 * halves: the whole duplicate-and-shift is one IMAD.
 */
__device__ __forceinline__ unsigned e8m0_to_bf16x2_reciprocal(unsigned e) {
  return (254u - e) * 0x00800080u;
}

// ---------------------------------------------------------------------------
// Activations, in TE's own operation order.
//
// These reproduce util/math.h term for term with non-contracted intrinsics, so
// the FP32 rounding sequence -- and therefore the quantized output -- matches
// the staged kernel bit for bit.  They are the reference the lookup table below
// is built from, and the fallback for inputs the table does not cover.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float act_dgelu(float v) {
  float a3 = __fadd_rn(1.0f, __fmul_rn(__fmul_rn(0.044715f, v), v));
  float a5 = __fmul_rn(__fmul_rn(0.79788456f, v), a3);
  float t = tanhf(a5);
  float c2 = __fsub_rn(1.0f, __fmul_rn(t, t));
  float d3 = __fadd_rn(0.79788456f, __fmul_rn(__fmul_rn(0.1070322243f, v), v));
  float f = __fmul_rn(__fmul_rn(0.5f, v), __fmul_rn(c2, d3));
  float h = __fmul_rn(0.5f, __fadd_rn(1.0f, t));
  return __fadd_rn(f, h);
}

// A packed pair of FP32 values, the operand form of the Blackwell `.f32x2`
// instructions.  ptx.cuh spells the arithmetic on ptx::floatx2 (add_2x, sub_2x,
// mul_2x, fma_2x); this kernel keeps the raw 64-bit view, because its dbias
// accumulators live in a register array that has to stay in registers and a
// struct-typed array does not reliably.  The two are the same bits; the thin
// wrappers below are the only place the reinterpret happens.
using f32x2 = unsigned long long;

__device__ __forceinline__ f32x2 make_f32x2(float lo, float hi) {
  f32x2 d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
  return d;
}
__device__ __forceinline__ void unpack_f32x2(f32x2 a, float& lo, float& hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(a));
}
__device__ __forceinline__ f32x2 splat_f32x2(float c) { return make_f32x2(c, c); }

#define NVTE_REGTILE_F32X2_OP(NAME, PTX_NAME)                                       \
  __device__ __forceinline__ f32x2 NAME(f32x2 a, f32x2 b) {                         \
    const ptx::floatx2 d = ptx::PTX_NAME(reinterpret_cast<const ptx::floatx2&>(a),  \
                                         reinterpret_cast<const ptx::floatx2&>(b)); \
    return reinterpret_cast<const f32x2&>(d);                                       \
  }
NVTE_REGTILE_F32X2_OP(add_f32x2, add_2x)
NVTE_REGTILE_F32X2_OP(sub_f32x2, sub_2x)
NVTE_REGTILE_F32X2_OP(mul_f32x2, mul_2x)
#undef NVTE_REGTILE_F32X2_OP

__device__ __forceinline__ f32x2 fma_f32x2(f32x2 a, f32x2 b, f32x2 c) {
  const ptx::floatx2 d = ptx::fma_2x(reinterpret_cast<const ptx::floatx2&>(a),
                                     reinterpret_cast<const ptx::floatx2&>(b),
                                     reinterpret_cast<const ptx::floatx2&>(c));
  return reinterpret_cast<const f32x2&>(d);
}

/*! \brief Widen both halves of a BF16 pair to a packed FP32 pair.
 *
 * ptx::up_cast does the same with two PRMTs; the shift-and-mask form here keeps
 * the value in the 64-bit register the `.f32x2` chain wants.
 */
__device__ __forceinline__ f32x2 bf16x2_to_f32x2(unsigned a) {
  return make_f32x2(bf16_lo(a), bf16_hi(a));
}

// ---------------------------------------------------------------------------
// Packed tanh.
//
// An instruction-for-instruction replica of CUDA libdevice `tanhf` evaluated on
// a packed pair: the two transcendental steps stay scalar, everything else is
// packed, and each lane is bit-identical to what `tanhf` would return.  Matching
// libdevice exactly is the whole point -- the staged kernel calls `tanhf`, and
// the golden rule for this kernel is to reproduce its output, not to improve on
// it.
// ---------------------------------------------------------------------------

//! 2 * log2(e), the argument scale of the exponential branch.
constexpr float kTanhLog2eX2 = 0x1.715476p+1f;
//! Coefficients of the |x| < 0.6 minimax polynomial, in Horner order.
constexpr float kTanhPoly4 = 0x1.01e104p-6f;
constexpr float kTanhPoly3 = -0x1.ac795cp-5f;
constexpr float kTanhPoly2 = 0x1.10b282p-3f;
constexpr float kTanhPoly1 = -0x1.5553dap-2f;
//! The branch threshold 0.6f, squared.  0.6f * 0.6f rounds exactly to 0.36f, so
//! the predicate can be tested on the already-computed square.
constexpr float kTanhBranchXSq = 0x1.70a3d8p-2f;

/*! \brief copysign(a, b) where a is known non-negative.
 *
 * Collapses to a single LOP3, (b & sign_mask) | a, with the immediate folded
 * into the LOP3 operand.
 */
__device__ __forceinline__ float copysign_nonneg(float a, float b) {
  unsigned d;
  asm("lop3.b32 %0, %1, %2, 0x80000000, 0xec;"
      : "=r"(d)
      : "r"(__float_as_uint(b)), "r"(__float_as_uint(a)));
  return __uint_as_float(d);
}

__device__ __forceinline__ f32x2 tanh_f32x2(f32x2 u) {
  float ua, ub;
  unpack_f32x2(u, ua, ub);
  // |x| folds into the FMUL operand modifier; the two MUFU steps stay scalar.
  float e0, e1;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(__fmul_rn(fabsf(ua), kTanhLog2eX2)));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(__fmul_rn(fabsf(ub), kTanhLog2eX2)));
  float f0, f1;
  unpack_f32x2(add_f32x2(make_f32x2(e0, e1), splat_f32x2(1.0f)), f0, f1);
  float r0, r1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(f0));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(f1));
  // libdevice clamps |x| >= 9.010914 to exactly 1.0; there 2*rcp(ex2(2ln2|x|))
  // is already below half an ulp of 1.0f, so fma(r,-2,1) rounds to 1.0f on its
  // own and the clamp select is dropped.
  float g0, g1;
  unpack_f32x2(fma_f32x2(make_f32x2(r0, r1), splat_f32x2(-2.0f), splat_f32x2(1.0f)), g0, g1);
  const float xa = copysign_nonneg(g0, ua);
  const float xb = copysign_nonneg(g1, ub);
  // The small-|x| polynomial branch is pure FP32 arithmetic, so it packs fully.
  const f32x2 s = mul_f32x2(u, u);
  f32x2 p = fma_f32x2(s, splat_f32x2(kTanhPoly4), splat_f32x2(kTanhPoly3));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly2));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly1));
  p = mul_f32x2(p, s);
  p = fma_f32x2(p, u, u);
  float pa, pb, sa, sb;
  unpack_f32x2(p, pa, pb);
  unpack_f32x2(s, sa, sb);
  return make_f32x2(sa >= kTanhBranchXSq ? xa : pa, sb >= kTanhBranchXSq ? xb : pb);
}

/*! \brief GeLU on a packed pair, matching act_gelu term for term. */
__device__ __forceinline__ f32x2 gelu_f32x2(f32x2 v) {
  const f32x2 t2 = mul_f32x2(mul_f32x2(splat_f32x2(0.03567741f), v), v);
  const f32x2 u = mul_f32x2(v, add_f32x2(splat_f32x2(0.79788456f), t2));
  const f32x2 t = tanh_f32x2(u);
  // 0.5f*t is exact, so fma(0.5,t,0.5) == fadd(0.5, fmul(0.5,t)) bit for bit.
  return mul_f32x2(v, fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f)));
}

/*! \brief dGeLU on a packed pair, matching act_dgelu term for term. */
__device__ __forceinline__ f32x2 dgelu_f32x2(f32x2 v) {
  const f32x2 a3 = add_f32x2(splat_f32x2(1.0f), mul_f32x2(mul_f32x2(splat_f32x2(0.044715f), v), v));
  const f32x2 a5 = mul_f32x2(mul_f32x2(splat_f32x2(0.79788456f), v), a3);
  const f32x2 t = tanh_f32x2(a5);
  const f32x2 c2 = sub_f32x2(splat_f32x2(1.0f), mul_f32x2(t, t));
  const f32x2 d3 =
      add_f32x2(splat_f32x2(0.79788456f), mul_f32x2(mul_f32x2(splat_f32x2(0.1070322243f), v), v));
  const f32x2 f = mul_f32x2(mul_f32x2(splat_f32x2(0.5f), v), mul_f32x2(c2, d3));
  // 0.5f*(1+t) and fma(0.5,t,0.5) round on the same grid (halving is exact).
  const f32x2 h = fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f));
  return add_f32x2(f, h);
}

/*! \brief GeLU on a BF16 pair, returned as a BF16 pair. */
__device__ __forceinline__ unsigned gelu_bf16x2(unsigned av) {
  float v0, v1;
  unpack_f32x2(gelu_f32x2(bf16x2_to_f32x2(av)), v0, v1);
  return ptx::cvt_bf16x2(v1, v0);
}

// ---------------------------------------------------------------------------
// dGeLU lookup table.
//
// The activation input is BF16, so dgelu is a function of a 16-bit key and can
// be tabulated EXACTLY: the table holds the very same FP32 value dgelu_f32x2
// produces, so routing a word through it is a pure instruction-count trade with
// no numerical consequence.
//
// Why trade at all: the probe is L1/shared-pipe work and the arithmetic body is
// FP32/MUFU work, so sending a fraction of every row through the table is what
// keeps both sides of the machine busy.  An all-table kernel saturates the
// shared pipe instead (57% of its wavefronts are bank-conflict replays, L1 88%
// busy against DRAM 40%) while the FP32 pipes idle.  Cfg::kLutSlots sets the
// split; see LutSlot.
//
// Only the dGeLU instantiations tabulate.  GeLU is short enough that its
// arithmetic body already fits under the memory latency, so it evaluates the
// closed form for every word and carries no table at all.
//
// A full 2^16 table would be 256 KB.  Range reduction shrinks it to 16 KB:
//   * |v| >= 5.5    -> dgelu has saturated to exactly 1.0f (v>0) / 0.0f (v<0),
//                      so every magnitude at or above BF16 8.0 shares one entry;
//   * |v| <  2^-12  -> rare enough (1.2% of warps for unit-scale data) that the
//                      arithmetic tail below handles it.
// Both tails are caught by a single "did the clamp move the value?" compare, and
// the fallback is exact for the whole pair, so correctness never depends on the
// range analysis being tight.
// ---------------------------------------------------------------------------

//! BF16 bits of 2^-12, the smallest tabulated magnitude.
constexpr unsigned kLutLoBits = 0x3980u;
//! BF16 bits of 8.0, the largest tabulated magnitude.
constexpr unsigned kLutHiBits = 0x4100u;
//! Entries per sign.  kLutHiBits - kLutLoBits + 1 = 1921 of them are reachable.
constexpr int kLutEntriesPerSign = 2048;
//! Both bounds broadcast, for the packed clamp.
constexpr unsigned kLutLoBitsX2 = 0x39803980u;
constexpr unsigned kLutHiBitsX2 = 0x41004100u;
//! Table size.  Reachable FP32 entries end at byte 15875; rounded up to the
//! bulk-copy granularity.
constexpr int kLutBytes = 15888;

__device__ __align__(16) unsigned char d_dgelu_table[kLutBytes];

/*! \brief Byte offsets of both halves of a BF16 pair into the dGeLU table.
 *
 * Clamps each magnitude into the tabulated window and turns it into a byte
 * offset, five ALU ops for the whole column pair.  The sign bit is worth exactly
 * kLutEntriesPerSign entries, which at four bytes each is 0x8000 -- so it drops
 * in with no shift, and every half stays below 0x10000, meaning the packed add
 * never carries across the pair.  Subtract-before-shift plus a packed sign shift
 * maps to a SHF+LEA-friendly sequence and avoids a mask/add chain.
 *
 * \param[in,out] bad  Accumulates every bit the clamp had to move, so a whole
 *                     row's worth of probes can share ONE out-of-window test
 *                     instead of branching per word.
 */
__device__ __forceinline__ unsigned lut_byte_offsets(unsigned b, unsigned& bad) {
  const unsigned mag = b & 0x7fff7fffu;
  const unsigned c = ptx::min_bf16x2(ptx::max_bf16x2(mag, kLutLoBitsX2), kLutHiBitsX2);
  bad |= c ^ mag;
  return ((c - kLutLoBitsX2) << 2) + ((b ^ mag) >> 2);
}

/*! \brief Probe the dGeLU table for both halves of a BF16 pair.
 *  \param[out] oor  True if either half fell outside the tabulated window.
 */
__device__ __forceinline__ f32x2 lut_dgelu(unsigned b, const unsigned char* __restrict__ tab,
                                           bool& oor) {
  unsigned bad = 0u;
  const unsigned d = lut_byte_offsets(b, bad);
  oor = bad != 0u;
  return make_f32x2(*(const float*)(tab + (d & 0xffffu)), *(const float*)(tab + (d >> 16)));
}

/*! \brief dGeLU below the tabulated window.
 *
 * For |v| < 2^-12 the cubic term of the series is under 2^-36 relative, so the
 * tanh call collapses to its argument and this three-operation form reproduces
 * dgelu_f32x2's rounding sequence term for term.  Keeping the tail this cheap is
 * what keeps the whole tanh chain out of the table path's register budget.
 */
__device__ __forceinline__ float dgelu_tiny(float v) {
  const float t = __fmul_rn(0.79788456f, v);
  const float f = __fmul_rn(__fmul_rn(0.5f, v), 0.79788456f);
  return __fadd_rn(f, __fmaf_rn(0.5f, t, 0.5f));
}

/*! \brief Repair a clamped dGeLU probe.
 *
 * Above the window the clamped entry is already exact (tanh has saturated), so
 * only the |v| < 2^-12 half needs the closed form.  Rare path, so branches are
 * free here.
 */
__device__ __forceinline__ f32x2 dgelu_lut_tail(unsigned av, f32x2 probe) {
  float d0, d1;
  unpack_f32x2(probe, d0, d1);
  const unsigned m = av & 0x7fff7fffu;
  if ((m & 0xffffu) < kLutLoBits) d0 = dgelu_tiny(bf16_lo(av));
  if ((m >> 16) < kLutLoBits) d1 = dgelu_tiny(bf16_hi(av));
  return make_f32x2(d0, d1);
}

/*! \brief dGeLU of one BF16 pair, by table or by arithmetic.
 *  \tparam USE_LUT  Which route this (row, word) slot takes; see LutSlot.
 */
template <bool USE_LUT>
__device__ __forceinline__ f32x2 dgelu_word(unsigned av, const unsigned char* __restrict__ tab) {
  if constexpr (USE_LUT) {
    bool oor;
    const f32x2 d = lut_dgelu(av, tab, oor);
    if (__builtin_expect(!oor, 1)) return d;
    return dgelu_lut_tail(av, d);
  }
  return dgelu_f32x2(bf16x2_to_f32x2(av));
}

/*! \brief dGeLU of one BF16 pair times the incoming gradient pair.
 *
 * The table produces two scalar FP32 registers, so the table route keeps them
 * scalar through the gradient multiply rather than packing table values and
 * converted BF16 values into two temporary FP32x2 operands only to unpack the
 * product immediately.  The arithmetic route is already naturally packed and
 * keeps its FP32x2 path.
 */
template <bool USE_LUT>
__device__ __forceinline__ f32x2 dgelu_grad_word(unsigned av, unsigned gv,
                                                 const unsigned char* __restrict__ tab) {
  if constexpr (USE_LUT) {
    unsigned bad = 0u;
    const unsigned d = lut_byte_offsets(av, bad);
    float d0 = *(const float*)(tab + (d & 0xffffu));
    float d1 = *(const float*)(tab + (d >> 16));
    if (__builtin_expect(bad != 0u, 0)) {
      const unsigned m = av & 0x7fff7fffu;
      if ((m & 0xffffu) < kLutLoBits) d0 = dgelu_tiny(bf16_lo(av));
      if ((m >> 16) < kLutLoBits) d1 = dgelu_tiny(bf16_hi(av));
    }
    return make_f32x2(__fmul_rn(d0, bf16_lo(gv)), __fmul_rn(d1, bf16_hi(gv)));
  }
  return mul_f32x2(dgelu_f32x2(bf16x2_to_f32x2(av)), bf16x2_to_f32x2(gv));
}

/*! \brief One-time build of the dGeLU table.
 *
 * Pure compile-time-constant data: it depends on nothing but the activation
 * formula, exactly like a trig table.
 */
__global__ void init_dgelu_table_kernel() {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * (int)(kLutHiBits - kLutLoBits + 1)) return;
  const int sgn = i > (int)(kLutHiBits - kLutLoBits);
  const unsigned mag = kLutLoBits + (unsigned)(sgn ? i - (int)(kLutHiBits - kLutLoBits + 1) : i);
  const unsigned bits = ((unsigned)sgn << 15) | mag;
  const unsigned idx = (mag - kLutLoBits) + (sgn ? (unsigned)kLutEntriesPerSign : 0u);
  *(float*)(d_dgelu_table + (idx << 2)) = act_dgelu(__uint_as_float(bits << 16));
}

// ---------------------------------------------------------------------------
// Tile geometry.
//
// A CTA covers 64 rows -- one dbias band -- by 256 columns, walked as two 32-row
// sub-tiles.  Thirty-two rows is exactly the colwise MX block height, which is
// the whole point: it closes the colwise reduction inside the CTA, so the tile
// can be read once into registers and drive both scaling directions from there
// with nothing staged for a second pass.
// ---------------------------------------------------------------------------

//! Columns a CTA covers.
constexpr int kTileCols = 256;
//! Elements in one MX block, and equivalently the colwise block height.
constexpr int kRowsPerMxBlock = 32;
//! The streaming tile instead gives each thread two columns to load.
constexpr int kStreamingThreadsPerCta = kTileCols / 2;
//! ...and stages a row of the tile as that many 32-bit words of shared memory.
constexpr int kStreamingWordsPerSharedRow = kTileCols / 2;
//! MX groups (32 columns each) across the tile.
constexpr int kMxGroupsPerRow = kTileCols / 32;
//! A register-resident tile spreads its 256 columns over the 32 lanes of a warp.
constexpr int kColsPerLane = kTileCols / 32;
//! ...which a lane holds as BF16 pairs, one 32-bit word each.
constexpr int kWordsPerLane = kColsPerLane / 2;
//! ...read as 128-bit vectors, so one per row.
constexpr int kVecLoadsPerRow = kWordsPerLane / 4;
//! Lanes that have to cooperate to cover one 32-column MX group.
constexpr int kLanesPerMxGroup = 32 / kColsPerLane;

/*! \brief Store one lane's quantized row: kColsPerLane FP8 bytes, one STG.64.
 *
 * Spelling the vector type out, rather than storing four bytes at a time and
 * hoping the compiler merges them, is what guarantees the single wide store the
 * lane decomposition was chosen for.
 */
__device__ __forceinline__ void store_quantized(unsigned char* p, const unsigned* v) {
  static_assert(kColsPerLane == 8, "One STG.64 covers exactly eight FP8 bytes.");
  *(uint2*)p = *(const uint2*)v;
}
// How many of the four column words of a row take the shared-memory table
// route; the rest run the exact packed arithmetic body.  The table probe is
// L1/shared-pipe work and the arithmetic body is FMA/MUFU work, so the split
// is what keeps both sides of the machine busy: an all-table kernel saturates
// the shared pipe (measured 57% of its wavefronts are bank-conflict replays,
// L1 88% busy against DRAM 40%) while the FP32 pipes idle.
// Table/arithmetic routing at (row, word) SLOT granularity inside one kRowsInFlight-row
// load group: `L` of the `S` slots take the shared-table route and the
// arithmetic ones are spread by a Bresenham step, so their MUFU chains
// interleave with the table probes instead of clumping at the end of a row.
template <int S, int L, int I>
struct LutSlot {
  static constexpr int A = S - L;
  static constexpr bool v = !((((I + 1) * A) % S) < A);
};
//! Of the eight slots in dGeLU's two-row load group, how many take the table.
constexpr int kLutSlotsDact = 6;
//! Of the four slots in dbias+dGeLU's one-row load group, how many take it.
constexpr int kLutSlotsDbiasDact = 3;

// ---------------------------------------------------------------------------
// In-flight load-group depth: how many rows of the tile are read from global
// memory, and converted, before the next batch is issued.  Deeper keeps more
// loads in flight; shallower frees the registers those loads occupy.
// ---------------------------------------------------------------------------

//! dbias+dGeLU carries four FP32x2 dbias accumulators on top of its sixteen
//! tile words, which put a full four-row group above its register tier and
//! bought an 8-byte stack frame.  One row at a time frees the second row's
//! activation/gradient pair outright and the spill disappears at the same 64
//! registers and the same four resident CTAs: the trade is in-flight bytes
//! against spill traffic, and here the spill was the expensive half.
constexpr int kRowsInFlightDbiasDact = 1;
//! dbias alone carries the same accumulators but on the streaming tile, where
//! a four-row group still fits.
constexpr int kRowsInFlightDbias = 4;
//! Rows the streaming tile loads per batch.
constexpr int kStreamingUnroll = 8;

// ---------------------------------------------------------------------------
// Occupancy targets, as the second __launch_bounds__ argument.
//
// These were selected by autotuning.  Note that ptxas silently ignores the
// bound when threads_per_cta * blocks exceeds what the target architecture
// allows, so a 256-thread instantiation asking for 6 blocks may end up
// unconstrained.
// ---------------------------------------------------------------------------

//! Cast-only prefers 6 resident CTAs to 8: the mode runs at the HBM ceiling, so
//! the residency it gives up is slack it was not using, and the ~10 extra
//! registers per thread stop the 8-wide load batch's addresses from being
//! rematerialised.
constexpr int kMinBlocksCast = 6;
//! GeLU runs the narrow 128-thread tile, so 8 blocks is 32 resident warps.
constexpr int kMinBlocksAct = 8;
constexpr int kMinBlocksDact = 6;
constexpr int kMinBlocksDbiasDact = 4;
//! dbias runs the streaming tile at 128 threads.
constexpr int kMinBlocksDbias = 8;

//! How many 32-row iterations a register-resident dbias CTA folds into ONE
//! workspace band.  The shared fold (8 KB out, 8 KB in) and its two barriers
//! are paid once per band instead of once per 64 rows, and the band count --
//! hence both the workspace round trip and the length of the reducer's
//! dependent add chain -- shrinks by the same factor.
constexpr int kBandFoldIters = 4;

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
struct Cfg {
  static constexpr bool kHasActivation = IS_ACT || IS_DACT;
  // CAST_DBIAS_DACT joins the register-resident tiling: its dbias tolerance
  // (TE's rtol_dbias = 4e-2 with atol 1e-4) absorbs the band-local regrouping
  // that a row-owned tile forces, whereas CAST_DBIAS's much tighter
  // {1e-5, 1e-2} does not, so that mode keeps the column-owned streaming tile
  // and its exact row-order fp32 chain.
  static constexpr bool kRegisterResident = !IS_DBIAS || IS_DACT;
  static constexpr bool kNarrowTile = IS_ACT && !IS_DACT && !IS_DBIAS;
  static constexpr int kThreadsPerCta =
      kRegisterResident ? (kNarrowTile ? 128 : 256) : kStreamingThreadsPerCta;
  static constexpr int kWarpsPerCta = kThreadsPerCta / 32;
  static constexpr int kRowsPerWarp = kRegisterResident ? 32 / kWarpsPerCta : 1;  // rows per warp
  //! Whether the table/arithmetic split is expressed at (row, word) slot
  //! granularity rather than per row; see LutSlot.
  static constexpr bool kSlotSplit = IS_DACT || kNarrowTile;
  // The narrow tile drops the split epilogue: with only four warps there is no
  // idle tid range left to drain the row-scale scratch inside the columnwise
  // barrier interval.
  // The split epilogue needs somewhere to drain the row-scale scratch inside
  // the columnwise barrier interval.  A tile wide enough to leave an idle tid
  // range uses it; a narrow one simply hands the drain to the SAME threads that
  // just folded the columnwise partials -- they are done and the interval is
  // already open, so no third CTA barrier is needed either way.  What the split
  // buys is that barrier: three per 32-row tile become two, and the rowwise
  // stores retire before the columnwise half so their scale registers die early.
  static constexpr bool kSplitEpilogue =
      (kHasActivation || IS_DBIAS) && (kThreadsPerCta >= kTileCols / 2);
  static constexpr int kDrainTidBase =
      (kThreadsPerCta >= kTileCols / 2 + kTileCols / 4) ? kTileCols / 2 : 0;
  static constexpr int kRowsInFlight = (IS_DACT && IS_DBIAS) ? kRowsInFlightDbiasDact
                                       : IS_DBIAS            ? kRowsInFlightDbias
                                       : kSlotSplit          ? 2
                                                             : kRowsPerWarp;
  //! (row, word) slots in one load group, the granularity the table/arithmetic
  //! split is expressed at.
  static constexpr int kSlots = kRowsInFlight * kWordsPerLane;
  //! How many of those slots take the table route; see LutSlot.
  static constexpr int kLutSlots = IS_DBIAS ? kLutSlotsDbiasDact : kLutSlotsDact;
  // Only IS_DBIAS && !IS_DACT leaves the register-resident tile, so the
  // streaming case needs no further discrimination.
  static constexpr int kMinBlocksPerSm = !kRegisterResident ? kMinBlocksDbias
                                         : IS_DBIAS         ? kMinBlocksDbiasDact
                                         : IS_DACT          ? kMinBlocksDact
                                         : IS_ACT           ? kMinBlocksAct
                                                            : kMinBlocksCast;
  // per-warp fp32 dbias partials, folded in warp order inside the 64-row band
  static constexpr int kDbiasScratchWords =
      IS_DBIAS && kRegisterResident ? kWarpsPerCta * kTileCols : 1;
  //! Only dGeLU tabulates; GeLU evaluates its closed form for every word.
  static constexpr bool kNeedsLut = IS_DACT;
  //! Shared-memory bytes for the table.  A non-tabulating instantiation still
  //! declares the array, so give it the smallest legal size rather than zero.
  static constexpr int kLutSharedBytes = kNeedsLut ? kLutBytes : 16;
};

// global -> shared copy of the activation table.
//
// The table is the ONLY shared/L1 traffic a CTA pays that is not proportional
// to the tile work it does, and on the activation instantiations that pipe is
// the binding resource (16 KB / 128 B = 128 LSU wavefronts per CTA, ~6% of a
// CTA's entire wavefront budget at 8192x8192, where a CTA walks only two row
// blocks).  Handing the copy to the bulk-async (TMA) engine takes all of them
// off the LSU: one instruction issued by one thread, the data lands in shared
// through the async proxy, and the CTA joins on an mbarrier it needed a
// __syncthreads for anyway.
// The join is deliberately NOT part of the issue: a CTA that walks two row
// blocks spends a measurable share of its life parked on this barrier, and
// nothing about the tile's global loads depends on the table.  Issuing the copy
// first, then the first row group's loads, then joining, puts the table's L2
// round trip underneath the input's DRAM round trip instead of in front of it.
// ---------------------------------------------------------------------------
// Split cross-warp join.  __syncthreads() is arrive-and-block: the whole
// epilogue's cross-warp dependency (every warp's 32-row columnwise partial)
// is ready the instant the activation loop ends, but the barrier that publishes
// it is placed after the ROWWISE half, so each warp pays the full arrival skew
// with nothing to do.  An mbarrier splits the two: publish the partials, arrive
// (non-blocking), run the entire rowwise half -- quantize, store, row-scale
// scratch -- and only then wait.  The join latency disappears under work the
// warp had to do anyway, and the CTA still pays exactly two joins per 32-row
// tile because the row-scale drain moves behind the second one.
__device__ __forceinline__ void mbar_init(unsigned long long* bar, int cnt) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned phase) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  unsigned ok;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(ok)
        : "r"(b), "r"(phase)
        : "memory");
  } while (!ok);
}

__device__ __forceinline__ void wait_lut(unsigned long long* bar) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  unsigned ok;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(ok)
        : "r"(b)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void load_lut(unsigned char* dst, const void* src, int bytes,
                                         unsigned long long* bar, int tid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  __syncthreads();
  if (tid == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(d),
        "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(b)
        : "memory");
  }
  wait_lut(bar);
}

// ---------------------------------------------------------------------------
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
__device__ __forceinline__ void quantize_streaming(
    const unsigned* __restrict__ input,      // primary tensor (x or grad)
    const unsigned* __restrict__ act_input,  // pre-activation tensor (x)
    unsigned char* __restrict__ out_rw, unsigned char* __restrict__ scale_rw,
    unsigned char* __restrict__ out_cw, unsigned char* __restrict__ scale_cw,
    float* __restrict__ dbias_ws, int K, int sc_rw_stride, int sc_cw_stride, int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  // Cfg::kRegisterResident is false only when IS_DBIAS && !IS_DACT, so this body
  // never has to evaluate dGeLU and carries no activation table.
  static_assert(!IS_DACT, "The streaming tile is the non-dGeLU path.");
  __shared__ __align__(16) unsigned tile[kRowsPerMxBlock * kStreamingWordsPerSharedRow];
  __shared__ __align__(16) unsigned cscale[kTileCols / 2];
  __shared__ __align__(4) unsigned char srs[kRowsPerMxBlock * kMxGroupsPerRow];

  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * kTileCols;
  constexpr int TROWS = IS_DBIAS ? 64 : 32;  // dbias needs a full band per CTA
  const int Kw = K >> 1;

  const int lane = tid & 31;
  const int warp = tid >> 5;

  // CAST_ONLY/CAST_DBIAS always launch one row block per CTA.  Keep that bound
  // compile-time in their template instantiations so the hot path has no loop
  // backedge and does not carry the runtime `iters` argument through address
  // generation.  Activation modes retain their tuned multi-block walks.
  const int loop_iters = (IS_ACT || IS_DACT) ? iters : 1;
#pragma unroll 1
  for (int it = 0; it < loop_iters; ++it) {
    const int row0 = (blockIdx.y * loop_iters + it) * TROWS;
    f32x2 dbp = make_f32x2(0.f, 0.f);
    const unsigned* ip = input + (size_t)row0 * Kw + (col0 >> 1) + tid;
    const unsigned* ap = act_input + (size_t)row0 * Kw + (col0 >> 1) + tid;

#pragma unroll 1
    for (int sub = 0; sub < TROWS / kRowsPerMxBlock; ++sub) {
      // ------------- phase 1: load, activate, colwise amax, dbias ---------
      {
        unsigned acc = 0u;
        constexpr int UN = 8;
        auto do_batch = [&](int b) {
          unsigned ra[UN], rg[UN];
          const size_t base = (size_t)(sub * kRowsPerMxBlock + b * UN) * Kw;
#pragma unroll
          for (int k = 0; k < UN; ++k)
            ra[k] = (IS_ACT || IS_DACT) ? ap[base + (size_t)k * Kw] : ip[base + (size_t)k * Kw];
          if constexpr (IS_DACT) {
#pragma unroll
            for (int k = 0; k < UN; ++k) rg[k] = ip[base + (size_t)k * Kw];
          }
// rows are consumed in strictly increasing k so the fp32 dbias chain keeps
// the reference's summation order; the LUT/arithmetic split rides along.
#define MXFP8_ROW(KO)                                                        \
  {                                                                          \
    constexpr int k = (KO);                                                  \
    unsigned packed;                                                         \
    if constexpr (IS_ACT) {                                                  \
      packed = gelu_bf16x2(ra[k]);                                           \
    } else {                                                                 \
      packed = ra[k];                                                        \
      if constexpr (IS_DBIAS) dbp = add_f32x2(dbp, bf16x2_to_f32x2(packed)); \
    }                                                                        \
    acc = abs_max_bf16x2(acc, packed);                                       \
    tile[(b * UN + k) * kStreamingWordsPerSharedRow + tid] = packed;         \
  }
          MXFP8_ROW(0)
          MXFP8_ROW(1) MXFP8_ROW(2) MXFP8_ROW(3) MXFP8_ROW(4) MXFP8_ROW(5) MXFP8_ROW(6) MXFP8_ROW(7)
#undef MXFP8_ROW
        };
        if constexpr (IS_ACT || IS_DACT) {
          // the closed-form tails freed enough registers to keep two 8-row
          // batches (32 loads) in flight, which is what this low-occupancy
          // shared-staged path needs to cover DRAM latency.
#pragma unroll 2
          for (int b = 0; b < kRowsPerMxBlock / UN; ++b) do_batch(b);
        } else {
#pragma unroll
          for (int b = 0; b < kRowsPerMxBlock / UN; ++b) do_batch(b);
        }
        unsigned m = acc & 0x7fff7fffu;
        const unsigned p =
            ptx::float_to_e8m0_2x(bf16_hi(m) * (1.0f / 448.0f), bf16_lo(m) * (1.0f / 448.0f));
        *(unsigned short*)(scale_cw + (size_t)(row0 / 32 + sub) * sc_cw_stride + col0 + 2 * tid) =
            (unsigned short)p;
        cscale[tid] = (0x00FE00FEu - __byte_perm(p, 0, 0x4140)) << 7;
      }

      __syncthreads();

      // ------------- phase 2: rowwise amax + both quantizations -----------
      {
        const uint4* tp = (const uint4*)tile;
        const uint4 csv = *(const uint4*)(cscale + 4 * lane);
        const unsigned* cw = (const unsigned*)&csv;
        const int coff = col0 + 8 * lane;

#pragma unroll
        // Two rows at a time: each row's 8-value magnitude fold stays local,
        // then the two per-lane maxima are packed into one bf16x2 word so a
        // single shuffle chain and one ptx::float_to_e8m0_2x serve both rows.  Halves the
        // shuffle and scale-conversion work of this phase.
        for (int i = 0; i < kRowsPerMxBlock / 4; i += 2) {
          const int r = warp + 4 * i;
          const int r1 = r + 4;
          const uint4 v0 = tp[r * (kStreamingWordsPerSharedRow / 4) + lane];
          const uint4 v1 = tp[r1 * (kStreamingWordsPerSharedRow / 4) + lane];
          auto row_mag = [](const uint4& v) {
            unsigned a = abs_max_bf16x2(abs_max_bf16x2(v.x, v.y), abs_max_bf16x2(v.z, v.w));
            return abs_max_bf16x2(a, __byte_perm(a, a, 0x1032));
          };
          unsigned p = __byte_perm(row_mag(v0), row_mag(v1), 0x5410) & 0x7fff7fffu;
          p = abs_max_bf16x2(p, __shfl_xor_sync(0xffffffffu, p, 1));
          p = abs_max_bf16x2(p, __shfl_xor_sync(0xffffffffu, p, 2));
          const unsigned er =
              ptx::float_to_e8m0_2x(bf16_hi(p) * (1.0f / 448.0f), bf16_lo(p) * (1.0f / 448.0f));
          const unsigned e0 = er & 0xffu;
          const unsigned e1 = (er >> 8) & 0xffu;
          if ((lane & 3) == 0) {
            srs[r * kMxGroupsPerRow + (lane >> 2)] = (unsigned char)e0;
            srs[r1 * kMxGroupsPerRow + (lane >> 2)] = (unsigned char)e1;
          }
          const unsigned rsp0 = e8m0_to_bf16x2_reciprocal(e0);
          const unsigned rsp1 = e8m0_to_bf16x2_reciprocal(e1);

          auto write_row = [&](const uint4& v, int rr, unsigned rsp) {
            const unsigned* vw = (const unsigned*)&v;
            unsigned ro[2], co[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              ro[j] = ptx::mul_cvt_e4m3x4(vw[2 * j], vw[2 * j + 1], rsp, rsp);
              co[j] = ptx::mul_cvt_e4m3x4(vw[2 * j], vw[2 * j + 1], cw[2 * j], cw[2 * j + 1]);
            }
            const size_t go = (size_t)(row0 + sub * kRowsPerMxBlock + rr) * K + coff;
            *(uint2*)(out_rw + go) = *(const uint2*)ro;
            *(uint2*)(out_cw + go) = *(const uint2*)co;
          };
          write_row(v0, r, rsp0);
          write_row(v1, r1, rsp1);
        }
      }

      __syncthreads();

      if (tid < kRowsPerMxBlock * 2) {
        const int r = tid >> 1;
        const int half = tid & 1;
        *(unsigned*)(scale_rw + (size_t)(row0 + sub * kRowsPerMxBlock + r) * sc_rw_stride +
                     (col0 >> 5) + half * 4) =
            *(const unsigned*)(srs + r * kMxGroupsPerRow + half * 4);
      }
    }

    if constexpr (IS_DBIAS) {
      // one fp32 partial per 64-row band, accumulated strictly in row order --
      // the reference's exact summation shape.
      float* wp = dbias_ws + (size_t)(row0 >> 6) * K + col0 + 2 * tid;
      ptx::st_global_b64(wp, dbp, ptx::create_l2_policy_evict_last());
    }
  }
}

// ---------------------------------------------------------------------------
// Register-resident tiling: the 32x256 chunk never touches shared memory, only
// the 32-row columnwise partials do.
// ---------------------------------------------------------------------------
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
__device__ __forceinline__ void quantize_regtile(
    const uint4* __restrict__ input, const uint4* __restrict__ act_input,
    unsigned char* __restrict__ out_rw, unsigned char* __restrict__ scale_rw,
    unsigned char* __restrict__ out_cw, unsigned char* __restrict__ scale_cw,
    float* __restrict__ dbias_ws, int K, int sc_rw_stride, int sc_cw_stride, int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  constexpr int kWarpsPerCta = C::kWarpsPerCta, kRowsPerWarp = C::kRowsPerWarp,
                kRowsInFlight = C::kRowsInFlight;

  // The cross-warp columnwise-amax scratch and the dbias band fold are never
  // live at the same time -- a barrier separates every use of one from the next
  // use of the other -- so they share one buffer.  That is 4 KB less per CTA,
  // which is what keeps CAST_DBIAS_DACT inside the 164 KB carveout at 6 blocks.
  constexpr int FOLDW = (kWarpsPerCta * (kTileCols / 2) > C::kDbiasScratchWords)
                            ? kWarpsPerCta * (kTileCols / 2)
                            : C::kDbiasScratchWords;
  __shared__ __align__(16) unsigned foldbuf[FOLDW];
  unsigned(*cpart)[kTileCols / 2] = (unsigned(*)[kTileCols / 2]) foldbuf;
  float* dbp_s = (float*)foldbuf;
  __shared__ __align__(16) unsigned cscale[kTileCols / 2];
  // With the split join the row-scale scratch of tile `it` is still being
  // drained (after the second join) while tile it+1's rowwise half is already
  // filling it, so it double-buffers on `it & 1`.  256 extra bytes per CTA and
  // no extra live register -- the alternative, hoisting the scales into
  // registers so the mbarrier could order them, costs kRowsPerWarp registers on exactly
  // the instantiations that sit on a register cliff.
  __shared__ __align__(8) unsigned char srs[kRowsPerMxBlock * kMxGroupsPerRow];
  __shared__ __align__(16) unsigned char lutmem[C::kLutSharedBytes];
  __shared__ __align__(8) unsigned long long lutbar;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * kTileCols;
  const int K4 = K >> 3;  // uint4 (8 bf16 values) per row

  // Only the DBIAS-fused instantiations get a dbias_ws scratch buffer: TE's
  // nvte_quantize (plain cast) and the ACT/DACT-only entry points never
  // allocate one, so zeroing it unconditionally null-derefs on every call
  // that isn't fused with dbias.
  if constexpr (IS_DBIAS) {
    if (blockIdx.y == 0) {
#pragma unroll
      for (int q = tid; q < kTileCols; q += C::kThreadsPerCta)
        ((unsigned short*)dbias_ws)[col0 + q] = 0;
    }
  }

  if constexpr (C::kNeedsLut) load_lut(lutmem, d_dgelu_table, C::kLutSharedBytes, &lutbar, tid);
  const unsigned char* __restrict__ atab = lutmem;
  // one packed fp32 pair per owned column pair: add.rn.f32x2 folds both lanes
  // of a column pair in a single instruction with exactly the scalar rounding
  f32x2 dba[IS_DBIAS ? 4 : 1];

  // The non-activation instantiation is a fixed single-block walk.  Expressing
  // that through the template constant removes the dynamic loop/control and
  // row-address multiply from every tiny CAST_ONLY CTA.
  const int loop_iters = (IS_ACT || IS_DACT) ? iters : 1;
#pragma unroll 1
  for (int it = 0; it < loop_iters; ++it) {
    // A CTA walks `iters` row blocks so the one-off activation-table copy
    // amortises, but WHICH row blocks matters for DRAM: taking them
    // consecutively (blockIdx.y*iters + it) makes the resident CTAs drift into
    // `iters` different 32-row windows, whereas taking them grid-strided keeps
    // every CTA that is on the same step covering one contiguous row band --
    // exactly the footprint the iters==1 CAST_ONLY instantiation has, which is
    // the only one running at the HBM ceiling.
    const int row0 = (it * gridDim.y + blockIdx.y) * kRowsPerMxBlock;
    unsigned dv[kRowsPerWarp *
                kWordsPerLane];  // this thread's kRowsPerWarp x kColsPerLane sub-block, bf16x2
    // one fp32 running sum per owned column, restarted at each 64-row band
    // boundary so the workspace keeps TE's per-band partial layout
    // The band is the CTA's whole row walk: the shared fold (8 KB out, 8 KB in)
    // and its two barriers are paid ONCE per CTA instead of once per kBandFoldIters
    // iterations, and the workspace round trip plus the reducer's dependent
    // chain shrink by the same factor.  Rows are still consumed in strictly
    // increasing order inside the band.
    if constexpr (IS_DBIAS) {
      if (it == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dba[i] = make_f32x2(0.f, 0.f);
      }
    }
    const size_t rbase =
        (size_t)(row0 + warp * kRowsPerWarp) * K4 + (col0 >> 3) + (size_t)kVecLoadsPerRow * lane;
    const uint4* ip = input + rbase;
    const uint4* ap = act_input + rbase;

    // ---- load + activate, kRowsInFlight rows in flight at a time --------------------
#pragma unroll
    for (int h = 0; h < kRowsPerWarp / kRowsInFlight; ++h) {
      uint4 a[kRowsInFlight][kVecLoadsPerRow], g[kRowsInFlight][kVecLoadsPerRow];
#pragma unroll
      for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
        for (int n = 0; n < kVecLoadsPerRow; ++n)
          a[t][n] = ap[(size_t)(h * kRowsInFlight + t) * K4 + n];
      if constexpr (IS_DACT) {
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
          for (int n = 0; n < kVecLoadsPerRow; ++n)
            g[t][n] = ip[(size_t)(h * kRowsInFlight + t) * K4 + n];
      }
      // Convert the load group in place: every (row, word) slot becomes the
      // BF16 pair that the epilogue will quantize, and the dbias
      // instantiations accumulate their FP32 column partial on the way past.
      if constexpr (!IS_DACT) {
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t) {
#pragma unroll
          for (int n = 0; n < kVecLoadsPerRow; ++n) {
            const unsigned* av = (const unsigned*)&a[t][n];
            const int dvb = (h * kRowsInFlight + t) * kWordsPerLane + n * 4;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              if constexpr (IS_ACT) {
                dv[dvb + m] = gelu_bf16x2(av[m]);
              } else {
                dv[dvb + m] = av[m];
                if constexpr (IS_DBIAS) dba[m] = add_f32x2(dba[m], bf16x2_to_f32x2(av[m]));
              }
            }
          }
        }
      } else {
// The (row, word) slots of a load group are split between the table and the
// arithmetic body so the two saturated pipes (ALU + shared) hand work to the
// two idle ones (FP32 + MUFU); LutSlot spreads the arithmetic slots with a
// Bresenham step so their MUFU chains interleave with the table probes instead
// of clumping at the end of a row.
//
// The two dGeLU instantiations multiply by the incoming gradient differently on
// purpose.  With dbias fused the product feeds an FP32 accumulator, so the table
// route keeps its two probe results scalar and multiplies them scalar; without
// dbias the result only has to be packed back to BF16, so the packed multiply is
// the cheaper end.
#define MXFP8_DACT_WORD(T, M)                                                      \
  {                                                                                \
    constexpr bool kUseLut =                                                       \
        LutSlot<C::kSlots, C::kLutSlots, (T) * 4 + (M) + (IS_DBIAS ? 0 : 2)>::v;   \
    f32x2 r_;                                                                      \
    if constexpr (IS_DBIAS) {                                                      \
      r_ = dgelu_grad_word<kUseLut>(av[M], gv[M], atab);                           \
      dba[M] = add_f32x2(dba[M], r_);                                              \
    } else {                                                                       \
      r_ = mul_f32x2(dgelu_word<kUseLut>(av[M], atab), bf16x2_to_f32x2(gv[M]));    \
    }                                                                              \
    float a_, b_;                                                                  \
    unpack_f32x2(r_, a_, b_);                                                      \
    dv[(h * kRowsInFlight + (T)) * kWordsPerLane + (M)] = ptx::cvt_bf16x2(b_, a_); \
  }
#define MXFP8_DACT_ROW(T)                           \
  {                                                 \
    const unsigned* av = (const unsigned*)&a[T][0]; \
    const unsigned* gv = (const unsigned*)&g[T][0]; \
    MXFP8_DACT_WORD(T, 0)                           \
    MXFP8_DACT_WORD(T, 1)                           \
    MXFP8_DACT_WORD(T, 2)                           \
    MXFP8_DACT_WORD(T, 3)                           \
  }
        MXFP8_DACT_ROW(0)
        if constexpr (kRowsInFlight > 1) MXFP8_DACT_ROW(1)
        if constexpr (kRowsInFlight > 2) {
          MXFP8_DACT_ROW(2) MXFP8_DACT_ROW(3)
        }
#undef MXFP8_DACT_ROW
#undef MXFP8_DACT_WORD
      }
    }

    // Nothing in the ROWWISE half of the epilogue depends on the columnwise
    // scale, so for the instantiations that can afford the registers it runs
    // BEFORE the cross-warp barrier.  The row-scale scratch is then drained
    // inside the SAME barrier interval that reduces the columnwise partials --
    // by the tid in [128,192) threads that reduction leaves idle -- which turns
    // three CTA barriers per iteration into two and retires the rowwise stores
    // early so their scale registers stop being live across the columnwise half.
    constexpr bool kSplitEpilogue = C::kSplitEpilogue;
    const size_t gbase = (size_t)(row0 + warp * kRowsPerWarp) * K + col0 + kColsPerLane * lane;
    unsigned char* const srsb = srs;
    auto row_scale = [&](const unsigned* v, int j) {
      unsigned a = abs_max_bf16x2(abs_max_bf16x2(v[0], v[1]), abs_max_bf16x2(v[2], v[3]));
#pragma unroll
      for (int m = 4; m < kWordsPerLane; m += 2)
        a = abs_max_bf16x2(a, abs_max_bf16x2(v[m], v[m + 1]));
        // A 32-value rowwise block spans 32/kColsPerLane lanes, so a lane owning sixteen
        // columns needs ONE cross-lane fold where an eight-column lane needs two.
#pragma unroll
      for (int b = 1; b < kLanesPerMxGroup; b <<= 1)
        a = abs_max_bf16x2(a, __shfl_xor_sync(0xffffffffu, a, b));
      unsigned er;
      if constexpr (IS_DBIAS) {
        // CAST_DBIAS_DACT is the instantiation sitting on the 5-block register
        // cliff: the masked-extract form costs two more integer ops but keeps
        // one fewer value live, which is the cheaper trade there.
        a &= 0x7fff7fffu;
        const float mx = fmaxf(bf16_lo(a), bf16_hi(a)) * (1.0f / 448.0f);
        er = ptx::float_to_e8m0_2x(mx, mx) & 0xffu;
      } else {
        // Fold the two bf16 halves with one PRMT + one packed max instead of a
        // mask, two extracts and an FMNMX: max.xorsign.abs leaves the magnitude
        // in BOTH halves, so masking the high half alone already yields the
        // fp32 bit pattern.  Converting with 0.0f in the upper slot then lands
        // the exponent in the low byte with no trailing mask.
        a = abs_max_bf16x2(a, __byte_perm(a, a, 0x1032));
        const float mx = __uint_as_float(a & 0x7fff0000u) * (1.0f / 448.0f);
        er = ptx::float_to_e8m0_2x(0.f, mx);
      }
      if ((lane & (kLanesPerMxGroup - 1)) == 0)
        srsb[(warp * kRowsPerWarp + j) * kMxGroupsPerRow + (lane / kLanesPerMxGroup)] =
            (unsigned char)er;
      return e8m0_to_bf16x2_reciprocal(er);
    };
    // On the 128-thread CAST_ACT tile each warp owns eight rows.  Pack the
    // maxima of two rows into one bf16x2 so the lane butterfly, 1/448 scale,
    // and native UE8M0 conversion serve both rows together.
    auto row_mag = [&](const unsigned* v) {
      unsigned a = abs_max_bf16x2(abs_max_bf16x2(v[0], v[1]), abs_max_bf16x2(v[2], v[3]));
#pragma unroll
      for (int m = 4; m < kWordsPerLane; m += 2)
        a = abs_max_bf16x2(a, abs_max_bf16x2(v[m], v[m + 1]));
      return abs_max_bf16x2(a, __byte_perm(a, a, 0x1032));
    };
    auto row_scale_pair = [&](const unsigned* va, const unsigned* vb, int j, unsigned& s0,
                              unsigned& s1) {
      unsigned p = __byte_perm(row_mag(va), row_mag(vb), 0x5410) & 0x7fff7fffu;
#pragma unroll
      for (int b = 1; b < kLanesPerMxGroup; b <<= 1)
        p = abs_max_bf16x2(p, __shfl_xor_sync(0xffffffffu, p, b));
      float m0, m1;
      unpack_f32x2(mul_f32x2(make_f32x2(bf16_lo(p), bf16_hi(p)), splat_f32x2(1.0f / 448.0f)), m0,
                   m1);
      const unsigned e = ptx::float_to_e8m0_2x(m1, m0);
      const unsigned e0 = e & 0xffu, e1 = (e >> 8) & 0xffu;
      if ((lane & (kLanesPerMxGroup - 1)) == 0) {
        unsigned char* q =
            srsb + (warp * kRowsPerWarp + j) * kMxGroupsPerRow + (lane / kLanesPerMxGroup);
        q[0] = (unsigned char)e0;
        q[kMxGroupsPerRow] = (unsigned char)e1;
      }
      s0 = e8m0_to_bf16x2_reciprocal(e0);
      s1 = e8m0_to_bf16x2_reciprocal(e1);
    };
    // ---- columnwise amax: kRowsPerWarp-row partial in registers, cross-warp in smem
    auto colpart = [&]() {
      unsigned c[kWordsPerLane];
#pragma unroll
      for (int m = 0; m < kWordsPerLane; ++m) c[m] = dv[m];
#pragma unroll
      for (int j = 1; j < kRowsPerWarp; ++j)
#pragma unroll
        for (int m = 0; m < kWordsPerLane; ++m)
          c[m] = abs_max_bf16x2(c[m], dv[j * kWordsPerLane + m]);
#pragma unroll
      for (int n = 0; n < kVecLoadsPerRow; ++n)
        *(uint4*)(&cpart[warp][kWordsPerLane * lane + 4 * n]) = *(const uint4*)(c + 4 * n);
    };
    constexpr bool PAIRSC =
        IS_ACT && !IS_DACT && !IS_DBIAS && C::kThreadsPerCta == 128 && kRowsPerWarp == 8;
    if constexpr (PAIRSC) {
      unsigned char* prw = out_rw + gbase;
#pragma unroll
      for (int j = 0; j < kRowsPerWarp; j += 2) {
        const unsigned* va = dv + j * kWordsPerLane;
        const unsigned* vb = va + kWordsPerLane;
        unsigned s0, s1;
        row_scale_pair(va, vb, j, s0, s1);
        unsigned ro[kWordsPerLane / 2];
#pragma unroll
        for (int m = 0; m < kWordsPerLane / 2; ++m)
          ro[m] = ptx::mul_cvt_e4m3x4(va[2 * m], va[2 * m + 1], s0, s0);
        store_quantized(prw + (unsigned)j * (unsigned)K, ro);
#pragma unroll
        for (int m = 0; m < kWordsPerLane / 2; ++m)
          ro[m] = ptx::mul_cvt_e4m3x4(vb[2 * m], vb[2 * m + 1], s1, s1);
        store_quantized(prw + (unsigned)(j + 1) * (unsigned)K, ro);
      }
    } else if constexpr (kSplitEpilogue) {
      unsigned char* prw = out_rw + gbase;
#pragma unroll
      for (int j = 0; j < kRowsPerWarp; ++j) {
        const unsigned* v = dv + j * kWordsPerLane;
        const unsigned rsp = row_scale(v, j);
        unsigned ro[kWordsPerLane / 2];
#pragma unroll
        for (int m = 0; m < kWordsPerLane / 2; ++m)
          ro[m] = ptx::mul_cvt_e4m3x4(v[2 * m], v[2 * m + 1], rsp, rsp);
        store_quantized(prw + (unsigned)j * (unsigned)K, ro);
      }
    }

    colpart();
    __syncthreads();
    if (tid < kTileCols / 2) {
      unsigned m = cpart[0][tid];
#pragma unroll
      for (int w = 1; w < kWarpsPerCta; ++w) m = abs_max_bf16x2(m, cpart[w][tid]);
      m &= 0x7fff7fffu;
      const unsigned p =
          ptx::float_to_e8m0_2x(bf16_hi(m) * (1.0f / 448.0f), bf16_lo(m) * (1.0f / 448.0f));
      *(unsigned short*)(scale_cw + (size_t)(row0 / 32) * sc_cw_stride + col0 + 2 * tid) =
          (unsigned short)p;
      cscale[tid] = (0x00FE00FEu - __byte_perm(p, 0, 0x4140)) << 7;
    }
    auto drain = [&]() {
      constexpr int DO = C::kDrainTidBase;
      // A row's eight group scales are eight CONTIGUOUS bytes of scale_rw, and
      // both ends are 8-byte aligned for every shape in the contract
      // (sc_rw_stride is a multiple of four groups; col0 >> 5 is a multiple of
      // eight), so one warp drains a whole 32-row tile with one STG.64 per lane
      // where two warps needed one STG.32 each.  Same sector count, half the
      // store instructions, and the released warp rejoins the columnwise
      // quantize inside the already-open barrier interval.
      if (tid >= DO && tid < DO + kRowsPerMxBlock) {
        const int r = tid - DO;
        *(f32x2*)(scale_rw + (size_t)(row0 + r) * sc_rw_stride + (col0 >> 5)) =
            *(const f32x2*)(srsb + r * kMxGroupsPerRow);
      }
    };
    // Without the split join the scratch is already published by the first
    // __syncthreads and the drain fills the fold's idle tid range.  With it,
    // only the SECOND join orders the scratch, so the drain moves behind it and
    // overlaps the columnwise quantization instead.
    if constexpr (kSplitEpilogue) drain();
    __syncthreads();

    // ---- columnwise (and, when not split, rowwise) quantization ----------
    {
      uint4 csv[kVecLoadsPerRow];
#pragma unroll
      for (int n = 0; n < kVecLoadsPerRow; ++n)
        csv[n] = *(const uint4*)(cscale + kWordsPerLane * lane + 4 * n);
      const unsigned* cw = (const unsigned*)csv;
      unsigned char* prw = out_rw + gbase;
      unsigned char* pcw = out_cw + gbase;
#pragma unroll
      for (int j = 0; j < kRowsPerWarp; ++j) {
        const unsigned* v = dv + j * kWordsPerLane;
        if constexpr (!kSplitEpilogue) {
          const unsigned rsp = row_scale(v, j);
          unsigned ro[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            ro[m] = ptx::mul_cvt_e4m3x4(v[2 * m], v[2 * m + 1], rsp, rsp);
          store_quantized(prw + (unsigned)j * (unsigned)K, ro);
        }
        unsigned co[kWordsPerLane / 2];
#pragma unroll
        for (int m = 0; m < kWordsPerLane / 2; ++m)
          co[m] = ptx::mul_cvt_e4m3x4(v[2 * m], v[2 * m + 1], cw[2 * m], cw[2 * m + 1]);
        store_quantized(pcw + (unsigned)j * (unsigned)K, co);
      }
    }
    if constexpr (!kSplitEpilogue) {
      __syncthreads();
      if (tid < kRowsPerMxBlock)
        *(f32x2*)(scale_rw + (size_t)(row0 + tid) * sc_rw_stride + (col0 >> 5)) =
            *(const f32x2*)(srsb + tid * kMxGroupsPerRow);
    }
    if constexpr (IS_DBIAS) {
      if (it == iters - 1) {
        // every warp holds a partial for the SAME columns over a disjoint row
        // set; folding them in warp order keeps the whole band's contribution
        // inside one partial, which is the granularity the reference reduction
        // consumes.
        f32x2* dst = (f32x2*)(&dbp_s[warp * kTileCols + 8 * lane]);
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = dba[i];
        __syncthreads();
        if (tid < kTileCols) {
          float s = dbp_s[tid];
#pragma unroll
          for (int w = 1; w < kWarpsPerCta; ++w) s = __fadd_rn(s, dbp_s[w * kTileCols + tid]);
          float* wp = dbias_ws + (size_t)blockIdx.y * K + col0 + tid;
          ptx::st_global_f32(wp, s, ptx::create_l2_policy_evict_last());
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// One kernel; IS_DBIAS / IS_DACT / IS_ACT compile out the unused paths and pick
// the tiling, mirroring TE's quantize_mxfp8_kernel template.
// ---------------------------------------------------------------------------
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
__global__ void __launch_bounds__(Cfg<IS_DBIAS, IS_DACT, IS_ACT>::kThreadsPerCta,
                                  Cfg<IS_DBIAS, IS_DACT, IS_ACT>::kMinBlocksPerSm)
    quantize_mxfp8_kernel(const unsigned* __restrict__ input,
                          const unsigned* __restrict__ act_input,
                          unsigned char* __restrict__ out_rw, unsigned char* __restrict__ scale_rw,
                          unsigned char* __restrict__ out_cw, unsigned char* __restrict__ scale_cw,
                          float* __restrict__ dbias_ws, unsigned short* __restrict__ dbias_out,
                          int K, int sc_rw_stride, int sc_cw_stride, int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if constexpr (C::kRegisterResident)
    quantize_regtile<IS_DBIAS, IS_DACT, IS_ACT>((const uint4*)input, (const uint4*)act_input,
                                                out_rw, scale_rw, out_cw, scale_cw, dbias_ws, K,
                                                sc_rw_stride, sc_cw_stride, iters);
  else
    quantize_streaming<IS_DBIAS, IS_DACT, IS_ACT>(input, act_input, out_rw, scale_rw, out_cw,
                                                  scale_cw, dbias_ws, K, sc_rw_stride, sc_cw_stride,
                                                  iters);
}

// ---------------------------------------------------------------------------
// dbias band reduction, sequential over bands (matches the reference order)
// ---------------------------------------------------------------------------
// The reducer has exactly one thread per column -- at K = 8192 that is 8192
// threads for the whole GPU -- so it lives or dies on loads in flight per
// thread, not on occupancy: with only K/64 CTAs the SMs are never full anyway.
// Widening the register batch from 32 to 64 bands puts 4x the read stream in
// flight and turns nbands = 64 (every dbias+dGeLU shape) into a SINGLE
// dependent DRAM round trip.
constexpr int kReduceBandBatch = 64;

__global__ void __launch_bounds__(64, 4)
    reduce_dbias_kernel(const float* __restrict__ ws, unsigned short* __restrict__ dbias, int K,
                        int nbands) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= K) return;
  float acc = 0.f;
  const float* p = ws + col;
  // The band fold has to stay strictly left-to-right to reproduce the
  // reference's fp32 accumulation, but nothing forces the LOADS to be
  // serialised with it: pulling a whole group into registers first turns one
  // dependent DRAM round trip per band into one per group, which is what this
  // narrow (one thread per column) kernel needs to reach memory rate.
  // The band fold has only ONE thread per column -- at K = 32768 that is 1024
  // warps for the whole GPU, so this kernel lives or dies on how many loads a
  // thread keeps in flight.  A 64-deep register batch puts ~8 MB of reads in
  // flight instead of ~2 MB, and the adds still run strictly left-to-right so
  // the fp32 accumulation order is bit-for-bit the validated one.
  int b = 0;
  if (nbands >= kReduceBandBatch) {
    float v0[kReduceBandBatch], v1[kReduceBandBatch];
#pragma unroll
    for (int i = 0; i < kReduceBandBatch; ++i) v0[i] = p[(size_t)i * K];
    b = kReduceBandBatch;
    // Keep the next kReduceBandBatch-band load group outstanding while the previous group
    // runs through its exact left-to-right dependent add chain.
    for (; b + 2 * kReduceBandBatch <= nbands; b += 2 * kReduceBandBatch) {
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) v1[i] = p[(size_t)(b + i) * K];
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) acc = __fadd_rn(acc, v0[i]);
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) v0[i] = p[(size_t)(b + kReduceBandBatch + i) * K];
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) acc = __fadd_rn(acc, v1[i]);
    }
    if (b + kReduceBandBatch <= nbands) {
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) v1[i] = p[(size_t)(b + i) * K];
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) acc = __fadd_rn(acc, v0[i]);
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) acc = __fadd_rn(acc, v1[i]);
      b += kReduceBandBatch;
    } else {
#pragma unroll
      for (int i = 0; i < kReduceBandBatch; ++i) acc = __fadd_rn(acc, v0[i]);
    }
  }
  for (; b < nbands; ++b) acc = __fadd_rn(acc, p[(size_t)b * K]);
  unsigned short o;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(acc));
  dbias[col] = o;
}

// ---------------------------------------------------------------------------
// pick how many row blocks one CTA walks so the activation table load amortises
// Each CTA copies the whole activation table into shared once, so a grid of
// one-block CTAs pays for it 8192 times: at 8192x8192 that is 64 MB (gelu) or
// 128 MB (dgelu) of table traffic against a 272-407 MB workload.  Walking more
// row blocks per CTA divides that cost directly, and the floor below keeps
// enough blocks in flight (~4 full residency waves) that the tail does not eat
// the saving back.
// CAST_ACT's table is half the size of the dGeLU one, so its walk can afford to
// be shorter than the shared 8192-CTA target: the extra staging traffic buys a
// grid deep enough that the activation work of the trailing partial residency
// wave stops showing up as tail.
// CAST_DACT gets its own target too: its table is twice as big, so the staging
// it pays per extra CTA is twice CAST_ACT's, but its tile is a 256-thread one at
// six blocks/SM, so it already starts from a deeper grid.
// CAST_DBIAS_DACT shares the walk with the band count its workspace and reducer
// see, so its target has to be applied identically here and in mxfp8_nbands --
// and it keeps the shared 8192 target: doubling its grid measured 2.9% SLOWER
// at 16384x32768, because a deeper grid also doubles the band count, hence the
// pinned fp32 workspace round trip and the reducer's dependent add chain.
// CAST_DBIAS_DACT's walk sets its band count, hence the size of the pinned fp32
// workspace round trip AND the length of the reducer's dependent add chain.
// Halving the target doubles the walk and halves both: at 16384x32768 the band
// count drops 64 -> 32 and the workspace 8.4 MB -> 4.2 MB, worth 1.1%.  Going
// further starts costing more in residency tail than it saves in bands.
// The two-row-block floor exists to amortise the table staging; on the shortest
// grids that trade runs the other way, so it is a per-instantiation knob.
//! Grid size, in CTAs, that the walk length aims for.  Fewer, longer CTAs
//! amortise the one-off activation-table staging; more, shorter ones keep the
//! trailing residency wave from showing up as tail.  dbias+dGeLU wants the
//! shorter grid for a second reason: its walk length also sets its band count,
//! hence the size of the pinned FP32 workspace round trip and the length of the
//! reducer's dependent add chain.  At 16384x32768 halving the target drops the
//! band count 64 -> 32 and the workspace 8.4 MB -> 4.2 MB, worth 1.1%; going
//! further costs more in residency tail than it saves in bands.
constexpr long long kCtaTargetActivation = 16384;
constexpr long long kCtaTargetDbiasDact = 4096;
//! Never leave a CTA with a single row block: at 8192x8192 that is where the
//! table copy is worth 24% (GeLU) to 31% (dGeLU) of the whole workload's
//! traffic, and halving the block count halves it for free -- the grid is still
//! 4096 CTAs, the same number of residency waves as 8192.  Beyond that the table
//! cost is already noise and fewer, longer blocks measurably lose.
constexpr int kMinIters = 2;
//! dbias+dGeLU goes further, for the band-count reason above.
constexpr int kMinItersDbiasDact = 4;
//! ...but its walk is capped, or the residency tail takes the saving back.
constexpr int kMaxItersDact = 64;

static int pick_iters_t(int rowblocks, int gridx, long long target, int minit = kMinIters) {
  int it = 1;
  while ((rowblocks / it) % 2 == 0 && (long long)gridx * (rowblocks / it) > target) it *= 2;
  // Never leave a CTA with a single row block: at 8192x8192 that is where the
  // table copy is worth 24% (gelu) to 31% (dgelu) of the whole workload's
  // traffic, and halving the block count halves it for free -- the grid is
  // still 4096 blocks, the same number of residency waves as 8192.  Beyond that
  // the table cost is already noise and fewer, longer blocks measurably lose.
  if (it < minit && rowblocks % minit == 0) it = minit;
  return it;
}

// Shared/L1 carveout.  B200 splits one 228 KB unified array between shared and
// L1, and the default split hands shared 135 KB -- which the activation
// instantiations needed only because a 16 KB activation table times six
// resident CTAs nearly filled it.  With the gelu table at its natural two bytes
// per entry CAST_ACT's six blocks need 77 KB, so asking for a SMALLER shared
// partition costs no residency and buys ~44 KB of L1 back.  That matters here:
// forcing the opposite extreme (all-shared, no L1) was measured at +22% on this
// same kernel, so L1 capacity is worth real time even at a low hit rate.
//! Percentage of the unified shared/L1 array to leave to shared memory.  Only
//! the two instantiations with room to spare ask; 0 means "keep the default".
constexpr int kCarveoutPercentCast = 40;
constexpr int kCarveoutPercentAct = 40;

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
static void set_carveout() {
  constexpr int PCT = (!IS_DBIAS && !IS_DACT && !IS_ACT)
                          ? kCarveoutPercentCast
                          : ((IS_ACT && !IS_DACT && !IS_DBIAS) ? kCarveoutPercentAct : 0);
  if constexpr (PCT > 0) {
    static bool done = false;
    if (!done) {
      done = true;
      cudaFuncSetAttribute((const void*)quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT>,
                           cudaFuncAttributePreferredSharedMemoryCarveout, PCT);
    }
  }
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
static void launch(const void* prim, const void* actin, void* orw, void* srw, void* ocw, void* scw,
                   float* ws, void* dbo, int M, int K, int rws, int cws, cudaStream_t st) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  set_carveout<IS_DBIAS, IS_DACT, IS_ACT>();
  constexpr int RB = C::kRegisterResident ? 32 : 64;
  const int gx = K / kTileCols;
  const int rb = M / RB;
  constexpr long long TGT = IS_DBIAS ? kCtaTargetDbiasDact : kCtaTargetActivation;
  // Cast-only and dbias-on-the-streaming-tile are fixed single-block walks.
  int iters =
      (IS_ACT || IS_DACT || (IS_DBIAS && C::kRegisterResident)) ? pick_iters_t(rb, gx, TGT) : 1;
  // Per-instantiation walk length.  The activation table is copied once per
  // CTA, so a longer walk amortises it -- but a longer walk also spreads one
  // CTA's stores over more rows, costing streaming locality.  The two effects
  // balance at a different point for each instantiation: dgelu's 16 KB table
  // reaches break-even sooner than gelu's, and the dbias instantiation wants
  // FEWER, LONGER walks because the band count (hence the workspace round trip
  // and the reducer's dependent add chain) shrinks with it.
  if constexpr (IS_DACT && !IS_DBIAS) {
    if (iters > kMaxItersDact) iters = kMaxItersDact;
  }
  if constexpr (IS_DBIAS && C::kRegisterResident) {
    if (iters < kMinItersDbiasDact && rb % kMinItersDbiasDact == 0) iters = kMinItersDbiasDact;
  }
  dim3 grid(gx, rb / iters);
  quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT>
      <<<grid, Cfg<IS_DBIAS, IS_DACT, IS_ACT>::kThreadsPerCta, 0, st>>>(
          (const unsigned*)prim, (const unsigned*)actin, (unsigned char*)orw, (unsigned char*)srw,
          (unsigned char*)ocw, (unsigned char*)scw, ws, (unsigned short*)dbo, K, rws, cws, iters);
}
}  // anonymous namespace

// ---------------------------------------------------------------------------
// TE-facing host shell
// ---------------------------------------------------------------------------
// Replaces the campaign harness's standalone shell. Three things changed and
// nothing else:
//   1. the harness cached a cudaMalloc'd workspace in a file-static pointer;
//      TE owns the dbias workspace, so that allocator is gone and the caller
//      passes workspace_ptr through.
//   2. the harness dispatched a RUNTIME `mode` to launch<IS_DBIAS,IS_DACT,IS_ACT>;
//      TE already carries those three as template parameters, so the runtime
//      switch is gone and TE's flags feed the template directly.
//   3. gating helpers added, so anything outside the validated envelope keeps
//      using the generic TMA kernel.

// The activation LUT is built once per TRANSLATION UNIT by a tiny setup
// kernel. Per translation unit, not per process: d_gelu_tab / d_dgelu_table are
// declared in the anonymous namespace above, so every TU that includes this
// header gets its own private copy of them, and each copy needs its own
// initialization.
//
// This guard must therefore have internal linkage too. As an `inline` function
// its function-local `static` would be a single entity shared by every TU,
// while the tables it guards would not be -- so the first TU to call this would
// initialize its own tables and flip the flag for everyone, leaving every other
// TU's tables permanently zero. That is a silent wrong-answer bug: a zeroed
// GeLU table makes the table-path words of the output come out zero while the
// arithmetic-path words stay correct. (An `inline` function referencing
// internal-linkage entities is also an ODR violation in its own right.)
//
// The guard is not atomic: a concurrent first call from two threads can launch
// the build twice, which is harmless (both write identical constants, and the
// stream ordering below still applies) but must stay idempotent if the table
// contents are ever made input-dependent.
static bool& regtile_tables_ready() {
  static bool ready = false;
  return ready;
}

static void ensure_act_tables(cudaStream_t stream) {
  if (!regtile_tables_ready()) {
    init_dgelu_table_kernel<<<(2 * kLutEntriesPerSign + 255) / 256, 256, 0, stream>>>();
    NVTE_CHECK_CUDA(cudaGetLastError());
    regtile_tables_ready() = true;
  }
}

// Rows folded into one dbias workspace band. Must agree exactly with the
// grid the kernel is launched on, since it sizes the workspace.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
inline int regtile_dbias_bands(int M, int K) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if (!C::kRegisterResident) return M / 64;  // streaming tile: one 64-row band per CTA
  const int rb = M / kRowsPerMxBlock;
  int it = pick_iters_t(rb, K / kTileCols, kCtaTargetDbiasDact);
  if (it < kMinItersDbiasDact && rb % kMinItersDbiasDact == 0) it = kMinItersDbiasDact;
  return rb / it;
}

// Activation gate. The kernel's activation path is a gelu/dgelu-specific
// lookup table, so it is only valid for those two ops -- qgelu/dqgelu/silu and
// friends must fall through to the generic kernel.
//
// This identifies the op by TEMPLATE MATCHING, not by comparing OP against
// &gelu<fp32, fp32>. The comparison would be the obvious spelling but it forms
// the address of a __device__ function in host code, which is not portable
// under nvcc. Naming the op as a template argument here is exactly how TE's own
// call sites already spell it (see activation/gelu_dbias.cu), so it stays on
// well-trodden ground.
enum class RegtileOp { kUnsupported, kGelu, kDgelu };

template <typename ParamOP, float (*OP)(float, const ParamOP&)>
struct RegtileOpKind {
  static constexpr RegtileOp value = RegtileOp::kUnsupported;
};

template <>
struct RegtileOpKind<Empty, gelu<fp32, fp32>> {
  static constexpr RegtileOp value = RegtileOp::kGelu;
};

template <>
struct RegtileOpKind<Empty, dgelu<fp32, fp32>> {
  static constexpr RegtileOp value = RegtileOp::kDgelu;
};

template <bool IS_ACT, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP&)>
struct RegtileOpSupported {
  static constexpr RegtileOp kKind = RegtileOpKind<ParamOP, OP>::value;
  static constexpr bool value = (!IS_ACT && !IS_DACT) ||
                                (IS_ACT && !IS_DACT && kKind == RegtileOp::kGelu) ||
                                (IS_DACT && !IS_ACT && kKind == RegtileOp::kDgelu);
};

// Shape gate. The kernel derives its grid by exact integer division, so a
// remainder in either dimension would silently drop the tail.
//   cols % kTileCols : grid.x  = K / kTileCols
//   rows % 64    : covers both tile variants (regtile 32, streaming 64)
// pick_iters_t only ever returns a power of two that divides rows/tile, so
// grid.y = rb / iters needs no separate check.
inline bool regtile_shape_supported(size_t rows, size_t cols) {
  return (cols % kTileCols == 0) && (rows % 64 == 0) && rows > 0 && cols > 0;
}

// Host entry. Mirrors the campaign harness's launch<>() plus its optional
// separate dbias reduction pass.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
void launch_regtile(const void* input, const void* act_input, void* out_rowwise,
                    void* scale_rowwise, void* out_colwise, void* scale_colwise,
                    float* workspace_ptr, void* dbias_ptr, int M, int K, int scale_stride_rowwise,
                    int scale_stride_colwise, cudaStream_t stream) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if constexpr (IS_ACT || IS_DACT) {
    ensure_act_tables(stream);
  }

  // The register-resident tile quantizes out of the ACT_INPUT slot, so
  // CAST_DBIAS must hand it the primary (grad) pointer there; x is unused by
  // that mode. Matches the campaign harness's mode-1 argument wiring.
  const void* act_slot = (IS_DBIAS && !IS_DACT && C::kRegisterResident) ? input : act_input;

  launch<IS_DBIAS, IS_DACT, IS_ACT>(input, act_slot, out_rowwise, scale_rowwise, out_colwise,
                                    scale_colwise, workspace_ptr, dbias_ptr, M, K,
                                    scale_stride_rowwise, scale_stride_colwise, stream);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if constexpr (IS_DBIAS) {
    const int nbands = regtile_dbias_bands<IS_DBIAS, IS_DACT, IS_ACT>(M, K);
    constexpr int threads = 64;  // narrow CTAs so the column fan-out reaches every SM
    reduce_dbias_kernel<<<(K + threads - 1) / threads, threads, 0, stream>>>(
        workspace_ptr, reinterpret_cast<unsigned short*>(dbias_ptr), K, nbands);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
