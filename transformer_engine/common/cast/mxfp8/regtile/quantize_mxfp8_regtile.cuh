/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_regtile.cuh
 *  \brief Register-resident MXFP8 bidimensional fused quantize kernel.
 *
 *  Kernel body below is the winning candidate of Kernel Factory campaign
 *  rv390dmap97kd7jaxfef2kjcmw (mxfp8-quantize-b200-bidim-fused-te-unified),
 *  candidate 12920c25eba5de8346062b46c75d5e041fdf2fc2d2a7e97f43b98be44ad59ba5,
 *  vendored VERBATIM from the campaign export so it can be diffed against it.
 *  Only the host-side shell (below the kernel body) is TE-specific.
 *
 *  Host helpers were renamed on the way in, so comments inside the vendored
 *  body still use the campaign names: mxfp8_nbands -> regtile_dbias_bands,
 *  mxfp8_fused_reduce -> regtile_fuses_reduction, mxfp8_launch -> launch_regtile.
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

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace regtile {
namespace {

#define DEVI __device__ __forceinline__

#ifndef STRIDED_WALK
#define STRIDED_WALK 1
#endif



#ifndef GELU_FAST
#define GELU_FAST 1
#endif

// ---------------------------------------------------------------------------
// low level helpers
// ---------------------------------------------------------------------------

// max(|a|,|b|) elementwise on packed bf16x2; sign of result is sign(a)^sign(b)
// (only magnitudes are accumulated, the sign is masked off at the end).
DEVI unsigned amax2(unsigned a, unsigned b) {
  unsigned d;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

DEVI unsigned max_bf16x2(unsigned a, unsigned b) {
  unsigned d;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVI unsigned min_bf16x2(unsigned a, unsigned b) {
  unsigned d;
  asm("min.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVI unsigned clamp_abs_bf16x2(unsigned a, unsigned limit) {
  unsigned d;
  asm("min.xorsign.abs.bf16x2 %0, %1, %2;"
      : "=r"(d) : "r"(a), "r"(limit));
  return d;
}

DEVI unsigned pack_bf16x2(float hi, float lo) {
  unsigned d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

DEVI unsigned cvt_e4m3x2(float hi, float lo) {
  unsigned short d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return (unsigned)d;
}

DEVI unsigned mul_bf16x2(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// merge the low halves of two e4m3x2 results into one 4-byte word
DEVI unsigned prmt2(unsigned a, unsigned b) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

DEVI unsigned cvt_e4m3x2_bf16x2(unsigned a) {
  unsigned short d;
  asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(a));
  return (unsigned)d;
}

// scale two bf16x2 column pairs and pack the four E4M3 bytes into one word.
// Keeping the two .b16 conversion results inside one asm block lets ptxas pair
// them straight into `mov.b32 {x,y}` with no zero-extension of the half
// registers, which is what the separate cvt + prmt form pays for.
// Widen the quantized store with the tile: an eight-column lane emits STG.64,
// a sixteen-column lane emits STG.128 for the same bytes.
template <int W>
DEVI void st_out(unsigned char* p, const unsigned* v) {
  if constexpr (W == 8)
    *(uint4*)p = *(const uint4*)v;
  else
    *(uint2*)p = *(const uint2*)v;
}

DEVI unsigned quant4(unsigned a, unsigned b, unsigned sa, unsigned sb) {
  unsigned d;
  asm("{\n\t.reg .b32 ta, tb;\n\t.reg .b16 x, y;\n\t"
      "mul.rn.bf16x2 ta, %1, %3;\n\t"
      "mul.rn.bf16x2 tb, %2, %4;\n\t"
      "cvt.rn.satfinite.e4m3x2.bf16x2 x, ta;\n\t"
      "cvt.rn.satfinite.e4m3x2.bf16x2 y, tb;\n\t"
      "mov.b32 %0, {x, y};\n\t}"
      : "=r"(d)
      : "r"(a), "r"(b), "r"(sa), "r"(sb));
  return d;
}

// Native TE-exact E8M0 encode: e = clamp(ceil(log2(x)) + 127, 0, 254) is
// precisely round-up-to-UE8M0 with saturation, so one instruction covers a
// column pair (a -> upper byte, b -> lower byte).
DEVI unsigned cvt_ue8m0x2(float hi, float lo) {
  unsigned short d;
  asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return (unsigned)d;
}

// bf16 bit pattern of 2^(127-e) duplicated into both halves.  (254-e) <= 254 so
// (254-e)<<7 < 2^15 and the multiply by 0x00800080 cannot carry across halves,
// making the whole duplicate-and-shift a single IMAD.
DEVI unsigned enc_scale_bf16x2(unsigned e) { return (254u - e) * 0x00800080u; }

DEVI float bf16_lo(unsigned v) { return __uint_as_float(v << 16); }
DEVI float bf16_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

// TE / reference E8M0: e = clamp(ceil(log2(amax/448)) + 127, 0, 254), 0 if amax==0
DEVI unsigned e8m0_from_amax(float amax) {
  float s = amax * (1.0f / 448.0f);
  // ceil(log2(s)) + 127 == (bits + 0x7fffff) >> 23 for every finite s >= 0
  unsigned e = (__float_as_uint(s) + 0x7fffffu) >> 23;
  return e > 254u ? 254u : e;
}

// encode scale = 2^(127 - e) as fp32
DEVI float enc_scale(unsigned e) { return __uint_as_float((254u - e) << 23); }

// ---------------------------------------------------------------------------
// activations: TE util/math.h op order, non-contracted so the fp32 rounding
// sequence matches the reference exactly.
// ---------------------------------------------------------------------------
DEVI float act_gelu(float v) {
  float t2 = __fmul_rn(__fmul_rn(0.03567741f, v), v);
  float t4 = __fmul_rn(v, __fadd_rn(0.79788456f, t2));
  float t = tanhf(t4);
  return __fmul_rn(v, __fadd_rn(0.5f, __fmul_rn(0.5f, t)));
}

DEVI float act_dgelu(float v) {
  float a3 = __fadd_rn(1.0f, __fmul_rn(__fmul_rn(0.044715f, v), v));
  float a5 = __fmul_rn(__fmul_rn(0.79788456f, v), a3);
  float t = tanhf(a5);
  float c2 = __fsub_rn(1.0f, __fmul_rn(t, t));
  float d3 = __fadd_rn(0.79788456f, __fmul_rn(__fmul_rn(0.1070322243f, v), v));
  float f = __fmul_rn(__fmul_rn(0.5f, v), __fmul_rn(c2, d3));
  float h = __fmul_rn(0.5f, __fadd_rn(1.0f, t));
  return __fadd_rn(f, h);
}


// --------------------------------------------------------------------------
// Blackwell packed fp32x2 arithmetic: one instruction computes both lanes of a
// column pair with exactly the scalar .rn rounding, halving the FP32 op count
// of the activation chain.
// --------------------------------------------------------------------------
typedef unsigned long long u64;
DEVI u64 l2_evict_last_policy() {
  u64 p;
  asm("createpolicy.fractional.L2::evict_last.b64 %0;" : "=l"(p));
  return p;
}
DEVI void store_l2_evict_last(float* p, float v, u64 policy) {
  asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;"
               :: "l"(p), "f"(v), "l"(policy) : "memory");
}
DEVI void store_l2_evict_last(u64* p, u64 v, u64 policy) {
  asm volatile("st.global.L2::cache_hint.b64 [%0], %1, %2;"
               :: "l"(p), "l"(v), "l"(policy) : "memory");
}
DEVI u64 mk2(float lo, float hi) {
  u64 d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
  return d;
}
DEVI void un2(u64 a, float& lo, float& hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(a));
}
DEVI u64 splat2(float c) { return mk2(c, c); }
DEVI u64 fmul2(u64 a, u64 b) {
  u64 d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d;
}
DEVI u64 fadd2(u64 a, u64 b) {
  u64 d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d;
}
DEVI u64 fsub2(u64 a, u64 b) {
  u64 d; asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d;
}
DEVI u64 ffma2(u64 a, u64 b, u64 c) {
  u64 d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d;
}
DEVI u64 bf16x2_to_f32x2(unsigned a) {
  return mk2(__uint_as_float(a << 16), __uint_as_float(a & 0xffff0000u));
}
#define UAF(x) __uint_as_float(x##u)

// Instruction-for-instruction replica of CUDA libdevice tanhf, evaluated on a
// packed pair (the two transcendental steps stay scalar; everything else is
// packed). Bit-identical to tanhf() for each lane.
DEVI float slct(float a, float b, float c) {
  float d;
  asm("slct.f32.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
// The exp-branch magnitude is always non-negative, so copysign collapses to a
// single LOP3: (b & sign_mask) | a, immediate folded into the LOP3 operand.
DEVI float orsign(float a, float b) {
  unsigned d;
  asm("lop3.b32 %0, %1, %2, 0x80000000, 0xec;"
      : "=r"(d)
      : "r"(__float_as_uint(b)), "r"(__float_as_uint(a)));
  return __uint_as_float(d);
}

DEVI u64 tanh2(u64 u) {
  float ua, ub;
  un2(u, ua, ub);
  // |x| folds into the FMUL operand modifier; the two MUFU steps stay scalar.
  float e0, e1;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(__fmul_rn(fabsf(ua), UAF(0x4038AA3B))));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(__fmul_rn(fabsf(ub), UAF(0x4038AA3B))));
  float f0, f1;
  un2(fadd2(mk2(e0, e1), splat2(1.0f)), f0, f1);
  float r0, r1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(f0));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(f1));
  // libdevice clamps |x| >= 9.010914 to exactly 1.0; there 2*rcp(ex2(2ln2|x|))
  // is already below half an ulp of 1.0f, so fma(r,-2,1) rounds to 1.0f on its
  // own and the clamp select is dropped.
  float g0, g1;
  un2(ffma2(mk2(r0, r1), splat2(-2.0f), splat2(1.0f)), g0, g1);
  float xa = orsign(g0, ua);
  float xb = orsign(g1, ub);
  // small-|x| polynomial branch is pure FP32 arithmetic -> fully packed
  u64 s = fmul2(u, u);
  u64 p = ffma2(s, splat2(UAF(0x3C80F082)), splat2(UAF(0xBD563CAE)));
  p = ffma2(p, s, splat2(UAF(0x3E085941)));
  p = ffma2(p, s, splat2(UAF(0xBEAAA9ED)));
  p = fmul2(p, s);
  p = ffma2(p, u, u);
  // branch predicate |u| >= 0.6f re-expressed on the already-computed square
  // (0.6f*0.6f rounds exactly to 0.36f): compares straight against the
  // immediate, so no packed subtract and no packed abs are needed.
  float pa, pb, sa, sb;
  un2(p, pa, pb);
  un2(s, sa, sb);
  return mk2(sa >= UAF(0x3EB851EC) ? xa : pa, sb >= UAF(0x3EB851EC) ? xb : pb);
}

// ---------------------------------------------------------------------------
// Cheap exact-enough packed GeLU.
//
//   0.5 + 0.5*tanh(u) == sigmoid(2u) == 1 / (1 + exp(-2u))
//
// so gelu(v) = v / (1 + ex2(-2*log2e*u)).  Unlike libdevice's tanh (which forms
// 1 - 2*rcp(1+E) and therefore cancels ~1.5 bits, forcing a separate polynomial
// branch below |u| = 0.6), this form has NO cancellation anywhere: the relative
// error is just ex2.approx's 2 ulp folded through E/(1+E) < 1, i.e. ~2^-22,
// against a bf16 half-ulp of 2^-9.  Thirteen instructions per bf16x2 instead of
// ~thirty, and no MUFU/FP32 chain longer than the table probe it replaces.
DEVI u64 ex2_2(u64 a) {
  float x, y, e0, e1;
  un2(a, x, y);
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(x));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(y));
  return mk2(e0, e1);
}
DEVI u64 rcp_2(u64 a) {
  float x, y, r0, r1;
  un2(a, x, y);
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(y));
  return mk2(r0, r1);
}
// -2*log2(e) folded into both polynomial coefficients so the whole exponent
// argument is one FMUL2 + one FFMA2 + one FMUL2.
#define GELU_K1 (-0.10294324495752462f)
#define GELU_K0 (-2.302208195827745f)
DEVI u64 act_gelu2_fast(u64 v) {
  // arg = -2*log2e * v * (0.79788456 + 0.03567741 v^2)
  const u64 p = ffma2(fmul2(splat2(GELU_K1), v), v, splat2(GELU_K0));
  const u64 D = fadd2(splat2(1.0f), ex2_2(fmul2(v, p)));
  // No Newton refinement and no clamp: rcp.approx is already 1 ulp, and leaving
  // the raw reciprocal in place is what makes the v -> -inf tail land on a
  // signed zero (rcp(inf) = 0, v*0 = -0) instead of inf*0 = NaN.
  return fmul2(v, rcp_2(D));
}

DEVI u64 act_gelu2_exact(u64 v) {
  u64 t2 = fmul2(fmul2(splat2(0.03567741f), v), v);
  u64 u = fmul2(v, fadd2(splat2(0.79788456f), t2));
  u64 t = tanh2(u);
  // 0.5f*t is exact, so fma(0.5,t,0.5) == fadd(0.5, fmul(0.5,t)) bit-for-bit.
  return fmul2(v, ffma2(splat2(0.5f), t, splat2(0.5f)));
}

// ---------------------------------------------------------------------------
// Cheap exact-enough packed dGeLU, the dgelu analogue of act_gelu2_fast.
//
// With s = sigmoid(2*a5) = 0.5*(1+t) the TE form rearranges EXACTLY:
//     h  = 0.5*(1+t)          = s
//     c2 = 1 - t^2            = 4*s*(1-s)
//     dgelu = s + 2*v*d3*s*(1-s)
// and 2*d3 folds into the polynomial coefficients.  Writing the sigmoid on the
// magnitude of the exponent argument -- Em = ex2(-|arg|) in (0,1], q = rcp(1+Em)
// in [0.5,1) -- keeps BOTH s and (1-s) as products of accurately-computed
// quantities: s*(1-s) = Em*q^2 with no subtraction anywhere, and s itself is
// just q (arg<0) or Em*q (arg>=0), one FSEL.  That matters: the naive
// s = rcp(1+ex2(arg)) followed by (1-s) loses all of (1-s)'s significance when
// s -> 1, and ex2(arg) overflowing to +inf would turn inf*0 into a NaN, whereas
// ex2(-|arg|) can only underflow to zero, which is the right answer there.
// The only remaining subtraction is s + t1*w, which is the reference's own
// h + f split, so no cancellation is introduced relative to TE.
// 9 packed ops + 4 MUFU + 2 FSEL against ~35 for the exact tanh replica.
#define DGELU_C1 (0.2140644486f)    // 2 * 0.1070322243
#define DGELU_C0 (1.59576912f)      // 2 * 0.79788456
DEVI float slct_ge(float a, float b, float c) {
  float d;
  asm("slct.f32.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
// ex2(-|x|) on a packed pair: the -|x| is one LOP3 (set the sign bit) that
// ptxas usually folds straight into the MUFU operand modifier.
DEVI u64 ex2_nabs2(u64 a) {
  float x, y, e0, e1;
  un2(a, x, y);
  asm("ex2.approx.ftz.f32 %0, %1;"
      : "=f"(e0)
      : "f"(__uint_as_float(__float_as_uint(x) | 0x80000000u)));
  asm("ex2.approx.ftz.f32 %0, %1;"
      : "=f"(e1)
      : "f"(__uint_as_float(__float_as_uint(y) | 0x80000000u)));
  return mk2(e0, e1);
}
// Clamping the bf16 argument to [-8, 8] before the fp32 chain costs two
// bf16x2 ops and removes the whole tail problem: dgelu is EXACTLY 1.0f (v>0)
// / 0.0f (v<0) from |v| >= 5.5 up, so the clamp is value-preserving, and it
// keeps the ex2 argument under 72 so E can never reach +inf.  With E finite,
// (1 - s) is just E*s -- a product of two accurately-computed quantities, no
// subtraction and no cancellation -- so s*(1-s) = (E*s)*s needs no select and
// no scalar/packed register shuffling.
DEVI u64 act_dgelu2_fast(unsigned avw) {
  const unsigned cl = clamp_abs_bf16x2(avw, 0x41004100u);
  const u64 v = bf16x2_to_f32x2(cl);
  const u64 v2 = fmul2(v, v);
  const u64 E = ex2_2(fmul2(v, ffma2(splat2(GELU_K1), v2, splat2(GELU_K0))));
  const u64 s = rcp_2(fadd2(splat2(1.0f), E));
  const u64 w = fmul2(fmul2(E, s), s);
  const u64 t1 = fmul2(v, ffma2(splat2(DGELU_C1), v2, splat2(DGELU_C0)));
  return ffma2(t1, w, s);
}

DEVI u64 act_dgelu2_exact(u64 v) {
  u64 a3 = fadd2(splat2(1.0f), fmul2(fmul2(splat2(0.044715f), v), v));
  u64 a5 = fmul2(fmul2(splat2(0.79788456f), v), a3);
  u64 t = tanh2(a5);
  u64 c2 = fsub2(splat2(1.0f), fmul2(t, t));
  u64 d3 = fadd2(splat2(0.79788456f), fmul2(fmul2(splat2(0.1070322243f), v), v));
  u64 f = fmul2(fmul2(splat2(0.5f), v), fmul2(c2, d3));
  // 0.5f*(1+t) and fma(0.5,t,0.5) round on the same grid (halving is exact).
  u64 h = ffma2(splat2(0.5f), t, splat2(0.5f));
  return fadd2(f, h);
}

#ifndef DGELU_FAST
#define DGELU_FAST 1
#endif
DEVI u64 act_dgelu2_w(unsigned avw) {
#if DGELU_FAST
  return act_dgelu2_fast(avw);
#else
  return act_dgelu2_exact(bf16x2_to_f32x2(avw));
#endif
}


// ---------------------------------------------------------------------------
// Activation lookup tables.
//
// The activation input is BF16, so gelu/dgelu are functions of a 16-bit key and
// can be tabulated EXACTLY -- the table stores the very same fp32 value the
// arithmetic path above produces, so this is a pure instruction-count
// optimisation with zero numerical consequence (the E8M0 scales stay
// bit-exact).
//
// A full 2^16 table would be 256 KB.  Range reduction shrinks it to 16 KB:
//   * |v| >= 5.5      -> dgelu is exactly 1.0f (v>0) / 0.0f (v<0) and
//                        gelu(v) is exactly v / -0.0f, so every magnitude at or
//                        above bf16 8.0 collapses onto one entry;
//   * |v| <  2^-12    -> rare enough (1.2% of warps for unit-scale data) that
//                        the arithmetic path handles it.
// Both tails are detected by a single "did the clamp move the value?" compare,
// and the arithmetic fallback is exact for the whole pair, so correctness never
// depends on the range analysis being tight.
// ---------------------------------------------------------------------------
#define LUT_LO 0x3980u          // bf16 bits of 2^-12
#define LUT_HI 0x4100u          // bf16 bits of 8.0
#define LUT_N 2048              // entries per sign (0x780+1 = 1921 used)
// ---------------------------------------------------------------------------
// Bank-partitioned replication.
//
// The probe is a 32-lane random gather, whose bank distribution is uniform no
// matter how the table is permuted: 32 balls into 32 bins has an expected max
// load of ~3.4, and NCU confirms 3.45 wavefronts per LDS request -- 51% of every
// shared wavefront the kernel issues is a replay.  Permutation cannot help, but
// PARTITIONING can: give each group of 32/NCOPY lanes its own private copy of
// the table laid out so that copy g only ever occupies banks
// [32/NCOPY*g, 32/NCOPY*(g+1)).  Each group then throws (32/NCOPY) balls into
// (32/NCOPY) bins, and the whole warp's wavefront count is the max over the
// groups: ~2.4 for four groups of eight instead of 3.45.
//
// The layout that achieves it: entry i of copy g lives at byte
//     8*i - SWZ*(i & 15) + GSP*g
// Writing i = 16q + r, the address is 128*q + (8-SWZ)*r + GSP*g, i.e. a
// 128-byte row per 16 entries whose g-th slice is a contiguous chunk, so the
// bank is (r*(8-SWZ)/4 + GSP*g/4) mod 32 -- independent of q, and confined to
// the group's own range.  gelu stores a bf16 (2 B, 4 copies), dgelu an fp32
// (4 B, 2 copies); both land on 31 KB.
// ---------------------------------------------------------------------------
#ifndef SWIZLUT
#define SWIZLUT 0
#endif
#if SWIZLUT
// gelu tabulates a bf16, so two bytes per entry are enough and TWO private
// copies of the whole 4096-slot table still fit in the SAME 16 KB the single
// 4-byte-entry table used -- occupancy is untouched (a four-copy/eight-bank
// version needs 32 KB, which costs half the resident blocks and measured 21%
// SLOWER end to end).  Copy g owns banks [16g, 16g+16): entry i sits at byte
//     4*i - 2*(i & 31) + 64*g
// i.e. a 128-byte row per 32 entries, split into two 64-byte group slices.
#define LUT_SPAN_H 15952
#define LUT_SPAN_F (2 * LUT_N * 4)
#define HSHF 2
#define HMSK 0x001F001Fu
#define HMUL 2
#define HGSP 64
#else
// Reachable entries end at byte 15875; round up to bulk-copy granularity.
#define LUT_SPAN_F 15888
// gelu tabulates a BF16, so an entry is two bytes.  The 32-bit-per-entry form
// this replaces wasted half of every cache line the bulk copy moved: each CTA
// copies the whole table once, and at the small square -- where pick_iters
// bottoms out at a two-block walk -- that one-off is ~24% of the workload's
// entire L2 traffic.  Reachable entries now end at byte 7937.
#define LUT_SPAN_H 7952
#endif
#define LUT_LO2 0x39803980u
#define LUT_HI2 0x41004100u
#define LUT_BIAS2 0x39803980u

// Every CTA stages the whole table into shared once.  At 8192x8192 -- where the
// walk bottoms out at two row blocks -- that is 4096 CTAs all pulling the SAME
// 8 KB (gelu) / 16 KB (dgelu) line range, so the copy is not limited by L2
// bandwidth but by the handful of L2 sets those lines hash into.  Holding
// LUT_REP identical copies and letting CTA blockIdx.x take copy
// blockIdx.x % LUT_REP spreads the identical traffic across LUT_REP times as
// many sets at a one-off 0.7 MB of device memory and no change to the probe.
#ifndef LUT_REP
#define LUT_REP 1
#endif
__device__ __align__(16) unsigned char d_dgelu_tab[LUT_REP][LUT_SPAN_F];
__device__ __align__(16) unsigned char d_gelu_tab[LUT_REP][LUT_SPAN_H];


// clamp both halves of a bf16x2 magnitude into the tabulated window and turn
// them into two table indices (sign selects the 2048-entry half).  Five ALU ops
// for a whole column pair.
// Both halves at once, as BYTE offsets into the table: the sign bit is worth
// exactly 0x8000 bytes there, so it drops straight in with no shift, and every
// half stays below 0x10000 so the packed add never carries across the pair.
// `bad` accumulates every bit the clamp had to move, so a whole row's worth of
// table probes can share ONE out-of-window test instead of branching per word.
template <int SH>
DEVI unsigned lut_offs2(unsigned b, unsigned& bad) {
  const unsigned mag = b & 0x7fff7fffu;
  const unsigned c = min_bf16x2(max_bf16x2(mag, LUT_LO2), LUT_HI2);
  bad |= c ^ mag;
  // sign bit -> +LUT_N entries in its own half; no cross-half carry because
  // every clamped magnitude is >= LUT_LO and the biased index stays < 0x1000.
  // For the four-byte dGeLU entry, subtract-before-shift plus a packed sign
  // shift maps to a SHF+LEA-friendly sequence and avoids the mask/add chain.
  if constexpr (SH == 2)
    return ((c - LUT_BIAS2) << 2) + ((b ^ mag) >> 2);
  return (c + ((b >> 4) & 0x08000800u) - LUT_BIAS2) << SH;
}
// gelu's two-byte entry lets the whole bias drop out of the datapath: the same
// -2*LUT_LO applies to BOTH halves, so folding it into the table base pointer
// leaves the offset as one shift plus one LEA.  The clamped magnitude is at
// most LUT_HI = 0x4100, so (c << 1) + 0x1000 stays below 0x10000 in either
// half and the packed add still cannot carry across the pair.
DEVI unsigned lut_offs2h(unsigned b, unsigned& bad) {
  const unsigned mag = b & 0x7fff7fffu;
  const unsigned c = min_bf16x2(max_bf16x2(mag, LUT_LO2), LUT_HI2);
  bad |= c ^ mag;
  return (c << 1) + ((b >> 3) & 0x10001000u);
}
#define LUT_H_BIAS (2 * LUT_LO)
// Same, but emitting the bank-partitioned byte offset of BOTH halves at once.
// The index per half is < 0x1000, so 8*idx < 0x8000 and the packed shift, the
// packed nibble multiply (<= 15*6 = 90 per half) and the packed subtract all
// stay inside their own 16 bits.
template <int SHFT, unsigned MSK, int MUL>
DEVI unsigned lut_offs2s(unsigned b, unsigned gsp, unsigned& bad) {
  const unsigned mag = b & 0x7fff7fffu;
  const unsigned c = min_bf16x2(max_bf16x2(mag, LUT_LO2), LUT_HI2);
  bad |= c ^ mag;
  const unsigned t = c + ((b >> 4) & 0x08000800u) - LUT_BIAS2;
  return ((t << SHFT) - (t & MSK) * MUL) + gsp;
}
#if SWIZLUT
#define LUT_OFF_F(b, gsp, bad) lut_offs2<2>((b), (bad))
#define LUT_OFF_H(b, gsp, bad) lut_offs2s<HSHF, HMSK, HMUL>((b), (gsp), (bad))
#define GSPLAT_F(lane) 0u
#define GSPLAT_H(lane) (((lane) >> 4) * (HGSP * 0x00010001u))
#else
#define LUT_OFF_F(b, gsp, bad) lut_offs2<2>((b), (bad))
#define LUT_OFF_H(b, gsp, bad) lut_offs2h((b), (bad))
#define GSPLAT_F(lane) 0u
#define GSPLAT_H(lane) 0u
#define LUT_H_SHIFT LUT_H_BIAS
#endif
#ifndef LUT_H_SHIFT
#define LUT_H_SHIFT 0
#endif

DEVI u64 lut_dgelu2(unsigned b, const unsigned char* __restrict__ tab,
                    unsigned gsp, bool& oor) {
  unsigned bad = 0u;
  const unsigned d = LUT_OFF_F(b, gsp, bad);
  oor = bad != 0u;
  return mk2(*(const float*)(tab + (d & 0xffffu)),
             *(const float*)(tab + (d >> 16)));
}

// The table entry is a whole 32-bit word even though only its low half carries
// the bf16 result: CAST_DACT's fp32 table costs half as much per probe as this
// one used to at an identical probe count, and the only structural difference
// was the sub-word load, so the gelu table pays the 8 KB to use LDS.32 too.
DEVI unsigned lut_gelu2(unsigned b, const unsigned char* __restrict__ tab,
                        unsigned gsp, unsigned& bad) {
  const unsigned d = LUT_OFF_H(b, gsp, bad);
  // the folded -2*LUT_LO lands in the LDS instruction's immediate offset
  const unsigned char* __restrict__ t0 = tab - LUT_H_SHIFT;
  const unsigned lo = *(const unsigned short*)(t0 + (d & 0xffffu));
  const unsigned hi = *(const unsigned short*)(t0 + (d >> 16));
  // both loads are zero-extended halves, so one PRMT rebuilds the bf16x2
  return __byte_perm(lo, hi, 0x5410);
}

// ---- clamped tails -------------------------------------------------------
// Above the window the clamped entry is already the exact answer for dgelu
// (tanh has saturated, so dgelu is exactly 1.0f / 0.0f from |v| >= 5.5 up);
// gelu instead returns v itself there.  Below the window (|v| < 2^-12) the
// cubic term of both series is under 2^-36 relative, so the tanh call in the
// body collapses to its argument and these three-operation forms reproduce the
// body's rounding sequence term for term.  Keeping the tails this cheap is what
// keeps the whole tanh chain out of the kernel's register budget.
DEVI float gelu_tiny(float v) {
  const float t = __fmul_rn(0.79788456f, v);
  return __fmul_rn(v, __fmaf_rn(0.5f, t, 0.5f));
}
DEVI float dgelu_tiny(float v) {
  const float t = __fmul_rn(0.79788456f, v);
  const float f = __fmul_rn(__fmul_rn(0.5f, v), 0.79788456f);
  return __fadd_rn(f, __fmaf_rn(0.5f, t, 0.5f));
}

// Straight-line (no branch) repair of one clamped half.  The three cases are
// resolved with selects rather than control flow: even though this code only
// runs when a lane actually left the window, every `if` here used to cost the
// whole warp a BSSY/BSYNC reconvergence pair on the hot path, and CAST_ACT was
// spending 27% of its static instructions on that scaffolding.
DEVI unsigned gelu_tail_half(unsigned bits, unsigned tabv) {
  const unsigned mag = bits & 0x7fffu;
  unsigned short o;
  const float g = gelu_tiny(__uint_as_float(bits << 16));
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(g));
  const unsigned small = (mag < LUT_LO) ? (unsigned)o : tabv;
  const unsigned large = (bits & 0x8000u) ? 0x8000u : bits;
  return (mag >= LUT_HI) ? large : small;
}
DEVI unsigned gelu_fix2(unsigned av, unsigned p) {
  return gelu_tail_half(av & 0xffffu, p & 0xffffu) |
         (gelu_tail_half(av >> 16, p >> 16) << 16);
}

// ---- one shared activation body; USE_LUT picks table vs arithmetic ---------
// The table route returns its raw probe and defers the tail repair to one
// per-row test; the arithmetic route is exact everywhere and leaves `bad` alone.
template <bool USE_LUT>
DEVI unsigned do_gelu2(unsigned av, const unsigned char* __restrict__ htab,
                       unsigned gsp, unsigned& bad) {
  if constexpr (USE_LUT) return lut_gelu2(av, htab, gsp, bad);
#if GELU_FAST
  const u64 r = act_gelu2_fast(bf16x2_to_f32x2(av));
#else
  const u64 r = act_gelu2_exact(bf16x2_to_f32x2(av));
#endif
  float v0, v1;
  un2(r, v0, v1);
  return pack_bf16x2(v1, v0);
}

// Raw table probe with NO tail repair: the clamp residue is accumulated into
// `bad` so a whole CH-row group's probes can share ONE out-of-window test,
// exactly the way the gelu side already does it.  The dgelu route used to pay
// three branches PER WORD (the oor test plus one per half), and at 3.5 table
// words per row that is ~10 BSSY/BSYNC reconvergence pairs on a hot path that
// only actually needs them for ~1% of warps.
DEVI u64 lut_dgelu2_raw(unsigned b, const unsigned char* __restrict__ tab,
                        unsigned gsp, unsigned& bad) {
  const unsigned d = LUT_OFF_F(b, gsp, bad);
  return mk2(*(const float*)(tab + (d & 0xffffu)),
             *(const float*)(tab + (d >> 16)));
}
template <bool USE_LUT>
DEVI u64 do_dgelu2_nb(unsigned av, const unsigned char* __restrict__ ftab,
                      unsigned gsp, unsigned& bad) {
  if constexpr (USE_LUT) return lut_dgelu2_raw(av, ftab, gsp, bad);
  return act_dgelu2_w(av);
}
// Repair of a clamped dgelu probe.  Above the window the clamped entry is
// already exact (tanh has saturated), so only the |v| < 2^-12 half needs the
// closed form.  Rare path -- branches here are free.
DEVI u64 dgelu_fix2(unsigned av, u64 probe) {
  float d0, d1;
  un2(probe, d0, d1);
  const unsigned m = av & 0x7fff7fffu;
  if ((m & 0xffffu) < LUT_LO) d0 = dgelu_tiny(__uint_as_float(av << 16));
  if ((m >> 16) < LUT_LO) d1 = dgelu_tiny(__uint_as_float(av & 0xffff0000u));
  return mk2(d0, d1);
}

template <bool USE_LUT>
DEVI u64 do_dgelu2(unsigned av, const unsigned char* __restrict__ ftab,
                   unsigned gsp) {
  if constexpr (USE_LUT) {
    bool oor;
    const u64 d = lut_dgelu2(av, ftab, gsp, oor);
    if (__builtin_expect(!oor, 1)) return d;
    float d0, d1;
    un2(d, d0, d1);
    const unsigned m = av & 0x7fff7fffu;
    if ((m & 0xffffu) < LUT_LO) d0 = dgelu_tiny(__uint_as_float(av << 16));
    if ((m >> 16) < LUT_LO) d1 = dgelu_tiny(__uint_as_float(av & 0xffff0000u));
    return mk2(d0, d1);
  }
  return act_dgelu2_w(av);
}

// The LUT produces two scalar f32 registers.  Keep them scalar through the
// gradient multiply instead of packing LUT values and converted bf16 values
// into two temporary f32x2 operands only to unpack the product immediately.
// The arithmetic route is already naturally packed and keeps its f32x2 path.
template <bool USE_LUT>
DEVI u64 mul_dgelu_grad2(unsigned av, unsigned gv,
                        const unsigned char* __restrict__ ftab,
                        unsigned gsp) {
  if constexpr (USE_LUT) {
    unsigned bad = 0u;
    const unsigned d = LUT_OFF_F(av, gsp, bad);
    float d0 = *(const float*)(ftab + (d & 0xffffu));
    float d1 = *(const float*)(ftab + (d >> 16));
    if (__builtin_expect(bad != 0u, 0)) {
      const unsigned m = av & 0x7fff7fffu;
      if ((m & 0xffffu) < LUT_LO)
        d0 = dgelu_tiny(__uint_as_float(av << 16));
      if ((m >> 16) < LUT_LO)
        d1 = dgelu_tiny(__uint_as_float(av & 0xffff0000u));
    }
    return mk2(__fmul_rn(d0, bf16_lo(gv)),
               __fmul_rn(d1, bf16_hi(gv)));
  }
  return fmul2(act_dgelu2_w(av), bf16x2_to_f32x2(gv));
}

// One-time build of the tables.  Pure compile-time-constant data: it depends on
// nothing but the activation formula, exactly like a trig table.
__global__ void init_act_tables_kernel() {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * (LUT_HI - LUT_LO + 1)) return;
  const int sgn = i > (int)(LUT_HI - LUT_LO);
  const unsigned mag = LUT_LO + (unsigned)(sgn ? i - (LUT_HI - LUT_LO + 1) : i);
  const unsigned bits = ((unsigned)sgn << 15) | mag;
  const float v = __uint_as_float(bits << 16);
  const unsigned idx = (mag - LUT_LO) + (sgn ? (unsigned)LUT_N : 0u);
  const float dg = act_dgelu(v);
  unsigned short o;
  const float g = act_gelu(v);
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(g));
  const unsigned offf = idx << 2;
#if SWIZLUT
  const unsigned baseh = (idx << HSHF) - (idx & (HMSK & 0xffffu)) * HMUL;
#endif
  for (int r = 0; r < LUT_REP; ++r) {
    *(float*)(d_dgelu_tab[r] + offf) = dg;
#if SWIZLUT
#pragma unroll
    for (int c = 0; c < 2; ++c)
      *(unsigned short*)(d_gelu_tab[r] + baseh + c * HGSP) = o;
#else
    *(unsigned short*)(d_gelu_tab[r] + (idx << 1)) = o;
#endif
  }
}

// ---------------------------------------------------------------------------
// tile geometry: CTA tile = 64 rows (one dbias band) x 256 cols, processed as
// two 32-row sub-tiles (one columnwise MX block each).
// ---------------------------------------------------------------------------
constexpr int TCOLS = 256;
constexpr int SUBR = 32;
constexpr int NTHR = TCOLS / 2;   // 128 threads, 2 columns each on load
constexpr int SW = TCOLS / 2;     // 128 u32 words per smem row
constexpr int NGROUP = TCOLS / 32;
// How many of the four column words of a row take the shared-memory table
// route; the rest run the exact packed arithmetic body.  The table probe is
// L1/shared-pipe work and the arithmetic body is FMA/MUFU work, so the split
// is what keeps both sides of the machine busy: an all-table kernel saturates
// the shared pipe (measured 57% of its wavefronts are bank-conflict replays,
// L1 88% busy against DRAM 40%) while the FP32 pipes idle.
// Table/arithmetic routing at (row, word) SLOT granularity inside one CH-row
// load group: `L` of the `S` slots take the shared-table route and the
// arithmetic ones are spread by a Bresenham step, so their MUFU chains
// interleave with the table probes instead of clumping at the end of a row.
template <int S, int L, int I>
struct LutSlot {
  static constexpr int A = S - L;
  static constexpr bool v = !((((I + 1) * A) % S) < A);
};
#ifndef LUTQ_A
#define LUTQ_A 8                 // of 16 slots (CH = 4 rows x 4 words)
#endif
#ifndef LUTQ_D
#define LUTQ_D 6                 // of 8 slots (CH = 2 rows x 4 words)
#endif
#ifndef LUTQ_DB
#define LUTQ_DB 3                // of 4 slots; mode-4 optimum with CH_DB=1
#endif
// CAST_DBIAS_DACT is the one instantiation whose live set (dv[16] plus four u64
// dbias accumulators) sat above its four-block register tier, so it kept an
// 8-byte stack frame.  Halving the in-flight load-group depth frees the second
// row's act/grad uint4 pair outright and the spill disappears at the SAME 64
// registers and the SAME four resident CTAs -- the trade is in-flight bytes
// against spill traffic, and here the spill was the expensive half.
#ifndef CH_DB
#define CH_DB 1
#endif
#ifndef LUTM_A
#define LUTM_A 2
#endif
#ifndef LUTM_D
#define LUTM_D 3
#endif
// CAST_DBIAS_DACT wants a different split from CAST_DACT even though the two
// share the dgelu body: it carries 8 more live registers for its dbias
// accumulators.  Its 4-block bound creates enough register headroom for the
// 5-of-8 table/arithmetic split selected above.
#ifndef LUTM_DB
#define LUTM_DB 4
#endif
// ---------------------------------------------------------------------------
// NARROW: a 128-thread CTA for the CAST_ACT instantiation.  It DROPS resident
// warps 48 -> 32, which is the opposite of an occupancy optimisation, but it
// halves the cross-warp columnwise-amax fold (NWARP x TCOLS/2 shared words
// written and then read back per 32x256 tile) and halves the width of the two
// CTA barriers bracketing it.  That fold, not occupancy, is what this epilogue
// pays for: the mirror-image 512-thread tile reaches 64 warps with zero spills
// and still loses 13%.  With RPT=8 the tile is dv[32] at 64 registers, which is
// exactly 8 resident CTAs, and the split then has to be expressed at slot
// granularity (3 of 8) to stay at the same effective 6/16 table share.
// ---------------------------------------------------------------------------
#ifndef NARROW_A
#define NARROW_A 8
#endif
#ifndef WIDE_A
#define WIDE_A 0
#endif
#ifndef MINBW
#define MINBW 4
#endif
#ifndef CHW
#define CHW 2
#endif
// CAST_DBIAS on the register-resident tile.  Its dbias tolerance was read as
// demanding TE's exact row-order fp32 chain, but the output is BF16: rtol 1e-2
// is four bf16 ulps wide, and the only columns where atol 1e-4 binds are the
// near-zero ones, where an fp32 walk over 32768 terms carries ~9e-6 of
// round-off against an 8e-5 bound.  Sharing mode 4's tile buys CAST_DBIAS the
// 32x256 register tile, its wide loads and its two-barrier epilogue in place of
// the column-owned shared-staged streaming tile.
#ifndef DB_REGTILE
#define DB_REGTILE 0
#endif
// In-flight load-group depth for CAST_DBIAS on the register tile.  Unlike the
// other non-activation instantiation, this one carries four u64 dbias
// accumulators (8 registers) on top of the sixteen dv words, so a full
// four-row group (16 more registers for the uint4 batch) sits above the
// six-block tier.  Halving the group trades in-flight bytes for the spill.
#ifndef CH_DB1
#define CH_DB1 4
#endif
#ifndef NSPLIT
#define NSPLIT 1
#endif
#ifndef MINBRB1
#define MINBRB1 6
#endif
#ifndef ACT_CH2
// CAST_ACT loads its whole four-row group at once (four uint4 = 16 live
// registers) on top of the sixteen dv words -- 32 of the 40 registers the
// six-block launch bound allows are pure data.  Halving the load group to two
// rows frees eight registers for the scheduler; the split then has to be
// expressed at slot granularity (3 of 8) to stay at the same effective 6/16.
#define ACT_CH2 0
#endif
#ifndef LUTQ_A2
#define LUTQ_A2 3
#endif
#ifndef LUTQ_AW
#define LUTQ_AW 6                // of 16 slots (CH = 2 rows x 8 words)
#endif
// Rows of a load group that keep the full LUTM_A table words; the rest send one
// fewer.  With the narrow tile's two-row group, ACT_HI = 1 lands the effective
// table share on 3 of 8 rather than 4 of 8.  That is a two-sided optimum: the
// shared pipe is the binding one (L1 81.5% against 74.3% issue and 69.4% ALU),
// so the first quarter-word moved off the gather is worth more than the MUFU it
// costs, but 2 of 8 hands the FP32/MUFU side more than it can absorb.
#ifndef ACT_HI
#define ACT_HI 2
#endif
#define ACT_ROWM(T) ((T) < ACT_HI ? LUTM_A : LUTM_A - 1)
#ifndef UNA
#define UNA 8
#endif
#ifndef MINBA
#define MINBA 8
#endif
#ifndef MINB
// CAST_ONLY prefers 6 resident CTAs to 8.  The mode runs at the HBM ceiling, so
// the residency it loses is slack it was not using, and the ~10 extra registers
// per thread stop the 8-wide load batch's addresses from being rematerialised.
#define MINB 6
#endif
#ifndef MINBD
#define MINBD 5
#endif
#ifndef MINBR
#if NARROW_A
#define MINBR NARROW_A
#else
#define MINBR 6
#endif
#endif
#ifndef MINBRD
#define MINBRD 6
#endif
#ifndef MINBRB
#define MINBRB 4
#endif
// How many 32-row iterations a register-resident dbias CTA folds into ONE
// workspace band.  The shared fold (8 KB out + 8 KB in) and its two barriers
// are paid once per FOLDP iterations instead of once per 64 rows, and the band
// count -- hence both the workspace round trip and the length of the reducer's
// dependent add chain -- shrinks by the same factor.
#ifndef FOLDP
#define FOLDP 4
#endif
#ifndef DHOIST_DB
#define DHOIST_DB 0
#endif
#ifndef DHOIST_EN
#define DHOIST_EN 0
#endif
#ifndef HIDEB_EN
#define HIDEB_EN 1
#endif

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
struct Cfg {
  static constexpr bool ACTV = IS_ACT || IS_DACT;
  // CAST_DBIAS_DACT joins the register-resident tiling: its dbias tolerance
  // (TE's rtol_dbias = 4e-2 with atol 1e-4) absorbs the band-local regrouping
  // that a row-owned tile forces, whereas CAST_DBIAS's much tighter
  // {1e-5, 1e-2} does not, so that mode keeps the column-owned streaming tile
  // and its exact row-order fp32 chain.
  static constexpr bool REGTILE = !IS_DBIAS || IS_DACT || DB_REGTILE;
  static constexpr bool NARROW =
      NARROW_A && !WIDE_A && IS_ACT && !IS_DACT && !IS_DBIAS;
  // WIDE: 16 columns per lane instead of 8, i.e. a 32 x 512 CTA tile.  Every
  // per-ROW cost of the epilogue -- the two column-fold shuffles, the E8M0
  // encode, the group-scale byte store, the row's store address increment --
  // is paid once per lane per row no matter how many columns that lane owns,
  // so doubling the columns halves all of them per element.  The 32-value
  // rowwise block also shrinks from four lanes to two, dropping one of the two
  // shuffles outright, and both quantized stores widen from STG.64 to STG.128.
  static constexpr bool WIDE = WIDE_A && IS_ACT && !IS_DACT && !IS_DBIAS;
  static constexpr int NTHRC = REGTILE ? (NARROW ? 128 : 256) : NTHR;
  static constexpr int TC = WIDE ? 2 * TCOLS : TCOLS;   // CTA tile columns
  static constexpr int CPL = TC / 32;                   // columns per lane
  static constexpr int WPR = CPL / 2;                   // bf16x2 words per lane
  static constexpr int NV = WPR / 4;                    // uint4 loads per row
  static constexpr int LGRP = 32 / CPL;                 // lanes per MX group
  static constexpr int NGRP = TC / 32;                  // MX groups per row
  static constexpr int NWARP = NTHRC / 32;
  static constexpr int RPT = REGTILE ? 32 / NWARP : 1;   // rows per warp
  static constexpr bool SLOTSPLIT = IS_DACT || (IS_ACT && (ACT_CH2 || NARROW));
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
  static constexpr bool SPLITEP = (ACTV || IS_DBIAS) && (NTHRC >= TC / 2) &&
                                  (NSPLIT || NTHRC >= TC / 2 + TC / 4);
  static constexpr int DRAINOFF = (NTHRC >= TC / 2 + TC / 4) ? TC / 2 : 0;
  static constexpr int CH =
      (IS_DACT && IS_DBIAS)
          ? CH_DB
          : (IS_DBIAS ? CH_DB1
                      : (WIDE ? CHW : (SLOTSPLIT ? 2 : RPT)));
  static constexpr int LUTM =
      IS_DACT ? (IS_DBIAS ? LUTM_DB : LUTM_D) : LUTM_A;  // table/arith split
  static constexpr int SLOTS = CH * WPR;
  static constexpr int LUTS = IS_DACT ? (IS_DBIAS ? LUTQ_DB : LUTQ_D) : LUTQ_A2;
  static constexpr int MINB_C =
      REGTILE ? (IS_DBIAS ? (IS_DACT ? MINBRB : MINBRB1)
                          : (IS_DACT ? MINBRD
                                     : (IS_ACT ? (WIDE ? MINBW : MINBR) : MINB)))
              : (IS_DACT ? MINBD : MINBA);
  // per-warp fp32 dbias partials, folded in warp order inside the 64-row band
  static constexpr int DBB = IS_DBIAS && REGTILE ? NWARP * TC : 1;
  // shared-memory activation table: fp32 dgelu (16 KB) or bf16 gelu (8 KB)
  static constexpr bool NEEDLUT =
      IS_DACT ? (LUTS > 0) : (IS_ACT && LUTM_A > 0);
  static constexpr int LUTB = NEEDLUT ? (IS_DACT ? LUT_SPAN_F : LUT_SPAN_H) : 16;
  // hoist the dgelu table's out-of-window test from per-word to per-row
  static constexpr bool DHOIST = DHOIST_EN && IS_DACT && (!IS_DBIAS || DHOIST_DB);
  // Split the first cross-warp join into arrive/wait around the rowwise half.
  // Only CAST_ACT takes it: the 256-thread instantiations park the row-scale
  // drain on the [TC/2, 3TC/4) tids the columnwise fold leaves idle, so moving
  // the drain behind the second join -- which is what the split forces, since
  // only that join orders the scratch -- costs them more than the hidden
  // arrival skew buys.  The 128-thread narrow tile has no idle tid range to
  // begin with, so it keeps the win with no offsetting loss.
  static constexpr bool HIDEB =
      HIDEB_EN && SPLITEP && NEEDLUT && IS_ACT && !IS_DACT && !IS_DBIAS;
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
DEVI void mbar_init(unsigned long long* bar, int cnt) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(cnt)
               : "memory");
}
DEVI void mbar_arrive(unsigned long long* bar) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
DEVI void mbar_wait(unsigned long long* bar, unsigned phase) {
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

DEVI void wait_lut(unsigned long long* bar) {
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
template <bool DEFER = false>
DEVI void load_lut(unsigned char* dst, const void* src, int bytes,
                   unsigned long long* bar, int tid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  __syncthreads();
  if (tid == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     b),
                 "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(d),
        "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(b)
        : "memory");
  }
  if constexpr (!DEFER) wait_lut(bar);
}


// ---------------------------------------------------------------------------
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
__device__ __forceinline__ void quantize_streaming(
    const unsigned* __restrict__ input,      // primary tensor (x or grad)
    const unsigned* __restrict__ act_input,  // pre-activation tensor (x)
    unsigned char* __restrict__ out_rw, unsigned char* __restrict__ scale_rw,
    unsigned char* __restrict__ out_cw, unsigned char* __restrict__ scale_cw,
    float* __restrict__ dbias_ws, int K, int sc_rw_stride, int sc_cw_stride,
    int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  __shared__ __align__(16) unsigned tile[SUBR * SW];
  __shared__ __align__(16) unsigned cscale[TCOLS / 2];
  __shared__ __align__(4) unsigned char srs[SUBR * NGROUP];
  __shared__ __align__(16) unsigned char lutmem[C::LUTB];
  __shared__ __align__(8) unsigned long long lutbar;

  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * TCOLS;
  constexpr int TROWS = IS_DBIAS ? 64 : 32;  // dbias needs a full band per CTA
  const int Kw = K >> 1;

  const int lane = tid & 31;
  const int warp = tid >> 5;

  if constexpr (IS_DACT)
    load_lut(lutmem, d_dgelu_tab[blockIdx.x & (LUT_REP - 1)], C::LUTB, &lutbar,
             tid);
  const unsigned char* __restrict__ atab = lutmem;
  // Which private copy of the bank-partitioned table this lane gathers
  // from.  Constant for the whole kernel, so a probe pays for it with one
  // extra IADD3 operand.
  const unsigned gsp = IS_DACT ? GSPLAT_F(lane) : GSPLAT_H(lane);

  // CAST_ONLY/CAST_DBIAS always launch one row block per CTA.  Keep that bound
  // compile-time in their template instantiations so the hot path has no loop
  // backedge and does not carry the runtime `iters` argument through address
  // generation.  Activation modes retain their tuned multi-block walks.
  const int loop_iters = (IS_ACT || IS_DACT) ? iters : 1;
#pragma unroll 1
  for (int it = 0; it < loop_iters; ++it) {
    const int row0 = (blockIdx.y * loop_iters + it) * TROWS;
    u64 dbp = mk2(0.f, 0.f);
    const unsigned* ip = input + (size_t)row0 * Kw + (col0 >> 1) + tid;
    const unsigned* ap = act_input + (size_t)row0 * Kw + (col0 >> 1) + tid;

#pragma unroll 1
    for (int sub = 0; sub < TROWS / SUBR; ++sub) {
      // ------------- phase 1: load, activate, colwise amax, dbias ---------
      {
        unsigned acc = 0u;
        constexpr int UN = 8;
        auto do_batch = [&](int b) {
          unsigned ra[UN], rg[UN];
          const size_t base = (size_t)(sub * SUBR + b * UN) * Kw;
#pragma unroll
          for (int k = 0; k < UN; ++k)
            ra[k] = (IS_ACT || IS_DACT) ? ap[base + (size_t)k * Kw]
                                        : ip[base + (size_t)k * Kw];
          if constexpr (IS_DACT) {
#pragma unroll
            for (int k = 0; k < UN; ++k) rg[k] = ip[base + (size_t)k * Kw];
          }
// rows are consumed in strictly increasing k so the fp32 dbias chain keeps
// the reference's summation order; the LUT/arithmetic split rides along.
#define MXFP8_ROW(KO)                                                       \
  {                                                                         \
    constexpr int k = (KO);                                                 \
    unsigned packed;                                                        \
    if constexpr (IS_DACT) {                                                \
      const u64 res = mul_dgelu_grad2<((KO) % 4 < C::LUTM)>(                \
          ra[k], rg[k], atab, gsp);                                         \
      if constexpr (IS_DBIAS) dbp = fadd2(dbp, res);                        \
      float a_, b_;                                                         \
      un2(res, a_, b_);                                                     \
      packed = pack_bf16x2(b_, a_);                                         \
    } else if constexpr (IS_ACT) {                                          \
      unsigned bs_ = 0u;                                                    \
      packed = do_gelu2<((KO) % 4 < C::LUTM)>(ra[k], atab, gsp, bs_);       \
      (void)bs_;                    \
    } else {                                                                \
      packed = ra[k];                                                       \
      if constexpr (IS_DBIAS) dbp = fadd2(dbp, bf16x2_to_f32x2(packed));    \
    }                                                                       \
    acc = amax2(acc, packed);                                               \
    tile[(b * UN + k) * SW + tid] = packed;                                 \
  }
          MXFP8_ROW(0) MXFP8_ROW(1) MXFP8_ROW(2) MXFP8_ROW(3)
          MXFP8_ROW(4) MXFP8_ROW(5) MXFP8_ROW(6) MXFP8_ROW(7)
#undef MXFP8_ROW
        };
        if constexpr (IS_ACT || IS_DACT) {
          // the closed-form tails freed enough registers to keep two 8-row
          // batches (32 loads) in flight, which is what this low-occupancy
          // shared-staged path needs to cover DRAM latency.
#pragma unroll 2
          for (int b = 0; b < SUBR / UN; ++b) do_batch(b);
        } else {
#pragma unroll
          for (int b = 0; b < SUBR / UN; ++b) do_batch(b);
        }
        unsigned m = acc & 0x7fff7fffu;
        const unsigned p = cvt_ue8m0x2(bf16_hi(m) * (1.0f / 448.0f),
                                       bf16_lo(m) * (1.0f / 448.0f));
        *(unsigned short*)(scale_cw + (size_t)(row0 / 32 + sub) * sc_cw_stride +
                           col0 + 2 * tid) = (unsigned short)p;
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
        // single shuffle chain and one cvt_ue8m0x2 serve both rows.  Halves the
        // shuffle and scale-conversion work of this phase.
        for (int i = 0; i < SUBR / 4; i += 2) {
          const int r = warp + 4 * i;
          const int r1 = r + 4;
          const uint4 v0 = tp[r * (SW / 4) + lane];
          const uint4 v1 = tp[r1 * (SW / 4) + lane];
          auto row_mag = [](const uint4& v) {
            unsigned a = amax2(amax2(v.x, v.y), amax2(v.z, v.w));
            return amax2(a, __byte_perm(a, a, 0x1032));
          };
          unsigned p = __byte_perm(row_mag(v0), row_mag(v1), 0x5410) &
                       0x7fff7fffu;
          p = amax2(p, __shfl_xor_sync(0xffffffffu, p, 1));
          p = amax2(p, __shfl_xor_sync(0xffffffffu, p, 2));
          const unsigned er = cvt_ue8m0x2(
              bf16_hi(p) * (1.0f / 448.0f),
              bf16_lo(p) * (1.0f / 448.0f));
          const unsigned e0 = er & 0xffu;
          const unsigned e1 = (er >> 8) & 0xffu;
          if ((lane & 3) == 0) {
            srs[r * NGROUP + (lane >> 2)] = (unsigned char)e0;
            srs[r1 * NGROUP + (lane >> 2)] = (unsigned char)e1;
          }
          const unsigned rsp0 = enc_scale_bf16x2(e0);
          const unsigned rsp1 = enc_scale_bf16x2(e1);

          auto write_row = [&](const uint4& v, int rr, unsigned rsp) {
            const unsigned* vw = (const unsigned*)&v;
            unsigned ro[2], co[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              ro[j] = quant4(vw[2 * j], vw[2 * j + 1], rsp, rsp);
              co[j] = quant4(vw[2 * j], vw[2 * j + 1], cw[2 * j], cw[2 * j + 1]);
            }
            const size_t go = (size_t)(row0 + sub * SUBR + rr) * K + coff;
            *(uint2*)(out_rw + go) = *(const uint2*)ro;
            *(uint2*)(out_cw + go) = *(const uint2*)co;
          };
          write_row(v0, r, rsp0);
          write_row(v1, r1, rsp1);
        }
      }

      __syncthreads();

      if (tid < SUBR * 2) {
        const int r = tid >> 1;
        const int half = tid & 1;
        *(unsigned*)(scale_rw + (size_t)(row0 + sub * SUBR + r) * sc_rw_stride +
                     (col0 >> 5) + half * 4) =
            *(const unsigned*)(srs + r * NGROUP + half * 4);
      }
    }

    if constexpr (IS_DBIAS) {
      // one fp32 partial per 64-row band, accumulated strictly in row order --
      // the reference's exact summation shape.
      float* wp = dbias_ws + (size_t)(row0 >> 6) * K + col0 + 2 * tid;
      const u64 policy = l2_evict_last_policy();
      store_l2_evict_last((u64*)wp, dbp, policy);
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
    float* __restrict__ dbias_ws, int K, int sc_rw_stride, int sc_cw_stride,
    int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  constexpr int NWARP = C::NWARP, RPT = C::RPT, CH = C::CH;

  // The cross-warp columnwise-amax scratch and the dbias band fold are never
  // live at the same time -- a barrier separates every use of one from the next
  // use of the other -- so they share one buffer.  That is 4 KB less per CTA,
  // which is what keeps CAST_DBIAS_DACT inside the 164 KB carveout at 6 blocks.
  constexpr int TC = C::TC, WPR = C::WPR, NV = C::NV, CPL = C::CPL;
  constexpr int LGRP = C::LGRP, NGRP = C::NGRP;
  constexpr int FOLDW = (NWARP * (TC / 2) > C::DBB) ? NWARP * (TC / 2) : C::DBB;
  __shared__ __align__(16) unsigned foldbuf[FOLDW];
  unsigned(*cpart)[TC / 2] = (unsigned(*)[TC / 2])foldbuf;
  float* dbp_s = (float*)foldbuf;
  __shared__ __align__(16) unsigned cscale[TC / 2];
  // With the split join the row-scale scratch of tile `it` is still being
  // drained (after the second join) while tile it+1's rowwise half is already
  // filling it, so it double-buffers on `it & 1`.  256 extra bytes per CTA and
  // no extra live register -- the alternative, hoisting the scales into
  // registers so the mbarrier could order them, costs RPT registers on exactly
  // the instantiations that sit on a register cliff.
  __shared__ __align__(8) unsigned char srs[C::HIDEB ? 2 * SUBR * NGRP
                                                     : SUBR * NGRP];
  __shared__ __align__(16) unsigned char lutmem[C::LUTB];
  __shared__ __align__(8) unsigned long long lutbar;
  __shared__ __align__(8) unsigned long long epbar;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * TC;
  const int K4 = K >> 3;  // uint4 (8 bf16 values) per row

  // Only the DBIAS-fused instantiations get a dbias_ws scratch buffer: TE's
  // nvte_quantize (plain cast) and the ACT/DACT-only entry points never
  // allocate one, so zeroing it unconditionally null-derefs on every call
  // that isn't fused with dbias.
  if constexpr (IS_DBIAS) {
    if (blockIdx.y == 0) {
#pragma unroll
      for (int q = tid; q < TC; q += C::NTHRC)
        ((unsigned short*)dbias_ws)[col0 + q] = 0;
    }
  }

  // the epilogue mbarrier rides on the table copy's own init/join
  if constexpr (C::HIDEB)
    if (tid == 0) mbar_init(&epbar, C::NTHRC);
  if constexpr (C::NEEDLUT) {
    const unsigned rep = blockIdx.x & (LUT_REP - 1);
    load_lut<!IS_DACT>(lutmem,
                       IS_DACT ? (const void*)d_dgelu_tab[rep]
                               : (const void*)d_gelu_tab[rep],
                       C::LUTB, &lutbar, tid);
  }
  const unsigned char* __restrict__ atab = lutmem;
  // Which private copy of the bank-partitioned table this lane gathers
  // from.  Constant for the whole kernel, so a probe pays for it with one
  // extra IADD3 operand.
  const unsigned gsp = IS_DACT ? GSPLAT_F(lane) : GSPLAT_H(lane);
  // one packed fp32 pair per owned column pair: add.rn.f32x2 folds both lanes
  // of a column pair in a single instruction with exactly the scalar rounding
  u64 dba[IS_DBIAS ? 4 : 1];

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
#if STRIDED_WALK
    const int row0 = (it * gridDim.y + blockIdx.y) * SUBR;
#else
    const int row0 = (blockIdx.y * loop_iters + it) * SUBR;
#endif
    unsigned dv[RPT * WPR];  // this thread's RPT x CPL sub-block, bf16x2
    // one fp32 running sum per owned column, restarted at each 64-row band
    // boundary so the workspace keeps TE's per-band partial layout
    // The band is the CTA's whole row walk: the shared fold (8 KB out, 8 KB in)
    // and its two barriers are paid ONCE per CTA instead of once per FOLDP
    // iterations, and the workspace round trip plus the reducer's dependent
    // chain shrink by the same factor.  Rows are still consumed in strictly
    // increasing order inside the band.
    if constexpr (IS_DBIAS) {
      if (it == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dba[i] = mk2(0.f, 0.f);
      }
    }
    const size_t rbase =
        (size_t)(row0 + warp * RPT) * K4 + (col0 >> 3) + (size_t)NV * lane;
    const uint4* ip = input + rbase;
    const uint4* ap = act_input + rbase;


    // ---- load + activate, CH rows in flight at a time --------------------
#pragma unroll
    for (int h = 0; h < RPT / CH; ++h) {
      uint4 a[CH][NV], g[CH][NV];
#pragma unroll
      for (int t = 0; t < CH; ++t)
#pragma unroll
        for (int n = 0; n < NV; ++n)
          a[t][n] = ap[(size_t)(h * CH + t) * K4 + n];
      if constexpr (IS_DACT) {
#pragma unroll
        for (int t = 0; t < CH; ++t)
#pragma unroll
          for (int n = 0; n < NV; ++n)
            g[t][n] = ip[(size_t)(h * CH + t) * K4 + n];
      }
      // Only CAST_ACT defers the join; the dGeLU paths keep the blocking join.
      if constexpr (C::NEEDLUT && !IS_DACT) {
        if (h == 0 && it == 0) wait_lut(&lutbar);
      }
if constexpr (!IS_DACT) {
      // Fractional table/arithmetic split for CAST_ACT expressed at ROW
      // granularity: the first ACT_HI rows of the load group send LUTM_A words
      // to the shared table and the rest send one fewer, so the effective split
      // lands between two multiples of 1/4 without interleaving the two bodies
      // inside a row (which is what lengthened register lifetimes when the
      // Bresenham slot spread was tried here).  L1 is the binding pipe at the
      // 2-of-4 setting (80.8% against 71.1% issue), so shaving a fraction of a
      // word off the gather is worth more than the MUFU it costs.
      // ACT_HI < CH is what makes the split fractional; with a two-row load
      // group (the narrow tile) ACT_HI = 1 puts LUTM_A words of row 0 and one
      // fewer of row 1 through the table, i.e. 3 of 8 rather than 4 of 8.
      if constexpr (IS_ACT && ACT_HI < CH && !ACT_CH2 && !C::WIDE && NV == 1) {
#define MXFP8_AWORD(T, M)                                                \
  dv[(h * CH + (T)) * WPR + (M)] =                                         \
      do_gelu2<((M) < ACT_ROWM(T))>(av_[M], atab, gsp, bad_);
#define MXFP8_AFIX(T, M)                                                 \
  if constexpr ((M) < ACT_ROWM(T))                                       \
    dv[(h * CH + (T)) * WPR + (M)] =                                       \
        gelu_fix2(av_[M], dv[(h * CH + (T)) * WPR + (M)]);
#define MXFP8_AROW(T)                                                    \
  {                                                                      \
    const unsigned* av_ = (const unsigned*)&a[T][0];                     \
    MXFP8_AWORD(T, 0) MXFP8_AWORD(T, 1)                                  \
    MXFP8_AWORD(T, 2) MXFP8_AWORD(T, 3)                                  \
  }
#define MXFP8_AROWFIX(T)                                                 \
  {                                                                      \
    const unsigned* av_ = (const unsigned*)&a[T][0];                     \
    MXFP8_AFIX(T, 0) MXFP8_AFIX(T, 1)                                    \
    MXFP8_AFIX(T, 2) MXFP8_AFIX(T, 3)                                    \
  }
        unsigned bad_ = 0u;
        MXFP8_AROW(0)
        if constexpr (CH > 1) MXFP8_AROW(1)
        if constexpr (CH > 2) { MXFP8_AROW(2) MXFP8_AROW(3) }
        if (__builtin_expect(bad_ != 0u, 0)) {
          MXFP8_AROWFIX(0)
          if constexpr (CH > 1) MXFP8_AROWFIX(1)
          if constexpr (CH > 2) { MXFP8_AROWFIX(2) MXFP8_AROWFIX(3) }
        }
#undef MXFP8_AROWFIX
#undef MXFP8_AROW
#undef MXFP8_AFIX
#undef MXFP8_AWORD
      } else {
#pragma unroll
      for (int t = 0; t < CH; ++t) {
#pragma unroll
      for (int n = 0; n < NV; ++n) {
        const unsigned* av = (const unsigned*)&a[t][n];
        const unsigned* gv = (const unsigned*)&g[t][n];
        unsigned bad = 0u;
        const int dvb = (h * CH + t) * WPR + n * 4;
// The four column words of a row are split between the table and the
// arithmetic body so the two saturated pipes (ALU + shared) hand work to the
// two idle ones (FP32 + MUFU); LUTM sets the split.
#define MXFP8_WORD(M)                                                    \
  if constexpr (!C::ACTV) {                                              \
    dv[dvb + (M)] = av[M];                                               \
    if constexpr (IS_DBIAS)                                              \
      dba[M] = fadd2(dba[M], bf16x2_to_f32x2(av[M]));                    \
  } else if constexpr (IS_ACT) {                                         \
    dv[dvb + (M)] = do_gelu2<((M) < C::LUTM)>(av[M], atab, gsp, bad);    \
  } else {                                                               \
    const u64 r_ = mul_dgelu_grad2<((M) < C::LUTM)>(                       \
        av[M], gv[M], atab, gsp);                                          \
    float a_, b_;                                                        \
    un2(r_, a_, b_);                                                     \
    if constexpr (IS_DBIAS) dba[M] = fadd2(dba[M], r_);                  \
    dv[dvb + (M)] = pack_bf16x2(b_, a_);                                 \
  }
        MXFP8_WORD(0)
        MXFP8_WORD(1)
        MXFP8_WORD(2)
        MXFP8_WORD(3)
#undef MXFP8_WORD
        // one branch for the whole row: the table probes above already left
        // their clamp residue in `bad`, and only the table-routed words can
        // need the closed-form tail
        if constexpr (IS_ACT) {
          if (__builtin_expect(bad != 0u, 0)) {
#pragma unroll
            for (int M = 0; M < C::LUTM; ++M)
              dv[dvb + M] = gelu_fix2(av[M], dv[dvb + M]);
          }
        }
      }
      }
      }
      } else {
// The (row, word) slots of a load group are split between the table and the
// arithmetic body so the two saturated pipes (ALU + shared) hand work to the
// two idle ones (FP32 + MUFU); LutSlot sets the split.
// ACT keeps the plain per-row split (the Bresenham interleave lengthens its
// register lifetimes enough to spill); only the DACT bodies, whose optimum sits
// strictly between two multiples of four, need slot granularity.
#define MXFP8_LUT(T, M)                                                  \
  (C::SLOTSPLIT ? (LutSlot<C::SLOTS, C::LUTS,                            \
                            (T) * 4 + (M) +                              \
                                ((IS_DACT && !IS_DBIAS) ? 2 : 0)>::v)   \
                : ((M) < C::LUTM))
#define MXFP8_WORD(T, M)                                                 \
  if constexpr (!C::ACTV) {                                              \
    dv[(h * CH + (T)) * WPR + (M)] = av[M];                                \
  } else if constexpr (IS_ACT) {                                         \
    dv[(h * CH + (T)) * WPR + (M)] =                                       \
        do_gelu2<MXFP8_LUT(T, M)>(av[M], atab, gsp, bad);                     \
  } else {                                                               \
    u64 r_;                                                              \
    if constexpr (C::DHOIST) {                                           \
      const u64 dg_ = do_dgelu2_nb<MXFP8_LUT(T, M)>(                     \
          av[M], atab, gsp, bad);                                        \
      r_ = fmul2(dg_, bf16x2_to_f32x2(gv[M]));                           \
    } else if constexpr (IS_DBIAS) {                                     \
      r_ = mul_dgelu_grad2<MXFP8_LUT(T, M)>(                             \
          av[M], gv[M], atab, gsp);                                      \
    } else {                                                             \
      r_ = fmul2(do_dgelu2<MXFP8_LUT(T, M)>(av[M], atab, gsp),           \
                  bf16x2_to_f32x2(gv[M]));                               \
    }                                                                    \
    float a_, b_;                                                        \
    un2(r_, a_, b_);                                                     \
    if constexpr (IS_DBIAS) dba[M] = fadd2(dba[M], r_);                  \
    dv[(h * CH + (T)) * WPR + (M)] = pack_bf16x2(b_, a_);                  \
  }
// rare path: re-probe (cheap here, it runs for ~1% of warps) and rebuild the
// word from the closed-form tail.  For the dbias instantiation the running
// fp32 sum is corrected by the delta so the band partial still sees the exact
// product.
#define MXFP8_DFIX(T, M)                                                 \
  if constexpr (C::DHOIST && MXFP8_LUT(T, M)) {                          \
    unsigned bd_ = 0u;                                                   \
    const u64 raw_ = lut_dgelu2_raw(av[M], atab, gsp, bd_);                   \
    if (bd_) {                                                           \
      const u64 g_ = bf16x2_to_f32x2(gv[M]);                             \
      const u64 rn_ = fmul2(dgelu_fix2(av[M], raw_), g_);                \
      if constexpr (IS_DBIAS)                                            \
        dba[M] = fadd2(dba[M], fsub2(rn_, fmul2(raw_, g_)));             \
      float a2_, b2_;                                                    \
      un2(rn_, a2_, b2_);                                                \
      dv[(h * CH + (T)) * WPR + (M)] = pack_bf16x2(b2_, a2_);              \
    }                                                                    \
  }
// one branch for the whole row: the table probes above already left their
// clamp residue in `bad`, and only the table-routed words can need the
// closed-form tail
#define MXFP8_FIX(T, M)                                                  \
  if constexpr (IS_ACT && MXFP8_LUT(T, M))                               \
    dv[(h * CH + (T)) * WPR + (M)] =                                       \
        gelu_fix2(av[M], dv[(h * CH + (T)) * WPR + (M)]);
#define MXFP8_TROW(T)                                                    \
  {                                                                      \
    const unsigned* av = (const unsigned*)&a[T][0];                      \
    const unsigned* gv = (const unsigned*)&g[T][0];                      \
    unsigned bad = 0u;                                                   \
    MXFP8_WORD(T, 0)                                                     \
    MXFP8_WORD(T, 1)                                                     \
    MXFP8_WORD(T, 2)                                                     \
    MXFP8_WORD(T, 3)                                                     \
    if constexpr (IS_ACT) {                                              \
      if (__builtin_expect(bad != 0u, 0)) {                              \
        MXFP8_FIX(T, 0) MXFP8_FIX(T, 1)                                  \
        MXFP8_FIX(T, 2) MXFP8_FIX(T, 3)                                  \
      }                                                                  \
    } else if constexpr (C::DHOIST) {                                    \
      if (__builtin_expect(bad != 0u, 0)) {                              \
        MXFP8_DFIX(T, 0) MXFP8_DFIX(T, 1)                                \
        MXFP8_DFIX(T, 2) MXFP8_DFIX(T, 3)                                \
      }                                                                  \
    }                                                                    \
  }
      MXFP8_TROW(0)
      if constexpr (CH > 1) MXFP8_TROW(1)
      if constexpr (CH > 2) { MXFP8_TROW(2) MXFP8_TROW(3) }
#undef MXFP8_TROW
#undef MXFP8_FIX
#undef MXFP8_DFIX
#undef MXFP8_WORD
#undef MXFP8_LUT
      }
    }

    // Nothing in the ROWWISE half of the epilogue depends on the columnwise
    // scale, so for the instantiations that can afford the registers it runs
    // BEFORE the cross-warp barrier.  The row-scale scratch is then drained
    // inside the SAME barrier interval that reduces the columnwise partials --
    // by the tid in [128,192) threads that reduction leaves idle -- which turns
    // three CTA barriers per iteration into two and retires the rowwise stores
    // early so their scale registers stop being live across the columnwise half.
    constexpr bool SPLITEP = C::SPLITEP;
    const size_t gbase =
        (size_t)(row0 + warp * RPT) * K + col0 + CPL * lane;
    unsigned char* const srsb =
        srs + (C::HIDEB ? (unsigned)(it & 1) * (SUBR * NGRP) : 0u);
    auto row_scale = [&](const unsigned* v, int j) {
      unsigned a = amax2(amax2(v[0], v[1]), amax2(v[2], v[3]));
#pragma unroll
      for (int m = 4; m < WPR; m += 2) a = amax2(a, amax2(v[m], v[m + 1]));
      // A 32-value rowwise block spans 32/CPL lanes, so a lane owning sixteen
      // columns needs ONE cross-lane fold where an eight-column lane needs two.
#pragma unroll
      for (int b = 1; b < LGRP; b <<= 1)
        a = amax2(a, __shfl_xor_sync(0xffffffffu, a, b));
      unsigned er;
      if constexpr (IS_DBIAS) {
        // CAST_DBIAS_DACT is the instantiation sitting on the 5-block register
        // cliff: the masked-extract form costs two more integer ops but keeps
        // one fewer value live, which is the cheaper trade there.
        a &= 0x7fff7fffu;
        const float mx = fmaxf(bf16_lo(a), bf16_hi(a)) * (1.0f / 448.0f);
        er = cvt_ue8m0x2(mx, mx) & 0xffu;
      } else {
        // Fold the two bf16 halves with one PRMT + one packed max instead of a
        // mask, two extracts and an FMNMX: max.xorsign.abs leaves the magnitude
        // in BOTH halves, so masking the high half alone already yields the
        // fp32 bit pattern.  Converting with 0.0f in the upper slot then lands
        // the exponent in the low byte with no trailing mask.
        a = amax2(a, __byte_perm(a, a, 0x1032));
        const float mx = __uint_as_float(a & 0x7fff0000u) * (1.0f / 448.0f);
        er = cvt_ue8m0x2(0.f, mx);
      }
      if ((lane & (LGRP - 1)) == 0)
        srsb[(warp * RPT + j) * NGRP + (lane / LGRP)] = (unsigned char)er;
      return enc_scale_bf16x2(er);
    };
    // On the 128-thread CAST_ACT tile each warp owns eight rows.  Pack the
    // maxima of two rows into one bf16x2 so the lane butterfly, 1/448 scale,
    // and native UE8M0 conversion serve both rows together.
    auto row_mag = [&](const unsigned* v) {
      unsigned a = amax2(amax2(v[0], v[1]), amax2(v[2], v[3]));
#pragma unroll
      for (int m = 4; m < WPR; m += 2) a = amax2(a, amax2(v[m], v[m + 1]));
      return amax2(a, __byte_perm(a, a, 0x1032));
    };
    auto row_scale_pair = [&](const unsigned* va, const unsigned* vb, int j,
                              unsigned& s0, unsigned& s1) {
      unsigned p = __byte_perm(row_mag(va), row_mag(vb), 0x5410) & 0x7fff7fffu;
#pragma unroll
      for (int b = 1; b < LGRP; b <<= 1)
        p = amax2(p, __shfl_xor_sync(0xffffffffu, p, b));
      float m0, m1;
      un2(fmul2(mk2(bf16_lo(p), bf16_hi(p)), splat2(1.0f / 448.0f)), m0, m1);
      const unsigned e = cvt_ue8m0x2(m1, m0);
      const unsigned e0 = e & 0xffu, e1 = (e >> 8) & 0xffu;
      if ((lane & (LGRP - 1)) == 0) {
        unsigned char* q = srsb + (warp * RPT + j) * NGRP + (lane / LGRP);
        q[0] = (unsigned char)e0;
        q[NGRP] = (unsigned char)e1;
      }
      s0 = enc_scale_bf16x2(e0);
      s1 = enc_scale_bf16x2(e1);
    };
    // ---- columnwise amax: RPT-row partial in registers, cross-warp in smem
    auto colpart = [&]() {
      unsigned c[WPR];
#pragma unroll
      for (int m = 0; m < WPR; ++m) c[m] = dv[m];
#pragma unroll
      for (int j = 1; j < RPT; ++j)
#pragma unroll
        for (int m = 0; m < WPR; ++m) c[m] = amax2(c[m], dv[j * WPR + m]);
#pragma unroll
      for (int n = 0; n < NV; ++n)
        *(uint4*)(&cpart[warp][WPR * lane + 4 * n]) =
            *(const uint4*)(c + 4 * n);
    };
    // The cross-warp partial depends on nothing the rowwise half produces, so
    // publish it first and arrive without blocking; the rowwise quantize/store
    // then runs inside the join's own latency.
    if constexpr (C::HIDEB) {
      colpart();
      mbar_arrive(&epbar);
    }
    constexpr bool PAIRSC = IS_ACT && !IS_DACT && !IS_DBIAS &&
                            C::NTHRC == 128 && RPT == 8;
    if constexpr (PAIRSC) {
      unsigned char* prw = out_rw + gbase;
#pragma unroll
      for (int j = 0; j < RPT; j += 2) {
        const unsigned* va = dv + j * WPR;
        const unsigned* vb = va + WPR;
        unsigned s0, s1;
        row_scale_pair(va, vb, j, s0, s1);
        unsigned ro[WPR / 2];
#pragma unroll
        for (int m = 0; m < WPR / 2; ++m)
          ro[m] = quant4(va[2 * m], va[2 * m + 1], s0, s0);
        st_out<WPR>(prw + (unsigned)j * (unsigned)K, ro);
#pragma unroll
        for (int m = 0; m < WPR / 2; ++m)
          ro[m] = quant4(vb[2 * m], vb[2 * m + 1], s1, s1);
        st_out<WPR>(prw + (unsigned)(j + 1) * (unsigned)K, ro);
      }
    } else if constexpr (SPLITEP) {
      unsigned char* prw = out_rw + gbase;
#pragma unroll
      for (int j = 0; j < RPT; ++j) {
        const unsigned* v = dv + j * WPR;
        const unsigned rsp = row_scale(v, j);
        unsigned ro[WPR / 2];
#pragma unroll
        for (int m = 0; m < WPR / 2; ++m)
          ro[m] = quant4(v[2 * m], v[2 * m + 1], rsp, rsp);
        st_out<WPR>(prw + (unsigned)j * (unsigned)K, ro);
      }
    }

    if constexpr (C::HIDEB) {
      mbar_wait(&epbar, (unsigned)(it & 1));
    } else {
      colpart();
      __syncthreads();
    }
    if (tid < TC / 2) {
      unsigned m = cpart[0][tid];
#pragma unroll
      for (int w = 1; w < NWARP; ++w) m = amax2(m, cpart[w][tid]);
      m &= 0x7fff7fffu;
      const unsigned p = cvt_ue8m0x2(bf16_hi(m) * (1.0f / 448.0f),
                                     bf16_lo(m) * (1.0f / 448.0f));
      *(unsigned short*)(scale_cw + (size_t)(row0 / 32) * sc_cw_stride + col0 +
                         2 * tid) = (unsigned short)p;
      cscale[tid] = (0x00FE00FEu - __byte_perm(p, 0, 0x4140)) << 7;
    }
    auto drain = [&]() {
      constexpr int DO = C::DRAINOFF;
      // A row's eight group scales are eight CONTIGUOUS bytes of scale_rw, and
      // both ends are 8-byte aligned for every shape in the contract
      // (sc_rw_stride is a multiple of four groups; col0 >> 5 is a multiple of
      // eight), so one warp drains a whole 32-row tile with one STG.64 per lane
      // where two warps needed one STG.32 each.  Same sector count, half the
      // store instructions, and the released warp rejoins the columnwise
      // quantize inside the already-open barrier interval.
      if constexpr (NGRP == 8) {
        if (tid >= DO && tid < DO + SUBR) {
          const int r = tid - DO;
          *(u64*)(scale_rw + (size_t)(row0 + r) * sc_rw_stride +
                  (col0 >> 5)) = *(const u64*)(srsb + r * NGRP);
        }
      } else if (tid >= DO && tid < DO + SUBR * (NGRP / 4)) {
          const int q = tid - DO;
          const int r = q / (NGRP / 4);
          const int half = q % (NGRP / 4);
          *(unsigned*)(scale_rw + (size_t)(row0 + r) * sc_rw_stride +
                       (col0 >> 5) + half * 4) =
              *(const unsigned*)(srsb + r * NGRP + half * 4);
      }
    };
    // Without the split join the scratch is already published by the first
    // __syncthreads and the drain fills the fold's idle tid range.  With it,
    // only the SECOND join orders the scratch, so the drain moves behind it and
    // overlaps the columnwise quantization instead.
    if constexpr (SPLITEP && !C::HIDEB) drain();
    __syncthreads();
    if constexpr (SPLITEP && C::HIDEB) drain();

    // ---- columnwise (and, when not split, rowwise) quantization ----------
    {
      uint4 csv[NV];
#pragma unroll
      for (int n = 0; n < NV; ++n)
        csv[n] = *(const uint4*)(cscale + WPR * lane + 4 * n);
      const unsigned* cw = (const unsigned*)csv;
      unsigned char* prw = out_rw + gbase;
      unsigned char* pcw = out_cw + gbase;
#pragma unroll
      for (int j = 0; j < RPT; ++j) {
        const unsigned* v = dv + j * WPR;
        if constexpr (!SPLITEP) {
          const unsigned rsp = row_scale(v, j);
          unsigned ro[WPR / 2];
#pragma unroll
          for (int m = 0; m < WPR / 2; ++m)
            ro[m] = quant4(v[2 * m], v[2 * m + 1], rsp, rsp);
          st_out<WPR>(prw + (unsigned)j * (unsigned)K, ro);
        }
        unsigned co[WPR / 2];
#pragma unroll
        for (int m = 0; m < WPR / 2; ++m)
          co[m] = quant4(v[2 * m], v[2 * m + 1], cw[2 * m], cw[2 * m + 1]);
        st_out<WPR>(pcw + (unsigned)j * (unsigned)K, co);
      }
    }
    if constexpr (!SPLITEP) {
      __syncthreads();
      if constexpr (NGRP == 8) {
        if (tid < SUBR)
          *(u64*)(scale_rw + (size_t)(row0 + tid) * sc_rw_stride +
                  (col0 >> 5)) = *(const u64*)(srsb + tid * NGRP);
      } else if (tid < SUBR * (NGRP / 4)) {
        const int r = tid / (NGRP / 4);
        const int half = tid % (NGRP / 4);
        *(unsigned*)(scale_rw + (size_t)(row0 + r) * sc_rw_stride +
                     (col0 >> 5) + half * 4) =
            *(const unsigned*)(srsb + r * NGRP + half * 4);
      }
    }
    if constexpr (IS_DBIAS) {
      if (it == iters - 1) {
        // every warp holds a partial for the SAME columns over a disjoint row
        // set; folding them in warp order keeps the whole band's contribution
        // inside one partial, which is the granularity the reference reduction
        // consumes.
        u64* dst = (u64*)(&dbp_s[warp * TC + 8 * lane]);
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = dba[i];
        __syncthreads();
        if (tid < TC) {
          float s = dbp_s[tid];
#pragma unroll
          for (int w = 1; w < NWARP; ++w)
            s = __fadd_rn(s, dbp_s[w * TC + tid]);
          float* wp = dbias_ws + (size_t)blockIdx.y * K + col0 + tid;
          store_l2_evict_last(wp, s, l2_evict_last_policy());
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// One kernel; IS_DBIAS / IS_DACT / IS_ACT compile out the unused paths and pick
// the tiling, mirroring TE's quantize_mxfp8_kernel template.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fused dbias tail.  The band reduction is a separate KERNEL only because the
// bands have to be complete before it runs -- but the grid already knows when
// that happened for a given column strip: every CTA sharing a blockIdx.x writes
// one band of the SAME 256 columns, so the last of them to arrive can simply
// run the reduction itself.  That removes a whole launch from the mode-1/mode-4
// call, and the gap between two launches is 2-3 us of pure wall clock here --
// 3.6% of CAST_DBIAS at 8192x8192, where the reduction itself is under 1 us.
// The fold stays strictly left-to-right over bands, so the fp32 accumulation is
// bit-for-bit the order the reference replicates.
// ---------------------------------------------------------------------------
#ifndef FUSE_RED
#define FUSE_RED 0
#endif
// total live loads a tail thread keeps outstanding, split across its columns
#ifndef RG
#define RG 32
#endif
__device__ unsigned g_dbctr[1024];

// Batched, order-preserving band fold for the CPT columns this thread owns.
// Same shape as the standalone reducer: pull a whole register group of bands in
// before running the dependent add chain, so one strip's tail is one DRAM round
// trip per group rather than per band.
template <int CPT, int NT>
DEVI void dbias_band_fold(const float* __restrict__ ws,
                          unsigned short* __restrict__ dbias, int K, int nbands,
                          int col0, int tid) {
  // The tail runs inside a kernel whose register budget is already fixed by the
  // quantize body, so the batch depth is expressed in TOTAL live loads (RG)
  // split across the columns a thread owns, not per column.
  constexpr int G = RG / CPT;
  float acc[CPT];
#pragma unroll
  for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
  const float* p = ws + col0 + tid;
  int b = 0;
  for (; b + G <= nbands; b += G) {
    float v[CPT][G];
#pragma unroll
    for (int i = 0; i < G; ++i)
#pragma unroll
      for (int c = 0; c < CPT; ++c) v[c][i] = p[(size_t)(b + i) * K + c * NT];
#pragma unroll
    for (int i = 0; i < G; ++i)
#pragma unroll
      for (int c = 0; c < CPT; ++c) acc[c] = __fadd_rn(acc[c], v[c][i]);
  }
  for (; b < nbands; ++b)
#pragma unroll
    for (int c = 0; c < CPT; ++c)
      acc[c] = __fadd_rn(acc[c], p[(size_t)b * K + c * NT]);
#pragma unroll
  for (int c = 0; c < CPT; ++c) {
    unsigned short o;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(acc[c]));
    dbias[col0 + tid + c * NT] = o;
  }
}

// Release the band partial, then claim the strip.  Exactly one CTA per
// blockIdx.x sees the full count and runs the tail; it also clears the counter
// so the next call starts from zero.
template <int TCW, int NT>
DEVI void dbias_tail(const float* __restrict__ ws,
                     unsigned short* __restrict__ dbias, int K, int nbands,
                     int col0, int tid) {
  __shared__ unsigned islast;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    islast = (atomicAdd(&g_dbctr[blockIdx.x], 1u) == (unsigned)(nbands - 1));
    if (islast) g_dbctr[blockIdx.x] = 0u;
  }
  __syncthreads();
  if (!islast) return;
  __threadfence();
  dbias_band_fold<TCW / NT, NT>(ws, dbias, K, nbands, col0, tid);
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
__global__ void __launch_bounds__(Cfg<IS_DBIAS, IS_DACT, IS_ACT>::NTHRC,
                                  Cfg<IS_DBIAS, IS_DACT, IS_ACT>::MINB_C)
quantize_mxfp8_kernel(
    const unsigned* __restrict__ input, const unsigned* __restrict__ act_input,
    unsigned char* __restrict__ out_rw, unsigned char* __restrict__ scale_rw,
    unsigned char* __restrict__ out_cw, unsigned char* __restrict__ scale_cw,
    float* __restrict__ dbias_ws, unsigned short* __restrict__ dbias_out, int K,
    int sc_rw_stride, int sc_cw_stride, int iters) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if constexpr (C::REGTILE)
    quantize_regtile<IS_DBIAS, IS_DACT, IS_ACT>(
        (const uint4*)input, (const uint4*)act_input, out_rw, scale_rw, out_cw,
        scale_cw, dbias_ws, K, sc_rw_stride, sc_cw_stride, iters);
  else
    quantize_streaming<IS_DBIAS, IS_DACT, IS_ACT>(
        input, act_input, out_rw, scale_rw, out_cw, scale_cw, dbias_ws, K,
        sc_rw_stride, sc_cw_stride, iters);
  constexpr bool FUSED = IS_DBIAS && (FUSE_RED == 1 ||
                         (FUSE_RED == 2 && C::REGTILE) ||
                         (FUSE_RED == 3 && !C::REGTILE));
  if constexpr (FUSED)
    dbias_tail<C::TC, C::NTHRC>(dbias_ws, dbias_out, K, (int)gridDim.y,
                                (int)blockIdx.x * C::TC, (int)threadIdx.x);
}

// ---------------------------------------------------------------------------
// dbias band reduction, sequential over bands (matches the reference order)
// ---------------------------------------------------------------------------
// The reducer has exactly one thread per column -- at K = 8192 that is 8192
// threads for the whole GPU -- so it lives or dies on loads in flight per
// thread, not on occupancy: with only K/64 CTAs the SMs are never full anyway.
// Widening the register batch from 32 to RDB bands puts 4x the read stream in
// flight and turns nbands = 64 (every CAST_DBIAS_DACT shape) into a SINGLE
// dependent DRAM round trip.
#ifndef RDB
#define RDB 64
#endif
__global__ void __launch_bounds__(64, 4) reduce_dbias_kernel(const float* __restrict__ ws,
                                    unsigned short* __restrict__ dbias, int K,
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
  if (nbands >= RDB) {
    float v0[RDB], v1[RDB];
#pragma unroll
    for (int i = 0; i < RDB; ++i) v0[i] = p[(size_t)i * K];
    b = RDB;
    // Keep the next RDB-band load group outstanding while the previous group
    // runs through its exact left-to-right dependent add chain.
    for (; b + 2 * RDB <= nbands; b += 2 * RDB) {
#pragma unroll
      for (int i = 0; i < RDB; ++i) v1[i] = p[(size_t)(b + i) * K];
#pragma unroll
      for (int i = 0; i < RDB; ++i) acc = __fadd_rn(acc, v0[i]);
#pragma unroll
      for (int i = 0; i < RDB; ++i) v0[i] = p[(size_t)(b + RDB + i) * K];
#pragma unroll
      for (int i = 0; i < RDB; ++i) acc = __fadd_rn(acc, v1[i]);
    }
    if (b + RDB <= nbands) {
#pragma unroll
      for (int i = 0; i < RDB; ++i) v1[i] = p[(size_t)(b + i) * K];
#pragma unroll
      for (int i = 0; i < RDB; ++i) acc = __fadd_rn(acc, v0[i]);
#pragma unroll
      for (int i = 0; i < RDB; ++i) acc = __fadd_rn(acc, v1[i]);
      b += RDB;
    } else {
#pragma unroll
      for (int i = 0; i < RDB; ++i) acc = __fadd_rn(acc, v0[i]);
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
#ifndef DACT_CAP
#define DACT_CAP 64
#endif
#ifndef ACT_FLOOR
#define ACT_FLOOR 1
#endif
#ifndef DB_FLOOR
#define DB_FLOOR 4
#endif
#ifndef CTA_TARGET
#define CTA_TARGET 8192
#endif
// CAST_ACT's table is half the size of the dGeLU one, so its walk can afford to
// be shorter than the shared 8192-CTA target: the extra staging traffic buys a
// grid deep enough that the activation work of the trailing partial residency
// wave stops showing up as tail.
#ifndef ACT_TARGET
#define ACT_TARGET 16384
#endif
// CAST_DACT gets its own target too: its table is twice as big, so the staging
// it pays per extra CTA is twice CAST_ACT's, but its tile is a 256-thread one at
// six blocks/SM, so it already starts from a deeper grid.
#ifndef DACT_TARGET
#define DACT_TARGET 16384
#endif
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
#ifndef DB_TARGET
#define DB_TARGET 4096
#endif
// The two-row-block floor exists to amortise the table staging; on the shortest
// grids that trade runs the other way, so it is a per-instantiation knob.
#ifndef ACT_MIN
#define ACT_MIN 2
#endif
static int pick_iters_t(int rowblocks, int gridx, long long target,
                        int minit = 2) {
  int it = 1;
  while ((rowblocks / it) % 2 == 0 && (long long)gridx * (rowblocks / it) > target)
    it *= 2;
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
#ifndef CARVE_A
#define CARVE_A 40
#endif
#ifndef CARVE_C
#define CARVE_C 40
#endif
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
static void set_carveout() {
  constexpr int PCT = (!IS_DBIAS && !IS_DACT && !IS_ACT)
                          ? CARVE_C
                          : ((IS_ACT && !IS_DACT && !IS_DBIAS) ? CARVE_A : 0);
  if constexpr (PCT > 0) {
    static bool done = false;
    if (!done) {
      done = true;
      cudaFuncSetAttribute(
          (const void*)quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT>,
          cudaFuncAttributePreferredSharedMemoryCarveout, PCT);
    }
  }
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
static void launch(const void* prim, const void* actin, void* orw, void* srw,
                   void* ocw, void* scw, float* ws, void* dbo, int M, int K,
                   int rws, int cws, cudaStream_t st) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  set_carveout<IS_DBIAS, IS_DACT, IS_ACT>();
  constexpr int RB = C::REGTILE ? 32 : 64;
  const int gx = K / C::TC;
  const int rb = M / RB;
  constexpr bool PACT = IS_ACT && !IS_DACT && !IS_DBIAS;
  constexpr long long TGT =
      PACT ? (long long)ACT_TARGET
           : ((IS_DACT && !IS_DBIAS) ? (long long)DACT_TARGET
                                     : (long long)DB_TARGET);
  int iters = (IS_ACT || IS_DACT || (IS_DBIAS && C::REGTILE))
                  ? pick_iters_t(rb, gx, TGT, PACT ? ACT_MIN : 2)
                  : 1;
  // Per-instantiation walk length.  The activation table is copied once per
  // CTA, so a longer walk amortises it -- but a longer walk also spreads one
  // CTA's stores over more rows, costing streaming locality.  The two effects
  // balance at a different point for each instantiation: dgelu's 16 KB table
  // reaches break-even sooner than gelu's, and the dbias instantiation wants
  // FEWER, LONGER walks because the band count (hence the workspace round trip
  // and the reducer's dependent add chain) shrinks with it.
  if constexpr (IS_DACT && !IS_DBIAS) {
    if (iters > DACT_CAP) iters = DACT_CAP;
  }
  // Every activation CTA copies the whole table into shared once, so at the
  // small square -- where pick_iters bottoms out at a two-block walk -- that
  // one-off is 16-24% of the workload's entire DRAM traffic.  A four-block
  // floor halves it and still leaves 2048 CTAs, more than two full residency
  // waves, which is what the tail needs.
  if constexpr ((IS_ACT || IS_DACT) && !IS_DBIAS) {
    if (iters < ACT_FLOOR && rb % ACT_FLOOR == 0) iters = ACT_FLOOR;
  }
  if constexpr (IS_DBIAS && C::REGTILE) {
    if (iters < DB_FLOOR && rb % DB_FLOOR == 0) iters = DB_FLOOR;
  }
  dim3 grid(gx, rb / iters);
  quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT>
      <<<grid, Cfg<IS_DBIAS, IS_DACT, IS_ACT>::NTHRC, 0, st>>>(
      (const unsigned*)prim, (const unsigned*)actin, (unsigned char*)orw,
      (unsigned char*)srw, (unsigned char*)ocw, (unsigned char*)scw, ws,
      (unsigned short*)dbo, K, rws, cws, iters);
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
// kernel. Per translation unit, not per process: d_gelu_tab / d_dgelu_tab are
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
static bool &regtile_tables_ready() {
  static bool ready = false;
  return ready;
}

static void ensure_act_tables(cudaStream_t stream) {
  if (!regtile_tables_ready()) {
    init_act_tables_kernel<<<(2 * LUT_N + 255) / 256, 256, 0, stream>>>();
    NVTE_CHECK_CUDA(cudaGetLastError());
    regtile_tables_ready() = true;
  }
}

// Rows folded into one dbias workspace band. Must agree exactly with the
// grid the kernel is launched on, since it sizes the workspace.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
inline int regtile_dbias_bands(int M, int K) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if (!C::REGTILE) return M / 64;  // streaming tile: one 64-row band per CTA
  const int rb = M / SUBR;
  int it = pick_iters_t(rb, K / TCOLS, DB_TARGET);
  if (it < DB_FLOOR && rb % DB_FLOOR == 0) it = DB_FLOOR;
  return rb / it;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
inline bool regtile_fuses_reduction() {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  return FUSE_RED == 1 || (FUSE_RED == 2 && C::REGTILE) ||
         (FUSE_RED == 3 && !C::REGTILE);
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

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
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

template <bool IS_ACT, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
struct RegtileOpSupported {
  static constexpr RegtileOp kKind = RegtileOpKind<ParamOP, OP>::value;
  static constexpr bool value = (!IS_ACT && !IS_DACT) ||
                                (IS_ACT && !IS_DACT && kKind == RegtileOp::kGelu) ||
                                (IS_DACT && !IS_ACT && kKind == RegtileOp::kDgelu);
};

// Shape gate. The kernel derives its grid by exact integer division, so a
// remainder in either dimension would silently drop the tail.
//   cols % TCOLS : grid.x  = K / TCOLS
//   rows % 64    : covers both tile variants (regtile 32, streaming 64)
// pick_iters_t only ever returns a power of two that divides rows/tile, so
// grid.y = rb / iters needs no separate check.
inline bool regtile_shape_supported(size_t rows, size_t cols) {
  return (cols % TCOLS == 0) && (rows % 64 == 0) && rows > 0 && cols > 0;
}

// Host entry. Mirrors the campaign harness's launch<>() plus its optional
// separate dbias reduction pass.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
void launch_regtile(const void *input, const void *act_input, void *out_rowwise,
                    void *scale_rowwise, void *out_colwise, void *scale_colwise,
                    float *workspace_ptr, void *dbias_ptr, int M, int K,
                    int scale_stride_rowwise, int scale_stride_colwise,
                    cudaStream_t stream) {
  using C = Cfg<IS_DBIAS, IS_DACT, IS_ACT>;
  if constexpr (IS_ACT || IS_DACT) {
    ensure_act_tables(stream);
  }

  // The register-resident tile quantizes out of the ACT_INPUT slot, so
  // CAST_DBIAS must hand it the primary (grad) pointer there; x is unused by
  // that mode. Matches the campaign harness's mode-1 argument wiring.
  const void *act_slot = (IS_DBIAS && !IS_DACT && C::REGTILE) ? input : act_input;

  launch<IS_DBIAS, IS_DACT, IS_ACT>(input, act_slot, out_rowwise, scale_rowwise,
                                    out_colwise, scale_colwise, workspace_ptr,
                                    dbias_ptr, M, K, scale_stride_rowwise,
                                    scale_stride_colwise, stream);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if constexpr (IS_DBIAS) {
    if (!regtile_fuses_reduction<IS_DBIAS, IS_DACT, IS_ACT>()) {
      const int nbands = regtile_dbias_bands<IS_DBIAS, IS_DACT, IS_ACT>(M, K);
      constexpr int threads = 64;  // narrow CTAs so the column fan-out reaches every SM
      reduce_dbias_kernel<<<(K + threads - 1) / threads, threads, 0, stream>>>(
          workspace_ptr, reinterpret_cast<unsigned short *>(dbias_ptr), K, nbands);
      NVTE_CHECK_CUDA(cudaGetLastError());
    }
  }
}

}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine


// The vendored kernel carries ~80 tuning macros. They are all consumed by the
// definitions above, and macros ignore namespaces, so drop them here rather
// than leaking them into every translation unit that includes this header.
// The build can still override any of them: each is #ifndef-guarded and this
// #undef runs after the last use.
#undef TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
#undef DEVI
#undef STRIDED_WALK
#undef GELU_FAST
#undef UAF
#undef GELU_K1
#undef GELU_K0
#undef DGELU_C1
#undef DGELU_C0
#undef DGELU_FAST
#undef LUT_LO
#undef LUT_HI
#undef LUT_N
#undef SWIZLUT
#undef LUT_SPAN_H
#undef LUT_SPAN_F
#undef HSHF
#undef HMSK
#undef HMUL
#undef HGSP
#undef LUT_LO2
#undef LUT_HI2
#undef LUT_BIAS2
#undef LUT_REP
#undef LUT_H_BIAS
#undef LUT_OFF_F
#undef LUT_OFF_H
#undef GSPLAT_F
#undef GSPLAT_H
#undef LUT_H_SHIFT
#undef LUTQ_A
#undef LUTQ_D
#undef LUTQ_DB
#undef CH_DB
#undef LUTM_A
#undef LUTM_D
#undef LUTM_DB
#undef NARROW_A
#undef WIDE_A
#undef MINBW
#undef CHW
#undef DB_REGTILE
#undef CH_DB1
#undef NSPLIT
#undef MINBRB1
#undef ACT_CH2
#undef LUTQ_A2
#undef LUTQ_AW
#undef ACT_HI
#undef ACT_ROWM
#undef UNA
#undef MINBA
#undef MINB
#undef MINBD
#undef MINBR
#undef MINBRD
#undef MINBRB
#undef FOLDP
#undef DHOIST_DB
#undef DHOIST_EN
#undef HIDEB_EN
#undef MXFP8_ROW
#undef MXFP8_AWORD
#undef MXFP8_AFIX
#undef MXFP8_AROW
#undef MXFP8_AROWFIX
#undef MXFP8_WORD
#undef MXFP8_LUT
#undef MXFP8_DFIX
#undef MXFP8_FIX
#undef MXFP8_TROW
#undef FUSE_RED
#undef RG
#undef RDB
#undef DACT_CAP
#undef ACT_FLOOR
#undef DB_FLOOR
#undef CTA_TARGET
#undef ACT_TARGET
#undef DACT_TARGET
#undef DB_TARGET
#undef ACT_MIN
#undef CARVE_A
#undef CARVE_C

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
