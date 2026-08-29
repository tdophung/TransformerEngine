// Bidimensional MXFP8 quantization kernel (BF16 → FP8E4M3, rowwise + colwise).
//
// Ported from KF campaign 72y0hf474s4z310akncqt5exn4, candidate
// 7fe40f55cd8b369fc5717577ec4b27f960409ef061ec3fac0138495fc8b28224 (round 11),
// 1.3456× speedup over TE's TMA-based kernel on B200 (45.117µs → 33.528µs geomean).
//
// Key architectural differences from the replaced TMA kernel:
//   - No TMA/cp.async.bulk: register-resident 32×512 (wide) or 32×256 (narrow) tiles
//   - 256 threads/CTA (8 warps × 32) vs. 64 threads (2 warps)
//   - Plain 256-bit vectorized loads (ld.global.nc.L2::cache_hint.v8.b32)
//   - Packed BF16 PTX throughout (max.xorsign.abs.bf16x2, mul.rn.bf16x2, cvt.*)
//   - Uniform L2::evict_last on all three streams (input + qrow + qcol outputs)
//   - Software L2 prefetch one resident CTA-wave ahead (CFG_PF = 148 × MINB)
//   - Shape-selective thread-block clusters via cudaLaunchKernelEx
//
// See OPTIMIZATION_LOG.md in campaign-bidim-te/ for the full round-by-round
// technique breakdown and known dead ends.

#include "mxfp8/kf_bidim/kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---- compile-time config (winning candidate values) -------------------------
// Warps per CTA (shared by wide and narrow tiles when CFG_WNW == CFG_NNW).
#ifndef CFG_NW
#define CFG_NW 8
#endif
// Minimum blocks/SM for the wide tile (__launch_bounds__ occupancy target).
#ifndef CFG_MINB
#define CFG_MINB 4
#endif
// Prefetch distance in CTAs (must equal resident CTA count: SMs × MINB).
#ifndef CFG_PF
#define CFG_PF 592
#endif
// Maximum prefetch lines issued per CTA.
#ifndef CFG_PFN
#define CFG_PFN 1024
#endif
// 1 = prefetched cache lines carry L2::evict_last (matches demand-load policy).
#ifndef CFG_PFPOL
#define CFG_PFPOL 1
#endif
// 1 = warp-0 regenerates the full scol row as one STG.128/STG.64 per lane.
#ifndef CFG_SCOL1
#define CFG_SCOL1 1
#endif
// Shared-memory column-fold layout.  1 = half-split (conflict-free STG.128/LDS.128).
#ifndef CFG_SLAY
#define CFG_SLAY 1
#endif
// 1 = srow (b8) and scol stores carry the same L2::cache_hint evict_last policy.
#ifndef CFG_SHINT
#define CFG_SHINT 1
#endif
// Columns owned by one lane: 16 → 32×512 tile (256-bit loads), 8 → 32×256 tile.
#ifndef CFG_CPL
#define CFG_CPL 16
#endif
// L2 policy for the two large output streams (qrow/qcol).
//   0 = evict_last (same as input) — winning policy confirmed in round 11.
#ifndef CFG_STPOL
#define CFG_STPOL 0
#endif
// Sub-tiles per CTA (software pipeline depth).  1 = no pipelining.
#ifndef CFG_TPC
#define CFG_TPC 1
#endif
// 0 = prefetch from the trailing write burst (winning timing — see OPTIMIZATION_LOG.md).
// 1 = right after own loads (confirmed worse, do not change).
#ifndef CFG_PFPOS
#define CFG_PFPOS 0
#endif
// 1 = compile out the software L2 prefetch entirely.
#ifndef CFG_NOPF
#define CFG_NOPF 0
#endif

// Per-shape warp/occupancy config (from campaign winning candidate).
#ifndef CFG_WNW
#define CFG_WNW 8
#endif
#ifndef CFG_WMINB
#define CFG_WMINB 4
#endif
#ifndef CFG_NNW
#define CFG_NNW 8
#endif
#ifndef CFG_NMINB
#define CFG_NMINB 6
#endif

// Cluster widths for the wide tile: shallow grids vs. deep grids.
#ifndef CFG_WCLS
#define CFG_WCLS 1
#endif
#ifndef CFG_WCLD
#define CFG_WCLD 4
#endif
// Cluster width for the narrow tile.
#ifndef CFG_NCL
#define CFG_NCL 8
#endif
// Narrow tile switch threshold in wide-tile waves.
#ifndef CFG_NFW
#define CFG_NFW 8
#endif

// Resident CTA counts: SM count × MINB (B200 = 148 SMs).
#define MXFP8_WIDE_RESIDENT   (148 * CFG_WMINB)
#define MXFP8_NARROW_RESIDENT (148 * CFG_NMINB)
#define MXFP8_NARROW_FROM_WAVES CFG_NFW

namespace transformer_engine {
namespace kf_bidim {

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

// Magnitude-max of two bf16x2 (sign of result is xor of input signs, ignored).
__device__ __forceinline__ uint32_t maxabs2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Scale two bf16x2 pairs and emit the four packed e4m3 bytes.
__device__ __forceinline__ uint32_t scale_cvt4(uint32_t p0, uint32_t s0,
                                               uint32_t p1, uint32_t s1) {
    uint32_t d;
    asm("{\n\t.reg .b32 t0, t1;\n\t.reg .b16 lo, hi;\n\t"
        "mul.rn.bf16x2 t0, %1, %2;\n\t"
        "mul.rn.bf16x2 t1, %3, %4;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 lo, t0;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 hi, t1;\n\t"
        "mov.b32 %0, {lo, hi};\n\t}"
        : "=r"(d) : "r"(p0), "r"(s0), "r"(p1), "r"(s1));
    return d;
}

// E8M0 biased exponent from a 15-bit |bf16| magnitude pattern.
__device__ __forceinline__ uint32_t e8m0_from_absbf16(uint32_t a) {
    int rr = (int)((a + 31u) >> 7) - 8;
    uint32_t r = rr < 0 ? 0u : (uint32_t)rr;
    return (a >= 0x7F80u) ? 254u : r;
}

// encode_scale = 2^(127 - r) as bf16 bits (r == 254 → subnormal 0x0040).
__device__ __forceinline__ uint32_t e8m0_scale_bf16(uint32_t r) {
    uint32_t s = (254u - r) << 7;
    return s ? s : 0x0040u;
}

__device__ __forceinline__ uint64_t pol_store() {
    uint64_t p;
#if CFG_STPOL == 1
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
#elif CFG_STPOL == 2
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
#else
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
#endif
    return p;
}

__device__ __forceinline__ uint64_t pol_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// 256-bit (v8.b32) global load through the non-coherent cache.
__device__ __forceinline__ void ldg256(const void* p,
                                       uint32_t& a0, uint32_t& a1,
                                       uint32_t& a2, uint32_t& a3,
                                       uint32_t& a4, uint32_t& a5,
                                       uint32_t& a6, uint32_t& a7,
                                       uint64_t lpol) {
    asm("ld.global.nc.L2::cache_hint.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3),
          "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7)
        : "l"(p), "l"(lpol));
}

// 128-bit (v4.b32) global load.
__device__ __forceinline__ void ldg128(const void* p,
                                       uint32_t& a0, uint32_t& a1,
                                       uint32_t& a2, uint32_t& a3,
                                       uint64_t lpol) {
    asm("ld.global.nc.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(p), "l"(lpol));
}

// L2 prefetch (one 128-byte cache line).
__device__ __forceinline__ void pf_l2(const void* p) {
#if CFG_PFPOL == 1
    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
#else
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
}

__device__ __forceinline__ void stg128(void* p, uint4 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
}

__device__ __forceinline__ void stg64(void* p, uint2 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1,%2}, %3;" ::"l"(p),
                 "r"(v.x), "r"(v.y), "l"(pol) : "memory");
}

__device__ __forceinline__ void stg8(void* p, uint32_t v, uint64_t pol) {
#if CFG_SHINT == 1
    unsigned short h = (unsigned short)v;
    asm volatile("st.global.L2::cache_hint.b8 [%0], %1, %2;" ::"l"(p),
                 "h"(h), "l"(pol) : "memory");
#else
    (void)pol;
    *reinterpret_cast<uint8_t*>(p) = (uint8_t)v;
#endif
}

__device__ __forceinline__ void stg16(void* p, uint32_t v, uint64_t pol) {
#if CFG_SHINT == 1
    unsigned short h = (unsigned short)v;
    asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;" ::"l"(p),
                 "h"(h), "l"(pol) : "memory");
#else
    (void)pol;
    *reinterpret_cast<uint16_t*>(p) = (uint16_t)v;
#endif
}

// ---------------------------------------------------------------------------
// Primary kernel: 32×(TPC×32×CPL) tile, NW warps, MINB min blocks/SM.
//   CPL=16 → 32×512 wide tile (256-bit loads), CPL=8 → 32×256 narrow tile.
//   KC=0 → K passed at runtime; KC!=0 → K is a compile-time constant.
// ---------------------------------------------------------------------------
template <int NW, int KC, int MINB, int CPL, int TPC, int PF>
__global__ __launch_bounds__(32 * NW, MINB) void mxfp8_bidim_wide(
    const uint8_t* __restrict__ x,
    uint8_t* __restrict__ qrow,
    uint8_t* __restrict__ srow,
    uint8_t* __restrict__ qcol,
    uint8_t* __restrict__ scol,
    int K_rt, int srow_stride_rt, int scol_stride_rt) {

    const int K           = KC ? KC : K_rt;
    const int srow_stride = KC ? (((KC / 32) + 3) / 4) * 4 : srow_stride_rt;
    const int scol_stride = KC ? ((KC + 127) / 128) * 128  : scol_stride_rt;

    constexpr int R     = 32 / NW;
    constexpr int NV    = CPL / 2;
    constexpr int NCOL  = 32 * CPL;
    constexpr int SPAN  = NCOL * TPC;
    constexpr int NSLOT = NCOL / 2;
    constexpr int HALF  = NSLOT / 2;
    constexpr int QUAD  = NV / 2;
    constexpr int LPG   = 32 / CPL;
    constexpr int NT    = 32 * NW;

    // SLAY=2: pad 4 uint32 every 32 slots for conflict-free STS/LDS.
    constexpr int PADQ  = (CFG_SLAY == 2) ? 4 : 0;
    constexpr int NSLOTP = NSLOT + (NSLOT / 32) * PADQ;
#define PIDX(j) ((j) + PADQ * ((j) >> 5))

    // Half-split layout: lane l, slot k lives at index HALF*(k/QUAD) + QUAD*l + (k%QUAD).
    __shared__ uint32_t s_part[NW][NSLOTP];
    __shared__ uint32_t s_scale[NSLOTP];

    const int tid = threadIdx.x;
    const int w   = tid >> 5;
    const int l   = tid & 31;

    const int row0 = blockIdx.y * 32 + w * R;
    const int col0 = blockIdx.x * SPAN;

    uint32_t v[TPC][R][NV];
    const uint64_t pol = pol_evict_last();
#if CFG_STPOL == 0
    const uint64_t spol = pol;
#else
    const uint64_t spol = pol_store();
#endif

    const uint8_t* p   = x + ((size_t)row0 * K + col0 + CPL * l) * 2;
    const size_t rstep = (size_t)K * 2;

    const size_t obase = (size_t)row0 * K + col0 + CPL * l;
    uint8_t* qr = qrow + obase;
    uint8_t* qc = qcol + obase;
    uint8_t* sr = srow + (size_t)row0 * srow_stride + (col0 >> 5) + (l / LPG);
    const bool srlane = (l % LPG) == 0;

#define MXFP8_LOAD(S)                                                                   \
    _Pragma("unroll") for (int i = 0; i < R; ++i) {                                    \
        if constexpr (CPL == 16)                                                        \
            ldg256(p + (size_t)(S) * NCOL * 2 + i * rstep,                             \
                   v[S][i][0], v[S][i][1], v[S][i][2], v[S][i][3],                     \
                   v[S][i][4 % NV], v[S][i][5 % NV],                                   \
                   v[S][i][6 % NV], v[S][i][7 % NV], pol);                             \
        else                                                                            \
            ldg128(p + (size_t)(S) * NCOL * 2 + i * rstep,                             \
                   v[S][i][0], v[S][i][1], v[S][i][2], v[S][i][3], pol);               \
    }

    // Software L2 prefetch: one resident CTA-wave ahead from this CTA's own span.
    // CFG_PFPOS=0: issued from the trailing write burst (winning timing — keeps
    // next tile L2-resident without holding it alive across the full CTA lifetime).
    // CFG_PFPOS=1: right after own loads — confirmed worse; do not change.
#if CFG_NOPF
#define MXFP8_PREFETCH {}
#else
#define MXFP8_PREFETCH                                                                   \
    {                                                                                    \
        const int gx = KC ? (KC / SPAN) : (int)gridDim.x;                               \
        const int lin = blockIdx.y * gx + blockIdx.x + PF;                              \
        constexpr int PFL = 32 * SPAN * 2 / 128;                                        \
        constexpr int PFC = (PFL < CFG_PFN ? PFL : CFG_PFN);                            \
        if (lin < gx * (int)gridDim.y) {                                                 \
            const int tby = lin / gx;                                                    \
            const int tbx = lin - tby * gx;                                              \
            _Pragma("unroll") for (int pt = tid; pt < PFC; pt += NT) {                  \
                const int prow = pt / (PFL / 32);                                        \
                const int poff = (pt % (PFL / 32)) << 7;                                 \
                pf_l2(x + ((size_t)(tby * 32 + prow) * K + tbx * SPAN) * 2 + poff);    \
            }                                                                            \
        }                                                                                \
    }
#endif

    MXFP8_LOAD(0)
#if CFG_PFPOS == 1
    MXFP8_PREFETCH
#endif

#pragma unroll
    for (int s = 0; s < TPC; ++s) {
        const int coff = s * NCOL;
        uint32_t pc[NV];

        // ---- rowwise pass ---------------------------------------------------
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uint32_t m = v[s][i][0];
#pragma unroll
            for (int k = 1; k < NV; ++k) m = maxabs2(m, v[s][i][k]);
#pragma unroll
            for (int d = 1; d < LPG; d <<= 1)
                m = maxabs2(m, __shfl_xor_sync(0xFFFFFFFFu, m, d));
            m = maxabs2(m, __byte_perm(m, m, 0x1032));

            const uint32_t am = m & 0x7FFFu;
            const uint32_t rs =
                am >= 0x7F80u ? 0x0040u
                              : (0x8300u - max((am + 31u) & 0x7F80u, 0x0400u));
            const uint32_t rsx = __byte_perm(rs, 0u, 0x1010);

            if constexpr (CPL == 16)
                stg128(qr + coff + (size_t)i * K,
                       make_uint4(scale_cvt4(v[s][i][0], rsx, v[s][i][1], rsx),
                                  scale_cvt4(v[s][i][2], rsx, v[s][i][3], rsx),
                                  scale_cvt4(v[s][i][4 % NV], rsx, v[s][i][5 % NV], rsx),
                                  scale_cvt4(v[s][i][6 % NV], rsx, v[s][i][7 % NV], rsx)),
                       spol);
            else
                stg64(qr + coff + (size_t)i * K,
                      make_uint2(scale_cvt4(v[s][i][0], rsx, v[s][i][1], rsx),
                                 scale_cvt4(v[s][i][2], rsx, v[s][i][3], rsx)),
                      spol);

            if (srlane)
                stg8(sr + (coff >> 5) + (size_t)i * srow_stride,
                     254u - (rs >> 7), spol);
        }

        // ---- pipeline: next sub-tile's loads go out after first sub-tile ----
        if constexpr (TPC == 2) { if (s == 0) { MXFP8_LOAD(1) } }

        // ---- column partial-max fold into shared memory ----------------------
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            uint32_t a = v[s][0][k];
#pragma unroll
            for (int i = 1; i < R; ++i) a = maxabs2(a, v[s][i][k]);
            pc[k] = a;
        }

#if CFG_SLAY == 1
        if constexpr (QUAD == 4) {
            *reinterpret_cast<uint4*>(&s_part[w][QUAD * l]) =
                make_uint4(pc[0], pc[1], pc[2], pc[3 % NV]);
            *reinterpret_cast<uint4*>(&s_part[w][HALF + QUAD * l]) =
                make_uint4(pc[4 % NV], pc[5 % NV], pc[6 % NV], pc[7 % NV]);
        } else {
            *reinterpret_cast<uint2*>(&s_part[w][QUAD * l]) =
                make_uint2(pc[0], pc[1]);
            *reinterpret_cast<uint2*>(&s_part[w][HALF + QUAD * l]) =
                make_uint2(pc[2 % NV], pc[3 % NV]);
        }
#else
        *reinterpret_cast<uint4*>(&s_part[w][PIDX(NV * l)]) =
            make_uint4(pc[0], pc[1], pc[2 % NV], pc[3 % NV]);
        if constexpr (NV == 8)
            *reinterpret_cast<uint4*>(&s_part[w][PIDX(NV * l) + 4]) =
                make_uint4(pc[4 % NV], pc[5 % NV], pc[6 % NV], pc[7 % NV]);
#endif
        __syncthreads();

        // ---- column fold: one stride-1 slot per thread, conflict-free --------
#pragma unroll
        for (int fs = tid; fs < NSLOT; fs += NT) {
            const uint32_t* sp = &s_part[0][0] + PIDX(fs);
            uint32_t a = sp[0];
#pragma unroll
            for (int ww = 1; ww < NW; ++ww) a = maxabs2(a, sp[ww * NSLOTP]);

            uint32_t sc;
            if (__builtin_expect(
                    (maxabs2(a, __byte_perm(a, a, 0x1032)) & 0x7FFFu) < 0x7F80u, 1)) {
                uint32_t t = a & 0x7FFF7FFFu;
                uint32_t y = maxabs2((t + 0x001F001Fu) & 0xFF80FF80u, 0x04000400u);
                sc = 0x83008300u - y;
            } else {
                uint32_t r0 = e8m0_from_absbf16(a & 0x7FFFu);
                uint32_t r1 = e8m0_from_absbf16((a >> 16) & 0x7FFFu);
                sc = e8m0_scale_bf16(r0) | (e8m0_scale_bf16(r1) << 16);
            }
            s_scale[PIDX(fs)] = sc;

#if CFG_SLAY != 1
            uint32_t rp = 0x00FE00FEu - ((sc >> 7) & 0x00FF00FFu);
            stg16(scol + (size_t)blockIdx.y * scol_stride + col0 + coff + 2 * fs,
                  __byte_perm(rp, 0u, 0x4420), spol);
#endif
        }
        __syncthreads();

        // ---- read back column scales for qcol stores ------------------------
        uint32_t cs[NV];
#if CFG_SLAY == 1
        if constexpr (QUAD == 4) {
            uint4 a = *reinterpret_cast<const uint4*>(&s_scale[QUAD * l]);
            uint4 b = *reinterpret_cast<const uint4*>(&s_scale[HALF + QUAD * l]);
            cs[0] = a.x; cs[1] = a.y; cs[2] = a.z; cs[3 % NV] = a.w;
            cs[4 % NV] = b.x; cs[5 % NV] = b.y; cs[6 % NV] = b.z; cs[7 % NV] = b.w;
        } else {
            uint2 a = *reinterpret_cast<const uint2*>(&s_scale[QUAD * l]);
            uint2 b = *reinterpret_cast<const uint2*>(&s_scale[HALF + QUAD * l]);
            cs[0] = a.x; cs[1] = a.y; cs[2 % NV] = b.x; cs[3 % NV] = b.y;
        }
#else
        {
            uint4 a = *reinterpret_cast<const uint4*>(&s_scale[PIDX(NV * l)]);
            cs[0] = a.x; cs[1] = a.y; cs[2 % NV] = a.z; cs[3 % NV] = a.w;
            if constexpr (NV == 8) {
                uint4 b = *reinterpret_cast<const uint4*>(&s_scale[PIDX(NV * l) + 4]);
                cs[4 % NV] = b.x; cs[5 % NV] = b.y; cs[6 % NV] = b.z; cs[7 % NV] = b.w;
            }
        }
#endif

        // ---- SLAY=1: warp-0 emits one coalesced scol row per lane -----------
#if CFG_SLAY == 1
        if (w == 0) {
            uint32_t bb[NV / 2];
#pragma unroll
            for (int j = 0; j < NV / 2; ++j) {
                uint32_t r0 = 0x00FE00FEu - ((cs[2 * j]     >> 7) & 0x00FF00FFu);
                uint32_t r1 = 0x00FE00FEu - ((cs[2 * j + 1] >> 7) & 0x00FF00FFu);
                bb[j] = __byte_perm(r0, 0u, 0x4420) |
                        (__byte_perm(r1, 0u, 0x4420) << 16);
            }
            uint8_t* sc0 = scol + (size_t)blockIdx.y * scol_stride
                           + col0 + coff + CPL * l;
            if constexpr (QUAD == 4)
                stg128(sc0, make_uint4(bb[0], bb[1],
                                       bb[2 % (NV / 2)], bb[3 % (NV / 2)]), spol);
            else
                stg64(sc0, make_uint2(bb[0], bb[1]), spol);
        }
#endif

#if CFG_PFPOS == 0
        if (s == TPC - 1) { MXFP8_PREFETCH }
#endif

        // ---- colwise quantized output ----------------------------------------
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if constexpr (CPL == 16)
                stg128(qc + coff + (size_t)i * K,
                       make_uint4(
                           scale_cvt4(v[s][i][0], cs[0], v[s][i][1], cs[1]),
                           scale_cvt4(v[s][i][2], cs[2], v[s][i][3], cs[3]),
                           scale_cvt4(v[s][i][4 % NV], cs[4 % NV],
                                      v[s][i][5 % NV], cs[5 % NV]),
                           scale_cvt4(v[s][i][6 % NV], cs[6 % NV],
                                      v[s][i][7 % NV], cs[7 % NV])),
                       spol);
            else
                stg64(qc + coff + (size_t)i * K,
                      make_uint2(
                          scale_cvt4(v[s][i][0], cs[0], v[s][i][1], cs[1]),
                          scale_cvt4(v[s][i][2], cs[2], v[s][i][3], cs[3])),
                      spol);
        }
    }
#undef MXFP8_LOAD
#undef MXFP8_PREFETCH
#undef PIDX
    (void)NT;
}

// ---------------------------------------------------------------------------
// Generic fallback kernel: 32×256 tile, CPL=8, no compile-time K specialization.
// Used when K is not a multiple of 512 or 256 that the wide/narrow dispatch covers.
// ---------------------------------------------------------------------------
template <int NW>
__global__ __launch_bounds__(32 * NW) void mxfp8_bidim_kernel(
    const uint4* __restrict__ x,
    uint8_t* __restrict__ qrow,
    uint8_t* __restrict__ srow,
    uint8_t* __restrict__ qcol,
    uint8_t* __restrict__ scol,
    int K, int srow_stride, int scol_stride) {

    constexpr int R    = 32 / NW;
    constexpr int NCOL = 256;

    __shared__ uint16_t s_part[NW][NCOL];
    __shared__ uint16_t s_scale[NCOL];

    const int tid = threadIdx.x;
    const int w   = tid >> 5;
    const int l   = tid & 31;

    const int row0 = blockIdx.y * 32 + w * R;
    const int col0 = blockIdx.x * NCOL;
    const int Kq   = K >> 3;

    uint32_t v[R][4];
    uint32_t pc[4];
    uint32_t rm[R];

#pragma unroll
    for (int k = 0; k < 4; ++k) pc[k] = 0u;

#pragma unroll
    for (int i = 0; i < R; ++i) {
        const uint4* pp = x + (size_t)(row0 + i) * Kq + (col0 >> 3) + l;
        uint4 t = *pp;
        v[i][0] = t.x; v[i][1] = t.y; v[i][2] = t.z; v[i][3] = t.w;
        pc[0] = maxabs2(pc[0], t.x);
        pc[1] = maxabs2(pc[1], t.y);
        pc[2] = maxabs2(pc[2], t.z);
        pc[3] = maxabs2(pc[3], t.w);
        uint32_t m = maxabs2(maxabs2(t.x, t.y), maxabs2(t.z, t.w));
        m = maxabs2(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
        m = maxabs2(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
        rm[i] = maxabs2(m, __byte_perm(m, m, 0x1032));
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
        const uint32_t am = rm[i] & 0x7FFFu;
        rm[i] = am >= 0x7F80u
            ? 0x0040u
            : (0x8300u - max((am + 31u) & 0x7F80u, 0x0400u));
    }

    *reinterpret_cast<uint4*>(&s_part[w][8 * l]) =
        make_uint4(pc[0], pc[1], pc[2], pc[3]);
    __syncthreads();

    constexpr int NV = NCOL / 8;
    if (tid < NV) {
        const uint4* sp = reinterpret_cast<const uint4*>(&s_part[0][0]) + tid;
        uint4 a = sp[0];
#pragma unroll
        for (int ww = 1; ww < NW; ++ww) {
            uint4 b = sp[ww * NV];
            a.x = maxabs2(a.x, b.x);
            a.y = maxabs2(a.y, b.y);
            a.z = maxabs2(a.z, b.z);
            a.w = maxabs2(a.w, b.w);
        }
        uint32_t mx = maxabs2(maxabs2(a.x, a.y), maxabs2(a.z, a.w));
        mx = maxabs2(mx, __byte_perm(mx, mx, 0x1032)) & 0x7FFFu;

        uint32_t sc[4], pk[4];
        if (__builtin_expect(mx < 0x7F80u, 1)) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t t = (&a.x)[k] & 0x7FFF7FFFu;
                uint32_t y = maxabs2((t + 0x001F001Fu) & 0xFF80FF80u, 0x04000400u);
                sc[k] = 0x83008300u - y;
                uint32_t rp = 0x00FE00FEu - ((sc[k] >> 7) & 0x00FF00FFu);
                pk[k] = __byte_perm(rp, 0u, 0x4420);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t t = (&a.x)[k];
                uint32_t r0 = e8m0_from_absbf16(t & 0x7FFFu);
                uint32_t r1 = e8m0_from_absbf16((t >> 16) & 0x7FFFu);
                sc[k] = e8m0_scale_bf16(r0) | (e8m0_scale_bf16(r1) << 16);
                pk[k] = r0 | (r1 << 8);
            }
        }
        *reinterpret_cast<uint4*>(&s_scale[tid * 8]) =
            make_uint4(sc[0], sc[1], sc[2], sc[3]);
        *reinterpret_cast<uint2*>(
            scol + (size_t)blockIdx.y * scol_stride + col0 + tid * 8) =
            make_uint2(__byte_perm(pk[0], pk[1], 0x5410),
                       __byte_perm(pk[2], pk[3], 0x5410));
    }
    __syncthreads();

    uint4 cs = *reinterpret_cast<const uint4*>(&s_scale[8 * l]);
    const int soff = (col0 >> 5) + (l >> 2);

#pragma unroll
    for (int i = 0; i < R; ++i) {
        const size_t base = (size_t)(row0 + i) * K + col0 + 8 * l;
        uint32_t rsx2 = __byte_perm(rm[i], 0u, 0x1010);
        uint32_t p0 = v[i][0], p1 = v[i][1], p2 = v[i][2], p3 = v[i][3];
        *reinterpret_cast<uint2*>(qrow + base) =
            make_uint2(scale_cvt4(p0, rsx2, p1, rsx2),
                       scale_cvt4(p2, rsx2, p3, rsx2));
        *reinterpret_cast<uint2*>(qcol + base) =
            make_uint2(scale_cvt4(p0, cs.x, p1, cs.y),
                       scale_cvt4(p2, cs.z, p3, cs.w));
        if ((l & 3) == 0)
            srow[(size_t)(row0 + i) * srow_stride + soff] =
                (uint8_t)(254u - (rm[i] >> 7));
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper: builds grid/cluster config and launches mxfp8_bidim_wide.
// ---------------------------------------------------------------------------
template <int CPL, int MINB, int PF, int NWV = CFG_NW>
static inline void mxfp8_dispatch(const void* x, void* qrow, void* srow,
                                  void* qcol, void* scol,
                                  int M, int K, int srow_stride, int scol_stride,
                                  int cluster_x, cudaStream_t stream) {
    dim3 grid(K / (32 * CPL), M / 32);

    cudaLaunchAttribute cattr;
    cattr.id = cudaLaunchAttributeClusterDimension;
    cattr.val.clusterDim.x = cluster_x;
    cattr.val.clusterDim.y = 1;
    cattr.val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim        = grid;
    cfg.blockDim       = dim3(32 * NWV);
    cfg.dynamicSmemBytes = 0;
    cfg.stream         = stream;
    cfg.attrs          = &cattr;
    cfg.numAttrs       = 1;

#define LAUNCH_WIDE(KC)                                                              \
    do {                                                                             \
        if (cluster_x > 1)                                                           \
            cudaLaunchKernelEx(&cfg,                                                 \
                mxfp8_bidim_wide<NWV, KC, MINB, CPL, 1, PF>,                        \
                reinterpret_cast<const uint8_t*>(x),                                 \
                reinterpret_cast<uint8_t*>(qrow),                                    \
                reinterpret_cast<uint8_t*>(srow),                                    \
                reinterpret_cast<uint8_t*>(qcol),                                    \
                reinterpret_cast<uint8_t*>(scol),                                    \
                K, srow_stride, scol_stride);                                        \
        else                                                                         \
            mxfp8_bidim_wide<NWV, KC, MINB, CPL, 1, PF>                             \
                <<<grid, 32 * NWV, 0, stream>>>(                                     \
                reinterpret_cast<const uint8_t*>(x),                                 \
                reinterpret_cast<uint8_t*>(qrow),                                    \
                reinterpret_cast<uint8_t*>(srow),                                    \
                reinterpret_cast<uint8_t*>(qcol),                                    \
                reinterpret_cast<uint8_t*>(scol),                                    \
                K, srow_stride, scol_stride);                                        \
    } while (0)

    switch (K) {
        case 4096:  LAUNCH_WIDE(4096);  break;
        case 8192:  LAUNCH_WIDE(8192);  break;
        case 16384: LAUNCH_WIDE(16384); break;
        case 2048:  LAUNCH_WIDE(2048);  break;
        case 32768: LAUNCH_WIDE(32768); break;
        default:    LAUNCH_WIDE(0);     break;
    }
#undef LAUNCH_WIDE
}

// ---------------------------------------------------------------------------
// Top-level launch: picks wide vs. narrow vs. fallback based on shape.
// ---------------------------------------------------------------------------
void launch_mxfp8_kf_bidim(const void* x, void* qrow, void* srow,
                            void* qcol, void* scol,
                            int M, int K, int srow_stride, int scol_stride,
                            cudaStream_t stream) {
    if ((K % 512) == 0) {
        const long long wide_ctas = (long long)(M / 32) * (K / 512);
        if (wide_ctas >= (long long)MXFP8_NARROW_FROM_WAVES * 592) {
            // Deep shape: narrow tile wins (more waves, better latency hiding).
            mxfp8_dispatch<8, CFG_NMINB, MXFP8_NARROW_RESIDENT, CFG_NNW>(
                x, qrow, srow, qcol, scol, M, K,
                srow_stride, scol_stride, CFG_NCL, stream);
        } else {
            // Shallow/wide shape: wide tile wins (fewer LSU ops per byte).
            mxfp8_dispatch<16, CFG_WMINB, MXFP8_WIDE_RESIDENT, CFG_WNW>(
                x, qrow, srow, qcol, scol, M, K, srow_stride, scol_stride,
                wide_ctas >= 4096 ? CFG_WCLD : CFG_WCLS, stream);
        }
        return;
    }
    if ((K % 256) == 0) {
        mxfp8_dispatch<8, CFG_NMINB, MXFP8_NARROW_RESIDENT, CFG_NNW>(
            x, qrow, srow, qcol, scol, M, K,
            srow_stride, scol_stride, CFG_NCL, stream);
        return;
    }
    // Fallback: K not divisible by 256.
    dim3 grid(K / 256, M / 32);
    mxfp8_bidim_kernel<CFG_NW><<<grid, 32 * CFG_NW, 0, stream>>>(
        reinterpret_cast<const uint4*>(x),
        reinterpret_cast<uint8_t*>(qrow),
        reinterpret_cast<uint8_t*>(srow),
        reinterpret_cast<uint8_t*>(qcol),
        reinterpret_cast<uint8_t*>(scol),
        K, srow_stride, scol_stride);
}

}  // namespace kf_bidim
}  // namespace transformer_engine
