// KF campaign rowwise MXFP8 quantization kernel, adapted for TransformerEngine.
// Original: kf_campaign/mxfp8_r12_record (B200 campaign round 12).
// Requires SM90+ at runtime (bf16x2 PTX: max.xorsign.abs, cvt.rn.satfinite.e4m3x2,
// cvt.rp.satfinite.ue8m0x2).  Device helpers are guarded; kernel templates are
// declared at file scope so the host <<<>>> launch stubs compile correctly.

#include <cuda_runtime.h>
#include <cstdint>
#include "mxfp8/kf_rowwise/kernel.h"

namespace transformer_engine {
namespace kf_rowwise {

// ---------------------------------------------------------------------------
// Tunables — B200 campaign best, verified ≥ default on SM107
// ---------------------------------------------------------------------------
#ifndef KF_NT0
#define KF_NT0 256
#endif
#ifndef KF_GG0
#define KF_GG0 1
#endif
#ifndef KF_EV0
#define KF_EV0 1
#endif
#ifndef KF_LP0
#define KF_LP0 0
#endif
#ifndef KF_QM0
#define KF_QM0 0
#endif
#ifndef KF_CV0
#define KF_CV0 (-1)
#endif

#ifndef KF_NT1
#define KF_NT1 256
#endif
#ifndef KF_GG1
#define KF_GG1 2
#endif
#ifndef KF_EV1
#define KF_EV1 1
#endif
#ifndef KF_LP1
#define KF_LP1 0
#endif
#ifndef KF_QM1
#define KF_QM1 0
#endif
#ifndef KF_CV1
#define KF_CV1 (-1)
#endif

#ifndef KF_NT2
#define KF_NT2 128
#endif
#ifndef KF_GG2
#define KF_GG2 2
#endif
#ifndef KF_EV2
#define KF_EV2 3
#endif
#ifndef KF_LP2
#define KF_LP2 60
#endif
#ifndef KF_QM2
#define KF_QM2 0
#endif
#ifndef KF_CV2
#define KF_CV2 (-1)
#endif

#ifndef KF_NT3
#define KF_NT3 256
#endif
#ifndef KF_GG3
#define KF_GG3 2
#endif
#ifndef KF_EV3
#define KF_EV3 3
#endif
#ifndef KF_LP3
#define KF_LP3 60
#endif
#ifndef KF_QM3
#define KF_QM3 0
#endif
#ifndef KF_CV3
#define KF_CV3 (-1)
#endif

#ifndef KF_R0MICRO
#define KF_R0MICRO 1
#endif

#define KF_R0_BYTES (24ll << 20)
#define KF_R1_BYTES (48ll << 20)
#define KF_R2_BYTES (96ll << 20)

// ---------------------------------------------------------------------------
// SM90+ device helpers — only compiled for __CUDA_ARCH__ >= 900
// ---------------------------------------------------------------------------
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#define KF_DEVI __device__ __forceinline__

KF_DEVI uint32_t kf_amax2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
KF_DEVI uint32_t kf_bmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
KF_DEVI uint32_t kf_cvt_pack(uint32_t lo, uint32_t hi) {
    uint16_t a, b;
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(a) : "r"(lo));
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(b) : "r"(hi));
    return (uint32_t)a | ((uint32_t)b << 16);
}
KF_DEVI uint32_t kf_scale_cvt_pack(uint32_t lo, uint32_t hi, uint32_t scale) {
    uint32_t d;
    asm("{ .reg .b16 a, b;\n\t"
        "mul.rn.bf16x2 %1, %1, %3;\n\t"
        "mul.rn.bf16x2 %2, %2, %3;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 a, %1;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 b, %2;\n\t"
        "mov.b32 %0, {a, b}; }"
        : "=r"(d), "+r"(lo), "+r"(hi) : "r"(scale));
    return d;
}
KF_DEVI uint32_t kf_f32_to_e8m0_rp(float v) {
    uint16_t d;
    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(d) : "f"(v), "f"(v));
    return (uint32_t)(d & 0xFFu);
}

#define KF_LD8N(p,a,b,c,d,e,f,g,h) \
    asm("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];" \
        :"=r"(a),"=r"(b),"=r"(c),"=r"(d),"=r"(e),"=r"(f),"=r"(g),"=r"(h):"l"(p))
#define KF_LD8F(p,a,b,c,d,e,f,g,h) \
    asm("ld.global.nc.L2::evict_first.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];" \
        :"=r"(a),"=r"(b),"=r"(c),"=r"(d),"=r"(e),"=r"(f),"=r"(g),"=r"(h):"l"(p))
#define KF_LD8H(p,pol,a,b,c,d,e,f,g,h) \
    asm("ld.global.nc.L2::cache_hint.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8],%9;" \
        :"=r"(a),"=r"(b),"=r"(c),"=r"(d),"=r"(e),"=r"(f),"=r"(g),"=r"(h):"l"(p),"l"(pol))
#define KF_LD4N(p,a,b,c,d) \
    asm("ld.global.nc.v4.b32 {%0,%1,%2,%3},[%4];" :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(p))
#define KF_LD4F(p,pol,a,b,c,d) \
    asm("ld.global.nc.L2::cache_hint.v4.b32 {%0,%1,%2,%3},[%4],%5;" \
        :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(p),"l"(pol))
#define KF_ST2H(p,pol,a,b) \
    asm volatile("st.global.L2::cache_hint.v2.b32 [%0],{%1,%2},%3;" \
        ::"l"(p),"r"(a),"r"(b),"l"(pol):"memory")
#define KF_ST4H(p,pol,a,b,c,d) \
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0],{%1,%2,%3,%4},%5;" \
        ::"l"(p),"r"(a),"r"(b),"r"(c),"r"(d),"l"(pol):"memory")

KF_DEVI uint64_t kf_policy_evict_last() {
    uint64_t p;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

template <bool EVF>
KF_DEVI void kf_load_half(const uint32_t* __restrict__ p, uint32_t* r) {
    if (EVF) { KF_LD8F(p, r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]); }
    else     { KF_LD8N(p, r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]); }
}

template <bool FUSED_CONV, bool SHORT_SCALE, bool FUSE_LAST_PAIR = false>
KF_DEVI void kf_proc_half(const uint32_t* r, uint32_t* __restrict__ q,
                           uint8_t* __restrict__ s, bool store_scale, uint64_t pol) {
    uint32_t a0 = kf_amax2(r[0],r[1]), a1 = kf_amax2(r[2],r[3]);
    uint32_t a2 = kf_amax2(r[4],r[5]), a3 = kf_amax2(r[6],r[7]);
    a0 = kf_amax2(a0,a1); a2 = kf_amax2(a2,a3);
    a0 = kf_amax2(a0,a2);
    a0 = kf_amax2(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
    a0 = kf_amax2(a0, __byte_perm(a0, a0, 0x1032));

    uint32_t biased, spk;
#if KF_R0MICRO
    if constexpr (SHORT_SCALE) {
        const float amax = fabsf(__int_as_float(a0 << 16));
        biased = kf_f32_to_e8m0_rp(amax * (1.0f / 448.0f));
        spk = 0x7F007F00u - biased * 0x00800080u;
    } else
#endif
    {
        const float amax = __int_as_float((a0 & 0x7FFFu) << 16);
        biased = kf_f32_to_e8m0_rp(amax * (1.0f / 448.0f));
        const uint32_t sb = 32512u - (biased << 7);
        spk = __byte_perm(sb, sb, 0x1010);
    }
    if (store_scale) *s = (uint8_t)biased;

    uint32_t o[4];
    if constexpr (FUSE_LAST_PAIR) {
#pragma unroll
        for (int i = 0; i < 3; ++i)
            o[i] = kf_cvt_pack(kf_bmul2(r[2*i],spk), kf_bmul2(r[2*i+1],spk));
        o[3] = kf_scale_cvt_pack(r[6], r[7], spk);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (FUSED_CONV) o[i] = kf_scale_cvt_pack(r[2*i], r[2*i+1], spk);
            else                      o[i] = kf_cvt_pack(kf_bmul2(r[2*i],spk), kf_bmul2(r[2*i+1],spk));
        }
    }
    KF_ST4H(q, pol, o[0], o[1], o[2], o[3]);
}

KF_DEVI void kf_proc_block(const uint32_t* r, uint32_t* __restrict__ q,
                            uint8_t* __restrict__ s) {
    uint32_t a0=kf_amax2(r[0],r[1]),   a1=kf_amax2(r[2],r[3]);
    uint32_t a2=kf_amax2(r[4],r[5]),   a3=kf_amax2(r[6],r[7]);
    uint32_t a4=kf_amax2(r[8],r[9]),   a5=kf_amax2(r[10],r[11]);
    uint32_t a6=kf_amax2(r[12],r[13]), a7=kf_amax2(r[14],r[15]);
    a0=kf_amax2(a0,a1); a2=kf_amax2(a2,a3); a4=kf_amax2(a4,a5); a6=kf_amax2(a6,a7);
    a0=kf_amax2(a0,a2); a4=kf_amax2(a4,a6); a0=kf_amax2(a0,a4);
    a0=kf_amax2(a0,__byte_perm(a0,a0,0x1032));
    const float amax = __int_as_float((a0 & 0x7FFFu) << 16);
    const uint32_t biased = kf_f32_to_e8m0_rp(amax * (1.0f / 448.0f));
    const uint32_t sb = 32512u - (biased << 7);
    const uint32_t spk = __byte_perm(sb, sb, 0x1010);
    *s = (uint8_t)biased;
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        o[i] = kf_cvt_pack(kf_bmul2(r[2*i],spk), kf_bmul2(r[2*i+1],spk));
    asm volatile("st.global.L2::evict_last.v8.b32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};"
        ::"l"(q),"r"(o[0]),"r"(o[1]),"r"(o[2]),"r"(o[3]),
          "r"(o[4]),"r"(o[5]),"r"(o[6]),"r"(o[7]):"memory");
}

#endif  // __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// Kernel templates — declared at file scope so the host <<< >>> stubs compile.
// Bodies are no-ops below SM90; device helpers above are unreachable there.
// ---------------------------------------------------------------------------
template <int NT, int G, int EVFM, int FUSE_MODE = 0>
__global__ __launch_bounds__(NT)
void kf_half_kernel(const uint32_t* __restrict__ xin, uint32_t* __restrict__ qout,
                    uint8_t* __restrict__ sout, uint32_t late) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t lane = threadIdx.x & 31u;
    const long long wid = (long long)blockIdx.x * (NT/32) + (threadIdx.x >> 5);
    const long long B   = wid * (16 * G);
    const uint64_t pol  = kf_policy_evict_last();
    const bool even     = (lane & 1u) == 0u;

    uint32_t r[8 * G];
    if (EVFM == 3) {
        const float frac = (blockIdx.x >= late) ? 1.0f : 0.0f;
        uint64_t rpol;
        asm("createpolicy.fractional.L2::evict_first.b64 %0,%1;" : "=l"(rpol) : "f"(frac));
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const uint32_t* pp = xin + 16*B + 256*g + 8*lane;
            uint32_t* d = r + 8*g;
            KF_LD8H(pp, rpol, d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
        }
    } else if (EVFM == 1 || (EVFM == 2 && blockIdx.x >= late)) {
#pragma unroll
        for (int g = 0; g < G; ++g)
            kf_load_half<true>(xin + 16*B + 256*g + 8*lane, r + 8*g);
    } else {
#pragma unroll
        for (int g = 0; g < G; ++g)
            kf_load_half<false>(xin + 16*B + 256*g + 8*lane, r + 8*g);
    }
    if constexpr (G == 1) {
        kf_proc_half<true, true>(r, qout+8*B+4*lane, sout+B+(lane>>1), even, pol);
    } else {
        static_assert(G == 2);
        kf_proc_half<false,false>(r, qout+8*B+4*lane, sout+B+(lane>>1), even, pol);
        if constexpr (FUSE_MODE == 2)
            kf_proc_half<false,false,true>(r+8, qout+8*B+128+4*lane, sout+B+16+(lane>>1), even, pol);
        else
            kf_proc_half<(FUSE_MODE==1),false>(r+8, qout+8*B+128+4*lane, sout+B+16+(lane>>1), even, pol);
    }
#endif
}

template <int NT, int G, int EVFM>
__global__ __launch_bounds__(NT)
void kf_quarter_kernel(const uint32_t* __restrict__ xin, uint32_t* __restrict__ qout,
                       uint8_t* __restrict__ sout, uint32_t late) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t lane = threadIdx.x & 31u;
    const long long wid = (long long)blockIdx.x * (NT/32) + (threadIdx.x >> 5);
    const long long W   = wid * G;
    const uint64_t pol  = kf_policy_evict_last();
    const bool leader   = (lane & 3u) == 0u;

    uint32_t r[4 * G];
    const bool evf = (EVFM == 1) || (EVFM == 2 && blockIdx.x >= late);
    uint64_t rpol = 0;
    if (evf) { asm("createpolicy.fractional.L2::evict_first.b64 %0,1.0;" : "=l"(rpol)); }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const uint32_t* p = xin + (W+g)*128 + 4*lane;
        uint32_t* d = r + 4*g;
        if (evf) { KF_LD4F(p, rpol, d[0],d[1],d[2],d[3]); }
        else     { KF_LD4N(p, d[0],d[1],d[2],d[3]); }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const uint32_t* d = r + 4*g;
        uint32_t a0 = kf_amax2(d[0],d[1]), a1 = kf_amax2(d[2],d[3]);
        a0 = kf_amax2(a0,a1);
        a0 = kf_amax2(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
        a0 = kf_amax2(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
        a0 = kf_amax2(a0, __byte_perm(a0, a0, 0x1032));
        const float amax = __int_as_float((a0 & 0x7FFFu) << 16);
        const uint32_t biased = kf_f32_to_e8m0_rp(amax * (1.0f / 448.0f));
        const uint32_t sb  = 32512u - (biased << 7);
        const uint32_t spk = __byte_perm(sb, sb, 0x1010);
        if (leader) sout[(W+g)*8 + (lane>>2)] = (uint8_t)biased;
        const uint32_t o0 = kf_cvt_pack(kf_bmul2(d[0],spk), kf_bmul2(d[1],spk));
        const uint32_t o1 = kf_cvt_pack(kf_bmul2(d[2],spk), kf_bmul2(d[3],spk));
        KF_ST2H(qout + (W+g)*64 + 2*lane, pol, o0, o1);
    }
#endif
}

__global__ __launch_bounds__(128)
void kf_tail_kernel(const uint32_t* __restrict__ xin, uint32_t* __restrict__ qout,
                    uint8_t* __restrict__ sout, long long first, long long nblocks) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const long long idx = first + (long long)blockIdx.x * 128 + threadIdx.x;
    if (idx >= nblocks) return;
    uint32_t r[16];
    KF_LD8N(xin+idx*16,   r[0],r[1],r[2], r[3], r[4], r[5], r[6], r[7]);
    KF_LD8N(xin+idx*16+8, r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15]);
    kf_proc_block(r, qout+idx*8, sout+idx);
#endif
}

__global__ __launch_bounds__(128)
void kf_padded_kernel(const uint32_t* __restrict__ xin, uint32_t* __restrict__ qout,
                      uint8_t* __restrict__ sout, int bpr, int sstride) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int colb = blockIdx.x * 128 + threadIdx.x;
    if (colb >= bpr) return;
    const long long idx = (long long)blockIdx.y * bpr + colb;
    uint32_t r[16];
    KF_LD8N(xin+idx*16,   r[0],r[1],r[2], r[3], r[4], r[5], r[6], r[7]);
    KF_LD8N(xin+idx*16+8, r[8],r[9],r[10],r[11],r[12],r[13],r[14],r[15]);
    kf_proc_block(r, qout+idx*8, sout+(long long)blockIdx.y*sstride+colb);
#endif
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
void launch_mxfp8_kf(const void* x, void* q, void* s,
                     int M, int K, int sstride,
                     cudaStream_t stream) {
    const int bpr = K >> 5;
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(x);
    uint32_t*       qp = reinterpret_cast<uint32_t*>(q);
    uint8_t*        sp = reinterpret_cast<uint8_t*>(s);

    if (sstride != bpr) {
        dim3 grid((bpr + 127) / 128, M);
        kf_padded_kernel<<<grid, 128, 0, stream>>>(xp, qp, sp, bpr, sstride);
        return;
    }

    const long long nblocks = (long long)M * bpr;
    const long long obytes  = (long long)M * K;

    int nt, gg, ev, lp, qm, cv;
    if      (obytes <= KF_R0_BYTES) { nt=KF_NT0; gg=KF_GG0; ev=KF_EV0; lp=KF_LP0; qm=KF_QM0; cv=KF_CV0; }
    else if (obytes <= KF_R1_BYTES) { nt=KF_NT1; gg=KF_GG1; ev=KF_EV1; lp=KF_LP1; qm=KF_QM1; cv=KF_CV1; }
    else if (obytes <= KF_R2_BYTES) { nt=KF_NT2; gg=KF_GG2; ev=KF_EV2; lp=KF_LP2; qm=KF_QM2; cv=KF_CV2; }
    else                            { nt=KF_NT3; gg=KF_GG3; ev=KF_EV3; lp=KF_LP3; qm=KF_QM3; cv=KF_CV3; }

    const long long per_cta = (long long)nt * gg / (qm ? 4 : 2);
    const long long grid    = nblocks / per_cta;
    const uint32_t  late    = (uint32_t)((grid * (100 - lp)) / 100);

#define KF_LAUNCH1(FN) \
    do { \
        if (cv >= 0) { \
            static bool _done = false; \
            if (!_done) { \
                cudaFuncSetAttribute((const void*)(FN), \
                    cudaFuncAttributePreferredSharedMemoryCarveout, cv); \
                _done = true; \
            } \
        } \
        FN<<<grid, nt, 0, stream>>>(xp, qp, sp, late); \
    } while (0)

#define KF_DISPATCH(NT_, G_, QM_, FUSE_) \
    do { \
        if (QM_) { \
            if      (ev == 1) KF_LAUNCH1((kf_quarter_kernel<NT_, G_, 1>)); \
            else if (ev == 2) KF_LAUNCH1((kf_quarter_kernel<NT_, G_, 2>)); \
            else              KF_LAUNCH1((kf_quarter_kernel<NT_, G_, 0>)); \
        } else { \
            if      (ev == 1) KF_LAUNCH1((kf_half_kernel<NT_, G_, 1, FUSE_>)); \
            else if (ev == 3) KF_LAUNCH1((kf_half_kernel<NT_, G_, 3, FUSE_>)); \
            else if (ev == 2) KF_LAUNCH1((kf_half_kernel<NT_, G_, 2, FUSE_>)); \
            else              KF_LAUNCH1((kf_half_kernel<NT_, G_, 0, FUSE_>)); \
        } \
    } while (0)

    if (grid > 0) {
        if      (obytes <= KF_R0_BYTES) KF_DISPATCH(KF_NT0, KF_GG0, KF_QM0, false);
        else if (obytes <= KF_R1_BYTES) KF_DISPATCH(KF_NT1, KF_GG1, KF_QM1, 2);
        else if (obytes <= KF_R2_BYTES) KF_DISPATCH(KF_NT2, KF_GG2, KF_QM2, 0);
        else                            KF_DISPATCH(KF_NT3, KF_GG3, KF_QM3, 0);
    }
#undef KF_DISPATCH
#undef KF_LAUNCH1

    const long long done = grid * per_cta;
    if (done < nblocks) {
        const int tgrid = (int)((nblocks - done + 127) / 128);
        kf_tail_kernel<<<tgrid, 128, 0, stream>>>(xp, qp, sp, done, nblocks);
    }
}

}  // namespace kf_rowwise
}  // namespace transformer_engine
