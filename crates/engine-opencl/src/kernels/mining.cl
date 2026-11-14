// Quantus PoW (Poseidon2 over Goldilocks) mining kernel, OpenCL C.
// Port of engine-cuda/src/kernels/mining.cu with the PTX carry chains written as
// portable C. Same lazy reduce128 as the CUDA kernel (bit-identical outputs);
// every candidate is re-verified on the CPU by the host.
typedef uint u32;
typedef ulong u64;

#define MAX_HITS 8

#define EPS64 0xFFFFFFFFUL
#define P64 0xFFFFFFFF00000001UL

// Round-constant tables carry one extra zero row (RC_INITIAL's last row holds
// the first internal constant in slot 0) so the round loops index them
// without per-round selects.
__constant u64 RC_INTERNAL[23] = {
    0x97f7798a784ad863UL, 0xd1d2bf082f60d4f0UL, 0x69a377a79f9ad206UL,
    0xa9d06906a3858e24UL, 0x295275001eede5b5UL, 0x5874e441117bd746UL,
    0x8a084bbba8ed86ccUL, 0x3defd7645cde6425UL, 0x3998cfe6871cc137UL,
    0x3e52ef8bca48314aUL, 0x964a209f85dc9eccUL, 0x3fcc9ee82cc4577eUL,
    0x8e79b4a5d0096d6dUL, 0x8492362ad2392556UL, 0xee72f470262574d6UL,
    0x1e0e18496da2444aUL, 0x0f3a74bf215eaac6UL, 0x1b061b76a1c0ded3UL,
    0x192c42d86803d7a6UL, 0xf6d49ff997ae0260UL, 0x3ec372e7a0fa3786UL,
    0x5538cdf4f23445d3UL, 0UL};

__constant u64 RC_INITIAL[5][12] = {
    {0xc002e770975b1607UL, 0xbca51a8dfe14593aUL, 0x72938dfbe774f7f9UL,
     0xe4f2fe29e03234acUL, 0xd5e0ba2f541b6449UL, 0xec33b868f3cc46c1UL,
     0x486dcb55419d475aUL, 0x6c1cb2a358cc24f1UL, 0xe3f30d509a1436bbUL,
     0xd9a64f068dca7c29UL, 0xe59b3f57aabba1aeUL, 0x2a3dd4505b478fdcUL},
    {0xada1f8dc7676ed25UL, 0x2711aa8b5509d516UL, 0x4ae6acd0c9c92897UL,
     0x56eb3d6b5256d67aUL, 0x1f7a9d55923bf51eUL, 0x3600427d397a7f68UL,
     0xe5076df75b72c3d0UL, 0xfcd59aa12c6090adUL, 0xcd895e8c68b57a9eUL,
     0x41df7ef9d730ae3eUL, 0xee3e2b889abe977dUL, 0xd29bb7edbeb9c405UL},
    {0x7d5c08eef608e382UL, 0x89ae889caaf0802cUL, 0xb35a8e976d2af617UL,
     0xdb14234eafaf5173UL, 0x78f04462d48b1c98UL, 0x265293b0e47ce88aUL,
     0x999a649b69b9d32fUL, 0x64b0a186698e01d3UL, 0xee0b22d0dfae8bb8UL,
     0x4fd53e50ca04a7eeUL, 0x5762bfe181f25047UL, 0xf51593e2beb5e3bdUL},
    {0x1e5e2b5760e32477UL, 0x622462a1f9aaaeedUL, 0xaa284b3ecdb222aeUL,
     0x63c8e72f542bf3fcUL, 0x3ba588cacb43b5e0UL, 0x23eda6f3c99150ddUL,
     0xaad3bea4baac9a5aUL, 0xe9da8d699b94184aUL, 0xcdb13f4cd93e024cUL,
     0x902cbd0956f655e3UL, 0x5b4e40ffc759532fUL, 0xde795c20a2357af7UL},
    {0x97f7798a784ad863UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
     0UL, 0UL}};

__constant u64 RC_TERMINAL[5][12] = {
    {0x7b72c539e0ea4c6eUL, 0x144573dae2ce9976UL, 0x802028b68f35fc88UL,
     0x6d36c5022c4fe7c2UL, 0xa205d0ffa9b9def3UL, 0xf6e7e38b1ea6ba2fUL,
     0x34f7909ae5258d64UL, 0xb0464d9d77b97fcaUL, 0x64ddb9d5de7e00a6UL,
     0x0ed0d75c27975d97UL, 0x1cbb36f11127338bUL, 0x6673e505cfd0b6baUL},
    {0x605f902830872e01UL, 0x3fd5eb927e95fe4fUL, 0xe81025b5a24c69cdUL,
     0xf7d0ce75de23f74eUL, 0xf39942b6a8585089UL, 0x6d808a08f7b71df6UL,
     0xf8806b6588f49a8bUL, 0x57df2d8c2a32107aUL, 0x16e7c2074d654a2dUL,
     0x213de241fcf33835UL, 0xb0f2b8905a0976f6UL, 0xd8e3cf2bbd355417UL},
    {0xe498691679d9330fUL, 0x763b45d2a3821b28UL, 0x0908bf65eb0a1f0dUL,
     0x7691eb2d194b24f4UL, 0x0e43551233ae13b2UL, 0x93c393dbfc2fe76fUL,
     0x98f607485d48cdeaUL, 0xe3d95f30309819c0UL, 0x1ef581a93eaf6acfUL,
     0x0b24c1b7a030fca4UL, 0x624370be5670b327UL, 0x5f1e28615a11e486UL},
    {0xfe04051f909e042bUL, 0x7257e5b147fd3803UL, 0xe6ae134bb82f2e78UL,
     0x5711fd5cf4784511UL, 0xf83a42660c08c0bcUL, 0x2cd8c96d9a3ce855UL,
     0x7d2ffb1bb0e17271UL, 0x85ae1528caea3811UL, 0x52a345d5c7adb0b8UL,
     0x504c4c51f3faee94UL, 0xbce34a649cfccaf9UL, 0xe0a3389266fb6dc9UL},
    {0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL}};

__constant u64 MDS_DIAG[12] = {
    0xc3b6c08e23ba9300UL, 0xd84b5de94a324fb6UL, 0x0d0c371c5b35b84fUL,
    0x7964f570e7188037UL, 0x5daf18bbd996604bUL, 0x6743bc47b9595257UL,
    0x5528b9362c59bb70UL, 0xac45e25b7127b68bUL, 0xa2077d7dfbb606b5UL,
    0xf3faac6faee378aeUL, 0x0c6388b51545e883UL, 0xd27dbb6944917b60UL};

// a + b with a single 2^64 -> EPS fold (fold overflow ignored, as in CUDA).
inline u64 gf64_add(u64 a, u64 b) {
    u64 s = a + b;
    return s + ((s < a) ? EPS64 : 0UL);
}

// 128 -> 64 fold, 2^64 = EPS, 2^96 = -1 (mod p). Same deliberate slips as CUDA:
//   t = (r1:r0) + r2*EPS, cf = carry; c = r3 + cf; t += cf<<32; t -= c.
inline u64 reduce128(u32 r0, u32 r1, u32 r2, u32 r3) {
    u64 lo = ((u64)r1 << 32) | (u64)r0;
    // r2 * (2^32 - 1) without a 64-bit multiply (AMD has none)
    u64 t = lo + (((u64)r2 << 32) - (u64)r2);
    u32 cf = (t < lo) ? 1u : 0u;
    u32 c = r3 + cf;
    t += (u64)cf << 32;
    return t - (u64)c;
}

inline u64 reduce_hl(u64 lo, u64 hi) {
    return reduce128((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
}

// Exact 128-bit a*b from four 32x32->64 products. a*b plus mul_hi(a, b) made AMD
// compilers build several partial products twice (RDNA has no 64-bit multiply).
inline void mul_wide(u64 a, u64 b, u64 *lo, u64 *hi) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 p00 = (u64)a0 * b0, p01 = (u64)a0 * b1, p10 = (u64)a1 * b0, p11 = (u64)a1 * b1;
    u64 mid = (p00 >> 32) + (u64)(u32)p01 + (u64)(u32)p10;
    *lo = (mid << 32) | (u64)(u32)p00;
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline u64 gf64_mul(u64 a, u64 b) {
    u64 lo, hi;
    mul_wide(a, b, &lo, &hi);
    return reduce_hl(lo, hi);
}

inline u64 gf64_sqr(u64 a) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u64 ll = (u64)a0 * a0;
    u64 lh = (u64)a0 * a1;
    u64 hh = (u64)a1 * a1;
    // (a1:a0)^2 = ll + 2*lh*2^32 + hh*2^64
    u64 mid_lo = lh << 33;            // low 64 bits of 2*lh*2^32
    u64 mid_hi = lh >> 31;            // high bits of 2*lh*2^32
    u64 lo = ll + mid_lo;
    u64 hi = hh + mid_hi + ((lo < ll) ? 1UL : 0UL);
    return reduce_hl(lo, hi);
}

inline u64 gf64_sbox(u64 x) {
    u64 x2 = gf64_sqr(x);
    u64 x3 = gf64_mul(x2, x);
    u64 x4 = gf64_sqr(x2);
    return gf64_mul(x4, x3);
}

inline u64 gf64_canon(u64 a) {
    return a - ((a >= P64) ? P64 : 0UL);
}

// 96-bit accumulator: l (low 64) + h * 2^64.
typedef struct { u64 l; u32 h; } Wide;

inline Wide wide_from(u64 x) { Wide w; w.l = x; w.h = 0u; return w; }

inline void wide_add(Wide *w, u64 x) {
    u64 s = w->l + x;
    w->h += (s < x) ? 1u : 0u;
    w->l = s;
}

inline void wide_add_wide(Wide *w, Wide x) {
    u64 s = w->l + x.l;
    w->h += x.h + ((s < x.l) ? 1u : 0u);
    w->l = s;
}

inline u64 wide_reduce(Wide w) {
    return reduce128((u32)w.l, (u32)(w.l >> 32), w.h, 0u);
}

// a*b + w as an exact 128-bit value, then reduced.
inline u64 mul_add_wide_reduce(u64 a, u64 b, Wide w) {
    u64 lo, hi;
    mul_wide(a, b, &lo, &hi);
    u64 s = lo + w.l;
    hi += (u64)w.h + ((s < lo) ? 1UL : 0UL);
    return reduce_hl(s, hi);
}

inline void ext_layer64(u64 *state, __constant const u64 *rc12) {
    Wide y[12];
    #pragma unroll
    for (int chunk = 0; chunk < 3; chunk++) {
        int o = chunk * 4;
        u64 x0 = state[o], x1 = state[o + 1], x2 = state[o + 2], x3 = state[o + 3];
        Wide t01 = wide_from(x0); wide_add(&t01, x1);
        Wide t23 = wide_from(x2); wide_add(&t23, x3);
        Wide t0123 = t01; wide_add_wide(&t0123, t23);
        Wide t01123 = t0123; wide_add(&t01123, x1);
        Wide t01233 = t0123; wide_add(&t01233, x3);
        y[o + 3] = t01233; wide_add(&y[o + 3], x0); wide_add(&y[o + 3], x0);
        y[o + 1] = t01123; wide_add(&y[o + 1], x2); wide_add(&y[o + 1], x2);
        y[o] = t01123; wide_add_wide(&y[o], t01);
        y[o + 2] = t01233; wide_add_wide(&y[o + 2], t23);
    }
    Wide sums[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        sums[k] = y[k];
        wide_add_wide(&sums[k], y[k + 4]);
        wide_add_wide(&sums[k], y[k + 8]);
    }
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        Wide w = y[i];
        wide_add_wide(&w, sums[i % 4]);
        wide_add(&w, rc12[i]);
        state[i] = wide_reduce(w);
    }
}

inline u64 int_round_p(u64 *state, u64 x, u64 rc0) {
    Wide s = wide_from(state[1]);
    #pragma unroll
    for (int i = 2; i < 12; i++) wide_add(&s, state[i]);
    wide_add(&s, x);
    Wide s0 = s;
    wide_add(&s0, rc0);
    u64 out0 = mul_add_wide_reduce(x, MDS_DIAG[0], s0);
    #pragma unroll
    for (int i = 1; i < 12; i++) state[i] = mul_add_wide_reduce(state[i], MDS_DIAG[i], s);
    return out0;
}

inline void permute64_after_initial(u64 *state) {
    #pragma unroll 1
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 12; i++) state[i] = gf64_sbox(state[i]);
        ext_layer64(state, RC_INITIAL[r + 1]);
    }
    u64 x = gf64_sbox(state[0]);
    #pragma unroll 1
    for (int r = 0; r < 21; r++) x = gf64_sbox(int_round_p(state, x, RC_INTERNAL[r + 1]));
    state[0] = int_round_p(state, x, 0UL);
    #pragma unroll
    for (int i = 0; i < 12; i++) state[i] = gf64_add(state[i], RC_TERMINAL[0][i]);
    #pragma unroll 1
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 12; i++) state[i] = gf64_sbox(state[i]);
        ext_layer64(state, RC_TERMINAL[r + 1]);
    }
}

inline void permute64(u64 *state) {
    ext_layer64(state, RC_INITIAL[0]);
    permute64_after_initial(state);
}

inline void permute64_twice_after_initial(u64 *state) {
    #pragma unroll 1
    for (int pass = 0; pass < 2; pass++) {
        if (pass != 0) ext_layer64(state, RC_INITIAL[0]);
        permute64_after_initial(state);
        if (pass == 0) {
            state[0] = gf64_add(state[0], 1UL);
            state[1] = gf64_add(state[1], 1UL);
        }
    }
}

inline u32 bswap32(u32 v) {
    return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
}

// Self-test kernel: full hash of count consecutive nonces (midstate = header +
// nonce high half, start = 16 LE u32 limbs). Not a verifier.
__kernel void hash_nonces(__global u32 *hashes, __global const u32 *midstate,
                          __global const u32 *start_nonce, u32 count) {
    u32 tid = get_global_id(0);
    if (tid >= count) return;
    u64 st[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) st[i] = ((u64)midstate[2 * i + 1] << 32) | (u64)midstate[2 * i];
    u32 cur[8];
    u32 carry = tid;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 v = start_nonce[i];
        u32 s = v + carry;
        carry = (s < v) ? 1u : 0u;
        cur[i] = s;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] = gf64_add(st[i], (u64)bswap32(cur[7 - i]));
    permute64(st);
    st[0] = gf64_add(st[0], 1UL);
    st[1] = gf64_add(st[1], 1UL);
    permute64(st);
    u32 out[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 c = gf64_canon(st[i]);
        out[15 - 2 * i] = bswap32((u32)(c & EPS64));
        out[14 - 2 * i] = bswap32((u32)(c >> 32));
    }
    permute64(st);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 c = gf64_canon(st[i]);
        out[7 - 2 * i] = bswap32((u32)(c & EPS64));
        out[6 - 2 * i] = bswap32((u32)(c >> 32));
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) hashes[tid * 16u + (u32)i] = out[i];
}

// params: prestate[24], start_nonce[16], difficulty_target[16], dispatch[3]
__kernel void mining_main(__global u32 *results, __constant const u32 *params) {
    __constant const u32 *prestate = params;
    __constant const u32 *start_nonce = params + 24;
    __constant const u32 *target = params + 40;
    __constant const u32 *dispatch = params + 56;
    u32 thread_id = get_global_id(0);
    u32 total_threads = dispatch[0];
    u32 nonces_per_thread = dispatch[1];
    u32 total_nonces = dispatch[2];
    if (thread_id >= total_threads) return;
    u32 base_index = thread_id * nonces_per_thread;

    u64 mid[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) mid[i] = ((u64)prestate[2 * i + 1] << 32) | (u64)prestate[2 * i];
    u32 tgt_hi[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) tgt_hi[i] = target[8 + i];
    u64 nonce_base_low = ((u64)start_nonce[1] << 32) | (u64)start_nonce[0];

    for (u32 j = 0; j < nonces_per_thread; j++) {
        u32 logical_index = base_index + j;
        if (logical_index >= total_nonces) break;
        u64 st[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) st[i] = mid[i];
        u64 nonce_low = nonce_base_low + (u64)logical_index;
        u64 x6 = (u64)bswap32((u32)(nonce_low >> 32));
        u64 x7 = (u64)bswap32((u32)nonce_low);
        u64 x6_2 = x6 + x6, x6_3 = x6_2 + x6, x6_4 = x6_2 + x6_2, x6_6 = x6_3 + x6_3;
        u64 x7_2 = x7 + x7, x7_3 = x7_2 + x7, x7_4 = x7_2 + x7_2, x7_6 = x7_3 + x7_3;
        u64 c0 = x6 + x7, c1 = x6_3 + x7, c2 = x6_2 + x7_3, c3 = x6 + x7_2;
        st[0] = gf64_add(st[0], c0);
        st[1] = gf64_add(st[1], c1);
        st[2] = gf64_add(st[2], c2);
        st[3] = gf64_add(st[3], c3);
        st[4] = gf64_add(st[4], x6_2 + x7_2);
        st[5] = gf64_add(st[5], x6_6 + x7_2);
        st[6] = gf64_add(st[6], x6_4 + x7_6);
        st[7] = gf64_add(st[7], x6_2 + x7_4);
        st[8] = gf64_add(st[8], c0);
        st[9] = gf64_add(st[9], c1);
        st[10] = gf64_add(st[10], c2);
        st[11] = gf64_add(st[11], c3);
        permute64_twice_after_initial(st);

        u32 cmp = 0u;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            u64 c = gf64_canon(st[i]);
            u32 w[2] = {(u32)(c & EPS64), (u32)(c >> 32)};
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                if (cmp == 0u) {
                    u32 h = bswap32(w[k]);
                    u32 t = tgt_hi[7 - (2 * i + k)];
                    if (h != t) cmp = (h > t) ? 1u : 2u;
                }
            }
        }
        if (cmp == 1u) continue;
        u32 slot = atomic_inc(&results[0]);
        if (slot < MAX_HITS) results[1 + slot] = logical_index;
    }
}
