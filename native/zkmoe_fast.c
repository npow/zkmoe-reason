#include <stdint.h>
#include <stddef.h>

#define P ((uint64_t)((1ULL << 61) - 1ULL))

static inline uint64_t fp_mod128(__uint128_t x) {
    // x can be up to ~126 bits, so x >> 61 can be 65 bits -> must stay __uint128_t!
    x = (x & (__uint128_t)P) + (x >> 61);
    x = (x & (__uint128_t)P) + (x >> 61);
    uint64_t s = (uint64_t)x;
    return (s >= P) ? (s - P) : s;
}

static inline uint64_t fp_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    return (s >= P) ? (s - P) : s;
}

static inline uint64_t fp_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + (P - b));
}

/*
 * Project N x d x d expert table at (eq_r, eq_c) in F_p^2:
 *   e[j] = sum_{r,c} eq_r[r] * experts[j, r, c] * eq_c[c]  (mod P)
 * Output written to out_e0[j], out_e1[j].
 */
void project_expert_table_c(
    const uint64_t *experts,
    size_t N,
    size_t d,
    const uint64_t *eq_r0,
    const uint64_t *eq_r1,
    const uint64_t *eq_c0,
    const uint64_t *eq_c1,
    uint64_t *out_e0,
    uint64_t *out_e1
) {
    for (size_t j = 0; j < N; j++) {
        const uint64_t *mat = experts + j * d * d;
        __uint128_t total0 = 0;
        __uint128_t total1 = 0;
        for (size_t r = 0; r < d; r++) {
            const uint64_t *row = mat + r * d;
            __uint128_t row_dot0 = 0;
            __uint128_t row_dot1 = 0;
            // Chunk by 16 elements so row_dot <= 16 * 2^122 = 2^126 < 2^128
            size_t c = 0;
            while (c < d) {
                size_t end = (c + 16 < d) ? (c + 16) : d;
                __uint128_t chunk0 = 0;
                __uint128_t chunk1 = 0;
                for (; c < end; c++) {
                    uint64_t m = row[c];
                    chunk0 += (__uint128_t)m * eq_c0[c];
                    chunk1 += (__uint128_t)m * eq_c1[c];
                }
                row_dot0 += fp_mod128(chunk0);
                row_dot1 += fp_mod128(chunk1);
            }
            uint64_t rc0 = fp_mod128(row_dot0);
            uint64_t rc1 = fp_mod128(row_dot1);
            uint64_t er0 = eq_r0[r];
            uint64_t er1 = eq_r1[r];
            uint64_t p00 = fp_mod128((__uint128_t)er0 * rc0);
            uint64_t p11 = fp_mod128((__uint128_t)er1 * rc1);
            uint64_t p01 = fp_mod128((__uint128_t)er0 * rc1);
            uint64_t p10 = fp_mod128((__uint128_t)er1 * rc0);
            total0 += fp_sub(p00, p11);
            total1 += fp_add(p01, p10);
        }
        out_e0[j] = fp_mod128(total0);
        out_e1[j] = fp_mod128(total1);
    }
}
