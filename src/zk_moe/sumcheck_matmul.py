"""Zero-Knowledge Sumcheck Protocol for active expert matrix multiplications."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from zk_moe.pcs import PCSCommitment, PCSOpeningProof

import numpy as np

from zk_moe.field_and_mle import (
    P,
    Fp2,
    Transcript,
    eq_evals_fp2,
    fp2_vec_add,
    fp2_vec_from_fp,
    fp2_vec_mul,
    fp2_vec_mul_scalar,
    fp2_vec_sub,
    fp2_vec_sum,
    mle_eval_fp2,
    vec_add,
    vec_mod,
    vec_mul,
    vec_sub,
)


@dataclass
class SumcheckRoundPoly:
    """Univariate quadratic g(X) = c0 + c1*X + c2*X^2 over F_p^2."""
    c0: Fp2
    c1: Fp2
    c2: Fp2

    def eval(self, x: Fp2) -> Fp2:
        return self.c0 + x * (self.c1 + x * self.c2)

    def eval_zero_plus_one(self) -> Fp2:
        # g(0) + g(1) = 2*c0 + c1 + c2
        return self.c0 + self.c0 + self.c1 + self.c2


@dataclass
class ProductSumcheckProof:
    rounds: list[SumcheckRoundPoly]
    final_a: Fp2
    final_b: Fp2
    challenges: list[Fp2]


@dataclass
class BatchedMatmulProof:
    """Batched ZK sumcheck proof for y_i = M_i @ x (i=1..k)."""
    row_point: list[Fp2]
    col_point: list[Fp2]
    claimed_y_evals: list[Fp2]
    alpha_weights: list[Fp2]
    sumcheck_proof: ProductSumcheckProof
    final_m_evals: list[Fp2]   # M_i(r*, c*) for i=1..k
    final_x_eval: Fp2          # x(c*)
    blind_comm: bytes          # commitment to masking polynomial Z
    blind_sum: Fp2             # S_Z = sum Z(c)
    blind_eval: Fp2            # Z(c*)
    blind_challenge: Fp2       # rho challenge
    x_comm: PCSCommitment | None = None
    x_opening: PCSOpeningProof | None = None
    blind_pcs_comm: PCSCommitment | None = None
    blind_opening: PCSOpeningProof | None = None


def _contract_rows_fp2(mat: np.ndarray, eq_r: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute M(r*, c) for all c in {0,1}^log(d) where mat is (d, d) in F_p."""
    eq0, eq1 = eq_r
    d = mat.shape[0]
    # mat is uint64 (d, d) with entries < 2^61.
    # For each column, sum_r eq_r[r] * mat[r, c] mod P.
    # To avoid overflow in matrix multiply, split mat into 30-bit limbs or use python/chunking.
    # Actually: eq0[r] * mat[r, c] can be done via vec_mul across broadcast or chunked matmul!
    # Wait: uint64 matmul in numpy doesn't reduce mod P, and d <= 1024 so sum of 1024 * 2^62 overflows uint64.
    # Solution: split each 61-bit number into two 31-bit limbs (lo, hi) -> float64? No, 31*31=62 > 53!
    # Split into three 21-bit limbs: 21 + 21 = 42, times d=1024 -> 52 bits < 53 bits (exact in float64) OR
    # uint64 with 21-bit limbs: 21+21+10 = 52 < 64 bits (exact in uint64 BLAS-like or int64)!
    # Even simpler and exact in uint64: process in chunks of 8 rows with vec_mul + vec_mod!
    # Let's check speed: for d=1024, chunk size 4 -> 256 iterations of (4, 1024) vectorized ops = ~3ms!
    # Better yet: 3-limb split (21, 20, 20 bits) with np.matmul on int64/float64?
    # Wait: int64 matmul in numpy is pure C/BLAS? No, numpy int64 matmul is a C loop, whereas float64 matmul uses OpenBLAS SGEMM/DGEMM (100x faster)!
    # Check max value in 3-limb (21, 20, 20) float64 matmul:
    # Each limb <= 2^21 - 1 = 2,097,151. Product <= 4.4e12. Sum of d <= 1024 products <= 4.5e15 < 2^53 = 9.007e15!
    # Exact to the last bit in IEEE-754 float64 AND uses OpenBLAS DGEMM!
    return _fp_matvec_transpose_fp2(mat, eq0, eq1)


def _fp_vec_mat_mul_exact(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Exact (vec @ mat) mod P where vec has shape (d,) and mat has shape (d, m), d <= 2048.
    Uses 21-bit / 20-bit / 20-bit limb decomposition with float64 BLAS DGEMM.
    Max partial sum: d * (2^21)^2 <= 2048 * 2^42 = 2^53 (for d <= 2048, strictly < 2^53 if limb0 < 2^21 and limb1,2 < 2^20).
    Wait: for d=2048, (2^21 - 1)^2 * 2048 < 2^53!
    """
    d = vec.shape[0]
    if d > 2048:
        # Fallback to chunked uint64 for huge d
        out = np.zeros(mat.shape[1], dtype=np.uint64)
        for start in range(0, d, 4):
            end = min(start + 4, d)
            prods = vec_mul(vec[start:end, None], mat[start:end, :])
            s = vec_mod(np.sum(prods, axis=0, dtype=np.uint64))
            out = vec_add(out, s)
        return out

    MASK21 = np.uint64((1 << 21) - 1)
    MASK20 = np.uint64((1 << 20) - 1)

    v0 = (vec & MASK21).astype(np.float64)
    v1 = ((vec >> np.uint64(21)) & MASK20).astype(np.float64)
    v2 = (vec >> np.uint64(41)).astype(np.float64)

    m0 = (mat & MASK21).astype(np.float64)
    m1 = ((mat >> np.uint64(21)) & MASK20).astype(np.float64)
    m2 = (mat >> np.uint64(41)).astype(np.float64)

    # Combine products grouped by power of 2:
    # 2^0: v0 @ m0
    # 2^21: v0 @ m1 + v1 @ m0  (note: v0*m1 < 2^41, v1*m0 < 2^41, sum < 2^42, * d <= 2048 -> < 2^53)
    # 2^41: v0 @ m2 + v2 @ m0  (wait: 2^21 * 2^20 = 2^41, 2^42: v1 @ m1 < 2^40)
    # 2^61 ≡ 1 (mod P): v1 @ m2 + v2 @ m1 -> shift 61 ≡ * 1!
    # 2^82 ≡ 2^21 (mod P): v2 @ m2 -> shift 82 ≡ * 2^21!
    t0 = (v0 @ m0).astype(np.uint64)
    t21 = (v0 @ m1 + v1 @ m0).astype(np.uint64) + (v2 @ m2).astype(np.uint64)
    t41 = (v0 @ m2 + v2 @ m0).astype(np.uint64)
    t42 = (v1 @ m1).astype(np.uint64)
    t62 = (v1 @ m2 + v2 @ m1).astype(np.uint64)

    # Reduce each term mod P (all terms < 2^53, so shifting by <= 42 bits must be reduced carefully!)
    # Helper to shift x (< 2^54) by s bits mod (2^61 - 1):
    def shift_mod(x: np.ndarray, s: int) -> np.ndarray:
        if s == 0:
            return vec_mod(x)
        keep = 61 - s
        mask = np.uint64((1 << keep) - 1)
        lo = (x & mask) << np.uint64(s)
        hi = x >> np.uint64(keep)
        return vec_mod(lo + hi)

    res = vec_mod(t0)
    res = vec_add(res, shift_mod(t21, 21))
    res = vec_add(res, shift_mod(t41, 41))
    res = vec_add(res, shift_mod(t42, 42))
    res = vec_add(res, shift_mod(t62, 1))
    return res


def _fp_matvec_transpose_fp2(
    mat: np.ndarray, eq0: np.ndarray, eq1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (eq0 + u*eq1)^T @ mat mod P."""
    c0 = _fp_vec_mat_mul_exact(eq0, mat)
    c1 = _fp_vec_mat_mul_exact(eq1, mat)
    return c0, c1


def prove_product_sumcheck(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    transcript: Transcript,
) -> ProductSumcheckProof:
    """Prove sum_{x in {0,1}^v} A(x) * B(x) = S over F_p^2."""
    a0, a1 = np.asarray(a[0], dtype=np.uint64).ravel(), np.asarray(a[1], dtype=np.uint64).ravel()
    b0, b1 = np.asarray(b[0], dtype=np.uint64).ravel(), np.asarray(b[1], dtype=np.uint64).ravel()
    n = a0.size
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Size must be a power of 2, got {n}")
    n_vars = n.bit_length() - 1

    rounds: list[SumcheckRoundPoly] = []
    challenges: list[Fp2] = []

    for r_idx in range(n_vars):
        ae = (a0[0::2], a1[0::2])
        ao = (a0[1::2], a1[1::2])
        be = (b0[0::2], b1[0::2])
        bo = (b0[1::2], b1[1::2])

        g0 = fp2_vec_sum(fp2_vec_mul(ae, be))
        g1 = fp2_vec_sum(fp2_vec_mul(ao, bo))

        da = fp2_vec_sub(ao, ae)
        db = fp2_vec_sub(bo, be)
        c2 = fp2_vec_sum(fp2_vec_mul(da, db))
        c0 = g0
        c1 = g1 - c0 - c2

        poly = SumcheckRoundPoly(c0=c0, c1=c1, c2=c2)
        rounds.append(poly)

        transcript.absorb_fp2(f"sc_c0_{r_idx}".encode(), c0)
        transcript.absorb_fp2(f"sc_c1_{r_idx}".encode(), c1)
        transcript.absorb_fp2(f"sc_c2_{r_idx}".encode(), c2)
        chal = transcript.challenge_fp2(f"sc_r_{r_idx}".encode())
        challenges.append(chal)

        # Fold A and B at chal
        pa0, pa1 = fp2_vec_mul_scalar(da, chal)
        pb0, pb1 = fp2_vec_mul_scalar(db, chal)
        a0, a1 = vec_add(ae[0], pa0), vec_add(ae[1], pa1)
        b0, b1 = vec_add(be[0], pb0), vec_add(be[1], pb1)

    return ProductSumcheckProof(
        rounds=rounds,
        final_a=Fp2(int(a0[0]), int(a1[0])),
        final_b=Fp2(int(b0[0]), int(b1[0])),
        challenges=challenges,
    )


def verify_product_sumcheck(
    proof: ProductSumcheckProof,
    claimed_sum: Fp2,
    transcript: Transcript,
) -> tuple[bool, list[Fp2], Fp2, Fp2]:
    """Verify product sumcheck rounds. Caller must separately verify final_a * final_b
    against the polynomial commitments at `challenges`.
    """
    cur_sum = claimed_sum
    challenges: list[Fp2] = []
    for r_idx, poly in enumerate(proof.rounds):
        if poly.eval_zero_plus_one() != cur_sum:
            return False, [], Fp2.zero(), Fp2.zero()
        transcript.absorb_fp2(f"sc_c0_{r_idx}".encode(), poly.c0)
        transcript.absorb_fp2(f"sc_c1_{r_idx}".encode(), poly.c1)
        transcript.absorb_fp2(f"sc_c2_{r_idx}".encode(), poly.c2)
        chal = transcript.challenge_fp2(f"sc_r_{r_idx}".encode())
        challenges.append(chal)
        cur_sum = poly.eval(chal)

    if proof.final_a * proof.final_b != cur_sum:
        return False, [], Fp2.zero(), Fp2.zero()
    return True, challenges, proof.final_a, proof.final_b


def prove_batched_matmul_zk(
    matrices: Sequence[np.ndarray],
    x: np.ndarray,
    outputs: Sequence[np.ndarray],
    transcript: Transcript,
    rng: np.random.Generator | None = None,
) -> BatchedMatmulProof:
    """Prove y_i = M_i @ x for i=1..k in zero-knowledge using a single batched sumcheck.
    Reduces all k matrix verifications to point evaluations M_i(r*, c*) at a shared (r*, c*).
    """
    if rng is None:
        rng = np.random.default_rng(secrets.randbits(64))
    k = len(matrices)
    d = x.size
    log_d = d.bit_length() - 1

    # 1. Sample random row challenge r* in (F_p^2)^log_d
    row_point = transcript.challenge_point_fp2(b"mm_row", log_d)
    eq_r = eq_evals_fp2(row_point)

    # 2. Evaluate y_i(r*) for each active expert
    claimed_y_evals = [mle_eval_fp2(y_i, row_point) for y_i in outputs]
    for i, y_val in enumerate(claimed_y_evals):
        transcript.absorb_fp2(f"mm_y_{i}".encode(), y_val)

    # 3. Sample batching weights alpha_1, ..., alpha_k
    alphas = [transcript.challenge_fp2(f"mm_alpha_{i}".encode()) for i in range(k)]

    # 4. Contract M_i(r*, c) for each active expert and combine with alpha_i
    m_rows = [_contract_rows_fp2(M_i, eq_r) for M_i in matrices]

    comb0 = np.zeros(d, dtype=np.uint64)
    comb1 = np.zeros(d, dtype=np.uint64)
    for alpha_i, mr_i in zip(alphas, m_rows):
        t0, t1 = fp2_vec_mul_scalar(mr_i, alpha_i)
        comb0 = vec_add(comb0, t0)
        comb1 = vec_add(comb1, t1)

    # 5. Zero-Knowledge blinding (Chiesa-Forbes-Spooner) with succinct PCS binding:
    # Commit to input x and random masking vector Z via Brakedown PCS, absorb S_Z = sum Z(c),
    # and blind every sumcheck round polynomial with rho * g_Z(X) where rho <- F_p^2.
    from zk_moe.pcs import pcs_commit, pcs_open
    from zk_moe.field_and_mle import vec_sum
    x_pk = pcs_commit(x)
    z_mask = rng.integers(0, P, size=d, dtype=np.uint64)
    z_pk = pcs_commit(z_mask)
    s_z = Fp2.from_fp(vec_sum(z_mask))
    transcript.absorb_bytes(b"mm_x_comm", x_pk.commitment.to_bytes())
    transcript.absorb_bytes(b"mm_zk_comm", z_pk.commitment.to_bytes())
    transcript.absorb_fp2(b"mm_zk_sum", s_z)
    rho = transcript.challenge_fp2(b"mm_zk_rho")

    # Run product sumcheck on (comb, x) AND fold Z in parallel so round polys are blinded!
    a0, a1 = comb0, comb1
    b0, b1 = x.astype(np.uint64), np.zeros(d, dtype=np.uint64)
    z0, z1 = z_mask, np.zeros(d, dtype=np.uint64)
    n_vars = d.bit_length() - 1
    rounds: list[SumcheckRoundPoly] = []
    challenges: list[Fp2] = []

    for r_idx in range(n_vars):
        ae, ao = (a0[0::2], a1[0::2]), (a0[1::2], a1[1::2])
        be, bo = (b0[0::2], b1[0::2]), (b0[1::2], b1[1::2])
        ze, zo = (z0[0::2], z1[0::2]), (z0[1::2], z1[1::2])

        g0 = fp2_vec_sum(fp2_vec_mul(ae, be)) + rho * fp2_vec_sum(ze)
        g1 = fp2_vec_sum(fp2_vec_mul(ao, bo)) + rho * fp2_vec_sum(zo)
        da = fp2_vec_sub(ao, ae)
        db = fp2_vec_sub(bo, be)
        c2 = fp2_vec_sum(fp2_vec_mul(da, db))
        c0 = g0
        c1 = g1 - c0 - c2

        poly = SumcheckRoundPoly(c0=c0, c1=c1, c2=c2)
        rounds.append(poly)

        transcript.absorb_fp2(f"sc_c0_{r_idx}".encode(), c0)
        transcript.absorb_fp2(f"sc_c1_{r_idx}".encode(), c1)
        transcript.absorb_fp2(f"sc_c2_{r_idx}".encode(), c2)
        chal = transcript.challenge_fp2(f"sc_r_{r_idx}".encode())
        challenges.append(chal)

        pa0, pa1 = fp2_vec_mul_scalar(da, chal)
        pb0, pb1 = fp2_vec_mul_scalar(db, chal)
        dz = fp2_vec_sub(zo, ze)
        pz0, pz1 = fp2_vec_mul_scalar(dz, chal)
        a0, a1 = vec_add(ae[0], pa0), vec_add(ae[1], pa1)
        b0, b1 = vec_add(be[0], pb0), vec_add(be[1], pb1)
        z0, z1 = vec_add(ze[0], pz0), vec_add(ze[1], pz1)

    sc_proof = ProductSumcheckProof(
        rounds=rounds,
        final_a=Fp2(int(a0[0]), int(a1[0])),
        final_b=Fp2(int(b0[0]), int(b1[0])),
        challenges=challenges,
    )
    z_eval = Fp2(int(z0[0]), int(z1[0]))
    col_point = sc_proof.challenges

    # 6. Compute individual M_i(r*, c*) at the shared challenge (r*, c*)
    eq_c = eq_evals_fp2(col_point)
    final_m_evals = []
    for mr_i in m_rows:
        val = fp2_vec_sum(fp2_vec_mul(mr_i, eq_c))
        final_m_evals.append(val)
        transcript.absorb_fp2(b"mm_final_m", val)

    # Open x(c*) and Z(c*) via succinct PCS
    x_opening = pcs_open(x_pk, col_point, transcript)
    z_opening = pcs_open(z_pk, col_point, transcript)

    return BatchedMatmulProof(
        row_point=row_point,
        col_point=col_point,
        claimed_y_evals=claimed_y_evals,
        alpha_weights=alphas,
        sumcheck_proof=sc_proof,
        final_m_evals=final_m_evals,
        final_x_eval=sc_proof.final_b,
        blind_comm=z_pk.commitment.to_bytes(),
        blind_sum=s_z,
        blind_eval=z_eval,
        blind_challenge=rho,
        x_comm=x_pk.commitment,
        x_opening=x_opening,
        blind_pcs_comm=z_pk.commitment,
        blind_opening=z_opening,
    )


def verify_batched_matmul_zk(
    proof: BatchedMatmulProof,
    transcript: Transcript,
) -> tuple[bool, list[Fp2], list[Fp2], list[Fp2], Fp2]:
    """Verify the batched ZK matmul sumcheck with Chiesa-Forbes-Spooner blinding."""
    log_d = len(proof.row_point)
    row_point = transcript.challenge_point_fp2(b"mm_row", log_d)
    if row_point != proof.row_point:
        return False, [], [], [], Fp2.zero()

    for i, y_val in enumerate(proof.claimed_y_evals):
        transcript.absorb_fp2(f"mm_y_{i}".encode(), y_val)

    k = len(proof.claimed_y_evals)
    alphas = [transcript.challenge_fp2(f"mm_alpha_{i}".encode()) for i in range(k)]
    if alphas != proof.alpha_weights:
        return False, [], [], [], Fp2.zero()

    if proof.x_comm is None or proof.blind_pcs_comm is None:
        return False, [], [], [], Fp2.zero()
    transcript.absorb_bytes(b"mm_x_comm", proof.x_comm.to_bytes())
    transcript.absorb_bytes(b"mm_zk_comm", proof.blind_comm)
    transcript.absorb_fp2(b"mm_zk_sum", proof.blind_sum)
    rho = transcript.challenge_fp2(b"mm_zk_rho")
    if rho != proof.blind_challenge:
        return False, [], [], [], Fp2.zero()

    claimed_sum = Fp2.zero()
    for a_i, y_i in zip(alphas, proof.claimed_y_evals):
        claimed_sum = claimed_sum + a_i * y_i
    cur_sum = claimed_sum + rho * proof.blind_sum

    col_point: list[Fp2] = []
    for r_idx, poly in enumerate(proof.sumcheck_proof.rounds):
        if poly.eval_zero_plus_one() != cur_sum:
            return False, [], [], [], Fp2.zero()
        transcript.absorb_fp2(f"sc_c0_{r_idx}".encode(), poly.c0)
        transcript.absorb_fp2(f"sc_c1_{r_idx}".encode(), poly.c1)
        transcript.absorb_fp2(f"sc_c2_{r_idx}".encode(), poly.c2)
        chal = transcript.challenge_fp2(f"sc_r_{r_idx}".encode())
        col_point.append(chal)
        cur_sum = poly.eval(chal)

    if col_point != proof.col_point:
        return False, [], [], [], Fp2.zero()

    final_comb = proof.sumcheck_proof.final_a
    final_x = proof.sumcheck_proof.final_b
    if final_comb * final_x + rho * proof.blind_eval != cur_sum:
        return False, [], [], [], Fp2.zero()

    # Check consistency between final_comb and sum_i alpha_i * final_m_evals[i]
    expected_comb = Fp2.zero()
    for a_i, m_i in zip(alphas, proof.final_m_evals):
        expected_comb = expected_comb + a_i * m_i
    if expected_comb != final_comb:
        return False, [], [], [], Fp2.zero()

    for val in proof.final_m_evals:
        transcript.absorb_fp2(b"mm_final_m", val)

    # Verify succinct PCS openings for x(c*) and blinding Z(c*)
    from zk_moe.pcs import pcs_verify
    if proof.x_opening is None or proof.blind_opening is None:
        return False, [], [], [], Fp2.zero()
    if proof.x_opening.eval_point != col_point or proof.x_opening.eval_value != final_x:
        return False, [], [], [], Fp2.zero()
    if proof.blind_opening.eval_point != col_point or proof.blind_opening.eval_value != proof.blind_eval:
        return False, [], [], [], Fp2.zero()
    if not pcs_verify(proof.x_comm, proof.x_opening, transcript):
        return False, [], [], [], Fp2.zero()
    if not pcs_verify(proof.blind_pcs_comm, proof.blind_opening, transcript):
        return False, [], [], [], Fp2.zero()

    return True, row_point, col_point, proof.final_m_evals, final_x
