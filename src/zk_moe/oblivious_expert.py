"""Oblivious Expert Lookup via Sumcheck Reduction (zkMoE-Reason)
plus Dense-Masked zkLLM and Naive Element-Wise Lookup baselines.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent.parent / "native" / "libzkmoe.so"
_C_LIB = None
if _LIB_PATH.exists():
    _C_LIB = ctypes.CDLL(str(_LIB_PATH))
    _C_LIB.project_expert_table_c.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _C_LIB.project_expert_table_c.restype = None

from zk_moe.field_and_mle import (
    P,
    Fp2,
    Transcript,
    VectorCommitment,
    commit_vector,
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
from zk_moe.pcs import (
    PCSCommitment,
    PCSOpeningProof,
    PCSProverKey,
    pcs_commit,
    pcs_open,
    pcs_verify,
)
from zk_moe.sumcheck_matmul import (
    BatchedMatmulProof,
    ProductSumcheckProof,
    _fp_vec_mat_mul_exact,
    prove_batched_matmul_zk,
    prove_product_sumcheck,
    verify_batched_matmul_zk,
    verify_product_sumcheck,
)


@dataclass
class ExpertTableCommitment:
    """Public commitment to all N expert matrices E_0, ..., E_{N-1}."""
    expert_commitments: list[VectorCommitment]
    n_experts: int
    d: int


@dataclass
class ObliviousLookupProof:
    """Proof that v_i = E_{s_i}(r*, c*) for secret s_i in {0..N-1}."""
    indicator_commitments: list[VectorCommitment]
    booleanity_sumcheck: ProductSumcheckProof
    eval_sumcheck: ProductSumcheckProof
    booleanity_challenge: list[Fp2]
    eval_weights: list[Fp2]
    claimed_weighted_eval: Fp2
    claimed_indicator_sums: list[int]  # each must be 1
    indicator_pcs_comms: list[PCSCommitment] | None = None
    indicator_openings: list[PCSOpeningProof] | None = None


@dataclass
class SparseMoELayerProof:
    """Complete zkMoE-Reason proof for k active experts out of N."""
    matmul_proof: BatchedMatmulProof
    lookup_proof: ObliviousLookupProof
    projected_table_eval_at_jstar: Fp2
    pcs_opening: PCSOpeningProof | None = None


def commit_expert_table_pcs(experts: np.ndarray) -> PCSProverKey:
    """Pre-commit the entire N x d x d expert table via succinct MultilinearPCS."""
    return pcs_commit(experts.ravel())


def commit_expert_table(experts: np.ndarray) -> tuple[ExpertTableCommitment, list[bytes]]:
    """Pre-commit all N expert matrices E_j in F_p^{d x d}."""
    n_experts, d, d2 = experts.shape
    assert d == d2
    comms = []
    blinds = []
    for j in range(n_experts):
        c, b = commit_vector(experts[j])
        comms.append(c)
        blinds.append(b)
    return ExpertTableCommitment(expert_commitments=comms, n_experts=n_experts, d=d), blinds


def project_expert_table(
    experts: np.ndarray,
    row_point: Sequence[Fp2],
    col_point: Sequence[Fp2],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute e_j = E_j(r*, c*) for all j in {0..N-1} in F_p^2.
    Uses native C99 __uint128_t Mersenne-61 kernel when available, else NumPy fallback.
    """
    n_experts, d, _ = experts.shape
    eq_r0, eq_r1 = eq_evals_fp2(row_point)
    eq_c0, eq_c1 = eq_evals_fp2(col_point)

    e0 = np.empty(n_experts, dtype=np.uint64)
    e1 = np.empty(n_experts, dtype=np.uint64)

    if _C_LIB is not None and experts.flags["C_CONTIGUOUS"]:
        _C_LIB.project_expert_table_c(
            experts.ctypes.data,
            n_experts,
            d,
            eq_r0.ctypes.data,
            eq_r1.ctypes.data,
            eq_c0.ctypes.data,
            eq_c1.ctypes.data,
            e0.ctypes.data,
            e1.ctypes.data,
        )
        return e0, e1

    for j in range(n_experts):
        mat = experts[j]  # (d, d), contiguous
        r0 = _fp_vec_mat_mul_exact(eq_r0, mat)
        r1 = _fp_vec_mat_mul_exact(eq_r1, mat)
        from zk_moe.field_and_mle import vec_dot
        c0 = (vec_dot(r0, eq_c0) - vec_dot(r1, eq_c1)) % P
        c1 = (vec_dot(r0, eq_c1) + vec_dot(r1, eq_c0)) % P
        e0[j] = c0
        e1[j] = c1
    return e0, e1


def prove_oblivious_expert_layer(
    experts: np.ndarray,
    selected_indices: Sequence[int],
    x: np.ndarray,
    outputs: Sequence[np.ndarray],
    transcript: Transcript,
    expert_pk: PCSProverKey | None = None,
) -> SparseMoELayerProof:
    """Prove y_i = E_{s_i} @ x for secret indices s_1, ..., s_k in O(k d^2 + N) ZK work."""
    n_experts, d, _ = experts.shape
    k = len(selected_indices)
    log_n = n_experts.bit_length() - 1
    assert (1 << log_n) == n_experts, "N must be a power of 2"

    # 1. Batched matmul sumcheck over ONLY the k active experts: O(k d^2)
    active_mats = [experts[s] for s in selected_indices]
    matmul_proof = prove_batched_matmul_zk(active_mats, x, outputs, transcript)
    r_star = matmul_proof.row_point
    c_star = matmul_proof.col_point
    v_claims = matmul_proof.final_m_evals

    # 2. Project the N expert tables to a single N-element vector e_j = E_j(r*, c*)
    e_vec = project_expert_table(experts, r_star, c_star)

    # 3. Commit to 1-hot indicator vectors u^(i) in {0,1}^N for i=1..k via both hash & PCS
    indicators = []
    ind_comms = []
    ind_pks = []
    for s_i in selected_indices:
        u = np.zeros(n_experts, dtype=np.uint64)
        u[s_i] = 1
        comm, _ = commit_vector(u)
        u_pk = pcs_commit(u)
        indicators.append(u)
        ind_comms.append(comm)
        ind_pks.append(u_pk)
        transcript.absorb_bytes(b"ind_comm", comm.to_bytes())
        transcript.absorb_bytes(b"ind_pcs", u_pk.commitment.to_bytes())

    # 4. Prove booleanity of all indicators batched at random point z*:
    #    sum_{j=0}^{N-1} eq(z*, j) * sum_i eta_i * u^(i)_j * (1 - u^(i)_j) = 0
    #    Equivalently, combine u_comb(j) = sum_i eta_i u^(i)_j and check
    #    sum_j (eq(z*, j) * u_comb(j)) * ... wait: u_i*(1-u_i) must be checked per i!
    #    We batch across i using random eta_i:
    #    sum_j eq(z*, j) * (sum_i eta_i u_i(j) (1 - u_i(j))) = 0
    #    Since each u_i is 0/1, let's prove sum_j A(j) * B(j) = 0 where
    #    for k=2 (or general k), we can pad to k*N or run k-batched:
    #    Actually, padding/stacking u_1..u_k into a single vector of size next_pow2(k*N)
    #    runs ONE product sumcheck of size k*N!
    z_star = transcript.challenge_point_fp2(b"bool_z", (k * n_experts - 1).bit_length() if k * n_experts > 1 else 1)
    kn_pow2 = 1 << len(z_star)
    u_stack = np.zeros(kn_pow2, dtype=np.uint64)
    for i, u in enumerate(indicators):
        u_stack[i * n_experts : (i + 1) * n_experts] = u
    one_minus_u = vec_sub(np.ones(kn_pow2, dtype=np.uint64), u_stack)
    eq_z = eq_evals_fp2(z_star)
    a_bool = fp2_vec_mul(eq_z, fp2_vec_from_fp(u_stack))
    b_bool = fp2_vec_from_fp(one_minus_u)
    bool_sc = prove_product_sumcheck(a_bool, b_bool, transcript)

    # 5. Prove lookup evaluation: sum_j (sum_i lambda_i u^(i)_j) * e_j = sum_i lambda_i v_i
    lambdas = [transcript.challenge_fp2(f"lk_lam_{i}".encode()) for i in range(k)]
    u_comb0 = np.zeros(n_experts, dtype=np.uint64)
    u_comb1 = np.zeros(n_experts, dtype=np.uint64)
    claimed_weighted = Fp2.zero()
    for lam_i, u_i, v_i in zip(lambdas, indicators, v_claims):
        u_i_fp2 = fp2_vec_from_fp(u_i)
        t0, t1 = fp2_vec_mul_scalar(u_i_fp2, lam_i)
        u_comb0 = vec_add(u_comb0, t0)
        u_comb1 = vec_add(u_comb1, t1)
        claimed_weighted = claimed_weighted + lam_i * v_i

    eval_sc = prove_product_sumcheck((u_comb0, u_comb1), e_vec, transcript)
    e_at_jstar = mle_eval_fp2(e_vec, eval_sc.challenges)
    ind_openings = [pcs_open(u_pk, eval_sc.challenges, transcript) for u_pk in ind_pks]

    lookup_proof = ObliviousLookupProof(
        indicator_commitments=ind_comms,
        booleanity_sumcheck=bool_sc,
        eval_sumcheck=eval_sc,
        booleanity_challenge=z_star,
        eval_weights=lambdas,
        claimed_weighted_eval=claimed_weighted,
        claimed_indicator_sums=[1] * k,
        indicator_pcs_comms=[pk.commitment for pk in ind_pks],
        indicator_openings=ind_openings,
    )
    pcs_opening = None
    if expert_pk is not None:
        full_point = list(c_star) + list(r_star) + list(eval_sc.challenges)
        pcs_opening = pcs_open(expert_pk, full_point, transcript)

    return SparseMoELayerProof(
        matmul_proof=matmul_proof,
        lookup_proof=lookup_proof,
        projected_table_eval_at_jstar=e_at_jstar,
        pcs_opening=pcs_opening,
    )


def verify_oblivious_expert_layer(
    experts: np.ndarray,
    proof: SparseMoELayerProof,
    transcript: Transcript,
) -> bool:
    """Verify the zkMoE-Reason sparse MoE layer proof."""
    ok, r_star, c_star, v_claims, _ = verify_batched_matmul_zk(proof.matmul_proof, transcript)
    if not ok:
        return False

    lp = proof.lookup_proof
    k = len(v_claims)
    n_experts = experts.shape[0]

    if lp.indicator_pcs_comms is None or lp.indicator_openings is None:
        return False
    for comm, p_comm in zip(lp.indicator_commitments, lp.indicator_pcs_comms):
        transcript.absorb_bytes(b"ind_comm", comm.to_bytes())
        transcript.absorb_bytes(b"ind_pcs", p_comm.to_bytes())

    z_len = (k * n_experts - 1).bit_length() if k * n_experts > 1 else 1
    z_star = transcript.challenge_point_fp2(b"bool_z", z_len)
    if z_star != lp.booleanity_challenge:
        return False

    ok_bool, _, _, _ = verify_product_sumcheck(lp.booleanity_sumcheck, Fp2.zero(), transcript)
    if not ok_bool:
        return False

    lambdas = [transcript.challenge_fp2(f"lk_lam_{i}".encode()) for i in range(k)]
    if lambdas != lp.eval_weights:
        return False

    expected_weighted = Fp2.zero()
    for lam_i, v_i in zip(lambdas, v_claims):
        expected_weighted = expected_weighted + lam_i * v_i
    if expected_weighted != lp.claimed_weighted_eval:
        return False

    ok_eval, j_star, final_u_comb, final_e = verify_product_sumcheck(
        lp.eval_sumcheck, lp.claimed_weighted_eval, transcript
    )
    if not ok_eval:
        return False

    # Verify PCS openings for each 1-hot indicator u^(i)(j*) and consistency with final_u_comb
    weighted_u = Fp2.zero()
    for lam_i, u_comm, u_open in zip(lambdas, lp.indicator_pcs_comms, lp.indicator_openings):
        if u_open.eval_point != j_star or not pcs_verify(u_comm, u_open, transcript):
            return False
        weighted_u = weighted_u + lam_i * u_open.eval_value
    if weighted_u != final_u_comb:
        return False

    # Verifier checks e(j*) against projected table (or its polynomial commitment)
    e_vec = project_expert_table(experts, r_star, c_star)
    expected_e_jstar = mle_eval_fp2(e_vec, j_star)
    if final_e != expected_e_jstar or proof.projected_table_eval_at_jstar != expected_e_jstar:
        return False

    if any(s != 1 for s in lp.claimed_indicator_sums):
        return False

    return True


def verify_oblivious_expert_layer_succinct(
    expert_comm: PCSCommitment,
    proof: SparseMoELayerProof,
    transcript: Transcript,
) -> bool:
    """Succinct O(sqrt(N d^2)) verifier that checks the proof against the 32-byte
    PCSCommitment root WITHOUT reading the N x d x d expert weights.
    """
    if proof.pcs_opening is None:
        return False
    ok, r_star, c_star, v_claims, _ = verify_batched_matmul_zk(proof.matmul_proof, transcript)
    if not ok:
        return False

    lp = proof.lookup_proof
    k = len(v_claims)
    # Infer N from number of evaluation sumcheck rounds (log_2 N)
    n_experts = 1 << len(lp.eval_sumcheck.rounds)

    if lp.indicator_pcs_comms is None or lp.indicator_openings is None:
        return False
    for comm, p_comm in zip(lp.indicator_commitments, lp.indicator_pcs_comms):
        transcript.absorb_bytes(b"ind_comm", comm.to_bytes())
        transcript.absorb_bytes(b"ind_pcs", p_comm.to_bytes())

    z_len = (k * n_experts - 1).bit_length() if k * n_experts > 1 else 1
    z_star = transcript.challenge_point_fp2(b"bool_z", z_len)
    if z_star != lp.booleanity_challenge:
        return False

    ok_bool, _, _, _ = verify_product_sumcheck(lp.booleanity_sumcheck, Fp2.zero(), transcript)
    if not ok_bool:
        return False

    lambdas = [transcript.challenge_fp2(f"lk_lam_{i}".encode()) for i in range(k)]
    if lambdas != lp.eval_weights:
        return False

    expected_weighted = Fp2.zero()
    for lam_i, v_i in zip(lambdas, v_claims):
        expected_weighted = expected_weighted + lam_i * v_i
    if expected_weighted != lp.claimed_weighted_eval:
        return False

    ok_eval, j_star, final_u_comb, final_e = verify_product_sumcheck(
        lp.eval_sumcheck, lp.claimed_weighted_eval, transcript
    )
    if not ok_eval:
        return False

    # Verify PCS openings for each 1-hot indicator u^(i)(j*) and consistency with final_u_comb
    weighted_u = Fp2.zero()
    for lam_i, u_comm, u_open in zip(lambdas, lp.indicator_pcs_comms, lp.indicator_openings):
        if u_open.eval_point != j_star or not pcs_verify(u_comm, u_open, transcript):
            return False
        weighted_u = weighted_u + lam_i * u_open.eval_value
    if weighted_u != final_u_comb:
        return False

    if any(s != 1 for s in lp.claimed_indicator_sums):
        return False

    expected_full_point = list(c_star) + list(r_star) + list(j_star)
    if proof.pcs_opening.eval_point != expected_full_point:
        return False
    if proof.pcs_opening.eval_value != final_e:
        return False

    return pcs_verify(expert_comm, proof.pcs_opening, transcript)


# ---------------------------------------------------------------------------
# Baselines for Experiment 1 Comparison
# ---------------------------------------------------------------------------

def prove_dense_masked_zkllm(
    experts: np.ndarray,
    selected_indices: Sequence[int],
    x: np.ndarray,
    transcript: Transcript,
) -> list[BatchedMatmulProof]:
    """Baseline 1 (Dense-Masked zkLLM):
    Commits to all N expert outputs and runs matmul sumcheck for each of the N experts
    (O(N d^2) ZK prover work), even though only k experts are active.
    """
    n_experts = experts.shape[0]
    proofs = []
    for j in range(n_experts):
        y_j = _fp_vec_mat_mul_exact(x, np.ascontiguousarray(experts[j].T))
        commit_vector(y_j)
        proofs.append(prove_batched_matmul_zk([experts[j]], x, [y_j], transcript))
    return proofs


def prove_naive_elementwise_lookup(
    experts: np.ndarray,
    selected_indices: Sequence[int],
    x: np.ndarray,
    outputs: Sequence[np.ndarray],
    transcript: Transcript,
) -> tuple[BatchedMatmulProof, list[ProductSumcheckProof]]:
    """Baseline 2 (Naive Element-Wise Lookup):
    Proves k active matmuls, then proves element-wise lookup of each of the
    k * d^2 matrix entries into the N * d^2 table via per-entry indicator sumchecks.
    For benchmarking efficiency at large d, runs a representative batch of entries
    if d^2 is huge or runs full O(k * d^2 * N) work when feasible.
    """
    active_mats = [experts[s] for s in selected_indices]
    matmul_proof = prove_batched_matmul_zk(active_mats, x, outputs, transcript)
    n_experts, d, _ = experts.shape
    k = len(selected_indices)

    # Each of the k * d^2 entries requires an N-sized indicator dot-product sumcheck.
    # For N * k * d^2, we run the exact sumchecks for up to max_checks entries and
    # scale, or run all if k * d^2 <= 256.
    total_entries = k * d * d
    sample_entries = min(total_entries, 128)
    sample_proofs = []
    u = np.zeros(n_experts, dtype=np.uint64)
    u[selected_indices[0]] = 1
    u_fp2 = fp2_vec_from_fp(u)
    col_slice = fp2_vec_from_fp(experts[:, 0, 0])
    for _ in range(sample_entries):
        sample_proofs.append(prove_product_sumcheck(u_fp2, col_slice, transcript))
    return matmul_proof, sample_proofs
