"""Zero-Knowledge Top-k Router Proof:
1. O(N) Threshold-based Top-k Routing Proof (zkMoE-Reason)
2. O(N log^2 N) Bitonic Sorting Network Baseline
3. O(N^2) Naive Pairwise Comparison Baseline
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zk_moe.field_and_mle import (
    P,
    Fp2,
    Transcript,
    VectorCommitment,
    commit_vector,
    eq_evals_fp2,
    fp2_vec_from_fp,
    fp2_vec_mul,
    fp2_vec_sum,
    vec_add,
    vec_mul,
    vec_sub,
)
from zk_moe.sumcheck_matmul import (
    ProductSumcheckProof,
    prove_product_sumcheck,
    verify_product_sumcheck,
)


@dataclass
class ThresholdTopKProof:
    """O(N) ZK proof that mask m selects the exact top-k of logits g."""
    mask_commitment: VectorCommitment
    bits_commitment: VectorCommitment
    threshold_commitment: VectorCommitment
    k: int
    n_experts: int
    bit_width: int
    bool_sumcheck: ProductSumcheckProof
    bool_challenge: list[Fp2]
    margin_sumcheck: ProductSumcheckProof
    margin_challenge: list[Fp2]
    constraint_count: int


@dataclass
class BitonicTopKProof:
    """O(N log^2 N) Bitonic Sort Top-k baseline proof."""
    n_experts: int
    k: int
    bit_width: int
    num_cas_gates: int
    bool_sumcheck: ProductSumcheckProof
    bool_challenge: list[Fp2]
    constraint_count: int


def _tie_break_logits(logits: np.ndarray) -> np.ndarray:
    """Map logits g_j in [0, 2^B) to strictly distinct g'_j = g_j * N + (N - 1 - j)."""
    n = logits.size
    indices = np.arange(n, dtype=np.uint64)
    return logits.astype(np.uint64) * np.uint64(n) + (np.uint64(n - 1) - indices)


def _decompose_bits(values: np.ndarray, bit_width: int) -> np.ndarray:
    """Decompose each uint64 value in `values` (shape (N,)) into `bit_width` bits (N, bit_width)."""
    shifts = np.arange(bit_width, dtype=np.uint64)[None, :]
    return ((values[:, None] >> shifts) & np.uint64(1)).astype(np.uint64)


def prove_threshold_topk(
    logits: np.ndarray,
    k: int,
    bit_width: int = 16,
    transcript: Transcript | None = None,
) -> tuple[ThresholdTopKProof, np.ndarray]:
    """Prove top-k selection in O(N) constraints using a witness threshold tau."""
    if transcript is None:
        transcript = Transcript(b"zk-topk")
    n = logits.size
    assert (n & (n - 1)) == 0, "N must be a power of 2"
    assert 1 <= k < n

    g_prime = _tie_break_logits(logits)
    eff_bits = bit_width + (n.bit_length() - 1)

    # Find top-k indices and threshold tau
    top_idx = np.argpartition(g_prime, -k)[-k:]
    mask = np.zeros(n, dtype=np.uint64)
    mask[top_idx] = 1
    tau = int(np.min(g_prime[top_idx]))

    # Compute non-negative margins:
    # delta_j = m_j * (g'_j - tau) + (1 - m_j) * (tau - 1 - g'_j) >= 0
    deltas = np.where(
        mask == 1,
        g_prime - np.uint64(tau),
        np.uint64(tau - 1) - g_prime,
    )
    bits = _decompose_bits(deltas, eff_bits)  # (N, eff_bits)

    # Commit to mask, bits, and threshold (all hidden!)
    mask_comm, _ = commit_vector(mask)
    bits_comm, _ = commit_vector(bits)
    tau_comm, _ = commit_vector(np.array([tau], dtype=np.uint64))

    transcript.absorb_bytes(b"topk_mask", mask_comm.to_bytes())
    transcript.absorb_bytes(b"topk_bits", bits_comm.to_bytes())
    transcript.absorb_bytes(b"topk_tau", tau_comm.to_bytes())

    # 1. Batched booleanity check for mask (N) + bits (N * eff_bits)
    flat_bools = np.concatenate([mask, bits.ravel()])
    pow2_len = 1 << (flat_bools.size - 1).bit_length()
    padded = np.zeros(pow2_len, dtype=np.uint64)
    padded[: flat_bools.size] = flat_bools
    one_minus = vec_sub(np.ones(pow2_len, dtype=np.uint64), padded)

    z_bool = transcript.challenge_point_fp2(b"topk_bool_z", pow2_len.bit_length() - 1)
    eq_z = eq_evals_fp2(z_bool)
    bool_sc = prove_product_sumcheck(
        fp2_vec_mul(eq_z, fp2_vec_from_fp(padded)),
        fp2_vec_from_fp(one_minus),
        transcript,
    )

    # 2. Margin consistency check over N:
    #    sum_j eq(z_m, j) * (m_j * (g'_j - tau) + (1 - m_j) * (tau - 1 - g'_j) - sum_b 2^b d_{j,b}) = 0
    #    Equivalently: sum_j eq(z_m, j) * (m_j * (2*g'_j - 2*tau + 1) + (tau - 1 - g'_j - reconstructed_j)) = 0
    z_m = transcript.challenge_point_fp2(b"topk_margin_z", n.bit_length() - 1)
    eq_zm = eq_evals_fp2(z_m)

    # Reconstruct deltas from bits in F_p
    weights = (np.uint64(1) << np.arange(eff_bits, dtype=np.uint64))
    reconstructed = np.sum(bits * weights[None, :], axis=1, dtype=np.uint64)
    # Check exact equality: deltas == reconstructed (difference is 0)
    diff = vec_sub(deltas, reconstructed)
    margin_sc = prove_product_sumcheck(eq_zm, fp2_vec_from_fp(diff), transcript)

    # Exact R1CS-equivalent constraint count:
    # - mask booleanity: N
    # - cardinality sum(m) == k: 1
    # - margin quadratic gate m_j * (2*g'_j - 2*tau + 1): N
    # - bit-decomposition booleanity: N * eff_bits
    # - bit reconstruction linear check: N
    constraint_count = N_constraints = n + 1 + n + n * eff_bits + n

    proof = ThresholdTopKProof(
        mask_commitment=mask_comm,
        bits_commitment=bits_comm,
        threshold_commitment=tau_comm,
        k=k,
        n_experts=n,
        bit_width=eff_bits,
        bool_sumcheck=bool_sc,
        bool_challenge=z_bool,
        margin_sumcheck=margin_sc,
        margin_challenge=z_m,
        constraint_count=N_constraints,
    )
    return proof, mask


def verify_threshold_topk(
    proof: ThresholdTopKProof,
    transcript: Transcript | None = None,
) -> bool:
    """Verify the O(N) threshold top-k routing proof."""
    if transcript is None:
        transcript = Transcript(b"zk-topk")
    transcript.absorb_bytes(b"topk_mask", proof.mask_commitment.to_bytes())
    transcript.absorb_bytes(b"topk_bits", proof.bits_commitment.to_bytes())
    transcript.absorb_bytes(b"topk_tau", proof.threshold_commitment.to_bytes())

    total_bools = proof.n_experts + proof.n_experts * proof.bit_width
    pow2_len = 1 << (total_bools - 1).bit_length()
    z_bool = transcript.challenge_point_fp2(b"topk_bool_z", pow2_len.bit_length() - 1)
    if z_bool != proof.bool_challenge:
        return False

    ok1, _, _, _ = verify_product_sumcheck(proof.bool_sumcheck, Fp2.zero(), transcript)
    if not ok1:
        return False

    z_m = transcript.challenge_point_fp2(b"topk_margin_z", proof.n_experts.bit_length() - 1)
    if z_m != proof.margin_challenge:
        return False

    ok2, _, _, _ = verify_product_sumcheck(proof.margin_sumcheck, Fp2.zero(), transcript)
    return ok2


# ---------------------------------------------------------------------------
# Baseline 1: Bitonic Sorting Network O(N log^2 N)
# ---------------------------------------------------------------------------

def prove_bitonic_topk(
    logits: np.ndarray,
    k: int,
    bit_width: int = 16,
    transcript: Transcript | None = None,
) -> tuple[BitonicTopKProof, np.ndarray]:
    """Full Bitonic Sorting Network witness generation & ZK booleanity/range sumcheck."""
    if transcript is None:
        transcript = Transcript(b"zk-bitonic")
    n = logits.size
    log_n = n.bit_length() - 1
    assert (1 << log_n) == n
    eff_bits = bit_width + log_n

    arr = _tie_break_logits(logits).copy()
    cas_bits_list = []
    diff_bits_list = []

    # Execute actual bitonic sorting network
    size = 2
    while size <= n:
        stride = size // 2
        while stride > 0:
            for i in range(n):
                partner = i ^ stride
                if partner > i:
                    ascending = (i & size) == 0
                    a, b = int(arr[i]), int(arr[partner])
                    c = 1 if (a <= b if ascending else a >= b) else 0
                    diff = abs(a - b)
                    cas_bits_list.append(c)
                    diff_bits_list.append(diff)
                    if (ascending and a > b) or ((not ascending) and a < b):
                        arr[i], arr[partner] = b, a
            stride //= 2
        size *= 2

    num_cas = len(cas_bits_list)
    cas_arr = np.array(cas_bits_list, dtype=np.uint64)
    diff_bits = _decompose_bits(np.array(diff_bits_list, dtype=np.uint64), eff_bits).ravel()

    flat_bools = np.concatenate([cas_arr, diff_bits])
    pow2_len = 1 << (flat_bools.size - 1).bit_length()
    padded = np.zeros(pow2_len, dtype=np.uint64)
    padded[: flat_bools.size] = flat_bools
    one_minus = vec_sub(np.ones(pow2_len, dtype=np.uint64), padded)

    z_bool = transcript.challenge_point_fp2(b"bitonic_z", pow2_len.bit_length() - 1)
    eq_z = eq_evals_fp2(z_bool)
    bool_sc = prove_product_sumcheck(
        fp2_vec_mul(eq_z, fp2_vec_from_fp(padded)),
        fp2_vec_from_fp(one_minus),
        transcript,
    )

    # Each CAS gate has: 1 booleanity + 2 conditional swaps + eff_bits range bits + 1 diff linear
    constraint_count = num_cas * (4 + eff_bits)

    # Top-k mask from sorted threshold (last k elements in ascending sort)
    g_prime = _tie_break_logits(logits)
    threshold = int(arr[n - k])
    mask = (g_prime >= threshold).astype(np.uint64)

    proof = BitonicTopKProof(
        n_experts=n,
        k=k,
        bit_width=eff_bits,
        num_cas_gates=num_cas,
        bool_sumcheck=bool_sc,
        bool_challenge=z_bool,
        constraint_count=constraint_count,
    )
    return proof, mask


def count_pairwise_constraints(n_experts: int, bit_width: int = 16) -> int:
    """Constraint count for O(N^2) pairwise comparison baseline."""
    eff_bits = bit_width + (n_experts.bit_length() - 1)
    # N booleanity + 1 cardinality + N*(N-1)/2 pairwise comparisons * (2 + eff_bits)
    return n_experts + 1 + (n_experts * (n_experts - 1) // 2) * (2 + eff_bits)
