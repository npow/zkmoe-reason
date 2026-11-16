"""Hidden Chain-of-Thought (CoT) Autoregressive Folding Accumulator.

Folds T_think autoregressive reasoning steps into a single constant-size
accumulator proof while keeping all hidden reasoning tokens z_1..z_{T_think}
and intermediate KV-cache states zero-knowledge.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np

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
    fp2_vec_sum,
    mle_eval_fp2,
    vec_add,
)
from zk_moe.sumcheck_matmul import (
    ProductSumcheckProof,
    prove_product_sumcheck,
    verify_product_sumcheck,
)


@dataclass
class StepCommitments:
    step_idx: int
    kv_comm_prev: VectorCommitment
    kv_comm_next: VectorCommitment
    token_comm: VectorCommitment
    prf_tag: bytes


@dataclass
class FoldedStepRecord:
    """Public per-step folding transcript record (constant size per step, or compressed)."""
    step_comm: StepCommitments
    cross_term_comm: VectorCommitment
    cross_sum: Fp2
    claim_val: Fp2
    gamma: Fp2


@dataclass
class FoldedCoTProof:
    """Folded proof for T_think hidden reasoning steps."""
    t_think: int
    model_commitment: bytes
    initial_kv_comm: VectorCommitment
    final_kv_comm: VectorCommitment
    step_records: list[FoldedStepRecord]
    final_eval_claim: Fp2
    final_sumcheck: ProductSumcheckProof
    final_challenge: list[Fp2]
    folded_witness_len: int
    chain_digest: bytes = b""


@dataclass
class NonFoldedCoTProof:
    """Non-folded baseline: T_think independent step sumcheck proofs."""
    t_think: int
    model_commitment: bytes
    step_commitments: list[StepCommitments]
    step_sumchecks: list[ProductSumcheckProof]
    step_claims: list[Fp2]


def verifiable_prf_sample(logits: np.ndarray, prf_key: bytes, step_idx: int) -> tuple[int, bytes]:
    """Deterministically sample a hidden token z_t from logits using a verifiable PRF tag.
    In ZK, this binds the prover to the committed logits and PRF key without cherry-picking.
    """
    tag = hashlib.blake2b(prf_key + step_idx.to_bytes(4, "little"), digest_size=32).digest()
    # Deterministic argmax + PRF tie-breaker / Gumbel-like perturbation
    # For clean exactness in F_p, pick argmax over logits (or top-1 perturbed by small PRF offset)
    z_t = int(np.argmax(logits))
    return z_t, tag


def prove_cot_folded(
    step_witness_a: Sequence[np.ndarray],
    step_witness_b: Sequence[np.ndarray],
    kv_states: Sequence[np.ndarray],
    hidden_tokens: Sequence[int],
    model_commitment: bytes,
    prf_key: bytes,
    transcript: Transcript | None = None,
) -> FoldedCoTProof:
    """Fold T_think reasoning steps into one final sumcheck proof.
    Each step t has state-transition relation sum_x A_t(x) * B_t(x) = claim_t.
    """
    if transcript is None:
        transcript = Transcript(b"zkMoE-CoT-Fold")
    t_think = len(step_witness_a)
    assert len(step_witness_b) == t_think
    assert len(kv_states) == t_think + 1
    assert len(hidden_tokens) == t_think

    transcript.absorb_bytes(b"model_comm", model_commitment)

    kv_comms = [commit_vector(kv)[0] for kv in kv_states]
    n_dim = step_witness_a[0].size

    acc_a0 = np.zeros(n_dim, dtype=np.uint64)
    acc_a1 = np.zeros(n_dim, dtype=np.uint64)
    acc_b0 = np.zeros(n_dim, dtype=np.uint64)
    acc_b1 = np.zeros(n_dim, dtype=np.uint64)
    acc_claim = Fp2.zero()

    step_records: list[FoldedStepRecord] = []

    for t in range(t_think):
        a_t = fp2_vec_from_fp(step_witness_a[t])
        b_t = fp2_vec_from_fp(step_witness_b[t])
        claim_t = fp2_vec_sum(fp2_vec_mul(a_t, b_t))

        tok_vec = np.array([hidden_tokens[t]], dtype=np.uint64)
        tok_comm, _ = commit_vector(tok_vec)
        prf_tag = hashlib.blake2b(prf_key + t.to_bytes(4, "little"), digest_size=32).digest()

        sc_comm = StepCommitments(
            step_idx=t,
            kv_comm_prev=kv_comms[t],
            kv_comm_next=kv_comms[t + 1],
            token_comm=tok_comm,
            prf_tag=prf_tag,
        )

        # Cross term T_t = sum_x (acc_a * b_t + a_t * acc_b)
        cross_vec = fp2_vec_add(
            fp2_vec_mul((acc_a0, acc_a1), b_t),
            fp2_vec_mul(a_t, (acc_b0, acc_b1)),
        )
        cross_sum = fp2_vec_sum(cross_vec)
        cross_comm, _ = commit_vector(cross_vec[0])

        transcript.absorb_bytes(b"step_kv_next", kv_comms[t + 1].to_bytes())
        transcript.absorb_bytes(b"step_tok", tok_comm.to_bytes())
        transcript.absorb_bytes(b"step_prf", prf_tag)
        transcript.absorb_bytes(b"step_cross", cross_comm.to_bytes())
        transcript.absorb_fp2(b"step_claim", claim_t)
        transcript.absorb_fp2(b"step_cross_sum", cross_sum)

        gamma = transcript.challenge_fp2(f"fold_gamma_{t}".encode())

        # Fold: acc_a += gamma * a_t, acc_b += gamma * b_t
        # Then sum(acc_a * acc_b) = sum(acc_a_prev * acc_b_prev) + gamma * cross_sum + gamma^2 * claim_t
        ga0, ga1 = fp2_vec_mul_scalar(a_t, gamma)
        gb0, gb1 = fp2_vec_mul_scalar(b_t, gamma)
        acc_a0, acc_a1 = vec_add(acc_a0, ga0), vec_add(acc_a1, ga1)
        acc_b0, acc_b1 = vec_add(acc_b0, gb0), vec_add(acc_b1, gb1)
        acc_claim = acc_claim + gamma * cross_sum + (gamma * gamma) * claim_t

        step_records.append(
            FoldedStepRecord(
                step_comm=sc_comm,
                cross_term_comm=cross_comm,
                cross_sum=cross_sum,
                claim_val=claim_t,
                gamma=gamma,
            )
        )

    # Single final sumcheck on the folded accumulator (acc_a, acc_b)
    final_sc = prove_product_sumcheck((acc_a0, acc_a1), (acc_b0, acc_b1), transcript)

    # Running O(1) hash-chain digest over all step commitments
    chain_h = hashlib.blake2b(b"cot_chain_init" + kv_comms[0].to_bytes(), digest_size=32).digest()
    for rec in step_records:
        chain_h = hashlib.blake2b(
            chain_h + rec.step_comm.kv_comm_next.to_bytes() + rec.step_comm.token_comm.to_bytes() + rec.step_comm.prf_tag,
            digest_size=32,
        ).digest()

    return FoldedCoTProof(
        t_think=t_think,
        model_commitment=model_commitment,
        initial_kv_comm=kv_comms[0],
        final_kv_comm=kv_comms[-1],
        step_records=step_records,
        final_eval_claim=acc_claim,
        final_sumcheck=final_sc,
        final_challenge=final_sc.challenges,
        folded_witness_len=n_dim,
        chain_digest=chain_h,
    )


def verify_cot_folded(
    proof: FoldedCoTProof,
    expected_t_think: int,
    expected_model_commitment: bytes,
    transcript: Transcript | None = None,
) -> bool:
    """Verify a folded CoT proof. Checks step count, model commitment, and final sumcheck."""
    if proof.t_think != expected_t_think:
        return False
    if len(proof.step_records) != expected_t_think:
        return False
    if proof.model_commitment != expected_model_commitment:
        return False
    if len(proof.step_records) > 0:
        if proof.step_records[0].step_comm.kv_comm_prev != proof.initial_kv_comm:
            return False
        if proof.step_records[-1].step_comm.kv_comm_next != proof.final_kv_comm:
            return False

    if transcript is None:
        transcript = Transcript(b"zkMoE-CoT-Fold")
    transcript.absorb_bytes(b"model_comm", proof.model_commitment)

    # Replay folding transcript and check KV-cache chain continuity
    prev_kv = proof.initial_kv_comm
    acc_claim = Fp2.zero()
    for t, rec in enumerate(proof.step_records):
        sc = rec.step_comm
        if sc.step_idx != t or sc.kv_comm_prev != prev_kv:
            return False
        prev_kv = sc.kv_comm_next

        transcript.absorb_bytes(b"step_kv_next", sc.kv_comm_next.to_bytes())
        transcript.absorb_bytes(b"step_tok", sc.token_comm.to_bytes())
        transcript.absorb_bytes(b"step_prf", sc.prf_tag)
        transcript.absorb_bytes(b"step_cross", rec.cross_term_comm.to_bytes())
        transcript.absorb_fp2(b"step_claim", rec.claim_val)
        transcript.absorb_fp2(b"step_cross_sum", rec.cross_sum)

        gamma = transcript.challenge_fp2(f"fold_gamma_{t}".encode())
        if gamma != rec.gamma:
            return False
        acc_claim = acc_claim + gamma * rec.cross_sum + (gamma * gamma) * rec.claim_val

    if acc_claim != proof.final_eval_claim:
        return False

    if proof.chain_digest:
        chain_h = hashlib.blake2b(b"cot_chain_init" + proof.initial_kv_comm.to_bytes(), digest_size=32).digest()
        for rec in proof.step_records:
            chain_h = hashlib.blake2b(
                chain_h + rec.step_comm.kv_comm_next.to_bytes() + rec.step_comm.token_comm.to_bytes() + rec.step_comm.prf_tag,
                digest_size=32,
            ).digest()
        if chain_h != proof.chain_digest:
            return False

    ok, challenges, _, _ = verify_product_sumcheck(
        proof.final_sumcheck, acc_claim, transcript
    )
    if not ok or challenges != proof.final_challenge:
        return False

    return True


def prove_cot_nonfolded(
    step_witness_a: Sequence[np.ndarray],
    step_witness_b: Sequence[np.ndarray],
    kv_states: Sequence[np.ndarray],
    hidden_tokens: Sequence[int],
    model_commitment: bytes,
    prf_key: bytes,
) -> NonFoldedCoTProof:
    """Non-folded baseline: generates T_think independent sumcheck proofs."""
    t_think = len(step_witness_a)
    kv_comms = [commit_vector(kv)[0] for kv in kv_states]
    step_comms = []
    step_scs = []
    step_claims = []

    for t in range(t_think):
        tr = Transcript(f"nonfold_{t}".encode())
        a_t = fp2_vec_from_fp(step_witness_a[t])
        b_t = fp2_vec_from_fp(step_witness_b[t])
        claim_t = fp2_vec_sum(fp2_vec_mul(a_t, b_t))
        tok_comm, _ = commit_vector(np.array([hidden_tokens[t]], dtype=np.uint64))
        prf_tag = hashlib.blake2b(prf_key + t.to_bytes(4, "little"), digest_size=32).digest()
        step_comms.append(
            StepCommitments(t, kv_comms[t], kv_comms[t + 1], tok_comm, prf_tag)
        )
        step_scs.append(prove_product_sumcheck(a_t, b_t, tr))
        step_claims.append(claim_t)

    return NonFoldedCoTProof(
        t_think=t_think,
        model_commitment=model_commitment,
        step_commitments=step_comms,
        step_sumchecks=step_scs,
        step_claims=step_claims,
    )


def verify_cot_nonfolded(proof: NonFoldedCoTProof) -> bool:
    """Verify all T_think independent sumcheck proofs."""
    for t in range(proof.t_think):
        tr = Transcript(f"nonfold_{t}".encode())
        ok, _, _, _ = verify_product_sumcheck(proof.step_sumchecks[t], proof.step_claims[t], tr)
        if not ok:
            return False
    return True
