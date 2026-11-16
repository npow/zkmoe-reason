"""Reference Sparse Mixture-of-Experts (MoE) Transformer over F_p
integrated with zkMoE-Reason proof generation and verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from zk_moe.cot_folding import (
    FoldedCoTProof,
    prove_cot_folded,
    verifiable_prf_sample,
    verify_cot_folded,
)
from zk_moe.field_and_mle import (
    P,
    Transcript,
    VectorCommitment,
    commit_vector,
    vec_add,
    vec_mod,
)
from zk_moe.oblivious_expert import (
    ExpertTableCommitment,
    SparseMoELayerProof,
    commit_expert_table,
    prove_oblivious_expert_layer,
    verify_oblivious_expert_layer,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact
from zk_moe.zk_topk_router import (
    ThresholdTopKProof,
    prove_threshold_topk,
    verify_threshold_topk,
)


@dataclass
class SparseMoEConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_experts: int = 8
    top_k: int = 2
    bit_width: int = 16


@dataclass
class ModelCommitment:
    expert_table_comm: ExpertTableCommitment
    router_comm: VectorCommitment
    attn_comm: VectorCommitment
    lm_head_comm: VectorCommitment
    digest: bytes


@dataclass
class EndToEndReasoningProof:
    model_digest: bytes
    prompt_tokens: list[int]
    answer_token: int
    t_think: int
    router_proofs: list[ThresholdTopKProof]
    moe_layer_proofs: list[SparseMoELayerProof]
    cot_proof: FoldedCoTProof


class SparseMoETransformer:
    """Reference Sparse MoE Transformer operating in fixed-point F_p."""

    def __init__(self, cfg: SparseMoEConfig, seed: int = 42) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        # Small quantized weights in [0, 2^8) so activations stay well within [0, 2^24)
        self.embed = rng.integers(0, 64, size=(cfg.vocab_size, cfg.d_model), dtype=np.uint64)
        self.w_qkv = rng.integers(0, 16, size=(cfg.d_model, cfg.d_model), dtype=np.uint64)
        self.w_router = rng.integers(0, 16, size=(cfg.d_model, cfg.n_experts), dtype=np.uint64)
        self.experts = rng.integers(
            0, 16, size=(cfg.n_experts, cfg.d_model, cfg.d_model), dtype=np.uint64
        )
        self.lm_head = rng.integers(0, 16, size=(cfg.d_model, cfg.vocab_size), dtype=np.uint64)

    def commit(self) -> ModelCommitment:
        exp_comm, _ = commit_expert_table(self.experts)
        r_comm, _ = commit_vector(self.w_router)
        a_comm, _ = commit_vector(self.w_qkv)
        h_comm, _ = commit_vector(self.lm_head)

        h = hashlib.blake2b(digest_size=32)
        for ec in exp_comm.expert_commitments:
            h.update(ec.to_bytes())
        h.update(r_comm.to_bytes())
        h.update(a_comm.to_bytes())
        h.update(h_comm.to_bytes())
        return ModelCommitment(
            expert_table_comm=exp_comm,
            router_comm=r_comm,
            attn_comm=a_comm,
            lm_head_comm=h_comm,
            digest=h.digest(),
        )

    def step_forward(
        self, token_id: int, kv_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[np.ndarray], np.ndarray]:
        """Run one autoregressive step: Embedding + Attention(KV) + Top-k MoE + LM Head."""
        x = self.embed[token_id]
        # Simple linear attention + KV cache update
        attn_out = _fp_vec_mat_mul_exact(x, self.w_qkv)
        new_kv = vec_add(kv_state, attn_out)
        h_t = vec_add(x, new_kv)

        # Router logits in [0, 2^B)
        raw_logits = _fp_vec_mat_mul_exact(h_t, self.w_router)
        router_logits = raw_logits & np.uint64((1 << self.cfg.bit_width) - 1)

        # Top-k selection
        top_idx = np.argsort(router_logits, kind="stable")[-self.cfg.top_k :].tolist()
        expert_outs = [
            _fp_vec_mat_mul_exact(h_t, np.ascontiguousarray(self.experts[idx].T))
            for idx in top_idx
        ]
        moe_sum = np.zeros(self.cfg.d_model, dtype=np.uint64)
        for out in expert_outs:
            moe_sum = vec_add(moe_sum, out)

        h_out = vec_add(h_t, moe_sum)
        vocab_logits = _fp_vec_mat_mul_exact(h_out, self.lm_head)
        return h_t, router_logits, new_kv, top_idx, expert_outs, vocab_logits

    def prove_reasoning(
        self,
        prompt_tokens: list[int],
        t_think: int,
        prf_key: bytes = b"zkmoe-prf-secret-key-0123456789ab",
        model_comm: ModelCommitment | None = None,
    ) -> EndToEndReasoningProof:
        """Execute T_think hidden reasoning steps and emit an end-to-end zkMoE-Reason proof."""
        if model_comm is None:
            model_comm = self.commit()
        kv_state = np.zeros(self.cfg.d_model, dtype=np.uint64)
        # Process prompt
        for tok in prompt_tokens[:-1]:
            _, _, kv_state, _, _, _ = self.step_forward(tok, kv_state)
        cur_tok = prompt_tokens[-1]

        kv_states = [kv_state.copy()]
        hidden_tokens: list[int] = []
        step_a: list[np.ndarray] = []
        step_b: list[np.ndarray] = []
        router_proofs: list[ThresholdTopKProof] = []
        moe_proofs: list[SparseMoELayerProof] = []

        for t in range(t_think):
            h_t, r_logits, next_kv, top_idx, exp_outs, v_logits = self.step_forward(cur_tok, kv_state)

            # Prove top-k routing in O(N)
            r_tr = Transcript(f"router_{t}".encode())
            r_proof, _ = prove_threshold_topk(r_logits, self.cfg.top_k, self.cfg.bit_width, r_tr)
            router_proofs.append(r_proof)

            # Prove active k-expert matmul + oblivious lookup in O(k d^2 + N)
            m_tr = Transcript(f"moe_{t}".encode())
            m_proof = prove_oblivious_expert_layer(self.experts, top_idx, h_t, exp_outs, m_tr)
            moe_proofs.append(m_proof)

            # Verifiable PRF sampling of hidden reasoning token z_t
            z_t, _ = verifiable_prf_sample(v_logits, prf_key, t)
            hidden_tokens.append(z_t)

            # Witness vectors for CoT folding accumulator
            step_a.append(h_t)
            step_b.append(exp_outs[0])
            kv_states.append(next_kv.copy())

            kv_state = next_kv
            cur_tok = z_t

        # Emit final visible answer token
        _, _, _, _, _, final_logits = self.step_forward(cur_tok, kv_state)
        answer_token = int(np.argmax(final_logits))

        cot_proof = prove_cot_folded(
            step_a, step_b, kv_states, hidden_tokens, model_comm.digest, prf_key
        )

        return EndToEndReasoningProof(
            model_digest=model_comm.digest,
            prompt_tokens=prompt_tokens,
            answer_token=answer_token,
            t_think=t_think,
            router_proofs=router_proofs,
            moe_layer_proofs=moe_proofs,
            cot_proof=cot_proof,
        )

    def verify_reasoning(
        self,
        proof: EndToEndReasoningProof,
        expected_model_digest: bytes,
        expected_t_think: int,
    ) -> bool:
        """Verify an end-to-end zkMoE-Reason proof without seeing CoT tokens or expert IDs."""
        if proof.model_digest != expected_model_digest:
            return False
        if proof.t_think != expected_t_think:
            return False
        if len(proof.router_proofs) != expected_t_think or len(proof.moe_layer_proofs) != expected_t_think:
            return False

        for t in range(expected_t_think):
            r_tr = Transcript(f"router_{t}".encode())
            if not verify_threshold_topk(proof.router_proofs[t], r_tr):
                return False
            m_tr = Transcript(f"moe_{t}".encode())
            if not verify_oblivious_expert_layer(self.experts, proof.moe_layer_proofs[t], m_tr):
                return False

        return verify_cot_folded(proof.cot_proof, expected_t_think, expected_model_digest)
