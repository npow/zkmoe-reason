"""zkMoE-Reason: Sublinear Zero-Knowledge Proofs for Sparse MoE & Hidden Test-Time Reasoning."""

from zk_moe.field_and_mle import P, Fp2, Transcript, commit_vector, mle_eval_fp, mle_eval_fp2
from zk_moe.sumcheck_matmul import prove_batched_matmul_zk, verify_batched_matmul_zk
from zk_moe.oblivious_expert import prove_oblivious_expert_layer, verify_oblivious_expert_layer
from zk_moe.zk_topk_router import prove_threshold_topk, verify_threshold_topk, prove_bitonic_topk
from zk_moe.cot_folding import prove_cot_folded, verify_cot_folded
from zk_moe.moe_transformer import SparseMoEConfig, SparseMoETransformer

__all__ = [
    "P",
    "Fp2",
    "Transcript",
    "commit_vector",
    "mle_eval_fp",
    "mle_eval_fp2",
    "prove_batched_matmul_zk",
    "verify_batched_matmul_zk",
    "prove_oblivious_expert_layer",
    "verify_oblivious_expert_layer",
    "prove_threshold_topk",
    "verify_threshold_topk",
    "prove_bitonic_topk",
    "prove_cot_folded",
    "verify_cot_folded",
    "SparseMoEConfig",
    "SparseMoETransformer",
]
