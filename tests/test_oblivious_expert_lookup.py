import numpy as np

from zk_moe.field_and_mle import Transcript
from zk_moe.oblivious_expert import (
    commit_expert_table_pcs,
    prove_oblivious_expert_layer,
    verify_oblivious_expert_layer,
    verify_oblivious_expert_layer_succinct,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact


def test_oblivious_expert_lookup_valid_and_sound():
    rng = np.random.default_rng(10)
    n_experts, d, k = 16, 32, 2
    experts = rng.integers(0, 500, size=(n_experts, d, d), dtype=np.uint64)
    x = rng.integers(0, 500, size=d, dtype=np.uint64)
    selected = [3, 11]
    outputs = [_fp_vec_mat_mul_exact(x, np.ascontiguousarray(experts[s].T)) for s in selected]

    tr_p = Transcript(b"oblivious")
    proof = prove_oblivious_expert_layer(experts, selected, x, outputs, tr_p)

    tr_v = Transcript(b"oblivious")
    assert verify_oblivious_expert_layer(experts, proof, tr_v)

    # Soundness: cheating with a perturbed expert table must fail
    experts_bad = experts.copy()
    experts_bad[3, 0, 0] += 1
    tr_v_bad = Transcript(b"oblivious")
    assert not verify_oblivious_expert_layer(experts_bad, proof, tr_v_bad)

    # Soundness: forging output (e.g. zeroing an expert output) must fail
    outputs_bad = [outputs[0], np.zeros_like(outputs[1])]
    tr_p_bad = Transcript(b"oblivious_bad")
    proof_bad = prove_oblivious_expert_layer(experts, selected, x, outputs_bad, tr_p_bad)
    tr_v_bad2 = Transcript(b"oblivious_bad")
    assert not verify_oblivious_expert_layer(experts, proof_bad, tr_v_bad2)


def test_succinct_pcs_oblivious_lookup():
    rng = np.random.default_rng(11)
    n_experts, d, k = 16, 32, 2
    experts = rng.integers(0, 500, size=(n_experts, d, d), dtype=np.uint64)
    x = rng.integers(0, 500, size=d, dtype=np.uint64)
    selected = [2, 13]
    outputs = [_fp_vec_mat_mul_exact(x, np.ascontiguousarray(experts[s].T)) for s in selected]

    pk = commit_expert_table_pcs(experts)
    proof = prove_oblivious_expert_layer(
        experts, selected, x, outputs, Transcript(b"succ"), expert_pk=pk
    )
    # Verifier ONLY receives the 32-byte PCSCommitment root, NOT `experts`!
    assert verify_oblivious_expert_layer_succinct(pk.commitment, proof, Transcript(b"succ"))

    # Wrong commitment root must fail
    experts_bad = experts.copy()
    experts_bad[2, 5, 5] += 1
    pk_bad = commit_expert_table_pcs(experts_bad)
    assert not verify_oblivious_expert_layer_succinct(pk_bad.commitment, proof, Transcript(b"succ"))


def test_full_61bit_signed_quantized_succinct():
    from zk_moe.field_and_mle import P
    rng = np.random.default_rng(12)
    n_experts, d = 16, 64
    experts = rng.integers(0, P, size=(n_experts, d, d), dtype=np.uint64)
    x = rng.integers(0, P, size=d, dtype=np.uint64)
    selected = [1, 14]
    outputs = [_fp_vec_mat_mul_exact(x, np.ascontiguousarray(experts[s].T)) for s in selected]
    pk = commit_expert_table_pcs(experts)
    proof = prove_oblivious_expert_layer(
        experts, selected, x, outputs, Transcript(b"full61"), expert_pk=pk
    )
    assert verify_oblivious_expert_layer_succinct(pk.commitment, proof, Transcript(b"full61"))
