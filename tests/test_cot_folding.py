import numpy as np

from zk_moe.cot_folding import (
    prove_cot_folded,
    prove_cot_nonfolded,
    verify_cot_folded,
    verify_cot_nonfolded,
)
from zk_moe.field_and_mle import Fp2
from zk_moe.moe_transformer import SparseMoEConfig, SparseMoETransformer


def test_cot_folding_valid_and_sound():
    rng = np.random.default_rng(30)
    t_think = 16
    d = 32
    a_steps = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think)]
    b_steps = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think)]
    kv_states = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think + 1)]
    tokens = [int(rng.integers(0, 64)) for _ in range(t_think)]
    model_comm = b"\xab" * 32
    prf_key = b"\xcd" * 32

    folded = prove_cot_folded(a_steps, b_steps, kv_states, tokens, model_comm, prf_key)
    assert verify_cot_folded(folded, t_think, model_comm)

    # Truncation attack: claim t_think + 1 steps
    assert not verify_cot_folded(folded, t_think + 1, model_comm)
    # Model substitution attack
    assert not verify_cot_folded(folded, t_think, b"\x00" * 32)
    # Tampered claim
    folded.final_eval_claim = folded.final_eval_claim + Fp2.one()
    assert not verify_cot_folded(folded, t_think, model_comm)

    nonfolded = prove_cot_nonfolded(a_steps, b_steps, kv_states, tokens, model_comm, prf_key)
    assert verify_cot_nonfolded(nonfolded)


def test_end_to_end_sparse_moe_reasoning():
    cfg = SparseMoEConfig(vocab_size=32, d_model=32, n_experts=8, top_k=2, bit_width=16)
    model = SparseMoETransformer(cfg, seed=7)
    model_comm = model.commit()

    proof = model.prove_reasoning(prompt_tokens=[1, 5, 12], t_think=4, model_comm=model_comm)
    assert model.verify_reasoning(proof, model_comm.digest, expected_t_think=4)

    # Wrong model digest must fail
    assert not model.verify_reasoning(proof, b"\x11" * 32, expected_t_think=4)
    # Wrong T_think must fail
    assert not model.verify_reasoning(proof, model_comm.digest, expected_t_think=5)
