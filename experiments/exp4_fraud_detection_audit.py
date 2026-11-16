"""Experiment 4: Adversarial Fraud Detection Audit (H3)
Evaluates soundness of zkMoE-Reason against 5 adversarial provider attacks:
1. Model-downgrade fraud (substituting smaller/perturbed expert weights)
2. Reasoning truncation (claiming T_think steps while running fewer)
3. Expert-skipping (zeroing out an expert output or running k-1 experts)
4. Router cheating (forcing non-top-k expert selection)
5. CoT KV-cache tampering (altering intermediate KV-cache commitment/claim)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zk_moe.field_and_mle import Fp2, Transcript, commit_vector
from zk_moe.moe_transformer import SparseMoEConfig, SparseMoETransformer
from zk_moe.oblivious_expert import prove_oblivious_expert_layer, verify_oblivious_expert_layer
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact


def run_experiment(n_trials: int = 100) -> dict:
    rng = np.random.default_rng(404)
    cfg = SparseMoEConfig(vocab_size=32, d_model=32, n_experts=8, top_k=2, bit_width=16)

    honest_accepts = 0
    attacks = {
        "model_downgrade": 0,
        "reasoning_truncation": 0,
        "expert_skipping": 0,
        "router_cheating": 0,
        "cot_kv_tampering": 0,
    }

    for trial in range(n_trials):
        model = SparseMoETransformer(cfg, seed=1000 + trial)
        model_comm = model.commit()
        t_think = 4
        prompt = [int(rng.integers(0, cfg.vocab_size)) for _ in range(3)]

        proof = model.prove_reasoning(prompt, t_think=t_think, model_comm=model_comm)

        # Honest check
        if model.verify_reasoning(proof, model_comm.digest, expected_t_think=t_think):
            honest_accepts += 1

        # Attack 1: Model-downgrade (prover uses perturbed/cheaper model, verifier checks committed model)
        cheap_model = SparseMoETransformer(cfg, seed=2000 + trial)
        cheap_comm = cheap_model.commit()
        cheap_proof = cheap_model.prove_reasoning(prompt, t_think=t_think, model_comm=cheap_comm)
        if not model.verify_reasoning(cheap_proof, model_comm.digest, expected_t_think=t_think):
            attacks["model_downgrade"] += 1

        # Attack 2: Reasoning truncation (prover runs 2 steps, claims 4)
        trunc_proof = model.prove_reasoning(prompt, t_think=2, model_comm=model_comm)
        if not model.verify_reasoning(trunc_proof, model_comm.digest, expected_t_think=t_think):
            attacks["reasoning_truncation"] += 1

        # Attack 3: Expert-skipping (prover zeros out one expert output)
        x = rng.integers(0, 64, size=cfg.d_model, dtype=np.uint64)
        selected = [0, 3]
        outs_real = [_fp_vec_mat_mul_exact(x, np.ascontiguousarray(model.experts[s].T)) for s in selected]
        outs_skip = [outs_real[0], np.zeros_like(outs_real[1])]
        p_skip = prove_oblivious_expert_layer(model.experts, selected, x, outs_skip, Transcript(b"atk3"))
        if not verify_oblivious_expert_layer(model.experts, p_skip, Transcript(b"atk3")):
            attacks["expert_skipping"] += 1

        # Attack 4: Router cheating (prover tampers threshold router proof to select non-top-k)
        proof_rcheat = model.prove_reasoning(prompt, t_think=t_think, model_comm=model_comm)
        # Tamper mask commitment in step 0
        fake_comm, _ = commit_vector(np.zeros(cfg.n_experts, dtype=np.uint64))
        proof_rcheat.router_proofs[0].mask_commitment = fake_comm
        if not model.verify_reasoning(proof_rcheat, model_comm.digest, expected_t_think=t_think):
            attacks["router_cheating"] += 1

        # Attack 5: CoT KV-cache tampering
        proof_kv = model.prove_reasoning(prompt, t_think=t_think, model_comm=model_comm)
        proof_kv.cot_proof.step_records[1].step_comm.kv_comm_next = fake_comm
        if not model.verify_reasoning(proof_kv, model_comm.digest, expected_t_think=t_think):
            attacks["cot_kv_tampering"] += 1

    summary = {
        "n_trials": n_trials,
        "honest_acceptance_rate": honest_accepts / n_trials,
        "attack_detection_rates": {k: v / n_trials for k, v in attacks.items()},
        "undetected_fraud_rates": {k: 1.0 - (v / n_trials) for k, v in attacks.items()},
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment(n_trials=100)
    with open(out_dir / "exp4_fraud_audit.json", "w") as f:
        json.dump(data, f, indent=2)
