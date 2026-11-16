"""Experiment 6: Realistic Quantized MoE Fidelity & End-to-End ZK Verification
Evaluates FP32 vs fixed-point F_p quantization (B in {8, 12, 16} bits) on a
Mixtral/DeepSeek-style Sparse MoE block (d=512, N=64, k=2) with RMSNorm and Softmax routing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from zk_moe.field_and_mle import P, Transcript
from zk_moe.oblivious_expert import (
    commit_expert_table_pcs,
    prove_oblivious_expert_layer,
    verify_oblivious_expert_layer_succinct,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact
from zk_moe.zk_topk_router import prove_threshold_topk, verify_threshold_topk


def rmsnorm_fp32(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x ** 2) + eps)


def softmax_fp32(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits)
    ex = np.exp(logits - m)
    return ex / np.sum(ex)


def run_experiment(n_tokens: int = 32) -> dict:
    rng = np.random.default_rng(606)
    d = 512
    n_experts = 64
    k = 2

    # Realistic Transformer weight initialization N(0, 1/sqrt(d))
    w_router_f32 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, n_experts)).astype(np.float32)
    experts_f32 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(n_experts, d, d)).astype(np.float32)

    results = []
    for bits in [8, 12, 16]:
        # Symmetric per-tensor quantization to signed integers, shifted to [0, 2^bits) for F_p
        q_max = (1 << (bits - 1)) - 1
        exp_scale = np.max(np.abs(experts_f32)) / q_max
        rtr_scale = np.max(np.abs(w_router_f32)) / q_max

        experts_q = np.round(experts_f32 / exp_scale).astype(np.int64)
        # Map signed int64 to F_p (uint64): negative values -> P + val
        experts_fp = np.where(experts_q >= 0, experts_q, P + experts_q).astype(np.uint64)
        pk_exp = commit_expert_table_pcs(experts_fp)

        w_router_q = np.round(w_router_f32 / rtr_scale).astype(np.int64)

        top_k_matches = 0
        cos_sims = []
        rel_errs = []
        verified_count = 0
        prover_times = []
        verifier_times = []

        for t in range(n_tokens):
            h_raw = rng.normal(0.0, 1.0, size=d).astype(np.float32)
            h_norm = rmsnorm_fp32(h_raw)

            # FP32 reference
            g_f32 = h_norm @ w_router_f32
            probs_f32 = softmax_fp32(g_f32)
            top_f32 = set(np.argsort(probs_f32, kind="stable")[-k:].tolist())
            y_f32 = sum(probs_f32[idx] * (experts_f32[idx] @ h_norm) for idx in top_f32)

            # Fixed-point F_p quantized inference + ZK proof
            h_scale = np.max(np.abs(h_norm)) / q_max
            h_q = np.round(h_norm / h_scale).astype(np.int64)
            h_fp = np.where(h_q >= 0, h_q, P + h_q).astype(np.uint64)

            g_q = h_q @ w_router_q
            # Shift router logits to non-negative [0, 2^B_rtr) preserving exact order
            g_min = int(np.min(g_q))
            g_nonneg = (g_q - g_min).astype(np.uint64)
            rtr_bits = int(np.max(g_nonneg)).bit_length() + 1

            t0 = time.perf_counter()
            r_proof, r_mask = prove_threshold_topk(
                g_nonneg, k=k, bit_width=max(rtr_bits, 16), transcript=Transcript(f"q_r_{t}".encode())
            )
            sel_q = [int(i) for i in np.where(r_mask == 1)[0]]
            outs_fp = [_fp_vec_mat_mul_exact(h_fp, np.ascontiguousarray(experts_fp[s].T)) for s in sel_q]
            m_proof = prove_oblivious_expert_layer(
                experts_fp, sel_q, h_fp, outs_fp, Transcript(f"q_m_{t}".encode()), expert_pk=pk_exp
            )
            prover_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            ok_r = verify_threshold_topk(r_proof, Transcript(f"q_r_{t}".encode()))
            ok_m = verify_oblivious_expert_layer_succinct(
                pk_exp.commitment, m_proof, Transcript(f"q_m_{t}".encode())
            )
            verifier_times.append((time.perf_counter() - t0) * 1000.0)
            if ok_r and ok_m:
                verified_count += 1

            if set(sel_q) == top_f32:
                top_k_matches += 1

            # Dequantize F_p outputs back to FP32 for fidelity measurement
            probs_q = softmax_fp32(g_q.astype(np.float64) * (h_scale * rtr_scale))
            y_q_f32 = np.zeros(d, dtype=np.float64)
            for s, out_fp in zip(sel_q, outs_fp):
                out_signed = np.where(out_fp <= P // 2, out_fp.astype(np.int64), out_fp.astype(np.int64) - P)
                y_q_f32 += probs_q[s] * (out_signed.astype(np.float64) * (h_scale * exp_scale))

            cos = float(np.dot(y_f32, y_q_f32) / (np.linalg.norm(y_f32) * np.linalg.norm(y_q_f32) + 1e-12))
            rel = float(np.linalg.norm(y_f32 - y_q_f32) / (np.linalg.norm(y_f32) + 1e-12))
            cos_sims.append(cos)
            rel_errs.append(rel)

        row = {
            "bits": bits,
            "n_tokens": n_tokens,
            "topk_routing_agreement_pct": round(100.0 * top_k_matches / n_tokens, 2),
            "mean_cosine_similarity": round(float(np.mean(cos_sims)), 6),
            "mean_rel_l2_error": round(float(np.mean(rel_errs)), 6),
            "zk_verified_pct": round(100.0 * verified_count / n_tokens, 2),
            "mean_prover_ms": round(float(np.mean(prover_times)), 2),
            "mean_succinct_verifier_ms": round(float(np.mean(verifier_times)), 2),
        }
        results.append(row)
        print(
            f"B={bits:2d}-bit | TopK Agree={row['topk_routing_agreement_pct']:6.2f}% "
            f"| CosSim={row['mean_cosine_similarity']:.6f} | RelErr={row['mean_rel_l2_error']:.5f} "
            f"| Prover={row['mean_prover_ms']:6.1f}ms | SuccVer={row['mean_succinct_verifier_ms']:5.1f}ms "
            f"| Verified={row['zk_verified_pct']:.0f}%"
        )

    return {"d": d, "N": n_experts, "k": k, "results": results}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment(n_tokens=32)
    with open(out_dir / "exp6_real_moe_quantization.json", "w") as f:
        json.dump(data, f, indent=2)
