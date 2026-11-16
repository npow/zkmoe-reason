"""Experiment 3: Hidden CoT Folding Overhead (H3)
Measures Folded vs Non-Folded proof size, prover time, and verifier time
across T_think in {8, 16, 32, 64, 128, 256, 512, 1024}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zk_moe.cot_folding import (
    FoldedCoTProof,
    NonFoldedCoTProof,
    prove_cot_folded,
    prove_cot_nonfolded,
    verify_cot_folded,
    verify_cot_nonfolded,
)


def folded_proof_bytes(p: FoldedCoTProof) -> int:
    # 1 final sumcheck: each round has 3 Fp2 elems (3 * 16 = 48 bytes) + final_a, final_b (32 bytes)
    sc_bytes = len(p.final_sumcheck.rounds) * 48 + 32
    # Per-step accumulator record: kv_next (40B) + tok_comm (40B) + prf (32B) + cross_comm (40B) + cross_sum (16B) + claim (16B) = 184B
    # Note: with recursive SNARK/hash-chain compression of step commitments, only 1 final sumcheck + 32B chain head is needed;
    # here we report both the un-compressed step-record size and the sumcheck-only proof size.
    return sc_bytes + len(p.step_records) * 184


def nonfolded_proof_bytes(p: NonFoldedCoTProof) -> int:
    sc_bytes = sum(len(sc.rounds) * 48 + 32 for sc in p.step_sumchecks)
    comm_bytes = len(p.step_commitments) * (40 + 40 + 32)
    return sc_bytes + comm_bytes + len(p.step_claims) * 16


def run_experiment() -> dict:
    rng = np.random.default_rng(303)
    t_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    d = 256
    model_comm = b"\xaa" * 32
    prf_key = b"\xbb" * 32
    results = []

    for t_think in t_list:
        a_steps = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think)]
        b_steps = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think)]
        kv_states = [rng.integers(0, 1000, size=d, dtype=np.uint64) for _ in range(t_think + 1)]
        tokens = [int(rng.integers(0, 1024)) for _ in range(t_think)]

        # Folded
        t0 = time.perf_counter()
        p_fold = prove_cot_folded(a_steps, b_steps, kv_states, tokens, model_comm, prf_key)
        t_fold_p_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok_f = verify_cot_folded(p_fold, t_think, model_comm)
        t_fold_v_ms = (time.perf_counter() - t0) * 1000.0
        assert ok_f

        # Non-Folded
        t0 = time.perf_counter()
        p_non = prove_cot_nonfolded(a_steps, b_steps, kv_states, tokens, model_comm, prf_key)
        t_non_p_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok_n = verify_cot_nonfolded(p_non)
        t_non_v_ms = (time.perf_counter() - t0) * 1000.0
        assert ok_n

        b_fold = folded_proof_bytes(p_fold)
        b_non = nonfolded_proof_bytes(p_non)

        row = {
            "T_think": t_think,
            "d": d,
            "folded_prover_ms": round(t_fold_p_ms, 2),
            "folded_verifier_ms": round(t_fold_v_ms, 2),
            "folded_proof_kb": round(b_fold / 1024.0, 2),
            "folded_sumcheck_rounds": len(p_fold.final_sumcheck.rounds),
            "nonfolded_prover_ms": round(t_non_p_ms, 2),
            "nonfolded_verifier_ms": round(t_non_v_ms, 2),
            "nonfolded_proof_kb": round(b_non / 1024.0, 2),
            "nonfolded_sumcheck_rounds": sum(len(sc.rounds) for sc in p_non.step_sumchecks),
            "verifier_speedup": round(t_non_v_ms / max(t_fold_v_ms, 1e-6), 2),
            "size_reduction": round(b_non / b_fold, 2),
        }
        results.append(row)
        print(
            f"T={t_think:4d} | Folded Ver={t_fold_v_ms:6.2f}ms ({row['folded_proof_kb']:6.2f}KB) "
            f"| NonFolded Ver={t_non_v_ms:7.2f}ms ({row['nonfolded_proof_kb']:7.2f}KB) "
            f"| VerSpeedup={row['verifier_speedup']:5.2f}x"
        )

    return {"d": d, "results": results}


def plot_results(data: dict, out_dir: Path) -> None:
    rows = data["results"]
    ts = [r["T_think"] for r in rows]
    vf = [r["folded_verifier_ms"] for r in rows]
    vn = [r["nonfolded_verifier_ms"] for r in rows]
    sf = [r["folded_proof_kb"] for r in rows]
    sn = [r["nonfolded_proof_kb"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax1.plot(ts, vf, marker="o", linewidth=2.2, label="zkMoE-Reason (Folded)", color="#1f77b4")
    ax1.plot(ts, vn, marker="s", linewidth=2.0, linestyle="--", label="Non-Folded (Per-Step)", color="#d62728")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("Verifier Time vs. Reasoning Length $T_{\\text{think}}$")
    ax1.set_xlabel("Hidden Reasoning Tokens $T_{\\text{think}}$")
    ax1.set_ylabel("Verifier Time (ms, log)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(ts, sf, marker="o", linewidth=2.2, label="zkMoE-Reason (Folded)", color="#1f77b4")
    ax2.plot(ts, sn, marker="s", linewidth=2.0, linestyle="--", label="Non-Folded (Per-Step)", color="#d62728")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Proof Size vs. Reasoning Length $T_{\\text{think}}$")
    ax2.set_xlabel("Hidden Reasoning Tokens $T_{\\text{think}}$")
    ax2.set_ylabel("Proof Size (KB, log)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_dir / "exp3_cot_folding.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "exp3_cot_folding.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    with open(out_dir / "exp3_cot_folding.json", "w") as f:
        json.dump(data, f, indent=2)
    plot_results(data, out_dir)
