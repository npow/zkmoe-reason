"""Experiment 2: Router Proof Comparison (H2)
Compares Threshold Top-k O(N) vs Bitonic Sort O(N log^2 N) vs Naive Pairwise O(N^2)
across N in {8, 16, 32, 64, 128, 256}, and measures overhead % relative to active expert matmul.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zk_moe.field_and_mle import Transcript
from zk_moe.oblivious_expert import prove_oblivious_expert_layer
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact
from zk_moe.zk_topk_router import (
    count_pairwise_constraints,
    prove_bitonic_topk,
    prove_threshold_topk,
    verify_threshold_topk,
)


def run_experiment() -> dict:
    rng = np.random.default_rng(202)
    n_list = [8, 16, 32, 64, 128, 256]
    k = 2
    bit_width = 16
    results = []

    # Measure reference active-expert matmul sumcheck time at d=512, k=2, N=64
    d_ref = 512
    exp_ref = rng.integers(0, 128, size=(64, d_ref, d_ref), dtype=np.uint64)
    x_ref = rng.integers(0, 128, size=d_ref, dtype=np.uint64)
    outs_ref = [_fp_vec_mat_mul_exact(x_ref, np.ascontiguousarray(exp_ref[s].T)) for s in [0, 1]]
    prove_oblivious_expert_layer(exp_ref, [0, 1], x_ref, outs_ref, Transcript(b"ref_mm"))  # warm-up
    _ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        prove_oblivious_expert_layer(exp_ref, [0, 1], x_ref, outs_ref, Transcript(b"ref_mm"))
        _ts.append((time.perf_counter() - t0) * 1000.0)
    ref_matmul_ms = sorted(_ts)[1]

    for n in n_list:
        logits = rng.integers(0, 1 << bit_width, size=n, dtype=np.uint64)

        # Threshold O(N)
        t0 = time.perf_counter()
        p_th, _ = prove_threshold_topk(logits, k=k, bit_width=bit_width)
        t_th_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok = verify_threshold_topk(p_th)
        t_th_ver_ms = (time.perf_counter() - t0) * 1000.0
        assert ok

        # Bitonic O(N log^2 N)
        t0 = time.perf_counter()
        p_bt, _ = prove_bitonic_topk(logits, k=k, bit_width=bit_width)
        t_bt_ms = (time.perf_counter() - t0) * 1000.0

        c_pair = count_pairwise_constraints(n, bit_width=bit_width)
        overhead_pct = (t_th_ms / ref_matmul_ms) * 100.0

        row = {
            "N": n,
            "k": k,
            "threshold_constraints": p_th.constraint_count,
            "bitonic_constraints": p_bt.constraint_count,
            "pairwise_constraints": c_pair,
            "threshold_prover_ms": round(t_th_ms, 3),
            "threshold_verifier_ms": round(t_th_ver_ms, 3),
            "bitonic_prover_ms": round(t_bt_ms, 3),
            "constraint_reduction_vs_bitonic": round(p_bt.constraint_count / p_th.constraint_count, 2),
            "overhead_vs_d512_matmul_pct": round(overhead_pct, 2),
        }
        results.append(row)
        print(
            f"N={n:3d} | Thresh={p_th.constraint_count:6d} ({t_th_ms:5.2f}ms, {overhead_pct:4.2f}% of d=512) "
            f"| Bitonic={p_bt.constraint_count:7d} ({t_bt_ms:6.2f}ms) | Pairwise={c_pair:8d}"
        )

    return {"k": k, "bit_width": bit_width, "ref_matmul_d512_ms": round(ref_matmul_ms, 2), "results": results}


def plot_results(data: dict, out_dir: Path) -> None:
    rows = data["results"]
    ns = [r["N"] for r in rows]
    c_th = [r["threshold_constraints"] for r in rows]
    c_bt = [r["bitonic_constraints"] for r in rows]
    c_pw = [r["pairwise_constraints"] for r in rows]

    t_th = [r["threshold_prover_ms"] for r in rows]
    t_bt = [r["bitonic_prover_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax1.plot(ns, c_th, marker="o", linewidth=2.2, label="Threshold Top-$k$ ($O(N)$)", color="#1f77b4")
    ax1.plot(ns, c_bt, marker="s", linewidth=2.0, linestyle="--", label="Bitonic Sort ($O(N \\log^2 N)$)", color="#d62728")
    ax1.plot(ns, c_pw, marker="^", linewidth=1.8, linestyle=":", label="Pairwise ($O(N^2)$)", color="#ff7f0e")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("Router Proof Constraint Count ($k=2, B=16$)")
    ax1.set_xlabel("Total Experts $N$")
    ax1.set_ylabel("R1CS-Equivalent Constraints (log)")
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(ns, t_th, marker="o", linewidth=2.2, label="Threshold Top-$k$ ($O(N)$)", color="#1f77b4")
    ax2.plot(ns, t_bt, marker="s", linewidth=2.0, linestyle="--", label="Bitonic Sort ($O(N \\log^2 N)$)", color="#d62728")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("Router Proving Time")
    ax2.set_xlabel("Total Experts $N$")
    ax2.set_ylabel("Prover Time (ms, log)")
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_dir / "exp2_router_comparison.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "exp2_router_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    with open(out_dir / "exp2_router_comparison.json", "w") as f:
        json.dump(data, f, indent=2)
    plot_results(data, out_dir)
