"""Experiment 1: Sparsity Scaling (H1)
Sweeps N in {4, 8, 16, 32, 64, 128} with k=2 and d in {256, 512, 1024}.
Compares zkMoE-Reason vs Dense-Masked zkLLM vs Naive Element-Wise Lookup.
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zk_moe.field_and_mle import Transcript
from zk_moe.oblivious_expert import (
    prove_dense_masked_zkllm,
    prove_oblivious_expert_layer,
    verify_oblivious_expert_layer,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact


def measure_fn(fn):
    res = fn()  # warm-up run (library load, page faults)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    dt = sorted(times)[1]  # median of 3, ms
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, dt, peak / (1024.0 * 1024.0)  # MB


def run_experiment() -> dict:
    rng = np.random.default_rng(101)
    n_list = [4, 8, 16, 32, 64, 128]
    d_list = [256, 512, 1024]
    k = 2
    results = []

    for d in d_list:
        for n in n_list:
            # Allocate N x d x d expert weights
            experts = rng.integers(0, 128, size=(n, d, d), dtype=np.uint64)
            x = rng.integers(0, 128, size=d, dtype=np.uint64)
            selected = [0, n - 1]
            outputs = [
                _fp_vec_mat_mul_exact(x, np.ascontiguousarray(experts[s].T))
                for s in selected
            ]

            # 1. zkMoE-Reason
            proof_zk, t_zk_ms, mem_zk_mb = measure_fn(
                lambda: prove_oblivious_expert_layer(
                    experts, selected, x, outputs, Transcript(b"exp1_zk")
                )
            )
            _, t_ver_ms, _ = measure_fn(
                lambda: verify_oblivious_expert_layer(
                    experts, proof_zk, Transcript(b"exp1_zk")
                )
            )

            # 2. Dense-Masked zkLLM
            # At large (N=128, d=1024), dense matmul sumcheck across 128 experts takes ~several seconds;
            # run directly for all (d, N) where N*d <= 65536, and for (d=1024, N=128) run directly too if fast.
            _, t_dense_ms, mem_dense_mb = measure_fn(
                lambda: prove_dense_masked_zkllm(
                    experts, selected, x, Transcript(b"exp1_dense")
                )
            )

            speedup = t_dense_ms / t_zk_ms
            row = {
                "d": d,
                "N": n,
                "k": k,
                "zkmoe_prover_ms": round(t_zk_ms, 2),
                "zkmoe_verifier_ms": round(t_ver_ms, 2),
                "zkmoe_mem_mb": round(mem_zk_mb, 2),
                "dense_prover_ms": round(t_dense_ms, 2),
                "dense_mem_mb": round(mem_dense_mb, 2),
                "speedup_vs_dense": round(speedup, 2),
            }
            results.append(row)
            print(
                f"d={d:4d} N={n:3d} | zkMoE={t_zk_ms:7.2f}ms | Dense={t_dense_ms:8.2f}ms "
                f"| Speedup={speedup:5.2f}x | Ver={t_ver_ms:6.2f}ms"
            )

    return {"k": k, "results": results}


def plot_results(data: dict, out_dir: Path) -> None:
    rows = data["results"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    d_vals = [256, 512, 1024]

    for ax, d in zip(axes, d_vals):
        sub = [r for r in rows if r["d"] == d]
        ns = [r["N"] for r in sub]
        zk = [r["zkmoe_prover_ms"] for r in sub]
        dense = [r["dense_prover_ms"] for r in sub]

        ax.plot(ns, zk, marker="o", linewidth=2.2, label="zkMoE-Reason (Ours)", color="#1f77b4")
        ax.plot(ns, dense, marker="s", linewidth=2.0, linestyle="--", label="Dense-Masked zkLLM", color="#d62728")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"Hidden Dim $d = {d}$ ($k=2$)")
        ax.set_xlabel("Total Experts $N$")
        if d == 256:
            ax.set_ylabel("Prover Time (ms, log scale)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        if d == 1024:
            ax.legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_dir / "exp1_sparsity_scaling.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "exp1_sparsity_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    with open(out_dir / "exp1_sparsity_scaling.json", "w") as f:
        json.dump(data, f, indent=2)
    plot_results(data, out_dir)
