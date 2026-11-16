"""Experiment 5: Ablations & Succinct PCS Verifier Scaling
1. Active experts k in {1, 2, 4, 8} at N=64, d=512
2. Succinct PCS verifier time O(sqrt(N d^2)) vs raw-table verifier O(N d^2)
3. Multi-layer end-to-end SparseMoETransformer scaling L in {1, 2, 4, 8}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zk_moe.field_and_mle import Transcript
from zk_moe.moe_transformer import SparseMoEConfig, SparseMoETransformer
from zk_moe.oblivious_expert import (
    commit_expert_table_pcs,
    prove_dense_masked_zkllm,
    prove_oblivious_expert_layer,
    verify_oblivious_expert_layer,
    verify_oblivious_expert_layer_succinct,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact


def run_experiment() -> dict:
    rng = np.random.default_rng(505)

    # Part A: Active experts k sweep at N=64, d=512
    n_a, d_a = 64, 512
    experts_a = rng.integers(0, 128, size=(n_a, d_a, d_a), dtype=np.uint64)
    x_a = rng.integers(0, 128, size=d_a, dtype=np.uint64)
    prove_dense_masked_zkllm(experts_a, [0, 1], x_a, Transcript(b"ab_dense"))  # warm-up
    _ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        prove_dense_masked_zkllm(experts_a, [0, 1], x_a, Transcript(b"ab_dense"))
        _ts.append((time.perf_counter() - t0) * 1000.0)
    dense_ms = sorted(_ts)[1]

    k_sweep = []
    for k in [1, 2, 4, 8]:
        sel = list(range(k))
        outs = [_fp_vec_mat_mul_exact(x_a, np.ascontiguousarray(experts_a[s].T)) for s in sel]
        _tz = []
        for _ in range(4):  # first run is warm-up
            t0 = time.perf_counter()
            prove_oblivious_expert_layer(experts_a, sel, x_a, outs, Transcript(f"ab_k{k}".encode()))
            _tz.append((time.perf_counter() - t0) * 1000.0)
        zk_ms = sorted(_tz[1:])[1]
        k_sweep.append({
            "k": k,
            "N": n_a,
            "d": d_a,
            "zkmoe_prover_ms": round(zk_ms, 2),
            "dense_prover_ms": round(dense_ms, 2),
            "speedup": round(dense_ms / zk_ms, 2),
        })
        print(f"[k-sweep] k={k} | zkMoE={zk_ms:6.2f}ms | Dense={dense_ms:7.2f}ms | Speedup={dense_ms/zk_ms:5.2f}x")

    # Part B: Succinct PCS verifier O(sqrt(N d^2)) vs raw-table verifier O(N d^2) at d=256
    d_b = 256
    pcs_sweep = []
    for n in [4, 8, 16, 32, 64, 128]:
        exp_b = rng.integers(0, 128, size=(n, d_b, d_b), dtype=np.uint64)
        x_b = rng.integers(0, 128, size=d_b, dtype=np.uint64)
        sel = [0, n - 1]
        outs = [_fp_vec_mat_mul_exact(x_b, np.ascontiguousarray(exp_b[s].T)) for s in sel]
        pk = commit_expert_table_pcs(exp_b)
        proof = prove_oblivious_expert_layer(exp_b, sel, x_b, outs, Transcript(b"pcs_b"), expert_pk=pk)

        t0 = time.perf_counter()
        ok_s = verify_oblivious_expert_layer_succinct(pk.commitment, proof, Transcript(b"pcs_b"))
        t_succ_ms = (time.perf_counter() - t0) * 1000.0
        assert ok_s

        t0 = time.perf_counter()
        ok_r = verify_oblivious_expert_layer(exp_b, proof, Transcript(b"pcs_b"))
        t_raw_ms = (time.perf_counter() - t0) * 1000.0
        assert ok_r

        # Proof size of PCS opening (bytes)
        op = proof.pcs_opening
        pcs_kb = (
            op.combined_row0.nbytes
            + op.combined_row1.nbytes
            + sum(c.nbytes for c in op.queried_columns)
            + sum(len(p) * 33 for p in op.merkle_paths)
        ) / 1024.0

        pcs_sweep.append({
            "N": n,
            "d": d_b,
            "succinct_verifier_ms": round(t_succ_ms, 2),
            "raw_verifier_ms": round(t_raw_ms, 2),
            "pcs_opening_kb": round(pcs_kb, 2),
        })
        print(f"[PCS-ver] N={n:3d} | SuccinctVer={t_succ_ms:5.2f}ms | RawVer={t_raw_ms:6.2f}ms | PCS={pcs_kb:6.1f}KB")

    # Part C: Multi-layer SparseMoETransformer scaling L in {1, 2, 4, 8}
    layer_sweep = []
    for n_layers in [1, 2, 4, 8]:
        cfg = SparseMoEConfig(vocab_size=32, d_model=64, n_experts=16, top_k=2, bit_width=16)
        model = SparseMoETransformer(cfg, seed=77)
        model_comm = model.commit()
        t0 = time.perf_counter()
        for _ in range(n_layers):
            proof = model.prove_reasoning([1, 5, 12], t_think=4, model_comm=model_comm)
        t_p_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        for _ in range(n_layers):
            assert model.verify_reasoning(proof, model_comm.digest, expected_t_think=4)
        t_v_ms = (time.perf_counter() - t0) * 1000.0
        layer_sweep.append({
            "n_layers": n_layers,
            "prover_ms": round(t_p_ms, 2),
            "verifier_ms": round(t_v_ms, 2),
        })
        print(f"[Layers] L={n_layers} | Prover={t_p_ms:7.2f}ms | Verifier={t_v_ms:6.2f}ms")

    return {"k_sweep": k_sweep, "pcs_sweep": pcs_sweep, "layer_sweep": layer_sweep}


def plot_results(data: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ks = [r["k"] for r in data["k_sweep"]]
    zk_k = [r["zkmoe_prover_ms"] for r in data["k_sweep"]]
    dn_k = [r["dense_prover_ms"] for r in data["k_sweep"]]
    ax1.plot(ks, zk_k, marker="o", linewidth=2.2, label="zkMoE-Reason", color="#1f77b4")
    ax1.plot(ks, dn_k, marker="s", linewidth=2.0, linestyle="--", label="Dense-Masked zkLLM", color="#d62728")
    ax1.set_title("Active Experts $k$ Sweep ($N=64, d=512$)")
    ax1.set_xlabel("Active Experts $k$")
    ax1.set_ylabel("Prover Time (ms)")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ns = [r["N"] for r in data["pcs_sweep"]]
    sv = [r["succinct_verifier_ms"] for r in data["pcs_sweep"]]
    rv = [r["raw_verifier_ms"] for r in data["pcs_sweep"]]
    ax2.plot(ns, sv, marker="o", linewidth=2.2, label="Succinct PCS Verifier $O(\\sqrt{Nd^2})$", color="#2ca02c")
    ax2.plot(ns, rv, marker="s", linewidth=2.0, linestyle="--", label="Raw-Table Verifier $O(Nd^2)$", color="#7f7f7f")
    ax2.set_xscale("log", base=2)
    ax2.set_title("Succinct PCS Verifier Scaling ($d=256$)")
    ax2.set_xlabel("Total Experts $N$")
    ax2.set_ylabel("Verifier Time (ms)")
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_dir / "exp5_ablations.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "exp5_ablations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_experiment()
    with open(out_dir / "exp5_ablations.json", "w") as f:
        json.dump(data, f, indent=2)
    plot_results(data, out_dir)
