# zkMoE-Reason: prototype cost study for verifiable sparse MoE inference

Research prototype (not audited). It measures (1) an active-experts-only matmul sumcheck plus a $\Theta(Nd^2)$ expert-table projection checked by one commitment opening, and (2) a threshold-witness top-k relation's constraint count against a bitonic network.

Not claimed: routing privacy (opened indicator evaluations reveal the indices), sublinear cost in N, soundness of the chain-of-thought folding accumulator in `cot_folding.py` (kept only as an unproven prototype), or a composed proof of full MoE inference. Baselines are our own re-implementations; all timings are medians of 3 after warm-up on one aarch64 machine with random weights. See `paper/main.tex`, Section "Limitations".

Reproduce: `gcc -O3 -march=native -shared -fPIC native/zkmoe_fast.c -o native/libzkmoe.so`, then the commands below. Raw results: `results/*.json`.

## Repository Structure

```text
2026-11-zk-sparse-moe-reasoning/
├── pyproject.toml
├── README.md
├── src/zk_moe/
│   ├── field_and_mle.py      # Mersenne-61 F_p, F_p^2, MLEs, Fiat-Shamir, commitments
│   ├── sumcheck_matmul.py    # Batched ZK sumcheck for active expert matmuls
│   ├── oblivious_expert.py   # Random-point reduced oblivious expert lookup + baselines
│   ├── zk_topk_router.py     # O(N) threshold top-k proof vs bitonic sort & pairwise
│   ├── cot_folding.py        # Hidden CoT autoregressive folding accumulator
│   └── moe_transformer.py    # Reference Sparse MoE Transformer
├── experiments/
│   ├── exp1_sparsity_scaling.py
│   ├── exp2_router_proof_comparison.py
│   ├── exp3_cot_folding_overhead.py
│   └── exp4_fraud_detection_audit.py
├── tests/
├── results/
└── paper/
    ├── main.tex
    └── references.bib
```

## Running Tests & Experiments

```bash
PYTHONPATH=src pytest -v
PYTHONPATH=src python3 experiments/exp1_sparsity_scaling.py
PYTHONPATH=src python3 experiments/exp2_router_proof_comparison.py
PYTHONPATH=src python3 experiments/exp3_cot_folding_overhead.py
PYTHONPATH=src python3 experiments/exp4_fraud_detection_audit.py
```
