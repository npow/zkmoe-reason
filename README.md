# zkMoE-Reason: Sublinear Zero-Knowledge Proofs for Sparse Mixture-of-Experts & Hidden Test-Time Reasoning

**zkMoE-Reason** is a commit-and-prove zero-knowledge protocol for frontier Sparse Mixture-of-Experts (MoE) Transformers with hidden test-time reasoning (Chain-of-Thought). It combines:

1. **Sumcheck-Reduced Oblivious Expert Lookup** (`src/zk_moe/oblivious_expert.py`): Reduces proving $k$ active experts out of $N$ total experts from $O(N d^2)$ (dense masked) or $O(k d^2 N)$ (naive element-wise lookup) to $O(k d^2 + N)$ by deferring expert-table lookups until after the batched matmul sumcheck emits a single random point $(r^*, c^*)$.
2. **Threshold-Based $O(N)$ Top-$k$ Router Proof** (`src/zk_moe/zk_topk_router.py`): Proves exact top-$k$ expert selection using a witness threshold $\tau$ and $N$ range checks rather than an $O(N \log^2 N)$ bitonic sorting network.
3. **Hidden CoT Autoregressive Folding** (`src/zk_moe/cot_folding.py`): Folds $T_{\text{think}}$ hidden reasoning steps and KV-cache commitments into a single constant-round sumcheck accumulator, preventing truncation, model-downgrade, and expert-skipping fraud without leaking hidden CoT tokens.

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
