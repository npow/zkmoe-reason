import numpy as np

from zk_moe.field_and_mle import Transcript
from zk_moe.zk_topk_router import (
    count_pairwise_constraints,
    prove_bitonic_topk,
    prove_threshold_topk,
    verify_threshold_topk,
)


def test_threshold_topk_exactness_and_ties():
    rng = np.random.default_rng(20)
    for n in [8, 16, 64]:
        for k in [1, 2, 4]:
            # Include ties intentionally
            logits = rng.integers(0, 100, size=n, dtype=np.uint64)
            logits[0] = logits[1] = 50
            proof_th, mask_th = prove_threshold_topk(logits, k=k, bit_width=16)
            proof_bt, mask_bt = prove_bitonic_topk(logits, k=k, bit_width=16)

            assert int(np.sum(mask_th)) == k
            assert np.array_equal(mask_th, mask_bt)
            assert verify_threshold_topk(proof_th)
            # Threshold constraints must be strictly fewer than bitonic and pairwise
            assert proof_th.constraint_count < proof_bt.constraint_count
            assert proof_bt.constraint_count < count_pairwise_constraints(n, 16)


def test_threshold_topk_soundness():
    logits = np.array([10, 50, 20, 80, 30, 90, 40, 5], dtype=np.uint64)
    tr_p = Transcript(b"router_test")
    proof, _ = prove_threshold_topk(logits, k=2, bit_width=16, transcript=tr_p)

    # Wrong transcript fails
    assert not verify_threshold_topk(proof, Transcript(b"wrong"))

    # Tampered proof fails
    proof.k = 3
    # Note: sumcheck doesn't directly check proof.k unless absorbed; let's tamper round poly
    proof.bool_sumcheck.rounds[0].c0 = proof.bool_sumcheck.rounds[0].c0 + 1
    assert not verify_threshold_topk(proof, Transcript(b"router_test"))
