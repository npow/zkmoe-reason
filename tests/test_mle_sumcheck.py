import numpy as np
import pytest

from zk_moe.field_and_mle import (
    P,
    Fp2,
    Transcript,
    eq_evals_fp2,
    fp2_vec_from_fp,
    fp2_vec_mul,
    fp2_vec_sum,
    fp_add,
    fp_inv,
    fp_mul,
    mle_eval_fp,
    mle_eval_fp2,
    vec_add,
    vec_mul,
    vec_sub,
)
from zk_moe.sumcheck_matmul import (
    _fp_vec_mat_mul_exact,
    prove_batched_matmul_zk,
    prove_product_sumcheck,
    verify_batched_matmul_zk,
    verify_product_sumcheck,
)


def test_field_arithmetic():
    rng = np.random.default_rng(1)
    a = rng.integers(0, P, size=100, dtype=np.uint64)
    b = rng.integers(0, P, size=100, dtype=np.uint64)
    c = vec_mul(a, b)
    for i in range(100):
        assert int(c[i]) == (int(a[i]) * int(b[i])) % P
        assert int(vec_add(a[i : i + 1], b[i : i + 1])[0]) == (int(a[i]) + int(b[i])) % P
        assert int(vec_sub(a[i : i + 1], b[i : i + 1])[0]) == (int(a[i]) - int(b[i])) % P

    x = Fp2(123456789, 987654321)
    x_inv = x.inv()
    assert x * x_inv == Fp2.one()


def test_mle_and_eq_consistency():
    rng = np.random.default_rng(2)
    f = rng.integers(0, P, size=32, dtype=np.uint64)
    r = [Fp2(int(rng.integers(0, P)), int(rng.integers(0, P))) for _ in range(5)]
    val_direct = mle_eval_fp2(f, r)
    eq_r = eq_evals_fp2(r)
    val_eq = fp2_vec_sum(fp2_vec_mul(fp2_vec_from_fp(f), eq_r))
    assert val_direct == val_eq

    r_fp = [int(rng.integers(0, P)) for _ in range(5)]
    val_fp = mle_eval_fp(f, r_fp)
    val_fp2 = mle_eval_fp2(f, [Fp2.from_fp(v) for v in r_fp])
    assert Fp2.from_fp(val_fp) == val_fp2


def test_product_sumcheck():
    rng = np.random.default_rng(3)
    a = rng.integers(0, P, size=64, dtype=np.uint64)
    b = rng.integers(0, P, size=64, dtype=np.uint64)
    a_fp2 = fp2_vec_from_fp(a)
    b_fp2 = fp2_vec_from_fp(b)
    true_sum = fp2_vec_sum(fp2_vec_mul(a_fp2, b_fp2))

    tr_p = Transcript(b"test_sc")
    proof = prove_product_sumcheck(a_fp2, b_fp2, tr_p)

    tr_v = Transcript(b"test_sc")
    ok, chals, fa, fb = verify_product_sumcheck(proof, true_sum, tr_v)
    assert ok
    assert fa == mle_eval_fp2(a, chals)
    assert fb == mle_eval_fp2(b, chals)

    # Soundness: wrong claimed sum must fail
    tr_bad = Transcript(b"test_sc")
    ok_bad, _, _, _ = verify_product_sumcheck(proof, true_sum + Fp2.one(), tr_bad)
    assert not ok_bad


def test_batched_matmul_zk():
    rng = np.random.default_rng(4)
    d = 32
    k = 3
    mats = [rng.integers(0, 1000, size=(d, d), dtype=np.uint64) for _ in range(k)]
    x = rng.integers(0, 1000, size=d, dtype=np.uint64)
    outputs = [_fp_vec_mat_mul_exact(x, np.ascontiguousarray(m.T)) for m in mats]

    # Verify _fp_vec_mat_mul_exact matches Python matmul
    for m, y in zip(mats, outputs):
        expected = (m.astype(object) @ x.astype(object)) % P
        assert np.array_equal(y, expected.astype(np.uint64))

    tr_p = Transcript(b"test_mm")
    proof = prove_batched_matmul_zk(mats, x, outputs, tr_p)

    tr_v = Transcript(b"test_mm")
    ok, r_star, c_star, final_ms, final_x = verify_batched_matmul_zk(proof, tr_v)
    assert ok
    assert final_x == mle_eval_fp2(x, c_star)
    for m, fm in zip(mats, final_ms):
        eq_r = eq_evals_fp2(r_star)
        eq_c = eq_evals_fp2(c_star)
        # Check M(r*, c*) directly
        row_mle = np.array([mle_eval_fp2(m[row], c_star) for row in range(d)])
        # Wait: compute via eq tensor
        m_rc = mle_eval_fp2(m.ravel(), list(c_star) + list(r_star))
        assert fm == m_rc
