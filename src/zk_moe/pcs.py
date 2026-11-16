"""Succinct Multilinear Polynomial Commitment Scheme (Brakedown-style Tensor Code + Merkle Tree).

Provides O(sqrt(M)) proof size and O(sqrt(M)) verifier time with no trusted setup,
operating over Mersenne-61 (F_p) with F_p^2 evaluation points.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from zk_moe.field_and_mle import (
    P,
    Fp2,
    Transcript,
    eq_evals_fp2,
    fp2_vec_from_fp,
    fp2_vec_mul,
    fp2_vec_sum,
    vec_add,
    vec_mul,
)
from zk_moe.sumcheck_matmul import _fp_vec_mat_mul_exact

CODE_DEGREE = 8
NUM_PROXIMITY_QUERIES = 16


@dataclass(frozen=True)
class PCSCommitment:
    root: bytes
    n_vars: int
    rows: int
    cols: int

    def to_bytes(self) -> bytes:
        return self.root + self.n_vars.to_bytes(4, "little")


@dataclass
class PCSOpeningProof:
    eval_point: list[Fp2]
    eval_value: Fp2
    combined_row0: np.ndarray  # length cols (c0)
    combined_row1: np.ndarray  # length cols (c1)
    queried_indices: list[int]
    queried_columns: list[np.ndarray]  # each length rows (uint64)
    merkle_paths: list[list[tuple[bytes, bool]]]  # (sibling_hash, is_right)


def _code_params(cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic sparse expander code indices (cols, CODE_DEGREE) and weights."""
    # Use deterministic PRNG seeded by cols so prover and verifier agree without storing matrix
    rng = np.random.default_rng(0x5A4B_0000 + cols)
    indices = rng.integers(0, cols, size=(cols, CODE_DEGREE), dtype=np.int64)
    weights = rng.integers(1, 256, size=(cols, CODE_DEGREE), dtype=np.uint64)
    return indices, weights


def _hash_leaf(col_idx: int, col_data: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=32)
    h.update(b"pcs_leaf")
    h.update(col_idx.to_bytes(4, "little"))
    h.update(np.ascontiguousarray(col_data, dtype=np.uint64).tobytes())
    return h.digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=32)
    h.update(b"pcs_node")
    h.update(left)
    h.update(right)
    return h.digest()


def _build_merkle_tree(leaves: list[bytes]) -> list[list[bytes]]:
    tree = [leaves]
    cur = leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else left
            nxt.append(_hash_node(left, right))
        tree.append(nxt)
        cur = nxt
    return tree


def _merkle_open(tree: list[list[bytes]], idx: int) -> list[tuple[bytes, bool]]:
    path = []
    cur_idx = idx
    for level in tree[:-1]:
        is_right = (cur_idx % 2) == 1
        sib_idx = cur_idx - 1 if is_right else cur_idx + 1
        sib_hash = level[sib_idx] if sib_idx < len(level) else level[cur_idx]
        path.append((sib_hash, is_right))
        cur_idx //= 2
    return path


def _merkle_verify(root: bytes, leaf_hash: bytes, path: list[tuple[bytes, bool]]) -> bool:
    cur = leaf_hash
    for sib, is_right in path:
        if is_right:
            cur = _hash_node(sib, cur)
        else:
            cur = _hash_node(cur, sib)
    return cur == root


@dataclass
class PCSProverKey:
    commitment: PCSCommitment
    encoded_matrix: np.ndarray  # (rows, 2 * cols)
    merkle_tree: list[list[bytes]]


def pcs_commit(evals: np.ndarray) -> PCSProverKey:
    """Commit to a multilinear polynomial of size 2^v in O(2^v) time."""
    flat = np.asarray(evals, dtype=np.uint64).ravel()
    m = flat.size
    if m == 0 or (m & (m - 1)) != 0:
        raise ValueError(f"Size must be power of 2, got {m}")
    n_vars = m.bit_length() - 1
    v_row = n_vars // 2
    v_col = n_vars - v_row
    rows = 1 << v_row
    cols = 1 << v_col

    # Reshape so row index is MSB (bits v_col..n_vars-1) and col index is LSB (bits 0..v_col-1)
    # Wait: in our LSB-first mle_eval, point[:v_col] splits LSBs (columns) and point[v_col:] splits MSBs (rows).
    # In C-order (rows, cols), axis 1 is contiguous (LSB), axis 0 is stride cols (MSB)!
    mat = flat.reshape(rows, cols)

    # Encode parity columns in O(M) using sparse expander code
    idx_map, w_map = _code_params(cols)
    parity = np.zeros((rows, cols), dtype=np.uint64)
    for d_i in range(CODE_DEGREE):
        gathered = mat[:, idx_map[:, d_i]]  # (rows, cols)
        scaled = vec_mul(gathered, w_map[:, d_i][None, :])
        parity = vec_add(parity, scaled)

    encoded = np.concatenate([mat, parity], axis=1)  # (rows, 2 * cols)
    leaves = [_hash_leaf(j, encoded[:, j]) for j in range(2 * cols)]
    tree = _build_merkle_tree(leaves)

    comm = PCSCommitment(root=tree[-1][0], n_vars=n_vars, rows=rows, cols=cols)
    return PCSProverKey(commitment=comm, encoded_matrix=encoded, merkle_tree=tree)


def pcs_open(
    pk: PCSProverKey,
    point: Sequence[Fp2],
    transcript: Transcript,
) -> PCSOpeningProof:
    """Produce a succinct O(sqrt(M)) opening proof for f(point)."""
    comm = pk.commitment
    assert len(point) == comm.n_vars
    v_col = comm.cols.bit_length() - 1
    pt_col = point[:v_col]
    pt_row = point[v_col:]

    eq_r0, eq_r1 = eq_evals_fp2(pt_row)
    eq_c0, eq_c1 = eq_evals_fp2(pt_col)

    mat = pk.encoded_matrix[:, : comm.cols]
    w0 = _fp_vec_mat_mul_exact(eq_r0, mat)
    w1 = _fp_vec_mat_mul_exact(eq_r1, mat)

    val = fp2_vec_sum(fp2_vec_mul((w0, w1), (eq_c0, eq_c1)))

    transcript.absorb_bytes(b"pcs_root", comm.root)
    transcript.absorb_fp2(b"pcs_val", val)
    transcript.absorb_vec(b"pcs_w0", w0)
    transcript.absorb_vec(b"pcs_w1", w1)

    total_cols = 2 * comm.cols
    n_queries = min(NUM_PROXIMITY_QUERIES, total_cols)
    q_indices = []
    for q_i in range(n_queries):
        chal = transcript.challenge_fp(f"pcs_q_{q_i}".encode())
        q_indices.append(chal % total_cols)

    q_cols = [np.ascontiguousarray(pk.encoded_matrix[:, j]) for j in q_indices]
    paths = [_merkle_open(pk.merkle_tree, j) for j in q_indices]

    return PCSOpeningProof(
        eval_point=list(point),
        eval_value=val,
        combined_row0=w0,
        combined_row1=w1,
        queried_indices=q_indices,
        queried_columns=q_cols,
        merkle_paths=paths,
    )


def pcs_verify(
    comm: PCSCommitment,
    proof: PCSOpeningProof,
    transcript: Transcript,
) -> bool:
    """Verify a succinct PCS opening proof in O(sqrt(M)) time without the polynomial."""
    if len(proof.eval_point) != comm.n_vars:
        return False
    if proof.combined_row0.size != comm.cols or proof.combined_row1.size != comm.cols:
        return False

    v_col = comm.cols.bit_length() - 1
    pt_col = proof.eval_point[:v_col]
    pt_row = proof.eval_point[v_col:]

    eq_r0, eq_r1 = eq_evals_fp2(pt_row)
    eq_c0, eq_c1 = eq_evals_fp2(pt_col)

    # 1. Check final evaluation <w, eq_col> == eval_value
    expected_val = fp2_vec_sum(
        fp2_vec_mul((proof.combined_row0, proof.combined_row1), (eq_c0, eq_c1))
    )
    if expected_val != proof.eval_value:
        return False

    # 2. Replay transcript to get queried column indices
    transcript.absorb_bytes(b"pcs_root", comm.root)
    transcript.absorb_fp2(b"pcs_val", proof.eval_value)
    transcript.absorb_vec(b"pcs_w0", proof.combined_row0)
    transcript.absorb_vec(b"pcs_w1", proof.combined_row1)

    total_cols = 2 * comm.cols
    n_queries = min(NUM_PROXIMITY_QUERIES, total_cols)
    if len(proof.queried_indices) != n_queries:
        return False

    idx_map, w_map = _code_params(comm.cols)

    for q_i, (j, col_data, path) in enumerate(
        zip(proof.queried_indices, proof.queried_columns, proof.merkle_paths)
    ):
        expected_j = transcript.challenge_fp(f"pcs_q_{q_i}".encode()) % total_cols
        if j != expected_j:
            return False
        if col_data.size != comm.rows:
            return False
        if not _merkle_verify(comm.root, _hash_leaf(j, col_data), path):
            return False

        # Check eq_row . col_data == encoded(w)[j]
        from zk_moe.field_and_mle import vec_dot
        lhs0 = vec_dot(eq_r0, col_data)
        lhs1 = vec_dot(eq_r1, col_data)

        if j < comm.cols:
            rhs0 = int(proof.combined_row0[j])
            rhs1 = int(proof.combined_row1[j])
        else:
            p_j = j - comm.cols
            rhs0 = 0
            rhs1 = 0
            for d_i in range(CODE_DEGREE):
                src = int(idx_map[p_j, d_i])
                wt = int(w_map[p_j, d_i])
                rhs0 = (rhs0 + wt * int(proof.combined_row0[src])) % P
                rhs1 = (rhs1 + wt * int(proof.combined_row1[src])) % P

        if lhs0 != rhs0 or lhs1 != rhs1:
            return False

    return True
