"""Finite field arithmetic (Mersenne-61 and quadratic extension F_p^2),
Multilinear Extension (MLE) evaluation, Fiat-Shamir Transcript, and
hiding/binding vector commitments.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Mersenne-61 prime: p = 2^61 - 1.
# Note: p ≡ 3 (mod 4), so -1 is a quadratic non-residue in F_p,
# making F_p^2 = F_p[u] / (u^2 + 1) a valid 122-bit extension field.
P: int = (1 << 61) - 1
P_U64 = np.uint64(P)
MASK_31 = np.uint64((1 << 31) - 1)
MASK_30 = np.uint64((1 << 30) - 1)


# ---------------------------------------------------------------------------
# Scalar base-field F_p helpers
# ---------------------------------------------------------------------------

def fp_add(a: int, b: int) -> int:
    s = (a % P) + (b % P)
    return s - P if s >= P else s


def fp_sub(a: int, b: int) -> int:
    d = (a % P) - (b % P)
    return d + P if d < 0 else d


def fp_mul(a: int, b: int) -> int:
    return ((a % P) * (b % P)) % P


def fp_neg(a: int) -> int:
    a = a % P
    return 0 if a == 0 else P - a


def fp_pow(a: int, e: int) -> int:
    return pow(a % P, e, P)


def fp_inv(a: int) -> int:
    a = a % P
    if a == 0:
        raise ZeroDivisionError("Cannot invert 0 in F_p")
    return pow(a, P - 2, P)


# ---------------------------------------------------------------------------
# Vectorized NumPy F_p helpers (uint64, zero-overflow 31-bit limb arithmetic)
# ---------------------------------------------------------------------------

def vec_mod(x: np.ndarray) -> np.ndarray:
    """Reduce uint64 array modulo P = 2^61 - 1."""
    x = np.asarray(x, dtype=np.uint64)
    lo = x & P_U64
    hi = x >> np.uint64(61)
    s = lo + hi
    return np.where(s >= P_U64, s - P_U64, s)


def vec_add(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    s = a + b
    return np.where(s >= P_U64, s - P_U64, s)


def vec_sub(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    return np.where(a >= b, a - b, a + (P_U64 - b))


def vec_neg(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    return np.where(a == np.uint64(0), np.uint64(0), P_U64 - a)


def vec_mul(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    """Vectorized multiplication mod 2^61 - 1 using 31-bit limb decomposition.
    All intermediate values are < 2^63, fitting safely in uint64 without overflow.
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    a0 = a & MASK_31
    a1 = a >> np.uint64(31)
    b0 = b & MASK_31
    b1 = b >> np.uint64(31)

    p0 = a0 * b0  # < 2^62
    m = a0 * b1 + a1 * b0  # < 2^62
    p2 = a1 * b1  # < 2^60

    # 2^31 * m ≡ (m_lo << 31) + m_hi (mod 2^61 - 1)
    m_lo = m & MASK_30
    m_hi = m >> np.uint64(30)
    mid = (m_lo << np.uint64(31)) + m_hi

    # 2^62 * p2 ≡ 2 * p2 (mod 2^61 - 1)
    hi_term = p2 << np.uint64(1)

    # Reduce p0 + mid + hi_term mod P
    s = vec_mod(p0) + vec_mod(mid)
    s = np.where(s >= P_U64, s - P_U64, s)
    s = s + vec_mod(hi_term)
    return np.where(s >= P_U64, s - P_U64, s)


def vec_dot(a: np.ndarray, b: np.ndarray) -> int:
    """Inner product of two F_p vectors."""
    prod = vec_mul(a, b)
    # Sum in Python int or chunked uint64 to avoid overflow for large vectors
    # Since prod elements < 2^61, summing up to 8 elements fits in uint64;
    # for general lengths, use object/int64 chunking or np.sum with modulo.
    n = prod.size
    if n == 0:
        return 0
    if n <= 4:
        return int(np.sum(prod)) % P
    # Pairwise tree reduction or Python int sum over uint64 chunks
    # Convert to Python int64/uint64 chunks of 4:
    rem = n % 4
    acc = int(np.sum(prod[:rem])) % P if rem else 0
    if n >= 4:
        reshaped = prod[rem:].reshape(-1, 4)
        # sum of 4 elements < 4 * 2^61 = 2^63 < 2^64
        s4 = vec_mod(np.sum(reshaped, axis=1, dtype=np.uint64))
        while s4.size > 4:
            r = s4.size % 4
            if r:
                acc = (acc + int(np.sum(s4[:r]))) % P
            s4 = vec_mod(np.sum(s4[r:].reshape(-1, 4), axis=1, dtype=np.uint64))
        acc = (acc + int(np.sum(s4))) % P
    return acc


def vec_sum(a: np.ndarray) -> int:
    """Sum of an F_p vector modulo P."""
    a = np.asarray(a, dtype=np.uint64).ravel()
    n = a.size
    if n == 0:
        return 0
    acc = 0
    while a.size > 4:
        r = a.size % 4
        if r:
            acc = (acc + int(np.sum(a[:r]))) % P
        a = vec_mod(np.sum(a[r:].reshape(-1, 4), axis=1, dtype=np.uint64))
    return (acc + int(np.sum(a))) % P


# ---------------------------------------------------------------------------
# Quadratic Extension Field F_p^2 = F_p[u] / (u^2 + 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Fp2:
    """Element c0 + c1 * u in F_p^2 where u^2 = -1."""
    c0: int
    c1: int

    @staticmethod
    def zero() -> Fp2:
        return Fp2(0, 0)

    @staticmethod
    def one() -> Fp2:
        return Fp2(1, 0)

    @staticmethod
    def from_fp(x: int) -> Fp2:
        return Fp2(x % P, 0)

    def __add__(self, other: Fp2 | int) -> Fp2:
        if isinstance(other, int):
            return Fp2(fp_add(self.c0, other), self.c1)
        return Fp2(fp_add(self.c0, other.c0), fp_add(self.c1, other.c1))

    def __radd__(self, other: int) -> Fp2:
        return self.__add__(other)

    def __sub__(self, other: Fp2 | int) -> Fp2:
        if isinstance(other, int):
            return Fp2(fp_sub(self.c0, other), self.c1)
        return Fp2(fp_sub(self.c0, other.c0), fp_sub(self.c1, other.c1))

    def __rsub__(self, other: int) -> Fp2:
        return Fp2(fp_sub(other, self.c0), fp_neg(self.c1))

    def __neg__(self) -> Fp2:
        return Fp2(fp_neg(self.c0), fp_neg(self.c1))

    def __mul__(self, other: Fp2 | int) -> Fp2:
        if isinstance(other, int):
            return Fp2(fp_mul(self.c0, other), fp_mul(self.c1, other))
        # (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u
        a0b0 = fp_mul(self.c0, other.c0)
        a1b1 = fp_mul(self.c1, other.c1)
        a0b1 = fp_mul(self.c0, other.c1)
        a1b0 = fp_mul(self.c1, other.c0)
        return Fp2(fp_sub(a0b0, a1b1), fp_add(a0b1, a1b0))

    def __rmul__(self, other: int) -> Fp2:
        return self.__mul__(other)

    def inv(self) -> Fp2:
        # (c0 + c1 u)^-1 = (c0 - c1 u) / (c0^2 + c1^2)
        norm = fp_add(fp_mul(self.c0, self.c0), fp_mul(self.c1, self.c1))
        norm_inv = fp_inv(norm)
        return Fp2(fp_mul(self.c0, norm_inv), fp_mul(fp_neg(self.c1), norm_inv))

    def to_bytes(self) -> bytes:
        return self.c0.to_bytes(8, "little") + self.c1.to_bytes(8, "little")


# ---------------------------------------------------------------------------
# Vectorized F_p^2 arrays (stored as pair (c0, c1) of uint64 arrays)
# ---------------------------------------------------------------------------

def fp2_vec_from_fp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.uint64)
    return a, np.zeros_like(a)


def fp2_vec_add(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return vec_add(a[0], b[0]), vec_add(a[1], b[1])


def fp2_vec_sub(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return vec_sub(a[0], b[0]), vec_sub(a[1], b[1])


def fp2_vec_mul(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    a0, a1 = a
    b0, b1 = b
    c0 = vec_sub(vec_mul(a0, b0), vec_mul(a1, b1))
    c1 = vec_add(vec_mul(a0, b1), vec_mul(a1, b0))
    return c0, c1


def fp2_vec_mul_scalar(
    a: tuple[np.ndarray, np.ndarray], s: Fp2
) -> tuple[np.ndarray, np.ndarray]:
    a0, a1 = a
    c0 = vec_sub(vec_mul(a0, s.c0), vec_mul(a1, s.c1))
    c1 = vec_add(vec_mul(a0, s.c1), vec_mul(a1, s.c0))
    return c0, c1


def fp2_vec_sum(a: tuple[np.ndarray, np.ndarray]) -> Fp2:
    return Fp2(vec_sum(a[0]), vec_sum(a[1]))


# ---------------------------------------------------------------------------
# Multilinear Extension (MLE) Engine
# ---------------------------------------------------------------------------

def mle_eval_fp(evals: np.ndarray, point: Sequence[int]) -> int:
    """Evaluate MLE of f: {0,1}^v -> F_p at point r in F_p^v in O(2^v) time."""
    cur = np.asarray(evals, dtype=np.uint64).ravel()
    expected = 1 << len(point)
    if cur.size != expected:
        raise ValueError(f"Expected {expected} evals for {len(point)} vars, got {cur.size}")
    for r_i in point:
        r = np.uint64(r_i % P)
        even = cur[0::2]
        odd = cur[1::2]
        diff = vec_sub(odd, even)
        cur = vec_add(even, vec_mul(diff, r))
    return int(cur[0])


def mle_eval_fp2(evals: np.ndarray | tuple[np.ndarray, np.ndarray], point: Sequence[Fp2]) -> Fp2:
    """Evaluate MLE of f: {0,1}^v -> F_p (or F_p^2) at point r in (F_p^2)^v in O(2^v)."""
    if isinstance(evals, tuple):
        c0 = np.asarray(evals[0], dtype=np.uint64).ravel()
        c1 = np.asarray(evals[1], dtype=np.uint64).ravel()
    else:
        c0 = np.asarray(evals, dtype=np.uint64).ravel()
        c1 = np.zeros_like(c0)
    expected = 1 << len(point)
    if c0.size != expected:
        raise ValueError(f"Expected {expected} evals for {len(point)} vars, got {c0.size}")
    for r_i in point:
        e0, e1 = c0[0::2], c1[0::2]
        o0, o1 = c0[1::2], c1[1::2]
        d0, d1 = vec_sub(o0, e0), vec_sub(o1, e1)
        p0, p1 = fp2_vec_mul_scalar((d0, d1), r_i)
        c0 = vec_add(e0, p0)
        c1 = vec_add(e1, p1)
    return Fp2(int(c0[0]), int(c1[0]))


def eq_evals_fp(point: Sequence[int]) -> np.ndarray:
    """Compute eq(x, r) for all x in {0,1}^v in O(2^v) time over F_p.
    Matches LSB-first indexing of mle_eval (point[0] is bit 0).
    """
    cur = np.array([1], dtype=np.uint64)
    for r_i in point:
        r = r_i % P
        one_minus_r = fp_sub(1, r)
        left = vec_mul(cur, one_minus_r)
        right = vec_mul(cur, r)
        cur = np.concatenate([left, right])
    return cur


def eq_evals_fp2(point: Sequence[Fp2]) -> tuple[np.ndarray, np.ndarray]:
    """Compute eq(x, r) for all x in {0,1}^v in O(2^v) time over F_p^2.
    Matches LSB-first indexing of mle_eval (point[0] is bit 0).
    """
    c0 = np.array([1], dtype=np.uint64)
    c1 = np.array([0], dtype=np.uint64)
    for r_i in point:
        one_minus_r = Fp2(fp_sub(1, r_i.c0), fp_neg(r_i.c1))
        l0, l1 = fp2_vec_mul_scalar((c0, c1), one_minus_r)
        r0, r1 = fp2_vec_mul_scalar((c0, c1), r_i)
        c0 = np.concatenate([l0, r0])
        c1 = np.concatenate([l1, r1])
    return c0, c1


def eq_eval_point_fp2(x: Sequence[Fp2], r: Sequence[Fp2]) -> Fp2:
    """Evaluate eq(x, r) at two single points in (F_p^2)^v in O(v) time."""
    if len(x) != len(r):
        raise ValueError("Point dimensions must match")
    acc = Fp2.one()
    one = Fp2.one()
    for xi, ri in zip(x, r):
        term = xi * ri + (one - xi) * (one - ri)
        acc = acc * term
    return acc


def int_to_bits(idx: int, n_bits: int) -> list[int]:
    """Little-endian or big-endian bit representation matching MLE indexing.
    In mle_eval, point[0] splits even/odd (LSB-first), so bit 0 is (idx & 1).
    """
    return [(idx >> b) & 1 for b in range(n_bits)]


# ---------------------------------------------------------------------------
# Fiat-Shamir Transcript
# ---------------------------------------------------------------------------

class Transcript:
    """Domain-separated Fiat-Shamir transcript backed by BLAKE2b-256."""

    def __init__(self, label: bytes = b"zkMoE-Reason-v1") -> None:
        self._state = hashlib.blake2b(label, digest_size=32).digest()

    def absorb_bytes(self, label: bytes, data: bytes) -> None:
        h = hashlib.blake2b(digest_size=32)
        h.update(self._state)
        h.update(len(label).to_bytes(4, "little"))
        h.update(label)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
        self._state = h.digest()

    def absorb_fp(self, label: bytes, val: int) -> None:
        self.absorb_bytes(label, (val % P).to_bytes(8, "little"))

    def absorb_fp2(self, label: bytes, val: Fp2) -> None:
        self.absorb_bytes(label, val.to_bytes())

    def absorb_vec(self, label: bytes, vec: np.ndarray) -> None:
        arr = np.asarray(vec, dtype=np.uint64).ravel()
        self.absorb_bytes(label, arr.tobytes())

    def challenge_fp(self, label: bytes) -> int:
        self.absorb_bytes(b"chal:" + label, b"")
        val = int.from_bytes(self._state[:8], "little") & P
        if val == P:
            val = 0
        return val

    def challenge_fp2(self, label: bytes) -> Fp2:
        self.absorb_bytes(b"chal2:" + label, b"")
        h1 = hashlib.blake2b(self._state + b"\x01", digest_size=16).digest()
        c0 = int.from_bytes(h1[:8], "little") & P
        c1 = int.from_bytes(h1[8:], "little") & P
        if c0 == P:
            c0 = 0
        if c1 == P:
            c1 = 0
        return Fp2(c0, c1)

    def challenge_point_fp2(self, label: bytes, n_vars: int) -> list[Fp2]:
        return [self.challenge_fp2(f"{label.decode()}_{i}".encode()) for i in range(n_vars)]


# ---------------------------------------------------------------------------
# Hiding & Binding Vector / MLE Commitment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VectorCommitment:
    """Salted BLAKE2b Merkle-root commitment to an F_p vector."""
    root: bytes
    length: int

    def to_bytes(self) -> bytes:
        return self.root + self.length.to_bytes(8, "little")


def commit_vector(vec: np.ndarray, blind: bytes | None = None) -> tuple[VectorCommitment, bytes]:
    """Commit to an F_p vector with a 32-byte hiding salt."""
    if blind is None:
        blind = secrets.token_bytes(32)
    arr = np.asarray(vec, dtype=np.uint64).ravel()
    h = hashlib.blake2b(digest_size=32)
    h.update(b"zkMoE-commit-v1")
    h.update(blind)
    h.update(arr.size.to_bytes(8, "little"))
    h.update(arr.tobytes())
    return VectorCommitment(root=h.digest(), length=arr.size), blind


def verify_vector_commitment(comm: VectorCommitment, vec: np.ndarray, blind: bytes) -> bool:
    recomputed, _ = commit_vector(vec, blind=blind)
    return recomputed == comm
