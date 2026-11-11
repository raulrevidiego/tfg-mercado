"""Mínimos cuadrados ordinarios (regresión lineal) en Python puro.

Se escribe a mano para entender qué hace: resuelve las "ecuaciones normales" (X'X) b = X'y
con eliminación gaussiana. Con pocas variables es instantáneo y no necesita NumPy.
"""

from __future__ import annotations


def fit(X: list[list[float]], y: list[float]) -> list[float]:
    """Ajusta y ≈ b0 + b1·x1 + ... + bk·xk. Devuelve [b0, b1, ..., bk]."""
    k = len(X[0]) + 1  # +1 por el intercepto b0
    A = [[0.0] * k for _ in range(k)]
    c = [0.0] * k
    for row, target in zip(X, y):
        r = [1.0, *row]
        for i in range(k):
            c[i] += r[i] * target
            for j in range(k):
                A[i][j] += r[i] * r[j]
    return _solve(A, c)


def predict(coef: list[float], x: list[float]) -> float:
    return coef[0] + sum(b * v for b, v in zip(coef[1:], x))


def _solve(A: list[list[float]], c: list[float]) -> list[float]:
    """Resuelve A·b = c por eliminación gaussiana con pivote parcial."""
    n = len(c)
    M = [row[:] + [c[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Matriz singular: regresores colineales o datos insuficientes")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    b = [0.0] * n
    for i in reversed(range(n)):
        b[i] = (M[i][n] - sum(M[i][j] * b[j] for j in range(i + 1, n))) / M[i][i]
    return b
