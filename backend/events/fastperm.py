"""Test de permutaciones con el núcleo C++ (si está compilado) o con Python (si no).

Mismas funciones que en `study.py`, con un parámetro más, `engine`:
- "python": la versión de siempre (`study.py`); mismos números que antes.
- "cpp": el núcleo C++ multihilo (error si no está compilado: `python core/build.py`).
- "auto": C++ si está compilado, Python si no.

Ojo: C++ y Python usan generadores de números aleatorios distintos, así que con la misma semilla NO dan
exactamente el mismo p, sino valores que difieren solo por el error de muestreo del propio Monte Carlo
(≈ √(p·(1−p)/remuestreos)). Lo que sí es exacto: C++ da el mismo p con 1 hilo que con 16.
"""

from __future__ import annotations

from backend.events import study

try:  # el módulo compilado vive en backend/_core*.so
    from backend import _core
except ImportError:  # sin compilar: se usa Python
    _core = None

ENGINES = ("auto", "python", "cpp")


def available() -> bool:
    """¿Está compilado el núcleo C++?"""
    return _core is not None


def max_threads() -> int:
    return _core.max_threads() if _core is not None else 1


def resolve_engine(engine: str) -> str:
    """Convierte "auto" en "cpp" o "python" y avisa si se pide C++ sin tenerlo compilado."""
    if engine not in ENGINES:
        raise ValueError(f"Motor desconocido: {engine!r} (usa {', '.join(ENGINES)})")
    if engine == "cpp" and _core is None:
        raise RuntimeError('El núcleo C++ no está compilado: pip install -e ".[core]" && python core/build.py')
    if engine == "auto":
        return "cpp" if _core is not None else "python"
    return engine


def _to_core(symbols: list[str], pools: dict[str, list[float]]) -> tuple[list[list[float]], list[int]]:
    names = list(pools)
    index = {name: i for i, name in enumerate(names)}
    return [list(pools[n]) for n in names], [index[s] for s in symbols]  # KeyError si falta una acción


def permutation_pvalue(
    observed_mean: float, symbols: list[str], pools: dict[str, list[float]], draws: int = 5000,
    seed: int = 7, engine: str = "auto", threads: int = 0,
) -> float:
    """p de dos colas sobre la media de CAR con signo (ver `study.permutation_pvalue`). threads=0: todos."""
    if resolve_engine(engine) == "python":
        return study.permutation_pvalue(observed_mean, symbols, pools, draws, seed)
    pool_lists, picks = _to_core(symbols, pools)
    return _core.permutation_pvalue(pool_lists, picks, observed_mean, draws, seed, False, threads)


def magnitude_pvalue(
    observed_mean_abs: float, symbols: list[str], pools: dict[str, list[float]], draws: int = 5000,
    seed: int = 7, engine: str = "auto", threads: int = 0,
) -> float:
    """p de una cola sobre la media de |CAR| (ver `study.magnitude_pvalue`). threads=0: todos."""
    if resolve_engine(engine) == "python":
        return study.magnitude_pvalue(observed_mean_abs, symbols, pools, draws, seed)
    pool_lists, picks = _to_core(symbols, pools)
    return _core.permutation_pvalue(pool_lists, picks, observed_mean_abs, draws, seed, True, threads)
