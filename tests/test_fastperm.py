"""Tests del test de permutaciones en C++ (backend/events/fastperm.py + core/permutation.cpp).

Los de "motor Python" corren siempre. Los de C++ se saltan solos si no has compilado el núcleo
(`python core/build.py`), igual que los de PyTorch se saltan sin PyTorch.
"""

from __future__ import annotations

import math
import random

import pytest

from backend.events import fastperm, study

needs_core = pytest.mark.skipif(not fastperm.available(), reason="núcleo C++ sin compilar (python core/build.py)")


def make_pools(seed: int = 3) -> dict[str, list[float]]:
    rnd = random.Random(seed)
    return {
        "A": [rnd.gauss(0.0, 0.01) for _ in range(200)],
        "B": [rnd.gauss(0.001, 0.03) for _ in range(150)],  # acción más volátil
    }


POOLS = make_pools()
SYMBOLS = ["A", "B", "A"]


# ------------------------------------------------------------------ siempre (sin C++)

def test_motor_python_da_lo_mismo_que_study():
    assert fastperm.permutation_pvalue(0.01, SYMBOLS, POOLS, 800, 5, engine="python") == study.permutation_pvalue(
        0.01, SYMBOLS, POOLS, 800, 5)
    assert fastperm.magnitude_pvalue(0.02, SYMBOLS, POOLS, 800, 5, engine="python") == study.magnitude_pvalue(
        0.02, SYMBOLS, POOLS, 800, 5)


def test_summarize_motor_python_es_el_de_siempre():
    res = [{"symbol": "A", "category": "x", "car1": 0.004, "abn_range": 1.5},
           {"symbol": "B", "category": "x", "car1": -0.02, "abn_range": 2.0}]
    assert study.summarize(res, POOLS, draws=300) == study.summarize(res, POOLS, draws=300, engine="python")


def test_motor_desconocido_da_error():
    with pytest.raises(ValueError, match="Motor desconocido"):
        fastperm.permutation_pvalue(0.0, SYMBOLS, POOLS, 10, engine="gpu")


def test_pedir_cpp_sin_compilar_explica_como_compilar(monkeypatch):
    monkeypatch.setattr(fastperm, "_core", None)
    assert fastperm.available() is False
    with pytest.raises(RuntimeError, match="core/build.py"):
        fastperm.magnitude_pvalue(0.0, SYMBOLS, POOLS, 10, engine="cpp")
    # "auto" sin núcleo cae a Python y sigue funcionando
    assert fastperm.permutation_pvalue(0.0, SYMBOLS, POOLS, 50, engine="auto") == pytest.approx(1.0)


# ------------------------------------------------------------------ solo con el núcleo compilado

M64 = (1 << 64) - 1


def _splitmix(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, (z ^ (z >> 31)) & M64


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & M64


class _RefRng:
    """Copia en Python del generador del núcleo (xoshiro256** + rechazo), para comprobarlo bit a bit."""

    def __init__(self, seed: int) -> None:
        self.s = []
        for _ in range(4):
            seed, v = _splitmix(seed)
            self.s.append(v)

    def next(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & M64, 7) * 9) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def below(self, n: int) -> int:
        limit = ((1 << 64) - n) % n
        while True:
            x = self.next()
            if x >= limit:
                return x % n


def _reference_p(observed, symbols, pools, draws, seed, magnitude, block):
    extreme = 0
    for b in range((draws + block - 1) // block):
        _, bs = _splitmix(seed ^ ((b * 0xD1342543DE82EF95) & M64))
        rng = _RefRng(bs)
        for _ in range(min(block, draws - b * block)):
            total = 0.0
            for s in symbols:
                v = pools[s][rng.below(len(pools[s]))]
                total += abs(v) if magnitude else v
            mean = total / len(symbols)
            extreme += (mean >= observed) if magnitude else (abs(mean) >= abs(observed))
    return (extreme + 1) / (draws + 1)


@needs_core
@pytest.mark.parametrize("magnitude", [False, True])
def test_cpp_coincide_bit_a_bit_con_la_referencia_en_python(magnitude):
    from backend import _core

    pools = {"A": POOLS["A"], "B": POOLS["B"]}
    observed = 0.012 if magnitude else -0.011
    draws = 2 * _core.BLOCK + 123  # tres bloques, el último incompleto
    got = fastperm.magnitude_pvalue(observed, SYMBOLS, pools, draws, 11, engine="cpp") if magnitude else \
        fastperm.permutation_pvalue(observed, SYMBOLS, pools, draws, 11, engine="cpp")
    assert got == _reference_p(observed, SYMBOLS, pools, draws, 11, magnitude, _core.BLOCK)
    assert 0.0 < got < 1.0  # que la comparación no sea trivial


@needs_core
def test_cpp_no_depende_del_numero_de_hilos():
    draws = 20_000
    ps = {t: fastperm.permutation_pvalue(0.006, SYMBOLS, POOLS, draws, 4, engine="cpp", threads=t) for t in (1, 2, 8)}
    assert len(set(ps.values())) == 1
    pm = {t: fastperm.magnitude_pvalue(0.015, SYMBOLS, POOLS, draws, 4, engine="cpp", threads=t) for t in (1, 3, 8)}
    assert len(set(pm.values())) == 1


@needs_core
def test_cpp_es_determinista_y_la_semilla_importa():
    a = fastperm.permutation_pvalue(0.006, SYMBOLS, POOLS, 8000, 1, engine="cpp")
    assert a == fastperm.permutation_pvalue(0.006, SYMBOLS, POOLS, 8000, 1, engine="cpp")
    assert a != fastperm.permutation_pvalue(0.006, SYMBOLS, POOLS, 8000, 2, engine="cpp")


@needs_core
def test_cpp_casos_limite_como_en_python():
    n = 3000
    # observado enorme: ningún remuestreo lo alcanza -> p mínimo 1/(n+1)
    assert fastperm.permutation_pvalue(0.5, SYMBOLS, POOLS, n, engine="cpp") == pytest.approx(1 / (n + 1))
    assert fastperm.magnitude_pvalue(0.5, SYMBOLS, POOLS, n, engine="cpp") == pytest.approx(1 / (n + 1))
    # observado 0: todos los remuestreos lo igualan o superan -> p = 1
    assert fastperm.permutation_pvalue(0.0, SYMBOLS, POOLS, n, engine="cpp") == 1.0
    assert fastperm.magnitude_pvalue(0.0, SYMBOLS, POOLS, n, engine="cpp") == 1.0
    # el p siempre es (extremas+1)/(n+1): p·(n+1) es entero
    p = fastperm.permutation_pvalue(0.004, SYMBOLS, POOLS, n, engine="cpp")
    assert p * (n + 1) == pytest.approx(round(p * (n + 1)))


@needs_core
@pytest.mark.parametrize("observed,kind", [(0.004, "signed"), (0.02, "magnitude")])
def test_cpp_y_python_coinciden_salvo_error_de_muestreo(observed, kind):
    draws = 40_000
    f = fastperm.permutation_pvalue if kind == "signed" else fastperm.magnitude_pvalue
    py_p = f(observed, SYMBOLS, POOLS, draws, 9, engine="python")
    cpp_p = f(observed, SYMBOLS, POOLS, draws, 9, engine="cpp")
    p = (py_p + cpp_p) / 2
    se_diff = math.sqrt(2 * p * (1 - p) / draws)  # error típico de la diferencia entre dos Monte Carlo
    assert 0.02 < p < 0.98
    assert abs(py_p - cpp_p) < 4 * se_diff


@needs_core
def test_cpp_magnitud_ignora_el_signo_de_los_placebos():
    pos = {"A": [abs(v) for v in POOLS["A"]]}
    neg = {"A": [-abs(v) for v in POOLS["A"]]}
    args = (0.008, ["A", "A"])
    assert fastperm.magnitude_pvalue(*args, pos, 5000, 3, engine="cpp") == fastperm.magnitude_pvalue(
        *args, neg, 5000, 3, engine="cpp")


@needs_core
def test_cpp_valida_los_datos():
    with pytest.raises(ValueError):
        fastperm.permutation_pvalue(0.0, SYMBOLS, POOLS, 0, engine="cpp")  # draws < 1
    with pytest.raises(ValueError):
        fastperm.permutation_pvalue(0.0, [], POOLS, 10, engine="cpp")  # sin eventos
    with pytest.raises(KeyError):
        fastperm.permutation_pvalue(0.0, ["Z"], POOLS, 10, engine="cpp")  # acción sin placebos
    with pytest.raises(ValueError):
        fastperm.permutation_pvalue(0.0, ["A"], {"A": []}, 10, engine="cpp")  # conjunto vacío
    with pytest.raises(ValueError):
        fastperm.permutation_pvalue(0.0, SYMBOLS, POOLS, 10, engine="cpp", threads=-1)


@needs_core
def test_summarize_con_motor_cpp_da_p_parecidos_a_los_de_python():
    res = [{"symbol": s, "category": "x", "car1": c, "abn_range": 1.0}
           for s, c in [("A", 0.004), ("B", -0.05), ("A", 0.001), ("B", 0.06)]]
    py_row = study.summarize(res, POOLS, draws=20_000)[0]
    cpp_row = study.summarize(res, POOLS, draws=20_000, engine="cpp")[0]
    for key in ("p", "p_abs"):
        p = (py_row[key] + cpp_row[key]) / 2
        assert abs(py_row[key] - cpp_row[key]) < 4 * math.sqrt(2 * p * (1 - p) / 20_000) + 1e-9
    assert py_row["mean_car1"] == cpp_row["mean_car1"]  # lo que no es Monte Carlo es idéntico
