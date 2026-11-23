"""Benchmark: test de permutaciones en Python frente al núcleo C++ (1 hilo y todos los hilos).

    python -m backend.events.benchmark_core
    python -m backend.events.benchmark_core --draws 1000000 --python-draws 100000

No necesita datos: usa CAR placebo sintéticos. Mide el tiempo de cada motor y comprueba que los p
coinciden salvo el error de muestreo del Monte Carlo. Python hace menos remuestreos (--python-draws)
porque es mucho más lento; la velocidad se compara por remuestreo y por segundo.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time

from backend.events import fastperm


def synthetic(n_events: int, pool_size: int, seed: int) -> tuple[dict[str, list[float]], list[str]]:
    rnd = random.Random(seed)
    vols = {"A": 0.012, "B": 0.02, "C": 0.015, "D": 0.01}
    pools = {s: [rnd.gauss(0.0, v) for _ in range(pool_size)] for s, v in vols.items()}
    return pools, [rnd.choice(list(vols)) for _ in range(n_events)]


def best_time(fn, repeat: int) -> tuple[float, float]:
    """(mejor tiempo en segundos, resultado) de `repeat` ejecuciones: el mejor descarta ruido del sistema."""
    times, result = [], 0.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compara el test de permutaciones en Python y en C++.")
    p.add_argument("--events", type=int, default=20, help="eventos por contraste")
    p.add_argument("--pool", type=int, default=300, help="CAR placebo por acción")
    p.add_argument("--draws", type=int, default=2_000_000, help="remuestreos del núcleo C++")
    p.add_argument("--python-draws", type=int, default=100_000, help="remuestreos de la versión Python")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if not fastperm.available():
        raise SystemExit('\nEl núcleo C++ no está compilado: pip install -e ".[core]" && python core/build.py')

    pools, symbols = synthetic(args.events, args.pool, args.seed)
    # |CAR| medio esperado de estos eventos, +3 %: así el p queda a media escala (ni 0 ni 1) y la comparación sirve
    observed = 1.03 * statistics.mean(statistics.mean(abs(v) for v in pools[s]) for s in symbols)
    hw = fastperm.max_threads()
    print(f"\n{args.events} eventos · {args.pool} placebos por acción · {hw} hilos de hardware\n")

    runs = [
        ("Python", args.python_draws, lambda: fastperm.magnitude_pvalue(
            observed, symbols, pools, args.python_draws, args.seed, engine="python")),
        ("C++ 1 hilo", args.draws, lambda: fastperm.magnitude_pvalue(
            observed, symbols, pools, args.draws, args.seed, engine="cpp", threads=1)),
    ]
    if hw > 1:
        runs.append((f"C++ {hw} hilos", args.draws, lambda: fastperm.magnitude_pvalue(
            observed, symbols, pools, args.draws, args.seed, engine="cpp", threads=hw)))
    results = []
    print(f"{'motor':<16}{'remuestreos':>13}{'tiempo (s)':>12}{'remuestreos/s':>16}{'p':>10}{'± 1,96·EE':>11}")
    for name, draws, fn in runs:
        seconds, pv = best_time(fn, args.repeat)
        se = math.sqrt(pv * (1 - pv) / draws)
        results.append((name, draws, seconds, pv, se))
        print(f"{name:<16}{draws:>13,}{seconds:>12.3f}{draws / seconds:>16,.0f}{pv:>10.4f}{1.96 * se:>11.4f}")

    rate = [d / sec for _, d, sec, _, _ in results]
    print(f"\nVelocidad frente a Python: C++ 1 hilo ×{rate[1] / rate[0]:.0f}", end="")
    if len(results) == 3:
        print(f" · C++ {hw} hilos ×{rate[2] / rate[0]:.0f} (el reparto entre hilos aporta ×{rate[2] / rate[1]:.1f}; ideal ×{hw})")
    else:
        print("\n(Esta máquina solo tiene 1 hilo de hardware: no hay comparación multihilo.)")
    last = results[-1]
    gap = abs(results[0][3] - last[3])
    limit = 1.96 * math.hypot(results[0][4], last[4])
    verdict = "coinciden dentro del error de muestreo" if gap <= limit else "DIFIEREN más de lo esperable: revisar"
    print(f"p Python {results[0][3]:.4f} vs p {last[0]} {last[3]:.4f} (diferencia {gap:.4f}, margen {limit:.4f}): {verdict}")
    print(
        "\nLectura: el C++ no cambia el resultado científico (el p es el mismo salvo ruido de Monte Carlo), solo lo\n"
        "calcula más deprisa, lo que permite usar más remuestreos y un p más fino. Los tiempos dependen de tu máquina."
    )


if __name__ == "__main__":
    main()
