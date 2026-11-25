"""Pruebas de robustez de la reacción en tamaño (|CAR|) por categoría de evento.

Un p muy pequeño sobre el conjunto de eventos puede deberse a UNA acción o a unos pocos eventos enormes.
Para cada categoría que supera el umbral de Bonferroni se repite el contraste de magnitud de tres formas:

1. Por acción: cada acción sola frente a sus propias fechas al azar. Con pocos eventos por acción el p
   de cada una es poco potente; lo que se mira es que el cociente (|CAR| real / |CAR| del azar) sea > 1 en todas.
2. Sin cada acción: se quita una acción y se repite el contraste con las demás. Si la categoría depende de
   una sola acción, el p se estropea al quitarla.
3. Sin los mayores: se quitan los 3 eventos con más reacción. Es una prueba CONSERVADORA: los quita de los
   eventos reales pero no de las fechas al azar, así que sesga el resultado en contra; si aun así aguanta, es sólido.

Veredicto (convenciones mías, no una regla estadística):
- "frágil": quitando una acción o los 3 mayores, el p deja de pasar el umbral de Bonferroni.
- "robusta con excepciones": aguanta esas pruebas, pero hay acciones donde el efecto no aparece (con 3 o más
  eventos, cociente < 1,1; con 10 o más, además, p ≥ 0,05 en solitario). El efecto es de unas acciones, no de todas.
- "robusta": aguanta y no hay excepciones.

Es la misma permutación del estudio de eventos (`study.magnitude_pvalue`; con `--motor cpp`, mucho más rápida).
"""

from __future__ import annotations

import statistics

from backend.events import fastperm

MIN_EVENTS = 10   # con menos eventos no se hacen estas pruebas
MIN_PER_STOCK = 3     # una acción con menos eventos no se juzga (ni como excepción)
EXCEPTION_RATIO = 1.1  # excepción: |CAR| real menos de un 10 % por encima del del azar
MIN_FOR_STOCK_TEST = 10  # desde aquí, la acción sola también debe dar p < 0,05
STOCK_P = 0.05


def _expected_abs(symbols: list[str], pools: dict[str, list[float]]) -> float:
    """|CAR| medio que se espera en fechas al azar de esas acciones (una por evento)."""
    return sum(sum(abs(v) for v in pools[s]) / len(pools[s]) for s in symbols) / len(symbols)


def _expected_median(symbols: list[str], pools: dict[str, list[float]]) -> float:
    return sum(statistics.median(abs(v) for v in pools[s]) for s in symbols) / len(symbols)


def analyze_category(
    group: list[dict], pools: dict[str, list[float]], draws: int = 5000, seed: int = 7,
    engine: str = "python", threads: int = 0, threshold: float = 0.005, trim: int = 3,
) -> dict:
    """Contraste de magnitud de una categoría y sus tres pruebas de robustez (ver el docstring del módulo)."""
    symbols = [r["symbol"] for r in group]
    cars = [abs(r["car1"]) for r in group]
    n = len(group)

    def test(idx: list[int]) -> dict:
        syms = [symbols[i] for i in idx]
        mean_abs = sum(cars[i] for i in idx) / len(idx)
        expected = _expected_abs(syms, pools)
        p = fastperm.magnitude_pvalue(mean_abs, syms, pools, draws, seed, engine, threads)
        return {"n": len(idx), "mean_abs": mean_abs, "expected_abs": expected, "ratio": mean_abs / expected, "p": p}

    everyone = list(range(n))
    out = {"n": n, "overall": test(everyone), "median_abs": statistics.median(cars),
           "expected_median": _expected_median(symbols, pools), "by_stock": [], "without_stock": [], "trimmed": None,
           "reasons": [], "exceptions": []}
    out["verdict"] = "sin efecto detectable"
    if out["overall"]["p"] >= threshold:
        return out

    stocks = sorted(set(symbols))
    for s in stocks:
        out["by_stock"].append({"symbol": s, **test([i for i in everyone if symbols[i] == s])})
    if len(stocks) > 1:
        for s in stocks:
            out["without_stock"].append({"without": s, **test([i for i in everyone if symbols[i] != s])})
    if n - trim >= 5:
        order = sorted(everyone, key=lambda i: -cars[i])
        dropped = order[:trim]
        out["trimmed"] = {**test([i for i in everyone if i not in set(dropped)]),
                          "dropped": [(symbols[i], cars[i]) for i in dropped]}

    reasons = [f"sin {x['without']}: p = {x['p']:.4f}" for x in out["without_stock"] if x["p"] >= threshold]
    if out["trimmed"] and out["trimmed"]["p"] >= threshold:
        reasons.append(f"sin los {len(out['trimmed']['dropped'])} mayores: p = {out['trimmed']['p']:.4f}")
    out["reasons"] = reasons
    out["exceptions"] = [
        x["symbol"] for x in out["by_stock"]
        if x["n"] >= MIN_PER_STOCK and (x["ratio"] < EXCEPTION_RATIO or (x["n"] >= MIN_FOR_STOCK_TEST and x["p"] >= STOCK_P))
    ]
    out["verdict"] = "frágil" if reasons else ("robusta con excepciones" if out["exceptions"] else "robusta")
    return out


def robustness(
    results: list[dict], pools: dict[str, list[float]], draws: int = 5000, seed: int = 7,
    engine: str = "python", threads: int = 0, threshold: float = 0.005, trim: int = 3,
) -> dict[str, dict]:
    """Pruebas de robustez de cada categoría con al menos MIN_EVENTS eventos (la clave es la categoría)."""
    out = {}
    for cat in sorted({r["category"] for r in results}):
        group = [r for r in results if r["category"] == cat]
        if len(group) >= MIN_EVENTS:
            out[cat] = analyze_category(group, pools, draws, seed, engine, threads, threshold, trim)
    return out


def _p(p: float, draws: int) -> str:
    return f"p < {1 / draws:.5f}" if p <= 1 / (draws + 1) + 1e-12 else f"p = {p:.4f}"


def _line(label: str, t: dict, draws: int) -> str:
    return (f"    {label:<24}{t['n']:>3} eventos   |CAR| {t['mean_abs'] * 100:>5.2f} % frente a "
            f"{t['expected_abs'] * 100:.2f} % del azar (×{t['ratio']:.2f})   {_p(t['p'], draws)}")


def format_robustness(report: dict[str, dict], threshold: float, draws: int) -> str:
    if not report:
        return f"\nRobustez: ninguna categoría tiene al menos {MIN_EVENTS} eventos."
    lines = [f"\nRobustez de la reacción en tamaño (|CAR|); umbral de Bonferroni p < {threshold:.4f}"]
    for cat, r in report.items():
        o = r["overall"]
        lines.append(f"\n  {cat} · {r['n']} eventos · |CAR| medio {o['mean_abs'] * 100:.2f} % frente a "
                     f"{o['expected_abs'] * 100:.2f} % del azar (×{o['ratio']:.2f}) · {_p(o['p'], draws)}")
        if r["verdict"] == "sin efecto detectable":
            lines.append("    No supera el umbral: no hay nada que comprobar.")
            continue
        lines.append(f"    Mediana de |CAR|: {r['median_abs'] * 100:.2f} % frente a {r['expected_median'] * 100:.2f} % del azar")
        lines += [_line(f"solo {x['symbol']}", x, draws) for x in r["by_stock"]]
        lines += [_line(f"sin {x['without']}", x, draws) for x in r["without_stock"]]
        if r["trimmed"]:
            big = ", ".join(f"{s} {c * 100:.1f} %" for s, c in r["trimmed"]["dropped"])
            lines.append(_line(f"sin los {len(r['trimmed']['dropped'])} mayores", r["trimmed"], draws))
            lines.append(f"      (quitados: {big}; prueba conservadora, no se quitan de las fechas al azar)")
        if r["verdict"] == "robusta":
            why = "sigue por debajo del umbral sin cada acción y sin los mayores, y el efecto aparece en todas las acciones."
        elif r["verdict"] == "robusta con excepciones":
            why = ("el efecto agregado aguanta sin cada acción y sin los mayores, pero no aparece en "
                   f"{', '.join(r['exceptions'])}: es un efecto de las demás acciones, no de todas.")
        else:
            why = "el p deja de pasar el umbral en: " + "; ".join(r["reasons"]) + ". No la des por segura."
        lines.append(f"    → {r['verdict'].upper()}: {why}")
    lines.append("\nLectura: con pocos eventos por acción, los p de 'solo X' son poco potentes; mira sobre todo el cociente de cada "
                 "acción (¿hay alguna sin efecto?) y que los p 'sin X' y 'sin los mayores' no se disparen.")
    return "\n".join(lines)
