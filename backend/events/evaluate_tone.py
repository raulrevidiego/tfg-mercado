"""¿Predice el tono de un comunicado de resultados el signo o el tamaño de la reacción?

Une tres cosas: la reacción de la acción (estudio de eventos), el texto de cada comunicado (descargado
con `python -m backend.data.edgar_text`) y su tono relativo. Contrastes, todos con p-valor por
permutaciones (se baraja el tono entre eventos):

1. Correlación de Spearman entre tono relativo y CAR con signo (¿el tono bueno acompaña a subidas?).
2. Correlación entre |tono relativo| y |CAR| (¿un tono inusual acompaña a reacciones mayores?).
3. Diferencia de CAR medio entre el tercio de tono más alto y el más bajo.
4. Acierto de signo: % de eventos en que el signo del tono relativo coincide con el del CAR.

Uso desde la raíz del repo:

    python -m backend.events.evaluate_tone
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import timezone
from pathlib import Path

from backend.data.store import load_series
from backend.events.study import analyze_event, prepare
from backend.events.table import first_affected_session, load_events
from backend.events.tone import relative_tone, tone_score
from backend.training.metrics import proportion_margin

MARKET = "SPY"


def _ranks(values: list[float]) -> list[float]:
    """Posición de cada valor al ordenar (los empates reparten la media de sus posiciones)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    """Correlación entre las posiciones: -1 a +1; no le afectan los valores extremos."""
    return pearson(_ranks(x), _ranks(y))


def permutation_p(stat, x: list[float], y: list[float], draws: int = 5000, seed: int = 7) -> float:
    """p-valor de dos colas: ¿con qué frecuencia da `stat` algo igual de extremo con `y` barajado?"""
    observed = abs(stat(x, y))
    rnd = random.Random(seed)
    shuffled = y[:]
    extreme = 0
    for _ in range(draws):
        rnd.shuffle(shuffled)
        extreme += abs(stat(x, shuffled)) >= observed
    return (extreme + 1) / (draws + 1)


def tercile_gap(x: list[float], y: list[float]) -> float:
    """Media de y en el tercio de x más alto menos la del tercio de x más bajo."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    k = max(len(x) // 3, 1)
    low, high = order[:k], order[-k:]
    return sum(y[i] for i in high) / k - sum(y[i] for i in low) / k


def load_records(args: argparse.Namespace) -> list[dict]:
    """Un registro por evento con reacción (car1) y tono (tone, tone_rel)."""
    events_path = args.events or args.data_dir / "events" / "edgar_8k.csv"
    index_path = args.data_dir / "events" / "texts_index.csv"
    for path, how in ((events_path, "python -m backend.data.edgar"), (index_path, "python -m backend.data.edgar_text")):
        if not path.exists():
            raise SystemExit(f"\nNo existe {path}. Genéralo con: {how}")
    with index_path.open(encoding="utf-8") as fh:
        texts = {(r["symbol"], r["datetime"]): Path(r["path"]) for r in csv.DictReader(fh)}

    market = load_series(MARKET, data_dir=args.data_dir)
    cache: dict[str, dict] = {}
    by_symbol: dict[str, list[dict]] = {}
    for ev in sorted(load_events(events_path), key=lambda e: e.when):
        key = (ev.symbol, ev.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        if ev.category != args.category or key not in texts or not texts[key].exists():
            continue
        if ev.symbol not in cache:
            cache[ev.symbol] = prepare(load_series(ev.symbol, data_dir=args.data_dir), market)
        data = cache[ev.symbol]
        session = first_affected_session(ev.when, data["session"])
        res = analyze_event(data, data["session"].index(session)) if session else None
        if res is None:
            continue
        body = texts[key].read_text(encoding="utf-8").partition("\n")[2]
        by_symbol.setdefault(ev.symbol, []).append({
            "symbol": ev.symbol, "session": session, "car1": res["car1"], **tone_score(body),
        })
    records = []
    for recs in by_symbol.values():
        for rec, rel in zip(recs, relative_tone([r["score"] for r in recs])):
            rec["tone_rel"] = rel
            records.append(rec)
    return [r for r in records if r["tone_rel"] is not None]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="¿Predice el tono de un comunicado la reacción del precio?")
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--category", default="resultados")
    p.add_argument("--draws", type=int, default=5000)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    recs = load_records(args)
    if len(recs) < 20:
        raise SystemExit(f"\nSolo {len(recs)} eventos con texto y tono: hacen falta al menos 20.")
    tone = [r["tone_rel"] for r in recs]
    car = [r["car1"] * 100 for r in recs]
    hits = sum((t > 0) == (c > 0) for t, c in zip(tone, car) if t != 0 and c != 0)
    calls = sum(1 for t, c in zip(tone, car) if t != 0 and c != 0)

    print(f"\n{len(recs)} comunicados de '{args.category}' con reacción y tono relativo "
          f"({', '.join(f'{s} {sum(r['symbol'] == s for r in recs)}' for s in sorted({r['symbol'] for r in recs}))})")
    print(f"Tono relativo: de {min(tone):+.2f} a {max(tone):+.2f}; CAR[0,+1] de {min(car):+.1f} % a {max(car):+.1f} %\n")
    rho = spearman(tone, car)
    rho_abs = spearman([abs(t) for t in tone], [abs(c) for c in car])
    gap = tercile_gap(tone, car)
    print(f"{'contraste':<62}{'resultado':>12}{'p':>9}")
    print(f"{'1. correlación tono relativo y CAR con signo (Spearman)':<62}{rho:>+12.3f}{permutation_p(spearman, tone, car, args.draws):>9.4f}")
    print(f"{'2. correlación |tono relativo| y |CAR| (Spearman)':<62}{rho_abs:>+12.3f}"
          f"{permutation_p(spearman, [abs(t) for t in tone], [abs(c) for c in car], args.draws):>9.4f}")
    print(f"{'3. CAR medio: tercio de tono más alto menos el más bajo (pp)':<62}{gap:>+12.2f}{permutation_p(tercile_gap, tone, car, args.draws):>9.4f}")
    acc = hits / calls
    print(f"{'4. signo del tono = signo del CAR (% de aciertos ± margen 95 %)':<62}{acc * 100:>8.1f} % ±{proportion_margin(acc, calls) * 100:.1f}")
    print(
        "\nLectura: con 4 contrastes, un p resiste la corrección de Bonferroni solo si es menor de 0,0125."
        f"\nCon {len(recs)} eventos, una correlación necesita rondar ±{1.96 / math.sqrt(len(recs) - 1):.2f} para ser distinguible del azar."
        "\nEste es el baseline de tono (lexicón propio); FinBERT tendrá que mejorarlo."
    )


if __name__ == "__main__":
    main()
