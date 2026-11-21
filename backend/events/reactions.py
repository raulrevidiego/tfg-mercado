"""Tabla de reacciones: qué hizo la acción tras cada evento, y qué hace en fechas normales.

`python -m backend.events.build_reactions` analiza todos los eventos (los mismos pasos que
`evaluate_events`: sesión de reacción, modelo de mercado, eventos independientes) y guarda una fila por
evento en `data/events/reactions.csv`, más filas "azar" (fechas sin evento) como referencia de un día
normal. La pantalla "¿y si…?" de la interfaz consulta esa tabla.

`what_if` responde, con la historia, a: "si pasa un evento de este tipo, ¿cuánto suele moverse la
acción, hacia dónde y con qué incertidumbre?". Describe lo que ha pasado en el pasado; no promete lo
que pasará.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import timezone
from pathlib import Path

from backend.data.store import load_series
from backend.events.study import analyze_event, placebo_pool, prepare, select_independent_events
from backend.events.table import first_affected_session, load_events
from backend.events.tone import relative_tone, tone_score

MARKET = "SPY"
PRIORITY = ["resultados", "directivos", "regfd", "otros_eventos", "otros"]
COLUMNS = ["kind", "symbol", "session", "category", "car0", "car1", "car4", "abn_range", "z1", "tone_rel"]
TONE_THRESHOLD = 0.05   # tono relativo por encima/debajo de ±0,05: positivo/negativo; en medio, neutro


def tone_group(tone_rel: float | None) -> str:
    if tone_rel is None:
        return "sin_dato"
    return "positivo" if tone_rel > TONE_THRESHOLD else "negativo" if tone_rel < -TONE_THRESHOLD else "neutro"


def _tone_map(texts_index: Path) -> dict[tuple[str, str], float]:
    """(símbolo, fecha ISO) -> tono relativo, comparando cada comunicado con los anteriores de su empresa."""
    if not texts_index.exists():
        return {}
    with texts_index.open(encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["datetime"])
    by_symbol: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        path = Path(r["path"])
        if path.exists():
            score = tone_score(path.read_text(encoding="utf-8").partition("\n")[2])["score"]
            by_symbol.setdefault(r["symbol"], []).append((r["datetime"], score))
    out = {}
    for symbol, items in by_symbol.items():
        for (iso, _), rel in zip(items, relative_tone([s for _, s in items])):
            if rel is not None:
                out[(symbol, iso)] = rel
    return out


def build_reactions(
    data_dir: Path, events_path: Path, min_gap: int = 5, placebo_n: int = 300, seed: int = 7
) -> list[dict]:
    market = load_series(MARKET, data_dir=data_dir)
    tones = _tone_map(data_dir / "events" / "texts_index.csv")
    cache: dict[str, dict] = {}
    analyzed = []
    for ev in load_events(events_path):
        if ev.symbol not in cache:
            cache[ev.symbol] = prepare(load_series(ev.symbol, data_dir=data_dir), market)
        data = cache[ev.symbol]
        session = first_affected_session(ev.when, data["session"])
        res = analyze_event(data, data["session"].index(session)) if session else None
        if res is None:
            continue
        iso = ev.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        analyzed.append({**res, "symbol": ev.symbol, "category": ev.category, "session": session,
                         "idx": data["session"].index(session), "tone_rel": tones.get((ev.symbol, iso))})
    kept, _, _ = select_independent_events(analyzed, min_gap, PRIORITY)

    rows = [{"kind": "evento", **{k: r.get(k) for k in COLUMNS if k != "kind"}} for r in kept]
    for symbol, data in cache.items():
        mine = [r["idx"] for r in analyzed if r["symbol"] == symbol]
        for p in placebo_pool(data, placebo_n, mine, seed):
            rows.append({"kind": "azar", "symbol": symbol, "session": data["session"][p["idx"]],
                         "category": "azar", "car0": p["car0"], "car1": p["car1"], "car4": p["car4"],
                         "abn_range": p["abn_range"], "z1": p["z1"], "tone_rel": None})
    return rows


def save_reactions(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in COLUMNS})


def load_reactions(path: Path) -> list[dict]:
    numeric = ("car0", "car1", "car4", "abn_range", "z1", "tone_rel")
    with path.open(encoding="utf-8") as fh:
        return [{**r, **{k: (float(r[k]) if r[k] != "" else None) for k in numeric}} for r in csv.DictReader(fh)]


def quantile(values: list[float], q: float) -> float:
    """Cuantil q (0-1) con interpolación lineal."""
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def what_if(rows: list[dict], symbol: str, category: str, tone: str = "any", scope: str = "symbol") -> dict:
    """Qué ha pasado históricamente tras eventos de esta categoría (y tono), frente a un día normal."""
    symbols = None if scope == "all" else {symbol}
    pick = lambda r: symbols is None or r["symbol"] in symbols  # noqa: E731
    events = [r for r in rows if r["kind"] == "evento" and r["category"] == category and pick(r)]
    if tone != "any":
        events = [r for r in events if tone_group(r["tone_rel"]) == tone]
    normal = [r for r in rows if r["kind"] == "azar" and pick(r)]
    out = {"symbol": symbol if scope == "symbol" else "(todas)", "category": category, "tone": tone,
           "scope": scope, "n": len(events), "warnings": []}
    if not events:
        out["warnings"].append("No hay eventos de este tipo en la historia: no se puede estimar nada.")
        return out

    cars = [r["car1"] for r in events]
    absc = [abs(c) for c in cars]
    normal_abs = [abs(r["car1"]) for r in normal]
    mean = sum(cars) / len(cars)
    sd = math.sqrt(sum((c - mean) ** 2 for c in cars) / (len(cars) - 1)) if len(cars) > 1 else None
    margin = 1.96 * sd / math.sqrt(len(cars)) if sd is not None else None
    mult = [r["abn_range"] for r in events]
    out["reaction"] = {
        "median_abs": quantile(absc, 0.5), "p75_abs": quantile(absc, 0.75), "p90_abs": quantile(absc, 0.9),
        "normal_median_abs": quantile(normal_abs, 0.5) if normal_abs else None,
    }
    out["direction"] = {"share_up": sum(c > 0 for c in cars) / len(cars), "mean": mean, "margin": margin,
                        "distinguishable": margin is not None and abs(mean) > margin}
    out["range"] = {"median_multiplier": quantile(mult, 0.5), "p90_multiplier": quantile(mult, 0.9)}
    if len(events) < 15:
        out["warnings"].append(f"Solo {len(events)} eventos: los números son orientativos y la incertidumbre es grande.")
    if not out["direction"]["distinguishable"]:
        out["warnings"].append("La dirección media no se distingue de cero: la historia no dice si sube o baja, solo cuánto se mueve.")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Construye data/events/reactions.csv para la pantalla '¿y si…?'.")
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--min-gap", type=int, default=5)
    p.add_argument("--placebo-n", type=int, default=300)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    events = args.events or args.data_dir / "events" / "edgar_8k.csv"
    if not events.exists():
        raise SystemExit(f"\nNo existe {events}. Genéralo con: python -m backend.data.edgar")
    rows = build_reactions(args.data_dir, events, args.min_gap, args.placebo_n)
    out = args.data_dir / "events" / "reactions.csv"
    save_reactions(rows, out)
    ev = [r for r in rows if r["kind"] == "evento"]
    toned = sum(r["tone_rel"] is not None for r in ev)
    print(f"\nGuardado {out}: {len(ev)} eventos independientes ({toned} con tono) y "
          f"{len(rows) - len(ev)} fechas al azar de referencia.")
    for cat in PRIORITY:
        n = sum(r["category"] == cat for r in ev)
        if n:
            print(f"  {cat:<14}{n:>4}")


if __name__ == "__main__":
    main()
