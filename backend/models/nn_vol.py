"""Red neuronal para predecir el rango de mañana (volatilidad), con walk-forward.

Idea: la red no predice el rango a ciegas, sino la CORRECCIÓN sobre la EWMA, que ya es un buen punto de
partida: rango previsto = EWMA de hoy × exp(salida de la red). Si la red no aprende nada, su salida es 0 y
el resultado es la EWMA. Esto además quita las diferencias de escala entre acciones, así que una única
red se entrena con las 4 a la vez (más datos).

Variables de cada día (la red ve una ventana de 22 días): sorpresa del rango frente a lo que esperaba la
EWMA, régimen actual frente a la media de siempre, régimen del mercado (SPY), volumen relativo y las dos
variables de calendario de resultados (hoy hubo / mañana hay).

Entrenamiento: pérdida QLIKE (la misma familia que se usa para evaluar), AdamW, parada anticipada con
una validación temporal (el último 20 % de los días de entrenamiento, con un margen de 22 días para no
solapar ventanas), 3 semillas promediadas, y reajuste cada 21 días solo con datos ya conocidos.

PyTorch solo se importa dentro de las funciones que lo necesitan.
"""

from __future__ import annotations

import math
import time

import polars as pl

from backend.events.table import Event
from backend.models.dataset import Standardizer, build_windows, refit_schedule, split_for_refit
from backend.models.har_e import forecast_har_e_frame
from backend.models.volatility import EWMA_LAMBDA

FEATURES = ["f_surprise", "f_regime", "f_market", "f_relvol", "earn_today", "earn_next"]
FLOOR = 1e-4
MIN_TRAIN = 300  # filas mínimas de entrenamiento para reajustar la red


# ----------------------------------------------------------------------------- datos

def build_panel(
    stocks: dict[str, pl.DataFrame],
    market: pl.DataFrame,
    events: list[Event],
    category: str = "resultados",
    warmup: int = 500,
    refit_every: int = 21,
) -> dict[str, pl.DataFrame]:
    """Una tabla por acción, todas con las mismas sesiones, con las variables de la red (`FEATURES`),
    el objetivo `z` = ln(rango de mañana / EWMA de hoy) y las predicciones de los modelos clásicos."""
    m = (
        market.sort("open_time")
        .select("open_time", mrng=(pl.col("high") / pl.col("low")).log().clip(lower_bound=FLOOR))
        .with_columns(
            m_ewma=pl.col("mrng").ewm_mean(alpha=1 - EWMA_LAMBDA, adjust=False),
            m_exp=pl.col("mrng").cum_sum() / pl.int_range(1, pl.len() + 1),
        )
        .select("open_time", f_market=(pl.col("m_ewma") / pl.col("m_exp")).log())
    )
    frames: dict[str, pl.DataFrame] = {}
    for symbol, stock in stocks.items():
        frame, _ = forecast_har_e_frame(stock, events, symbol, category, warmup, refit_every)
        volume = stock.select("open_time", "volume").unique(subset="open_time")
        frame = (
            frame.join(volume, on="open_time", how="left")
            .join(m, on="open_time", how="inner")
            .sort("open_time")
            .with_columns(
                session=pl.col("open_time").dt.convert_time_zone("America/New_York").dt.date(),
                ewma_c=pl.col("ewma_094").clip(lower_bound=FLOOR),
                rng_c=pl.col("rng").clip(lower_bound=FLOOR),
                vmean=pl.col("volume").rolling_mean(22),
            )
            .with_columns(
                ewma_prev=pl.col("ewma_c").shift(1),
                f_relvol=pl.when((pl.col("volume") > 0) & (pl.col("vmean") > 0))
                .then((pl.col("volume") / pl.col("vmean")).log()),
                f_regime=(pl.col("ewma_c") / pl.col("expanding_mean").clip(lower_bound=FLOOR)).log(),
            )
            .with_columns(
                f_surprise=(pl.col("rng_c") / pl.col("ewma_prev")).log(),
                z=(pl.col("target").clip(lower_bound=FLOOR) / pl.col("ewma_c")).log(),
            )
        )
        frames[symbol] = frame
    common = sorted(set.intersection(*(set(f["session"].to_list()) for f in frames.values())))
    return {s: f.filter(pl.col("session").is_in(common)) for s, f in frames.items()}


def prepare_windows(panel: dict[str, pl.DataFrame], window: int) -> dict[str, dict]:
    """Ventanas de `window` días por acción: x (ventanas), y (objetivo z), day (día final) y ewma de cada día."""
    data = {}
    for symbol, f in panel.items():
        xs, ys, days = build_windows([list(r) for r in f.select(FEATURES).rows()], f["z"].to_list(), window)
        data[symbol] = {"x": xs, "y": ys, "day": days, "ewma": f["ewma_c"].to_list()}
    return data


def time_split(days: list[int], val_fraction: float = 0.2, embargo: int = 22) -> tuple[list[int], list[int]]:
    """Posiciones de entrenamiento y de validación POR ORDEN TEMPORAL.

    Validación = el último `val_fraction` de los días. Se deja un margen (`embargo`, igual a la ventana)
    entre ambos para que ninguna ventana de entrenamiento comparta días con una de validación.
    """
    unique = sorted(set(days))
    start = unique[int(len(unique) * (1 - val_fraction))]
    train = [i for i, d in enumerate(days) if d < start - embargo]
    val = [i for i, d in enumerate(days) if d >= start]
    return train, val


# ---------------------------------------------------------------------------- PyTorch

def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch no está instalado: pip install torch numpy (ver README)") from e
    return torch


def resolve_device(name: str = "auto") -> str:
    torch = _torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Pediste --device cuda pero PyTorch no ve la GPU (prueba python -m backend.training.check_gpu)")
    return name


def make_model(arch: str, n_features: int, window: int, hidden: int = 64, dropout: float = 0.2):
    """'mlp': perceptrón multicapa sobre la ventana aplanada. 'gru': red recurrente que lee la ventana día a día."""
    nn = _torch().nn
    if arch == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(window * n_features, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
    if arch == "gru":
        class GruHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(n_features, hidden, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.out = nn.Linear(hidden, 1)

            def forward(self, x):
                h, _ = self.gru(x)
                return self.out(self.drop(h[:, -1]))

        return GruHead()
    raise ValueError(f"Arquitectura desconocida: {arch} (usa 'mlp' o 'gru')")


def qlike(pred, z):
    """Pérdida QLIKE sobre la corrección logarítmica: con d = z - pred vale exp(2d) - 2d - 1 (0 si acierta).
    Castiga más quedarse corto (d > 0, subestimar el riesgo) que pasarse."""
    torch = _torch()
    d = (z - pred).clamp(-4.0, 4.0)
    return (torch.exp(2 * d) - 2 * d - 1).mean()


def fit_predict(x_tr, y_tr, x_va, y_va, x_pr, arch, hidden, dropout, epochs, patience, lr, weight_decay, batch, seed):
    """Entrena una red con parada anticipada y predice `x_pr`. Devuelve (predicciones, pérdida de validación)."""
    torch = _torch()
    torch.manual_seed(seed)
    model = make_model(arch, x_tr.shape[2], x_tr.shape[1], hidden, dropout).to(x_tr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_tr), device=x_tr.device)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            qlike(model(x_tr[idx]).squeeze(-1), y_tr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = qlike(model(x_va).squeeze(-1), y_va).item()
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_pr).squeeze(-1).cpu().tolist(), best


def walk_forward(
    panel: dict[str, pl.DataFrame],
    window: int = 22,
    warmup: int = 500,
    refit_every: int = 21,
    arch: str = "mlp",
    hidden: int = 64,
    dropout: float = 0.2,
    epochs: int = 60,
    patience: int = 8,
    seeds: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch: int = 256,
    device: str = "auto",
    progress=None,
) -> dict[str, list[float | None]]:
    """Reentrena la red cada `refit_every` días con lo ya conocido y predice el rango de los días siguientes.

    Devuelve, por acción, una lista con el rango previsto para cada día (None antes del `warmup`).
    Cada tramo entrena `seeds` redes con las 4 acciones juntas y promedia sus predicciones.
    """
    torch = _torch()
    dev = resolve_device(device)
    symbols = list(panel)
    n = panel[symbols[0]].height
    data = prepare_windows(panel, window)
    out: dict[str, list[float | None]] = {s: [None] * n for s in symbols}
    schedule = refit_schedule(n, warmup, refit_every)
    started = time.perf_counter()

    for k, (t0, t1) in enumerate(schedule):
        windows, targets, days, to_predict = [], [], [], {}
        for s in symbols:
            d = data[s]
            tr_idx, pr_idx = split_for_refit(d["day"], d["y"], t0, t1)
            windows += [d["x"][i] for i in tr_idx]
            targets += [d["y"][i] for i in tr_idx]
            days += [d["day"][i] for i in tr_idx]
            to_predict[s] = pr_idx
        if len(windows) < MIN_TRAIN:
            continue
        tr, va = time_split(days)
        if not tr or not va:
            continue
        scaler = Standardizer().fit([windows[i] for i in tr])  # el escalado usa SOLO datos de entrenamiento

        def tensor(rows, scaler=scaler):
            return torch.tensor(scaler.transform(rows), dtype=torch.float32, device=dev)

        x_tr, x_va = tensor([windows[i] for i in tr]), tensor([windows[i] for i in va])
        y_tr = torch.tensor([targets[i] for i in tr], dtype=torch.float32, device=dev)
        y_va = torch.tensor([targets[i] for i in va], dtype=torch.float32, device=dev)

        # Una sola matriz con las ventanas a predecir de todas las acciones; se reparte después.
        owners = [(s, i) for s in symbols for i in to_predict[s]]
        if not owners:
            continue
        x_pr = tensor([data[s]["x"][i] for s, i in owners])
        total = [0.0] * len(owners)
        for j in range(seeds):
            zhat, _ = fit_predict(x_tr, y_tr, x_va, y_va, x_pr, arch, hidden, dropout, epochs, patience,
                                  lr, weight_decay, batch, seed=1000 * k + j)
            total = [a + b for a, b in zip(total, zhat)]
        for (s, i), z_sum in zip(owners, total):
            day = data[s]["day"][i]
            out[s][day] = data[s]["ewma"][day] * math.exp(z_sum / seeds)  # rango = EWMA × exp(corrección)
        if progress:
            progress(k + 1, len(schedule), t0, time.perf_counter() - started)
    return out
