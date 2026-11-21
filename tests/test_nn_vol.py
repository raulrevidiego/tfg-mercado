"""Tests de la red neuronal de volatilidad.

Los de datos, ventanas y calendario no usan PyTorch. Los demás se saltan solos si PyTorch no está disponible.
"""

import math
import re

import polars as pl
import pytest

from backend.models import nn_vol
from backend.training import evaluate_nn
from test_har_e import _bars, _escenario, _evento  # noqa: E402


def _torch_or_skip():
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - cualquier fallo al importar significa "no utilizable aquí"
        pytest.skip(f"PyTorch no utilizable: {e}")
    return torch


def _datos(n=900, seed_b=9):
    a, eventos_a, sesiones, _ = _escenario(n=n, seed=4)
    b, _, _, reaccion_b = _escenario(n=n, seed=seed_b)
    eventos = eventos_a + [_evento("BBB", sesiones[k]) for k in reaccion_b]
    mercado = _bars(sesiones, [0.01 + 0.004 * math.sin(i / 20) for i in range(n)])
    return a, b, mercado, eventos, sesiones


# --------------------------------------------------------------------- datos (sin PyTorch)

def test_el_panel_tiene_las_variables_y_un_calendario_comun():
    a, b, mercado, eventos, _ = _datos()
    quitado = b["open_time"][300]
    b = b.filter(pl.col("open_time") != quitado)                       # a BBB le falta una sesión
    panel = nn_vol.build_panel({"AAA": a, "BBB": b}, mercado, eventos, warmup=300)
    assert set(panel) == {"AAA", "BBB"}
    assert panel["AAA"].height == panel["BBB"].height == a.height - 1  # solo las sesiones comunes
    assert panel["AAA"]["open_time"].to_list() == panel["BBB"]["open_time"].to_list()
    assert quitado not in panel["AAA"]["open_time"].to_list()
    for col in (*nn_vol.FEATURES, "z", "har", "har_enext", "ewma_c", "target"):
        assert col in panel["AAA"].columns


def test_las_variables_se_calculan_como_dicen_las_formulas():
    a, b, mercado, eventos, _ = _datos()
    f = nn_vol.build_panel({"AAA": a, "BBB": b}, mercado, eventos, warmup=300)["AAA"]
    r = f.row(100, named=True)
    assert r["z"] == pytest.approx(math.log(r["target"] / r["ewma_c"]))
    assert r["f_surprise"] == pytest.approx(math.log(r["rng_c"] / r["ewma_prev"]))
    assert r["f_relvol"] == pytest.approx(0.0)                         # volumen constante: igual a su media
    assert r["f_regime"] == pytest.approx(math.log(r["ewma_c"] / r["expanding_mean"]))
    assert math.isfinite(r["f_market"]) and r["earn_today"] in (0.0, 1.0) and r["earn_next"] in (0.0, 1.0)
    assert f["z"][-1] is None                                          # el último día no tiene "mañana"


def test_las_ventanas_cuadran_con_el_panel():
    a, b, mercado, eventos, _ = _datos()
    panel = nn_vol.build_panel({"AAA": a, "BBB": b}, mercado, eventos, warmup=300)
    data = nn_vol.prepare_windows(panel, window=10)["AAA"]
    assert len(data["x"][0]) == 10 and len(data["x"][0][0]) == len(nn_vol.FEATURES)
    assert all(v is not None for w in data["x"] for row in w for v in row)
    assert data["y"][0] == panel["AAA"]["z"][data["day"][0]]            # el objetivo es el de su último día
    assert len(data["ewma"]) == panel["AAA"].height


def test_la_validacion_es_temporal_y_deja_margen():
    train, val = nn_vol.time_split(list(range(100)), val_fraction=0.2, embargo=22)
    assert val == list(range(80, 100)) and train == list(range(0, 58))   # 22 días de margen: 58..79
    assert max(train) + 22 < min(val)


# ------------------------------------------------------------------- PyTorch (se saltan si falta)

def test_modelos_y_perdida():
    torch = _torch_or_skip()
    x = torch.randn(4, 10, len(nn_vol.FEATURES))
    for arch in ("mlp", "gru"):
        assert nn_vol.make_model(arch, len(nn_vol.FEATURES), 10, hidden=16)(x).shape == (4, 1)
    with pytest.raises(ValueError):
        nn_vol.make_model("transformer", 6, 10)
    z = torch.tensor([0.5])
    assert nn_vol.qlike(torch.tensor([0.5]), z).item() == pytest.approx(0.0, abs=1e-6)
    corto = nn_vol.qlike(torch.tensor([0.0]), z).item()                  # prever menos de lo real
    largo = nn_vol.qlike(torch.tensor([1.0]), z).item()                  # prever más, la misma distancia
    assert corto > largo > 0


def test_dispositivo(monkeypatch):
    torch = _torch_or_skip()
    assert nn_vol.resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nn_vol.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="check_gpu"):
        nn_vol.resolve_device("cuda")


def test_la_red_aprende_una_relacion_sencilla():
    torch = _torch_or_skip()
    torch.manual_seed(0)
    x = torch.randn(500, 5, 3)
    y = 0.8 * x[:, -1, 0]
    pred, _ = nn_vol.fit_predict(x[:400], y[:400], x[400:450], y[400:450], x[450:], "mlp", 16, 0.0, 150, 30, 5e-3, 0.0, 64, seed=1)
    corr = torch.corrcoef(torch.stack([torch.tensor(pred), y[450:]]))[0, 1].item()
    assert corr > 0.8


def test_walk_forward_pequeno_es_reproducible_y_empieza_tras_el_warmup():
    _torch_or_skip()
    a, b, mercado, eventos, _ = _datos()
    panel = nn_vol.build_panel({"AAA": a, "BBB": b}, mercado, eventos, warmup=300)
    kw = dict(window=10, warmup=300, refit_every=63, epochs=3, seeds=1, hidden=16, device="cpu")
    p1, p2 = nn_vol.walk_forward(panel, **kw), nn_vol.walk_forward(panel, **kw)
    assert p1 == p2
    for s, preds in p1.items():
        assert len(preds) == panel[s].height and all(v is None for v in preds[:300])
        asignadas = [v for v in preds[300:] if v is not None]
        assert len(asignadas) > 400 and all(math.isfinite(v) and v > 0 for v in asignadas)


def test_sin_fuga_del_futuro_cambiar_precios_posteriores_no_cambia_el_pasado():
    _torch_or_skip()
    a, b, mercado, eventos, _ = _datos()
    corte = a["open_time"][700]
    a2 = a.with_columns(high=pl.when(pl.col("open_time") >= corte).then(pl.col("high") * 1.5).otherwise(pl.col("high")))
    kw = dict(window=10, warmup=300, refit_every=63, epochs=3, seeds=1, hidden=16, device="cpu")
    antes = nn_vol.walk_forward(nn_vol.build_panel({"AAA": a, "BBB": b}, mercado, eventos, warmup=300), **kw)
    despues = nn_vol.walk_forward(nn_vol.build_panel({"AAA": a2, "BBB": b}, mercado, eventos, warmup=300), **kw)
    for s in antes:
        for dia in range(300, 600):                                      # tramos cuyo entrenamiento acaba antes del día 700
            x, y = antes[s][dia], despues[s][dia]
            assert (x is None and y is None) or x == pytest.approx(y, rel=1e-6), f"{s} cambia en el día {dia}"


def test_cli_de_extremo_a_extremo(tmp_path, capsys):
    _torch_or_skip()
    a, b, mercado, eventos, _ = _datos()
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    for nombre, df in {"AAA": a, "BBB": b, "SPY": mercado}.items():
        df.write_parquet(out / f"{nombre}_1d.parquet")
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "edgar_8k.csv").write_text(
        "symbol,datetime,category,source,note\n" + "\n".join(f"{e.symbol},{e.when.isoformat()},{e.category},," for e in eventos) + "\n")
    evaluate_nn.main(["--symbols", "AAA", "BBB", "--epochs", "2", "--seeds", "1", "--hidden", "16", "--window", "10",
                      "--warmup", "300", "--refit-every", "63", "--device", "cpu", "--data-dir", str(tmp_path)])
    salida = capsys.readouterr().out
    assert "Red mlp · 1 semillas" in salida and "AAA · todos los días" in salida
    assert re.search(r"\bnn\b", salida) and "CONJUNTO · víspera de resultados" in salida
    assert (tmp_path / "models" / "nn_predictions.parquet").exists()


def test_cli_sin_eventos_o_sin_pytorch_da_un_mensaje_claro(tmp_path):
    with pytest.raises(SystemExit, match="python -m backend.data.edgar"):
        evaluate_nn.main(["--data-dir", str(tmp_path)])
