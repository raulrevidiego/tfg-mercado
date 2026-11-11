"""Tests de la preparación de datos para redes neuronales y del chequeo de GPU."""

import math
import sys
import types

import pytest

from backend.models.dataset import Standardizer, build_windows, refit_schedule, split_for_refit
from backend.training import check_gpu


def test_calendario_sin_solapes_ni_huecos():
    tramos = refit_schedule(n=100, start=40, refit_every=21)
    assert tramos == [(40, 61), (61, 82), (82, 100)]
    dias = [d for t0, t1 in tramos for d in range(t0, t1)]
    assert dias == list(range(40, 100))                      # cada día se predice exactamente una vez


def test_ventanas_terminan_en_t_y_el_objetivo_es_el_dia_siguiente():
    feats = [[float(t), float(10 * t)] for t in range(8)]
    target = [float(t + 1) for t in range(7)] + [None]      # el objetivo de t es "el valor de t+1"
    xs, ys, days = build_windows(feats, target, window=3)
    assert days == [2, 3, 4, 5, 6, 7]
    assert xs[0] == [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]  # los 3 días que terminan en t=2
    assert ys[:2] == [3.0, 4.0] and ys[-1] is None


def test_las_ventanas_con_datos_ausentes_se_descartan():
    feats = [[1.0], [2.0], [None], [4.0], [5.0], [6.0]]
    xs, _, days = build_windows(feats, [0.0] * 6, window=2)
    assert days == [1, 4, 5]                                 # las dos ventanas que tocan el hueco desaparecen


def test_el_reajuste_solo_entrena_con_lo_que_ya_se_conocia():
    feats = [[float(t)] for t in range(50)]
    target = [float(t + 1) for t in range(49)] + [None]
    _, ys, days = build_windows(feats, target, window=5)
    t0, t1 = 30, 40
    train, predict = split_for_refit(days, ys, t0, t1)
    assert max(days[i] for i in train) == t0 - 1             # nada de t0 en adelante
    assert [days[i] for i in predict] == list(range(t0, t1))
    # El objetivo del último día de entrenamiento (t0-1) es el valor de t0: ya conocido en t0.
    assert all(ys[i] <= t0 for i in train)


def test_estandarizador_usa_solo_datos_de_entrenamiento():
    train = [[[1.0, 100.0], [3.0, 300.0]], [[5.0, 500.0], [7.0, 700.0]]]
    test = [[[1000.0, -50.0], [2000.0, -60.0]]]
    sc = Standardizer().fit(train)
    assert sc.mean == pytest.approx([4.0, 400.0])
    assert sc.std == pytest.approx([math.sqrt(5), math.sqrt(50000)])
    out = sc.transform(train)
    flat = [row[0] for w in out for row in w]
    assert sum(flat) / len(flat) == pytest.approx(0.0, abs=1e-12)
    # Cambiar los datos de test no altera los parámetros ya ajustados.
    sc.transform(test)
    assert sc.mean == pytest.approx([4.0, 400.0])
    assert sc.transform(test)[0][0][0] == pytest.approx((1000.0 - 4.0) / math.sqrt(5))


def test_estandarizador_con_variable_constante_no_divide_por_cero():
    sc = Standardizer().fit([[[2.0], [2.0]], [[2.0], [2.0]]])
    assert sc.transform([[[2.0]]])[0][0][0] == pytest.approx(0.0)


def test_check_gpu_sin_pytorch(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)          # "import torch" falla
    assert check_gpu.main() == 1
    assert "PyTorch no está instalado" in capsys.readouterr().out


def test_check_gpu_con_pytorch_sin_gpu(monkeypatch, capsys):
    falso = types.SimpleNamespace(
        __version__="2.9.0+cpu", version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", falso)
    assert check_gpu.main() == 2
    salida = capsys.readouterr().out
    assert "NO ve la GPU" in salida and "nvidia-smi" in salida


# ----------------------------------------------------- check_gpu con distintas GPU simuladas

class _Tensor:
    def __matmul__(self, otro):
        return self


def _torch_falso(gpu_name="GeForce RTX 4080", cc=(8, 9), arch=("sm_75", "sm_80", "sm_86", "sm_90"), falla=False):
    def randn(*a, device=None, **k):
        if falla and device is not None and device.type == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return _Tensor()

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: types.SimpleNamespace(name=gpu_name, total_memory=12 * 1024 ** 3, major=cc[0], minor=cc[1]),
        get_arch_list=lambda: list(arch), synchronize=lambda: None,
    )
    return types.SimpleNamespace(
        __version__="2.15.0+cu132", version=types.SimpleNamespace(cuda="13.2"),
        cuda=cuda, device=lambda nombre: types.SimpleNamespace(type=nombre), randn=randn,
    )


def test_check_gpu_con_una_gpu_compatible(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _torch_falso())
    assert check_gpu.main() == 0
    assert "RTX 4080" in capsys.readouterr().out


def test_check_gpu_con_una_gpu_demasiado_antigua_explica_en_vez_de_fallar(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _torch_falso("GeForce MX250", (6, 1)))
    assert check_gpu.main() == 3
    out = capsys.readouterr().out
    assert "MX250" in out and "6.1" in out and "7.5" in out and "demasiado antigua" in out


def test_check_gpu_si_cuda_falla_al_calcular_lo_cuenta(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _torch_falso(falla=True))
    assert check_gpu.main() == 4
    assert "no kernel image" in capsys.readouterr().out


def test_check_gpu_avisa_si_falta_numpy(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _torch_falso())
    monkeypatch.setitem(sys.modules, "numpy", None)      # None en sys.modules hace que 'import numpy' falle
    assert check_gpu.main() == 0
    assert "pip install numpy" in capsys.readouterr().out
