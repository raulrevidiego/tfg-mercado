"""Preparación de datos para redes neuronales: ventanas, escalado y calendario de reajustes.

Todo esto es Python puro, sin PyTorch, para poder comprobar con tests que NO se filtra
información del futuro al entrenar. Convenciones (las mismas que HAR):

- La ventana del día t contiene las variables de los `window` días que terminan en t.
- El objetivo del día t es el rango del día t+1.
- Al reajustar el día t0 solo se entrena con días t <= t0-1: su objetivo (el rango de t+1 <= t0)
  ya se conocía en t0.
"""

from __future__ import annotations

import math


def refit_schedule(n: int, start: int, refit_every: int) -> list[tuple[int, int]]:
    """Calendario walk-forward: [(t0, t1), ...].

    En cada tramo se entrena con lo conocido hasta t0 y se predicen los días t0 .. t1-1.
    Los tramos no se solapan, no dejan huecos y el último termina en n.
    """
    return [(t0, min(t0 + refit_every, n)) for t0 in range(start, n, refit_every)]


def build_windows(
    features: list[list[float | None]], target: list[float | None], window: int
) -> tuple[list[list[list[float]]], list[float | None], list[int]]:
    """Convierte la tabla diaria de variables en ventanas de `window` días.

    Devuelve (ventanas, objetivos, día de cada ventana). Las ventanas con algún dato ausente
    se descartan. El objetivo puede ser None (el último día no tiene "mañana").
    """
    xs: list[list[list[float]]] = []
    ys: list[float | None] = []
    days: list[int] = []
    for t in range(window - 1, len(features)):
        rows = features[t - window + 1 : t + 1]
        if any(v is None for row in rows for v in row):
            continue
        xs.append([list(row) for row in rows])
        ys.append(target[t])
        days.append(t)
    return xs, ys, days


def split_for_refit(
    days: list[int], targets: list[float | None], t0: int, t1: int
) -> tuple[list[int], list[int]]:
    """Posiciones (en las listas de `build_windows`) para entrenar y para predecir en el tramo."""
    train = [i for i, (t, y) in enumerate(zip(days, targets)) if t <= t0 - 1 and y is not None]
    predict = [i for i, t in enumerate(days) if t0 <= t < t1]
    return train, predict


class Standardizer:
    """Escala cada variable a media 0 y desviación 1.

    La media y la desviación se calculan SOLO con los datos de entrenamiento (`fit`); después
    se aplican igual a los datos nuevos (`transform`). Calcularlas con todos los datos filtraría
    información del futuro.
    """

    def __init__(self) -> None:
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit(self, windows: list[list[list[float]]]) -> "Standardizer":
        n_features = len(windows[0][0])
        values = [[row[j] for w in windows for row in w] for j in range(n_features)]
        self.mean = [sum(v) / len(v) for v in values]
        self.std = [
            max(math.sqrt(sum((x - m) ** 2 for x in v) / len(v)), 1e-12)
            for v, m in zip(values, self.mean)
        ]
        return self

    def transform(self, windows: list[list[list[float]]]) -> list[list[list[float]]]:
        return [
            [[(x - m) / s for x, m, s in zip(row, self.mean, self.std)] for row in w]
            for w in windows
        ]
