"""Las entregas (docs/entregas) no deben quedarse obsoletas: los comandos de reproducir.md tienen que apuntar a
módulos y opciones que existen, y los documentos no deben llevar marcas de borrador sin terminar."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ENTREGAS = Path("docs/entregas")
ARCHIVOS = ["README.md", "resultados.md", "reproducir.md"]


def _comandos() -> list[tuple[str, list[str]]]:
    """(módulo, opciones) de cada línea o celda con `python -m modulo --opcion ...` de reproducir.md."""
    texto = (ENTREGAS / "reproducir.md").read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"python -m ([\w.]+)([^\n`|]*)", texto):
        opciones = re.findall(r"(--[a-z][\w-]*)", m.group(2).split("#")[0])
        out.append((m.group(1), opciones))
    return out


def test_existen_los_archivos_de_entrega():
    for nombre in ARCHIVOS:
        assert (ENTREGAS / nombre).read_text(encoding="utf-8").strip(), nombre


def test_los_comandos_de_reproducir_apuntan_a_modulos_que_existen():
    modulos = sorted({m for m, _ in _comandos()})
    assert len(modulos) >= 14  # datos, baselines, volatilidad, HAR-X/E, red, eventos, tono, apoyo
    assert [m for m in modulos if importlib.util.find_spec(m) is None] == []


def test_las_opciones_de_reproducir_existen_en_cada_modulo():
    for modulo, opciones in _comandos():
        if not modulo.startswith("backend."):  # p. ej. `python -m venv`: es de la librería estándar
            continue
        fuente = Path(*modulo.split(".")).with_suffix(".py").read_text(encoding="utf-8")
        for opcion in opciones:
            assert f'"{opcion}"' in fuente, f"{modulo} no define {opcion}"


def test_los_resultados_cubren_las_cuatro_variantes_de_tono_y_no_dejan_marcas_de_borrador():
    texto = (ENTREGAS / "reproducir.md").read_text(encoding="utf-8")
    variantes = set(re.findall(r"--metodo (\w+) --parte (\w+) --variantes 4", texto))
    assert variantes == {("lexicon", "all"), ("lexicon", "quotes"), ("finbert", "all"), ("finbert", "quotes")}
    for nombre in ARCHIVOS:
        contenido = (ENTREGAS / nombre).read_text(encoding="utf-8")
        for marca in ("TODO", "FIXME", "XXX", "pendiente de ejecutar"):
            assert marca not in contenido, f"{nombre} contiene {marca!r}"
