"""Vigila que la API no tenga rutas repetidas (un bloque pegado varias veces registraba /api/whatif 3 veces)."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from backend.api.main import app


def test_ninguna_ruta_esta_repetida():
    pares = Counter((r.path, tuple(sorted(r.methods or ()))) for r in app.routes if hasattr(r, "methods"))
    assert [par for par, n in pares.items() if n > 1] == []


def test_el_modulo_de_la_api_no_repite_definiciones_ni_imports():
    arbol = ast.parse(Path("backend/api/main.py").read_text(encoding="utf-8"))
    nombres = Counter(n.name for n in arbol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    imports = Counter(ast.unparse(n) for n in arbol.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert [k for k, v in nombres.items() if v > 1] == []
    assert [k for k, v in imports.items() if v > 1] == []
