"""Compila el núcleo C++ y deja `backend/_core*.so` junto al código Python.

Desde la raíz del repo (necesita un compilador de C++: g++ o clang++):

    pip install -e ".[core]"      # instala pybind11 y setuptools
    python core/build.py

Si no lo compilas, todo sigue funcionando: el código usa la versión en Python (más lenta).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
        from setuptools import setup
    except ImportError:
        raise SystemExit('Falta pybind11/setuptools: pip install -e ".[core]"')

    windows = sys.platform == "win32"
    ext = Pybind11Extension(
        "backend._core",
        ["core/permutation.cpp"],
        cxx_std=17,
        extra_compile_args=[] if windows else ["-O3"],
        extra_link_args=[] if windows else ["-pthread"],
    )
    os.chdir(ROOT)
    setup(
        name="tfg-core",
        packages=[],  # solo compilamos la extensión; evita que setuptools "descubra" carpetas
        ext_modules=[ext],
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace", "--build-temp", "build/core"],
    )
    print("\nListo: núcleo C++ compilado en backend/")


if __name__ == "__main__":
    main()
