"""Configuración: claves de API desde variables de entorno o un archivo .env."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> None:
    """Carga líneas CLAVE=valor de `.env` en os.environ.

    No pisa las variables que ya estén definidas en el entorno. El archivo `.env` está en
    .gitignore: las claves nunca deben subirse a git.
    """
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))
