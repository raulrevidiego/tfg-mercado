"""Comprueba si PyTorch está instalado y puede usar tu GPU.

Uso desde la raíz del repo:

    python -m backend.training.check_gpu

Códigos de salida: 0 = todo bien, 1 = PyTorch no está instalado, 2 = PyTorch no ve la GPU,
3 = la GPU es demasiado antigua para esta versión de PyTorch, 4 = la GPU falló al calcular.
"""

from __future__ import annotations

import sys
import time
import warnings

HELP_INSTALL = """PyTorch no está instalado en este entorno.

Lo más simple (la rueda de PyPI ya incluye CUDA):

  pip install torch

Después vuelve a ejecutar este comando. Si `nvidia-smi` muestra una "CUDA Version" 12.x (driver más
antiguo), instala la rueda que corresponda a tu driver:

  CUDA 12.8:  pip install torch --index-url https://download.pytorch.org/whl/cu128
  CUDA 12.6:  pip install torch --index-url https://download.pytorch.org/whl/cu126

Si pip dice "No matching distribution found", puede que aún no haya ruedas para tu versión de Python:
crea un entorno con Python 3.13 o 3.12 (ver README)."""

HELP_NO_CUDA = """PyTorch está instalado pero NO ve la GPU. Causas habituales:
  - Se instaló la versión solo-CPU (`pip list | grep torch` no muestra +cu...).
  - El driver de NVIDIA no está cargado: ejecuta `nvidia-smi`; si falla, hay que arreglar el driver.
  - El driver es más antiguo que el CUDA de la rueda: instala la rueda cu128 o cu126 (ver arriba)."""

HELP_NO_NUMPY = """NumPy no está instalado. PyTorch funciona sin él, pero lo necesita para convertir tensores
y arrays: instálalo con `pip install numpy`."""

HELP_UNSUPPORTED = """Tu GPU ({name}, capacidad {cc}) es demasiado antigua para esta versión de PyTorch, que
solo incluye kernels desde la capacidad {minimum} (serie RTX 20 en adelante). No es un fallo de tu
instalación: esa GPU no se puede usar para entrenar con PyTorch actual. En este equipo usa la CPU."""


def _min_supported_capability(torch) -> tuple[int, int] | None:
    """Menor capacidad de GPU incluida en esta versión de PyTorch (p. ej. 'sm_75' -> (7, 5))."""
    try:
        caps = [(int(a[3:-1]), int(a[-1])) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    except Exception:
        return None
    return min(caps) if caps else None


def _cpu_time() -> float | None:
    try:
        import torch

        return _timed_matmul(torch.device("cpu"))
    except Exception:
        return None


def _timed_matmul(device, size: int = 4096, repeats: int = 5) -> float:
    import torch

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(2):  # calentamiento
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PyTorch avisa de mucho; aquí damos nuestros propios mensajes
        try:
            import torch
        except ImportError:
            print(HELP_INSTALL)
            return 1
        print(f"PyTorch {torch.__version__} (compilado con CUDA {torch.version.cuda})")
        try:
            import numpy  # noqa: F401
        except ImportError:
            print("\n" + HELP_NO_NUMPY + "\n")

        cpu = _cpu_time()
        if cpu is not None:
            print(f"CPU: multiplicación de matrices 4096x4096 en {cpu * 1000:.0f} ms")
        if not torch.cuda.is_available():
            print(HELP_NO_CUDA)
            return 2

        props = torch.cuda.get_device_properties(0)
        capability = (props.major, props.minor)
        print(f"GPU: {props.name} · {props.total_memory / 1024 ** 3:.1f} GB de VRAM · capacidad {props.major}.{props.minor}")
        minimum = _min_supported_capability(torch)
        if minimum is not None and capability < minimum:
            print(HELP_UNSUPPORTED.format(
                name=props.name, cc=f"{props.major}.{props.minor}", minimum=f"{minimum[0]}.{minimum[1]}"))
            return 3
        try:
            gpu = _timed_matmul(torch.device("cuda"))
        except Exception as e:  # noqa: BLE001 - cualquier fallo de CUDA se explica igual
            print(f"La GPU falló al ejecutar un cálculo de prueba: {e}")
            return 4

    if cpu is not None:
        print(f"GPU: la misma multiplicación en {gpu * 1000:.1f} ms (unas {cpu / gpu:.0f} veces más rápida que la CPU)")
    else:
        print(f"GPU: multiplicación de matrices 4096x4096 en {gpu * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
