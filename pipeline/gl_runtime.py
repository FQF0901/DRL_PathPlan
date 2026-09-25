"""venv 本地 glvnd 运行库路径保障（spawn worker 的 GL 修复，p2-contract §8 前置）。

背景（P0 实测，见 `.slim/deepwork/planner-rl-feasibility.md`）
--------------------------------------------------------------
本机没有系统 ``libGL.so.1``；MetaDrive/Panda3D 初始化 engine 时两个 GL pipe 都加载失败 →
``n_pipes=0`` → MetaDrive 0.4.3 的 ``logger.info("Known Pipes: {}".format(*[]))`` 抛
``IndexError: Replacement index 0 out of range for positional args tuple``。

修复分两层：

1. **LD_LIBRARY_PATH**：``tools/venv-python`` 给当前进程加 ``.venv/gl/usr/lib/x86_64-linux-gnu``；
   但 ``multiprocessing`` spawn 的子进程只继承**父进程当时的 environ**——父进程若没经
   wrapper 启动（或子进程被直接 spawn），worker 就缺少该路径 → pooled 训练在 worker 里崩溃。
   :func:`ensure_gl_library_path` 从**仓库根**计算目录并写回 ``os.environ``，保证 spawn 前生效。
2. **进程内预加载**（``preload=True``）：Linux 的 ``LD_LIBRARY_PATH`` 只在进程启动时被 ld.so
   读取，运行时改 environ 不影响**当前进程**的 ``dlopen``。父进程自己也要建 env 时（``--pool local``），
   在 ``import panda3d`` 之前用 ``ctypes`` 按依赖序把 venv 里的 glvnd 库 ``RTLD_GLOBAL`` 预载，
   等价于 wrapper 的效果。

两个入口都幂等、缺目录时静默返回 ``None``（非本项目环境不受影响）。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

__all__ = ["GL_LIB_RELPATH", "repo_root", "gl_library_dir", "ensure_gl_library_path"]

#: 仓库内 glvnd 库目录（相对仓库根；由 ``tools/setup_gl_libs.sh`` 解包）
GL_LIB_RELPATH = Path(".venv") / "gl" / "usr" / "lib" / "x86_64-linux-gnu"

#: 预载顺序 = 依赖序（libGL 依赖 libGLdispatch/libGLX；EGL 供 headless pipe）
_PRELOAD_ORDER = ("libGLdispatch.so.0", "libGLX.so.0", "libEGL.so.1", "libGL.so.1")


def repo_root() -> Path:
    """仓库根（本文件位于 ``<repo>/pipeline/gl_runtime.py``）。"""
    return Path(__file__).resolve().parents[1]


def gl_library_dir() -> Optional[Path]:
    """返回 venv 本地 glvnd 目录；不存在时返回 ``None``。"""
    candidate = repo_root() / GL_LIB_RELPATH
    return candidate if candidate.is_dir() else None


def ensure_gl_library_path(*, preload: bool = False) -> Optional[str]:
    """确保 ``LD_LIBRARY_PATH`` 含 venv 本地 glvnd 目录（幂等），返回该目录或 ``None``。

    Args:
        preload: 额外尝试把 glvnd 库预载进**当前进程**（仅在 ``panda3d`` 尚未导入时；
            用于未经过 ``tools/venv-python`` 启动、但自己也要建 env 的父进程）。
    """
    gl_dir = gl_library_dir()
    if gl_dir is None:
        return None
    entry = str(gl_dir)
    parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    if entry not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([entry, *parts])
    if preload and "panda3d.core" not in sys.modules:
        for name in _PRELOAD_ORDER:
            library = gl_dir / name
            if not library.is_file():
                continue
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:  # pragma: no cover - 依赖缺失时保持可诊断但不致命
                break
    return entry
