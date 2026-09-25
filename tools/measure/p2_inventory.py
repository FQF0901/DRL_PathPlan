#!/usr/bin/env python3
"""P2 接口清单：导入各 P2 模块并打印公开 API（类/函数签名）。

用途：五条实现线落地后，用它一次性看清实际接口（对照 p2-contract 检查偏差），
再写真正的集成检查。纯导入，不建 env、不跑仿真。

Usage:
    tools/venv-python tools/measure/p2_inventory.py
"""
import importlib
import inspect
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

MODULES = [
    "net.encoders", "net.temporal", "net.spatial", "net.moe", "net.world_model",
    "net.policy", "net.model",
    "reward_model.registry", "reward_model.terms", "reward_model.aggregation", "reward_model.kpi",
    "env.tracking",
    "pipeline.vector_env", "pipeline.buffer", "pipeline.rollout",
    "pipeline.trainer", "pipeline.stages", "pipeline.eval_runner", "pipeline.monitoring",
]


def describe(module_name: str) -> None:
    print(f"== {module_name} ==")
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
        return
    root = module_name.split(".")[0]
    found = False
    for name, obj in sorted(vars(mod).items()):
        if name.startswith("_"):
            continue
        is_local = getattr(obj, "__module__", "").startswith(root)
        if inspect.isclass(obj) and is_local:
            try:
                sig = str(inspect.signature(obj.__init__)).replace("(self, ", "(").replace("(self)", "()")
            except Exception:  # noqa: BLE001
                sig = "(?)"
            doc = (inspect.getdoc(obj) or "").splitlines()
            print(f"  class {name}{sig}")
            if doc:
                print(f"        # {doc[0][:100]}")
            found = True
        elif inspect.isfunction(obj) and is_local:
            try:
                sig = str(inspect.signature(obj))
            except Exception:  # noqa: BLE001
                sig = "(?)"
            print(f"  def   {name}{sig}")
            found = True
    if not found:
        print("  (no public API found)")


def main() -> None:
    for module_name in MODULES:
        describe(module_name)
        print()


if __name__ == "__main__":
    main()
