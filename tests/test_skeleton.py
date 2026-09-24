"""骨架测试：确认 P0 配置文件存在且可解析为 YAML。"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ("default.yaml", "env.yaml", "model.yaml", "train.yaml", "eval.yaml")


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_files_exist_and_parse(name: str) -> None:
    path = ROOT / "config" / name
    assert path.is_file(), f"缺少配置文件: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, f"配置为空或不是映射: {path}"
