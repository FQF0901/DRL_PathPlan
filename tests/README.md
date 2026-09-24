# tests

目的：验证骨架与后续回归。

- `test_skeleton.py`：校验配置文件存在且可被 YAML 解析。
- 运行：`python3 -m pytest tests/ -q`。
- 约定：测试无外部依赖、可离线运行；不加载 MetaDrive。

## 待补（P1/P2）
- spec 生成 / 校验的端到端用例。
- 数据集抽检：帧间位移 / 航向与记录运动一致、物体速度有界、无时间反转。
- 各阶段 loss / KPI 落在预声明区间内的冒烟测试。
