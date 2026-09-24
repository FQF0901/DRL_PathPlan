# env

目的：非侵入式封装 MetaDrive，提供场景、观测、钩子与专家接口。

- 仿真：headless、仅物理；策略步 0.5 s（2 Hz），MPC/插值参考 10 Hz，预览 6 点 / 3 s。
- 观测：6 帧历史（对齐当前 ego 的 SE(2) 系）、OD/LD top-16、导航（`get_checkpoints()` +
  `navigation_command`，不算到转弯距离）、信号占位（恒“无灯”）、限速、ego 动力学。
- 场景：3–5 个 block，按显式 `BIG_BLOCK_SEQUENCE` 组装，构建后设限速；有界 LRU 地图缓存；
  不含信号灯与多智能体。

接口：`scenario/`（spec 生成/校验/注册表）、`obs/`（观测通道）、`hooks/`（生命周期探针）、
`expert/`（BC 用专家）。

可插拔：观测通道、场景构建器、钩子均经注册表挂载；不修改 metadrive 源码与系统 Python。
