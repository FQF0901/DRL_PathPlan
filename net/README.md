# net

目的：P2 驾驶策略网络（编码 / 时序 / 空间 / MoE / 世界模型 / 策略头），主入口 `net.model.DrivingModel`。

## 模块
- `encoders.py`：各通道线性投影到 H + 类型嵌入 + LayerNorm；节点顺序 `[ego, 16 OD, 16 LD]`。
- `temporal.py`：逐节点 GRU 过 6 帧历史，`hist_valid` 门控预热补位帧（补位帧复制最旧帧但掩码=1）。
- `spatial.py`：2 层消息传递（OD↔ego、LD↔ego、LD↔LD 相邻、OD↔OD 近邻，边=相对位姿）。
- `moe.py`：primary 恒激活（不受门控）+ 8 个零初始化残差 expert；sigmoid 多标签路由（BCE）。
- `world_model.py`：ego 条件化 OD/LD 动力学（OD_PRED_DIM=5、LD_PRED_DIM=4，输出残差），直接多步损失 + mask + 噪声。
- `policy.py`：sigmoid 压缩的有界高斯策略（ds∈[0,10] m、dθ∈[±0.6] rad）+ value 头。
- `model.py`：装配、`arc_step` 运动学单一实现、B1 自回归 rollout、`SUPERVISED_LABELS`（8 个受监督标签）。
- `param_probe.py`：参数探针 + 前向/rollout 形状冒烟（断言 ≤1.5M）。

## I/O（B 维在前）
- 输入：ego(8，末 2 维 = 上一策略步 `(ds, dθ)`)、od(16,9)、ld(16,7)、nav(11)、signal(4)；
  历史 `od/ld_hist (6,16,F)` + 掩码 + `hist_valid(6)`（6 帧 @0.5 s）。
- 输出：`action_mu/action_logstd (2)`、`value (1)`、`traj_xy (6,2)`（t=0.5..3.0 s，t0 自车系）、
  `od_pred (6,16,5)`、`ld_pred (6,16,4)`、`router_logits (8)`、`plan (6,2)`（rollout 实际执行的 6 个动作）。
- `forward(..., rollout=False, world_model=False)` 为 PPO cheap path：省略 traj/WM 键，其余输出逐位一致。

## 规模与用法
- 配置（`config/model.yaml`）H=128、experts 128→256→128 → 训练 ckpt 实测 1,174,875 个标量
  （110 tensors，含 buffer），占 1.5M 参数预算的 78.3%。
- `tools/venv-python net/param_probe.py`：打印各模块参数并断言 ≤1.5M；默认构造走模块默认
  H=96 / experts 192 → 显示 666,263（与当前 config 不是同一配置）。
- 按 config 构建用 `pipeline.stages.build_model`（includes 平铺合并后的根配置）。
