# Round-32 decision（Level-1 草案；待用户/CG 裁决）

**decision: `GO_ACTION_CONDITIONING`**

## 依据（Level-1 证据）

- **Gate E（P1，50 ep，MoE-off）**：D0→D1 **2.9%** / D1→D2 **0.0%** / **D2→D3 97.0%** ⇒ 主瓶颈=latent transition；pose/执行与动态图几何非主因。
- **Gate B（P2）**：FD 敏感度 ds 通道 **26–50× 弱于 dθ/κ**；Action Swap matched **top-1=10.2%、margin≈0** ⇒ 动作条件化未过门。
- **Gate C（P3）**：real latent 充分（lane 目标 ld_slots 视图 0.90/0.93）⇒ 排除 state representation 支路；WM 保留 ego-dynamics ✓（0.91–1.10）、lane ✗（0.61/0.77）。
- **Gate D（P4）**：A_k 温和（p50 ×2.1）；Lipschitz 中位比 0.73–0.78、仅尾部 P95≈6× ⇒ 接口为次要项。

## 未选支路及理由

- `GO_ROLLOUT_PROTOCOL_FIX`：D0→D1=2.9%（pose/执行非主因）。
- `GO_DYNAMIC_GRAPH_FIX`：D1→D2=0.0%（且位姿通道敏感度自检弱）。
- `GO_STATE_REPRESENTATION_FIX`：real latent 充分（Gate C 正证）。
- `GO_POLICY_INTERFACE_ROBUSTNESS`：A_k 温和，仅尾部放大（次要）。
- `GO_ALTERNATING_WM_POLICY_TRAINING`：Level-3，未到。
- `BLOCKED_INSTRUMENTATION`：Gate A 全绿（逐位复现/自检=0/clip=0）。

## 建议下一步（待批）

1. **action-conditioning canary**（主）：显式动作残差（ds/dθ/κ 归一化特征 + 零初始化 `alpha=0`，前向逐位兼容）；数据 60/40（expert/policy）；保持 detach/无 BPTT/不扩容；R4 选 ckpt；目标沿用 GPT 裁决（W2/P1 门）。
2. **并行只读**：统一口径（MoE-off）A_k 重录；lane 保留率深化（Level-2 条件项：ego 物理锚/车道锚/prior-residual/rank）。
3. 其余 Level-2/3 项按门触发。

## 纪律

Level-1 全程零训练、零生产代码改动（Gate A 全绿）；本 decision 为**草案**，等待裁决后执行 canary。
