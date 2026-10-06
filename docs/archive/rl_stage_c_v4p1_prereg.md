# Stage C v4.1 批 · 预注册（2026-10-01 03:50 冻结；基于 v4 批结果）

> 目的：在 v4 批（7 臂均未达"不损伤"）结论上，用 **V-data（500 池）底座 + 单杠杆** 的组合臂冲击"不损伤（clean500 net ≥ −10）"。

## 背景（v4 批全轨迹，子集150 配对 net；详见 `/tmp/opencode/rl_v4_results.md` + `rl_v4_posthoc2.md`）
| 臂 | u25 | u50 | u75 | u100 | 判定 |
|---|---|---|---|---|---|
| V0（critic bundle） | −62 | −22 ✗ | −8 | −34 | early-collapse |
| V1（去 bundle） | −66 | −20 ✗ | −28 | −40 | early-collapse |
| V-anchor（真锚 0.01） | −32 | −27 ✗ | −26 | **−12**（offΔ +0.067） | early-collapse（破线后逐点恢复，端点最近闸内） |
| **V-data（500 池）** | −20 | **−8 ✓** | −11 | −9 | **terminal:failed**（clean −26 / eval −23；唯一全程稳定） |
| V-lr（1e-4） | −13 | −67 ✗ | −67 | −67 | 爬行塌缩（succ 0） |
| V-ttc | −62 | −16 ✓ | −59 | −58 | terminal:failed（clean **−187**；假过→崩） |
| V-boundary | −31 | −67 ✗ | −67 | −67 | early-collapse（drift 1.31 最大、offΔ +0.51） |

要点：① **pool 是唯一防塌杠杆**；② **真锚能"破线后恢复"**（V-anchor 端点 −12）；③ 闸弱筛选（V0 漏报 / V-ttc 误报）⇒ 全轨迹 + keep-best 为标配；④ 爬行盆地真实存在（lr 臂）。

## 底座（两臂一致）
- **V-data recipe**：v0 flags（critic bundle: `--critic-warmup-updates 10 --value-lr-scale 5 --target-kl 0.05`）+ `--spec env/specs/scenarios_train_dagger_r1.json`（500 条）；其余 pins 同 v4（`--spec-rotation episode --plan-reference plan --kl-anchor-coef 0.0 --kl-anchor-final-coef 0.0 --pool local --envs 1 --trainable-scope design --updates 100 --rollout-steps 256 --ppo-epochs 2 --minibatch-size 1024 --seed 0 --device cuda --monitor --monitor-legacy-tags --ckpt-every 25`）。

## 臂集（2 臂，单变量于底座）
| 臂 | 变量 | 依据 |
|---|---|---|
| **v41pa** | ＋真锚 `--kl-anchor-coef 0.01 --kl-anchor-final-coef 0.01` | pool（防塌）× 锚（防漂、破线后恢复）——两个最佳杠杆合流 |
| **v41pk** | ＋`--target-kl 0.03`（更紧 KL 守门） | 针对崩解模式（V-ttc u75 崩、V-boundary 崩、V0 末段回退） |
| ~~v41pb（弃）~~ | ~~＋lane_boundary~~ | V-boundary@v4 直接崩（−67）⇒ 无正信号，弃 |

## 判据（同 v4）
- u25 观察；u50 闸（net < −20 或 offΔ ≥ +0.10 ⇒ 停）；过闸 ⇒ u75/u100 + 终评 clean500+eval500。
- **不损伤** = clean500 net ≥ −10；**正向** = net ≥ +20 且 z ≥ 1.96；**正向 ⇒ seed=11 复现**（优先）。
- 报全轨迹、fixed/broken/z、offΔ/collΔ、探针读数；单 seed 仅方向性。

## 顺序与预算
- 顺序：**v41pa → v41pk**；每臂 ≈35–45min；预计 04:00 起 ~1.5h。
- 驱动 `rl_v41_driver.py` + `rl_v41_chain.sh`；结果档 `/tmp/opencode/rl_v41_results.{md,json}`。

## 边界
- 单 seed；基于 V-data 底座（不与 v4 单杠杆臂直接混比）；闸弱筛选已知（以全轨迹为准）；GPU 串行（诊断评测允许并行并标注）。
