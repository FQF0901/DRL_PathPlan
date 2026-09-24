# env/obs

目的：把环境状态编码为定长观测（含 6 帧历史），供网络直接消费。

## 通道（实测形状，单帧）

| 通道 | 形状 | 内容 |
|---|---|---|
| ego | (1,8) | v, a_long, a_lat, yaw_rate, steer, curvature + 2 保留 |
| od | (16,9) | dx, dy, vx, vy, cosθ, sinθ, L, W, type_id（100 m 内、距离+TTC 排序取 top-16，排除自车）|
| ld | (16,7) | dx, dy, heading_rel, curvature, speed_limit, left_line_type_id, right_line_type_id |
| nav | (1,11) | 2 个 checkpoint（**自车系** x,y）+ 6 维命令 one-hot + route_completion |
| signal | (1,4) | 占位，恒 `[0,0,0,1]`（无交通灯）|

> 注意：**没有单独的 speed_limit 通道**，限速包含在 LD 特征里。

## 历史与对齐
6 帧 @0.5 s（每 5 个 env step 存 1 帧）；每帧按"该帧自车系特征 + 该帧 ego 位姿"存储，
取用时做 SE(2) 对齐到当前自车系（约定：**x 前向 / y 左向**；实测对齐误差 ~5e-6 m）。
预热期重复最旧真实帧，`*_hist_mask` / `hist_valid` 标记有效帧。

## 性能（单实例 headless）
obs 0.53–1.06 ms/step；env.step 0.42–2.4 ms（≈600–1000 FPS）。

## 可插拔
实现 `ObservationChannel` 并 `register()` 即可扩展；不改动已有通道。
