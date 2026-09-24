# env/scenario

目的：由显式 block 序列构建小地图，生成可复现的场景 spec。

## spec schema（JSON）
- `map`：`BIG_BLOCK_SEQUENCE` 的 block 列表与随机种子。
- `traffic`：cut-in / cut-out 脚本与参与车辆参数。
- `ego`：初始位姿与目标路线（checkpoints）。
- `limits`：构建后写入的限速。
- `meta`：生成器版本、参数哈希（保证可复现）。

## 接口
- generator：按 spec 构建 MetaDrive 地图与交通。
- validator：校验字段与可执行性，失败即报错。
- registry：场景构建器注册 / 取用。

## 可插拔
新增 block / 行为 / 构建器时注册即可；地图规模限 3–5 个 block，地图缓存为有界 LRU。
