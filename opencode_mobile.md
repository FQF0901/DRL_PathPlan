# 手机企业微信远程指挥 OpenCode（V2 方案）

> 适用环境：Ubuntu + OpenCode v2.x（实测 v2.0.25，2026-10）。
> V1 的 `opencode attach`、`--username/--password`、`/global/*`、`/doc` 在 V2 中已不存在，不要再按旧文档部署。
> 场景：单聊（一对一）；任务常跑数小时～数十小时。
> 目标：手机企微与 Ubuntu 本地终端操作同一个 OpenCode Server、同一个 session；不需要公网 IP、域名备案、VPN 或付费服务。

## 1. 架构

```text
手机企业微信
   │  WebSocket 长连接（BotID + Secret）
   ▼
Ubuntu: wecom-opencode-bridge（新增的唯一组件，纯出站）
   │  HTTP Basic + SSE，127.0.0.1
   ▼
OpenCode Server（v2，仅监听 127.0.0.1）
   ▲
   │  opencode / opencode -s ses_xxx（自动发现本地服务）
Ubuntu 本地人工交互
```

- 桥接器负责：企微消息 → OpenCode API；OpenCode 事件/权限请求 → 企微消息。
- 本地终端不加任何参数，`opencode` 自动连到同一个 Server，因此双端天然共享 session。
- 服务器只需要的出站能力：443 到 `wss://openws.work.weixin.qq.com` + 本机 127.0.0.1。
- tmux + SSH 只留作应急维护，不参与企微交互。

## 2. OpenCode V2 关键事实（已实测）

| 项 | 结论 |
|---|---|
| 启动 Server | `opencode serve`（默认 127.0.0.1）；常驻用 `opencode serve --service`，即 V2 的“共享后台服务”本体 |
| 本地连接 | TUI 直接 `opencode`；指定会话 `opencode -s ses_xxx`；脚本 `opencode run -s ses_xxx "消息"` |
| 认证 | HTTP Basic；用户名固定 `opencode`；密码 = 服务注册文件 `~/.local/state/opencode/service.json`（0600）中的 `password` |
| 事件流 | `GET /api/event`（SSE，**live-only**：无重放、无自动重连，含 `: heartbeat` 注释行） |
| 接口定义 | `GET /openapi.json`，实现以它为准 |
| 版本差异 | `OPENCODE_SERVER_USERNAME`、`/global/health`、`/global/event`、`/doc` 均无效 |

三个必须遵守的结论：

1. **全系统只有一个 Server 实例**。用 systemd 跑 `opencode serve --service` 后，不要再手动 `opencode service start` 或另起 `opencode serve`，否则多进程共用同一个 `opencode.db`，且企微/事件/权限状态会不一致。
2. **服务密码每次重启可能变化**（V2 未固定密码时随机生成并重写注册文件）。桥接器在每次重连时重新读取 `service.json`，不要长期缓存。
3. **服务与桥用同一个 Linux 用户**。桥需要读上述 0600 注册文件；同用户时零配置。独立用户读不到该文件，需要额外维护密码共享（固定服务密码 + 双方可读文件或 systemd LoadCredential），而隔离收益很小——桥本来就能让 OpenCode 以该用户身份执行任意命令。

## 3. Ubuntu 部署

### 3.1 OpenCode Server 常驻

```ini
# /etc/systemd/system/opencode-server.service
[Unit]
Description=OpenCode Server (V2 shared service)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu                          # 改成你的登录用户
ExecStart=/usr/local/bin/opencode serve --service --hostname 127.0.0.1 --port 4096
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# ExecStart 的路径用 command -v opencode 的实际输出替换
sudo systemctl daemon-reload
sudo systemctl enable --now opencode-server

# 验证（自动发现 + 自动带认证）
opencode api get /api/info
opencode                             # 本机 TUI，应直接连上
opencode -s ses_xxx                  # 进入指定会话
```

### 3.2 桥接器常驻

```ini
# /etc/systemd/system/wecom-opencode-bridge.service
[Unit]
Description=WeCom <-> OpenCode bridge
After=network-online.target opencode-server.service
Wants=opencode-server.service
# 60 秒内超过 10 次启动即视为崩溃循环，停止重启（需人工 reset-failed）
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/etc/wecom-opencode-bridge.env
WorkingDirectory=/opt/wecom-opencode-bridge
ExecStart=/usr/bin/python3 -m bridge
Restart=always
# 认证失败会快速退出，间隔放宽避免频繁向企微重新订阅
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
# /etc/wecom-opencode-bridge.env （chmod 600，属主 ubuntu）
WECOM_BOT_ID=aibXXXXXXXX
WECOM_BOT_SECRET=XXXXXXXXXXXXXXXX
WECOM_ALLOWED_USERS=userid1,userid2            # 白名单，写真实回调里出现过的 userid
WECOM_ALLOWED_PROJECTS=/home/ubuntu/code        # 允许桥操作的项目根目录（逗号分隔）
# 默认模型由 opencode.json 决定，桥不覆盖；仅当要给企微新建会话单独指定时才设置：
# WECOM_DEFAULT_MODEL=provider/model
STATE_DB=/var/lib/wecom-bridge/state.db
TZ=Asia/Shanghai
```

- 桥接器不需要监听任何端口，也不需要 root；与 Server 同用户运行（见第 2 节结论 3）。
- 依赖安装：`pip3 install -r /opt/wecom-opencode-bridge/requirements.txt`（服务器上也可用 venv，systemd 单元里的 ExecStart 换成 venv 的 python 即可）。
- 无 systemd 的环境（容器等）用 `run.sh`：flock 单实例 + 崩溃自动重启（认证失败退避 60s）。启动：`setsid nohup /opt/wecom-opencode-bridge/run.sh >/dev/null 2>&1 &`，日志 `/var/log/wecom-opencode-bridge.log`。
- 推荐实现栈：Python 3.10+，官方企微 SDK `wecom-aibot-python-sdk`（负责长连接/心跳/重连），`httpx` 调 OpenCode API，`sqlite3` 标准库存状态。Node 等价方案：`@wecom/aibot-node-sdk`。

### 3.3 企业微信侧

1. 企业微信 PC 客户端 → 通讯录 → 智能机器人 → 创建 → **API 模式** → 选择**长连接**，记录 BotID / Secret。**一定要用页面上的「复制」按钮**——手抄极易混淆 `I/l`、`K/k`、`O/0`（实测抄错过 3 个字符）。保存后确认连接方式仍是「长连接」。
2. 可见范围只放自己和白名单成员；无需公网地址、无需 Token/EncodingAESKey。
3. 只用单聊：不需要建群，也不需要群聊权限模型。
4. 白名单引导：先把 `WECOM_ALLOWED_USERS` 留空，手机给机器人发消息会回复你的 userid；填入后重启桥即可。
5. 认证结果看日志：出现 `wecom authenticated` 即成功；失败会打印 `errcode=853000 invalid bot_id or secret`（凭据错/抄错/不是长连接模式），桥每 60s 自动重试。
6. 回调里的 `from.userid` 可能是**加密值**（除非机器人创建者是超管）；白名单用真实回调里出现过的值。

## 4. 桥接器设计

### 4.1 状态（SQLite）

| 表 | 字段（最少） |
|---|---|
| bindings | `wecom_userid` → `project_dir` → `session_id` |
| jobs | `job_id, session_id, msgid, status, started_at, finished_at, last_event_at, last_summary` |
| dedup | `msgid` 唯一键（企微重推防重） |
| audit | 时间、来源 userid、动作、参数摘要 |

### 4.2 OpenCode API 清单（全部走 Basic 认证）

| 用途 | 请求 |
|---|---|
| 健康检查 | `GET /api/info` |
| 列会话 | `GET /api/session?directory=<dir>&order=desc&limit=20`（加 `parentID=null` 只看根会话，与 TUI 对齐） |
| 建会话 | `POST /api/session` `{"location":{"directory":"<dir>"},"title":"企微任务 MM-DD HH:MM"}` |
| 发消息 | `POST /api/session/<ses>/prompt` `{"text":"...","delivery":"queue"}` |
| 等完成 | `POST /api/experimental/session/<ses>/wait`（只作短超时辅助；长任务见 4.4/4.7） |
| 取消息 | `GET /api/session/<ses>/message?limit=50&order=desc`（取最终回复、/tail） |
| 中断 | `POST /api/session/<ses>/interrupt` |
| 活跃会话 | `GET /api/session/active` |
| 权限列表 | `GET /api/session/<ses>/permission` |
| 权限答复 | `POST /api/session/<ses>/permission/<per>/reply` `{"decision":"once"\|"always"\|"reject"}` |
| 本轮改动 | `GET /api/session/<ses>/diff`（汇总“改动文件”） |
| 事件流 | `GET /api/event`（SSE） |

```bash
# 认证示例：密码从注册文件读取
PW=$(python3 -c "import json;print(json.load(open('$HOME/.local/state/opencode/service.json'))['password'])")
curl -u "opencode:$PW" http://127.0.0.1:4096/api/info
```

### 4.3 消息与命令处理

- 普通文本 → 发 `prompt`，默认 `delivery:"queue"`（服务端排队，不打断正在执行的任务）。
- 文本以 `/` 开头 → 本地命令（见第 5 节），不透传给模型。
- 发消息后：立即回执（第 4.4 节），长任务转入事件驱动监控，不再阻塞等待。
- `/status`、`/tail` 基于 `session/active` + `message` 接口即时查询。

### 4.4 输出与通知（按长任务设计）

官方硬限制：回调后 **5 秒内**首次响应；单条流式消息最多持续 **10 分钟**；每会话 **30 条/分钟、1000 条/小时**；收到用户消息后 24 小时内可回复；主动推送 `aibot_send_msg` 要求用户曾在该会话给机器人发过消息（发过一次即可）。

任务常跑数小时到数十小时，分两段处理：

1. **活跃期（前 10 分钟）**：`aibot_respond_msg` 流式消息（`finish=false`，透传回调的 `req_id`），每 20–30 秒把“当前步骤 + 已运行时长 + 最近一行输出”覆盖到同一条消息。
2. **长跑期（10 分钟后）**：第 ~9 分钟 `finish=true` 收尾（“仍在运行，后续在本会话汇报”），改为事件驱动：
   - **必须立即推**：权限审批、agent 提问、运行失败、任务完成；
   - **常规心跳**：默认每 60 分钟一条简短状态（可配置/关闭）；
   - 其余时间安静，用户可随时 `/status`。
3. **完成/失败通知**：用 `aibot_send_msg` 主动推送（不依赖 24 小时回复窗口），markdown 摘要（≤20480 字节）：状态、耗时、结论、改动文件（来自 `session/diff`）、建议下一步。

所有推送按会话合并计数，避免触发 30 条/分钟限流；内容超长只发截断摘要。

### 4.5 权限审批（安全关键）

- 桥接器**不得**给 OpenCode 传 `--auto` 或自动批准权限。
- 任务执行期间轮询 `GET /api/session/<ses>/permission`（或收到 SSE 事件后立即拉取）；发现待批请求立即推给手机（属于 4.4 的“必须立即推”）：

```text
需要授权：bash 执行 `rm -rf build/`
回复 /approve per_xxx 或 /reject per_xxx（不答复则任务一直阻塞）
```

- 收到命令后调用 `permission/<per>/reply`（`once` / `always` / `reject`）。`always` 只对可信的低危操作开放。

### 4.6 并发

- 桥接器对每个 session 串行：同一 session 同时只有一个由手机发起的任务，其余排队并在回执中说明队列位置。
- 本机 TUI 与手机同时操作同一 session 时，靠 `delivery:"queue"` 排队；手机端 `/abort` 可中断。
- 无需复制“自定义高危正则 + 审批码”系统，全部交给 OpenCode 原生权限机制（4.5）。

### 4.7 断线与恢复（长任务的生命线）

- 企微长连接：指数退避重连 + 30s 心跳；收到 `disconnected_event` 说明有第二个连接踢人，检查是否多实例。
- OpenCode SSE 是 live-only（无重放、无自动重连）：断开后重新订阅，并立即 REST 对账：
  - `GET /api/session/active` 判断任务是否还在执行；
  - 对 `jobs.status=running` 且已不在 active 列表的任务，用 `message` 复查最后结果后补发完成通知（凭 `finished_at` 去重）。
- 长任务不要依赖单个 `wait` 请求挂住：当前实现用自适应轮询（前 30s 每 5s → 20s → 长跑期 60s）查 `/api/session/active` + `message`；后续可叠加 SSE 事件进一步降延迟。
- 桥重启是常态：job 状态全部落 SQLite，启动时先对账、再恢复通知。
- Server 重启后 `service.json` 密码可能变化：重连前重新读取。

### 4.8 实战踩坑与规避（上线实录）

1. **BotID/Secret 手抄必错**：`I/l`、`K/k`、`O/0` 极易混淆（实测 Secret 抄错 3 个字符）。用页面「复制」按钮；错误特征是日志里的 `errcode=853000`，桥会 60s 退避重试。
2. **企微 Markdown 会吃掉下划线**：`/workspace/01_Proj/...` 的 `_` 被当斜体后可能在显示/复制时消失（变成 `01 Proj` 或 `01Proj`）。桥已做：路径与配置名一律反引号包裹；`/new` 目录不存在时给最接近的正确路径建议。用户侧重要操作优先用完整会话 ID。
3. **TUI 只显示根会话**：`opencode session list` / TUI 默认仅 top-level；fork/子任务会话有 `parentID`。桥 `/sessions` 已对齐（仅根），`/sessions all` 看子会话（带 `↳`）。
4. **emoji 兼容性**：🟢/⚪ 等圈形 emoji 在企微客户端可能显示异常（实测全变红圈）；状态一律用文字（`【运行中】/【空闲】`），关键信息不依赖 emoji。
5. **双端同会话互相等待**：手机任务与 TUI 任务在同一会话时按 `delivery=queue` 排队；会话忙时再发消息会提示"已有任务在运行"；完成判定以会话整体空闲为准。
6. **桥重启恢复已实测**：排队中的任务在桥重启后自动恢复监控，并正常推送完成汇报。

## 5. 手机指令一览

| 指令 | 作用 |
|---|---|
| `/sessions [all]` | 列出会话（默认仅根会话，与 TUI 对齐；`all` 含子会话） |
| `/use ses_xxx` | 绑定会话（短前缀限当前项目；完整 ID 可跨项目、可绑 fork 子会话） |
| `/new [项目目录]` | 新建并绑定会话（校验目录存在、自动命名"企微任务 MM-DD HH:MM"） |
| `/current` | 查看当前绑定与任务状态 |
| `/status` | 当前任务状态 + 已运行时长 + 最近进度 |
| `/sys` | 主机资源：CPU/负载、内存、Swap、磁盘、GPU、Top 进程（本地直读，不走模型） |
| `/tail [n]` | 最近 n 条消息摘要（默认 5） |
| `/abort` | 中断当前生成 |
| `/approve per_xxx` / `/reject per_xxx` | 处理权限审批 |
| `/local` | 返回本机 TUI 进入命令：`opencode -s ses_xxx` |
| `/unbind` | 解除会话绑定 |
| 其他文本 | 作为 prompt 发给当前 session |

> 提示：企业微信会把消息里的 `_` 当 Markdown 斜体处理，显示/复制时可能"消失"。发路径前先确认下划线还在，或直接用完整会话 ID。

## 6. 安全清单

- Server 仅监听 127.0.0.1；桥不开放端口；不把 Server 暴露公网（禁止 `--hostname 0.0.0.0`）。
- 企微用户白名单 + 项目目录白名单（单聊同样强制）。
- `msgid` 去重；所有动作写审计日志。
- `service.json`、`/etc/wecom-opencode-bridge.env` 权限 600，不写进日志；journal 输出脱敏。
- 不在企微消息里返回密码、API Key、`.env` 内容。
- 桥与 Server 同用户运行时不要额外加 root；systemd 单元不加 `User=root`。

## 7. 实施顺序

- **P0 ✅ 已完成**：桥骨架（长连接、去重、绑定、prompt、长任务完成检测与主动通知、`/status` `/tail` `/abort`）+ 权限审批转发。
- **P1 ✅ 已完成**：`/sessions [all]` `/use` `/new`（目录校验 + 自动标题）、`session/diff` 汇总、根/子会话对齐。
- **P2 待做**：入站图片/文件（aeskey 解密、512KB 分片上传）、模板卡片审批按钮、`/unbind` 后可浏览、生产服务器 systemd 部署、Web UI 备用面板。

## 8. 备选通道（均非主链路）

- `opencode pair`：一次性链接 + 二维码，手机浏览器直接打开官方 Web UI 当监控面板；`--remote` 走 OpenTunnel（境外服务，大陆延迟不稳定，不建议依赖）。
- SSH + tmux：应急维护。
- 升级注意：`opencode upgrade` 后 `sudo systemctl restart opencode-server`，并核对 `/openapi.json` 是否与桥接器使用的接口一致。

## 9. 运行手册（已上线，2026-10-08）

- 代码 `/opt/wecom-opencode-bridge/`；配置 `/etc/wecom-opencode-bridge.env`（0600）；日志 `/var/log/wecom-opencode-bridge.log`。
- 启动 / 重启（容器，无 systemd）：

  ```bash
  # 停掉旧进程后执行
  setsid nohup /opt/wecom-opencode-bridge/run.sh >/dev/null 2>&1 &
  tail -f /var/log/wecom-opencode-bridge.log     # 看到 wecom authenticated 即成功
  ```

  `run.sh` 自带 flock 单实例；崩溃自动重启；认证失败退避 60s 重试。
- 改配置（白名单/项目/模型）后重启桥生效；生产 Ubuntu 服务器用 `deploy/` 下两个 systemd 单元。
- 当前状态：企微认证成功；白名单 `FangQiFei`；绑定 `/workspace/01_Proj/DRL_PathPlan`；`/sessions` 仅根会话（与 TUI 一致），`/sessions all` 含 fork 子会话。

**验收清单**：日志出现 `wecom authenticated`；手机 `/help` 有回复；`/sessions` 与 TUI 根会话一致；`/new` 正确目录成功、错目录给建议；只读任务走完"回执 → 进度 → 完成汇报"；`/abort` 汇报"已中断"；`/sys` 返回主机资源；本机 `opencode -s <会话ID>` 与手机同会话。

**已知限制 / 下一步**：容器重启后需手动启动（生产机器用 systemd 规避）；`/unbind` 后无法 `/sessions` 浏览（可记住上次项目）；入站图片/文件暂不支持；模板卡片审批未做（当前文本 `/approve` `/reject`）。
