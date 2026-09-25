"""训练向量环境池（P2 契约 §4 / §8.1）：spawn 子进程 + 常驻 env + 按 spec 数回收。

为什么需要它
------------
MetaDrive **每进程只能有一个 engine**（``envs/base_env.py`` reset 里的 singleton 断言），
多场景训练必须多进程；而"每条 spec 都 ``build_env`` + ``close``"的路径实测残留
≈3.5 MB/spec（validator 长跑 RSS 1.2GB → 4GB+）。因此本模块实现 §8.1 的两条防线：

1. **常驻 env（默认路径）**：worker 内的 env 只按**地图相关字段**重建
   （blocks / ego spawn 覆盖 / random_traffic / seed_pool；**无 seed_pool 时 seed 也进入
   复用键** → 逐 spec 重建，退化为下面的 B 路径）。其余 spec 变化走
   ``env.bind_spec(spec)`` + ``reset``；``traffic_density`` 在 reset 前写入
   ``env.config["traffic_density"]``——已核实 ``PGTrafficManager.before_reset``
   会重读 ``engine.global_config["traffic_density"]``（``traffic_manager.py:130``），
   且 ``env.config is engine.global_config``（``base_env.py:546`` 断言）。
   ``store_map=False`` 由 ``build_env`` 固定，地图每次 reset 重建但**不泄漏 env 实例**。
2. **按 spec 数回收（兜底）**：``recycle_every_specs``（默认 150，与
   ``config/eval.yaml`` 同口径）判定池级重建，进程池整体重启把 RSS 拉回基线。

对外接口
--------
- :class:`VectorEnvPool`：``reset(specs) / reset_workers(ids, specs) / step(actions) /
  close() / recycle_stats()``，``soak(specs)`` 打印/返回"RSS vs spec 数"曲线（验收用）。
  ``reset`` = 整池 reset（episode 起点）；``reset_workers`` 只 reset 指定 worker
  （终局回收：其余 worker 的 episode 状态原样继续），池级回收改由调用方在整池
  reset 边界用 :attr:`VectorEnvPool.recycle_due` 触发（见方法 docstring）。
- :class:`ResidentEnv`：单 worker 的常驻 env 逻辑（不依赖 IPC；可单进程直接实例化测试）。
- :class:`SpecAssigner`：按 env 复用 key 分组的确定性 spec 分配（最小化 env 重建）。
- :func:`env_key` / :func:`mem_available_mb` / :func:`process_rss_mb`。

命令返回的 record dict：``obs``（观测 dict，附加 ``pose``）/ ``pose`` / ``reward`` /
``terminated`` / ``truncated`` / ``env_builds`` / ``resets`` / ``prev_action``（本策略步
``(ds, dθ)``，float32 ``(2,)``；reset 后为 0）/ ``router_labels``（float32 ``(8,)``，
顺序 = ``label_order``）/ ``has_router_labels``（bool；标签模块不可用时 False 降级）/
``info``（仍保留 ``router_labels`` 供既有消费方）；口径与 ``pipeline/trainer.py`` 一致。

策略动作注入（§8.4）：``step(actions, policy_actions=...)`` 时 worker 在构建步后观测**之前**
把 ``(ds, dθ)`` 写进 ``env.prev_policy_action``（ego 通道 reserved 6:8 承载），
``policy_actions=None`` 时保持 worker 内现值（reset 后为 0，兼容旧调用方）。

内存纪律（§8.1）：``mem_floor_mb`` 给定时，创建池之前程序化检查 ``MemAvailable``，
不足直接报错（eval/train 互斥由 config/eval.yaml 的 ``train_pool_policy`` 约束）。

动作口径：``step`` 接受**env-step（0.1 s）动作**（``[steer, throttle]``，2 维），
也接受长度 k 的动作序列（如经 ``LqrTracker``/``ExactTracker`` 展开的一整个 0.5 s
策略步），worker 内顺序执行，返回**最后一步**的观测与累计回报。
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from env.metadrive_env import (
    _spec_blocks,
    _spec_ego_overrides,
    _spec_random_traffic,
    _spec_seed,
    _spec_traffic_density,
    build_env,
)
from pipeline.gl_runtime import ensure_gl_library_path

logger = logging.getLogger(__name__)

__all__ = [
    "mem_available_mb",
    "process_rss_mb",
    "env_key",
    "load_supervised_labels",
    "ResidentEnv",
    "VectorEnvPool",
    "SpecAssigner",
]

#: 池级回收阈值（每个 worker 的 spec 数）；0/None = 不回收。
DEFAULT_RECYCLE_EVERY_SPECS = 150
#: 单 env 进程经验 RSS（P1a 实测 ≈0.65GB，用于创建前告警）
ENV_RSS_PER_WORKER_MB = 650.0
#: 默认观测配置：None → ObservationBuilder 默认（ego/od/ld/nav/signal + 6 帧历史）
DEFAULT_OBS_CONFIG: Optional[dict] = None
#: router 受监督标签兜底顺序（``config/model.yaml`` 缺失/不可解析时的固定副本）。
SUPERVISED_LABELS: tuple[str, ...] = (
    "cutin_active",
    "cutout_active",
    "crowded",
    "car_following",
    "on_curve",
    "merging",
    "roundabout_near",
    "near_intersection",
)
#: 默认模型配置路径（与 ``pipeline.trainer._MODEL_CONFIG_DEFAULT`` 同口径）。
MODEL_CONFIG_PATH = os.path.join("config", "model.yaml")


def load_supervised_labels(path: Optional[str] = None) -> tuple[str, ...]:
    """读取 ``config/model.yaml::moe.router.supervised_labels``（8 标签顺序单一出处）。

    只在 pool/Renv 初始化时调用（不做模块级读取，保持导入无副作用）；YAML 依赖或文件
    缺失、长度非 8 时回退 :data:`SUPERVISED_LABELS`——配置问题绝不阻塞训练。
    """
    candidates: list[str] = []
    if path:
        candidates.append(str(path))
    else:
        candidates.append(MODEL_CONFIG_PATH)  # CWD 相对（兼容 trainer 口径）
        candidates.append(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_CONFIG_PATH)
        )
    for candidate in candidates:
        try:
            import yaml  # type: ignore

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            labels = payload.get("moe", {}).get("router", {}).get("supervised_labels")
        except Exception:  # noqa: BLE001
            continue
        if labels:
            order = tuple(str(item) for item in labels)
            if len(order) == 8:
                return order
            logger.warning("supervised_labels 必须是 8 个（实际 %d），忽略 %s", len(order), candidate)
    return SUPERVISED_LABELS


# ======================================================================================
# 系统资源 / spec 访问
# ======================================================================================
def mem_available_mb() -> float:
    """``/proc/meminfo`` 的 ``MemAvailable``（MB）；读取失败返回 ``inf``（放行）。"""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - 非 Linux / 容器限制下不阻塞启动
        return float("inf")
    return float("inf")


def process_rss_mb(pid: Optional[int] = None) -> float:
    """进程常驻内存（MB）；默认当前进程。读取失败返回 ``nan``。"""
    target = os.getpid() if pid is None else int(pid)
    try:
        with open(f"/proc/{target}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def env_key(spec: Any, *, seed_pool: Optional[Sequence[int]] = None) -> tuple:
    """env 复用键：只有**构建期固化**的字段参与（密度不参与）。

    - ``blocks`` / ego spawn 覆盖 / ``random_traffic``：``build_env`` 构造期固化；
    - ``seed_pool``：``num_scenarios`` 与 ``start_seed`` 固化在 config 里；
    - ``seed``：**给定时不参与**（池覆盖多个 seed，reset 时切换场景即可）；
      **未给定时参与**——单 seed env 的 ``num_scenarios=1``，reset 到别的 seed 会触发
      MetaDrive 的 ``start_index <= seed`` 断言，因此必须逐 spec 重建
      （§8.1-B 路径，由 ``recycle_every_specs``/``recycle_stats`` 兜住残留增长）；
    - ``traffic_density``：reset 前注入即可（§8.1-A，已验证上游 before_reset 重读）。
    """
    pool = None if seed_pool is None else (int(seed_pool[0]), int(seed_pool[1]))
    return (
        _spec_blocks(spec),
        repr(_spec_ego_overrides(spec)),
        bool(_spec_random_traffic(spec)),
        pool,
        None if pool is not None else _spec_seed(spec),
    )


def _normalize_env_steps(actions: Any) -> list:
    """把 step 输入统一成 ``[[steer, throttle], ...]``（单个动作或动作序列）。"""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"动作需要 2 维 (steer, throttle)，收到 shape={arr.shape}")
        return [arr]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [row for row in arr]
    raise ValueError(f"step 动作需要 (2,) 或 (k,2)，收到 shape={arr.shape}")


def _sanitize(value: Any, dropped: set) -> Any:
    """把 info 转换成可 pickle 的纯 Python 结构；无法转换的对象丢弃并记录 key。

    为什么需要：MetaDrive 的 step info 由多个 manager 的 step_info 合并而来，可能携带
    引擎对象（不可 pickle，跨进程 ``conn.send`` 会炸）。下游奖励模型需要的
    ``crash*``/``out_of_road``/``arrive_dest``/``route_completion``/``velocity`` 等
    都是标量/布尔，本函数保证这些字段原样通过。
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, dropped) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize(item, dropped) for key, item in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    dropped.add(type(value).__name__)
    return None


def sanitize_info(info: Any) -> dict:
    """info dict → 可跨进程传输的纯 Python dict（不可转换的键被丢弃并 warning）。"""
    if not isinstance(info, dict):
        return {}
    dropped: set = set()
    out = {str(key): _sanitize(value, dropped) for key, value in info.items()}
    if dropped:
        logger.warning("info 含不可序列化的值类型 %s，已丢弃对应键（仅提示一次/步）", sorted(dropped))
    return out


# ======================================================================================
# 单进程常驻 env（worker 逻辑，可在主进程直接实例化做单进程验证）
# ======================================================================================
class ResidentEnv:
    """常驻 env：按 :func:`env_key` 复用，密度注入，可选观测构建。

    ``seed_pool`` 为 ``None`` 时复用键含 seed（单 seed env 不能 reset 到别的 seed），
    多 seed 常驻复用必须给出覆盖所有 spec seed 的 ``seed_pool``。
    """

    def __init__(
        self,
        worker_id: int = 0,
        *,
        obs_config: Optional[dict] = None,
        build_obs: bool = True,
        seed_pool: Optional[Sequence[int]] = None,
        lru_size: int = 0,
        traffic_density: Optional[float] = None,
        label_order: Optional[Sequence[str]] = None,
    ):
        self.worker_id = int(worker_id)
        self.obs_config = obs_config
        self.build_obs = bool(build_obs)
        self.seed_pool = None if seed_pool is None else (int(seed_pool[0]), int(seed_pool[1]))
        self.lru_size = int(lru_size or 0)
        self.traffic_density = None if traffic_density is None else float(traffic_density)
        if label_order is None:
            # 默认 = config/model.yaml::moe.router.supervised_labels（8 标签，固定顺序）
            self.label_order: Optional[tuple[str, ...]] = load_supervised_labels()
        else:
            order = tuple(str(name) for name in label_order)
            self.label_order = order if order else None  # 空序列 = 明确关闭标签
        self._labels_warned = False
        self._prev_warned = False

        self._env: Any = None
        self._builder: Any = None
        self._key: Optional[tuple] = None
        self._spec: Any = None
        self._seed: Optional[int] = None
        self.env_builds = 0
        self.resets = 0
        self.specs_done = 0

    # ------------------------------------------------------------------ env 生命周期
    @property
    def env(self) -> Any:
        return self._env

    def _density_for(self, spec: Any) -> float:
        return float(self.traffic_density) if self.traffic_density is not None else float(
            _spec_traffic_density(spec)
        )

    def _ensure_env(self, spec: Any) -> Any:
        """按 env_key 复用；命中则只注入密度（不重建），未命中则重建。"""
        key = env_key(spec, seed_pool=self.seed_pool)
        density = self._density_for(spec)
        if self._env is not None and key == self._key:
            # §8.1-A：before_reset 会重读 global_config["traffic_density"]（traffic_manager.py:130）
            self._env.config["traffic_density"] = density
        else:
            self.close()
            self._env = build_env(
                spec,
                traffic_density=density,
                use_render=False,
                lru_size=self.lru_size,
                seed_pool=self.seed_pool,
            )
            self._builder = None
            self._key = key
            self.env_builds += 1
        self._env.bind_spec(spec)
        self._spec = spec
        self._seed = _spec_seed(spec)
        return self._env

    def _build_obs(self) -> Optional[dict]:
        if not self.build_obs:
            return None
        if self._builder is None:
            from env.obs.builder import ObservationBuilder  # 惰性导入：主进程不必拉起 obs 依赖

            self._builder = ObservationBuilder(self.obs_config)
        return self._builder.build(self._env, self._spec)

    def close(self) -> None:
        """关闭 env（保留 worker 存活，下次 reset 会重建）。"""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:  # noqa: BLE001 - close 失败不应吞掉业务错误
                logger.warning("worker=%d env.close 异常：%r", self.worker_id, exc)
            self._env = None
        self._builder = None
        self._key = None

    # ------------------------------------------------------------------ 对外命令
    def reset(self, spec: Any = None) -> dict:
        """reset 到一个 spec；返回记录 dict（含 obs/info/构建计数）。

        ``spec=None`` = 复用当前 spec（仅限已经 reset 过的 worker；配合
        :meth:`VectorEnvPool.reset_workers` 的 ``specs=None`` 语义）。
        """
        if spec is None:
            if self._spec is None:
                raise ValueError("首次 reset 必须给出 spec（无法复用）")
            spec = self._spec
        env = self._ensure_env(spec)
        _, info = env.reset(seed=self._seed)
        self.resets += 1
        self.specs_done += 1
        # §8.4：新 episode 首帧没有上一策略步动作 → ego reserved 6:8 保持 0
        env.prev_policy_action = np.zeros(2, dtype=np.float64)
        return self._record(info=info, reward=0.0, terminated=False, truncated=False)

    def step(self, actions: Any, prev_action: Optional[Sequence[float]] = None) -> dict:
        """执行一个动作或一串 env-step 动作（0.5 s 策略步），返回最后一步结果。

        ``prev_action`` = 本策略步 ``(ds, dθ)``（§8.4）；给定时在步进前写入
        ``env.prev_policy_action``，使**步后观测**的 ego reserved 6:8 承载它。``None``
        （默认）时保持现值（reset 后为 0），兼容只传子步动作的旧调用方。
        """
        if self._env is None:
            raise RuntimeError("ResidentEnv.step 前必须先 reset(spec)")
        if prev_action is not None:
            self._set_prev_policy_action(prev_action)
        steps = _normalize_env_steps(actions)
        reward_total = 0.0
        terminated = truncated = False
        info: dict = {}
        for action in steps:
            _, reward, terminated, truncated, info = self._env.step([float(action[0]), float(action[1])])
            reward_total += float(reward)
            if terminated or truncated:
                break
        return self._record(info=info, reward=reward_total, terminated=terminated, truncated=truncated)

    def stats(self) -> dict:
        """worker 资源/计数快照。"""
        return {
            "worker": self.worker_id,
            "pid": os.getpid(),
            "rss_mb": process_rss_mb(),
            "env_builds": int(self.env_builds),
            "resets": int(self.resets),
            "specs": int(self.specs_done),
            "last_spec": None if self._spec is None else getattr(self._spec, "id", None),
            "last_seed": self._seed,
            "env_key": None if self._key is None else repr(self._key),
        }

    # ------------------------------------------------------------------ 内部
    def _pose(self) -> np.ndarray:
        """当前 ego 位姿 ``(x, y, theta)`` float32（历史窗口 SE(2) 对齐需要）。"""
        ego = None
        if self._env is not None:
            try:
                ego = self._env.agent
            except Exception:  # noqa: BLE001 - 终局/agent 未就绪
                ego = None
        if ego is None:
            return np.zeros(3, dtype=np.float32)
        return np.array(
            [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)], dtype=np.float32
        )

    def _router_labels(self) -> Optional[np.ndarray]:
        """逐步可观测 router 标签（``label_order`` 为空/标签模块不可用时返回 None）。"""
        if self.label_order is None or self._env is None:
            return None
        try:
            from env.scenario.labels import compute_step_labels

            raw = compute_step_labels(self._env, self._spec)
            return np.asarray([float(raw.get(name, 0.0)) for name in self.label_order], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - 标签失败不应中断训练
            if not self._labels_warned:
                logger.warning("worker=%d router 标签计算失败（后续静默跳过）：%r", self.worker_id, exc)
                self._labels_warned = True
            return None

    def _set_prev_policy_action(self, action: Any) -> None:
        """把策略动作 ``(ds, dθ)`` 写入 env（非法输入告警一次并置 0，绝不中断训练）。"""
        vector: Optional[np.ndarray] = None
        try:
            candidate = np.asarray(action, dtype=np.float64).reshape(-1)
            if candidate.shape[0] == 2:
                vector = candidate
        except (TypeError, ValueError):
            vector = None
        if vector is None:
            if not self._prev_warned:
                logger.warning(
                    "worker=%d prev_action 非法（需要 2 维 (ds,dtheta)，收到 %r），置 0", self.worker_id, action
                )
                self._prev_warned = True
            vector = np.zeros(2, dtype=np.float64)
        self._env.prev_policy_action = vector

    def _prev_action(self) -> np.ndarray:
        """``env.prev_policy_action`` 的 float32 ``(2,)`` 快照（缺失/非法/非 2 维 → 0）。"""
        try:
            vector = np.asarray(getattr(self._env, "prev_policy_action", None), dtype=np.float32).reshape(-1)
            if vector.shape[0] == 2:
                return vector
        except (TypeError, ValueError):
            pass
        return np.zeros(2, dtype=np.float32)

    def _record(self, *, info: Any, reward: float, terminated: bool, truncated: bool) -> dict:
        spec = self._spec
        pose = self._pose()
        obs = self._build_obs()
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["pose"] = pose.copy()  # trainer/缓冲的历史重建直接可用（NON_OBS_KEYS）
        record = {
            "worker": self.worker_id,
            "spec_id": int(getattr(spec, "id", -1)) if spec is not None else -1,
            "seed": self._seed,
            "obs": obs,
            "pose": pose,
            "info": sanitize_info(info),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "env_builds": int(self.env_builds),
            "resets": int(self.resets),
        }
        labels = self._router_labels()
        record["router_labels"] = labels
        record["has_router_labels"] = bool(labels is not None)
        record["prev_action"] = self._prev_action()
        if labels is not None:
            record["info"]["router_labels"] = labels  # 兼容既有 info 消费口径
        return record


# ======================================================================================
# spawn worker 入口
# ======================================================================================
def _worker_main(conn: Connection, worker_id: int, cfg: dict) -> None:
    """spawn 子进程入口：命令循环（reset/step/stats/close）。"""
    state: Optional[ResidentEnv] = None
    try:
        state = ResidentEnv(worker_id, **cfg)
        conn.send((True, {"ready": True, **state.stats()}))
        while True:
            try:
                command, payload = conn.recv()
            except EOFError:
                break
            if command == "reset":
                conn.send((True, state.reset(payload)))
            elif command == "step":
                # 新 payload = (子步动作, 策略动作 (ds,dθ) 数组 | None)；旧 payload = 纯子步动作
                if (
                    isinstance(payload, tuple)
                    and len(payload) == 2
                    and (payload[1] is None or np.ndim(payload[1]) >= 1)
                ):
                    actions, prev_action = payload
                else:
                    actions, prev_action = payload, None
                conn.send((True, state.step(actions, prev_action)))
            elif command == "stats":
                conn.send((True, state.stats()))
            elif command == "close":
                conn.send((True, True))
                break
            else:
                conn.send((False, f"未知命令：{command!r}"))
    except BaseException as exc:  # noqa: BLE001 - 任何失败都回报给父进程而不是静默退出
        try:
            conn.send((False, f"worker {worker_id} 异常：{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if state is not None:
            try:
                state.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ======================================================================================
# 进程池
# ======================================================================================
class VectorEnvPool:
    """spawn 子进程向量环境池（常驻 env + 按 spec 数池级回收）。"""

    def __init__(
        self,
        num_workers: int = 4,
        *,
        seed_pool: Optional[Sequence[int]] = None,
        obs_config: Optional[dict] = None,
        build_obs: bool = True,
        recycle_every_specs: Optional[int] = DEFAULT_RECYCLE_EVERY_SPECS,
        lru_size: int = 0,
        traffic_density: Optional[float] = None,
        label_order: Optional[Sequence[str]] = None,
        mem_floor_mb: Optional[float] = None,
        start_timeout: float = 180.0,
        recv_timeout: float = 600.0,
        auto_restart_dead: bool = True,
    ):
        """
        Args:
            num_workers: worker 数（每个 worker 一个 engine；§8.1 建议 ≤4，孤立时上限 8）。
            seed_pool: ``(start, stop)``；常驻复用多 seed 场景的必要条件（见 ``build_env``）。
                为 ``None`` 时 env 按 seed 重建（§8.1-B 路径，配合 ``recycle_every_specs``）。
            obs_config: ``ObservationBuilder`` 配置；``None`` = 默认通道与 6 帧历史。
            build_obs: False 时只跑 env 生命周期（内存/吞吐测量）。
            recycle_every_specs: 池级回收阈值（每个 worker 处理这么多 spec 后重建进程池）；
                ``None``/0 关闭。
            lru_size: 传给 ``build_env`` 的地图 LRU（默认 0，见 config/env.yaml）。
            traffic_density: 覆盖所有 spec 的密度（``None`` = 逐 spec 注入）。
            label_order: router 标签顺序；``None``（默认）= 读
                ``config/model.yaml::moe.router.supervised_labels``（8 标签）；空序列 =
                不算标签（零开销）。worker 每步附 record 级 ``router_labels`` /
                ``has_router_labels``（并保留 ``info["router_labels"]``）。
            mem_floor_mb: 创建池前要求的最小 ``MemAvailable``（MB）；不足直接报错。
            start_timeout / recv_timeout: 子进程握手 / 单命令等待超时（秒）。
            auto_restart_dead: reset 前发现 worker 已死时自动重启（计入回收次数）。
        """
        if int(num_workers) < 1:
            raise ValueError(f"num_workers 必须 >= 1，收到 {num_workers}")
        if mem_floor_mb is not None:
            available = mem_available_mb()
            if available < float(mem_floor_mb):
                raise RuntimeError(
                    f"MemAvailable={available:.0f}MB < mem_floor_mb={float(mem_floor_mb):.0f}MB；"
                    "按 §8.1 先暂停/缩减其它 env-heavy 任务（eval/train 互斥）"
                )
        self.num_workers = int(num_workers)
        self.seed_pool = None if seed_pool is None else (int(seed_pool[0]), int(seed_pool[1]))
        self.build_obs = bool(build_obs)
        self.recycle_every_specs = None if not recycle_every_specs else max(1, int(recycle_every_specs))
        self.start_timeout = float(start_timeout)
        self.recv_timeout = float(recv_timeout)
        self.auto_restart_dead = bool(auto_restart_dead)
        # 默认标签顺序在父进程解析一次（worker 收到显式 tuple，不在子进程重复读配置）
        if label_order is None:
            label_order = load_supervised_labels()
        resolved_order = tuple(str(name) for name in label_order)
        self.worker_cfg = {
            "obs_config": obs_config if obs_config is not None else DEFAULT_OBS_CONFIG,
            "build_obs": self.build_obs,
            "seed_pool": self.seed_pool,
            "lru_size": int(lru_size or 0),
            "traffic_density": None if traffic_density is None else float(traffic_density),
            "label_order": resolved_order if resolved_order else None,
        }
        estimated = self.num_workers * ENV_RSS_PER_WORKER_MB
        if mem_available_mb() < estimated:
            logger.warning(
                "MemAvailable=%.0fMB < %d×%.0fMB 估算需求；若同机有其它 env-heavy 任务请先停止（§8.1）",
                mem_available_mb(),
                self.num_workers,
                ENV_RSS_PER_WORKER_MB,
            )

        self._ctx = get_context("spawn")
        self._conns: list[Optional[Connection]] = [None] * self.num_workers
        self._procs: list[Any] = [None] * self.num_workers
        #: 各 worker 最近一次 reset 生效的 spec（``reset_workers(specs=None)`` 复用 / 死进程自愈用）。
        self._worker_specs: list[Any] = [None] * self.num_workers
        #: 最近一次 :meth:`reset_workers` 里因死进程自愈被**额外** reset 的 worker 记录
        #: （``{worker_index: record}``；正常情况为空 dict）。
        self.last_forced_resets: dict = {}
        self._active = 0
        self.generation = 0
        self.recycles = 0
        self._specs_since_recycle = 0
        self._cum_specs = 0
        self._cum_builds = 0
        self._closed = False
        self._launch_all()

    # ------------------------------------------------------------------ 生命周期
    def _launch(self, index: int) -> None:
        # spawn 子进程继承父进程 environ：worker 建 engine 前必须拿到 venv 本地 glvnd 路径，
        # 否则 Panda3D n_pipes=0 → MetaDrive "Known Pipes" IndexError（pooled 运行阻塞性崩溃）。
        ensure_gl_library_path()
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child, index, self.worker_cfg),
            daemon=True,
            name=f"envpool-{index}",
        )
        proc.start()
        child.close()
        try:
            ok, payload = self._recv_from(parent, proc, self.start_timeout)
        except BaseException:
            proc.terminate()
            proc.join(timeout=5.0)
            parent.close()
            raise
        if not ok:
            proc.terminate()
            proc.join(timeout=5.0)
            parent.close()
            raise RuntimeError(f"worker {index} 启动失败：{payload}")
        self._conns[index] = parent
        self._procs[index] = proc

    def _launch_all(self) -> None:
        for index in range(self.num_workers):
            self._launch(index)

    def _recv_from(self, conn: Connection, proc: Any, timeout: float) -> tuple:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"等待 worker pid={proc.pid} 超时（{timeout}s）")
            if conn.poll(min(remaining, 1.0)):
                return conn.recv()
            if not proc.is_alive():
                raise RuntimeError(f"worker pid={proc.pid} 提前退出（exitcode={proc.exitcode}）")

    def _request(self, index: int, command: str, payload: Any = None) -> Any:
        conn = self._conns[index]
        proc = self._procs[index]
        if conn is None or proc is None or not proc.is_alive():
            raise RuntimeError(f"worker {index} 不可用（pid={None if proc is None else proc.pid}）")
        conn.send((command, payload))
        ok, reply = self._recv_from(conn, proc, self.recv_timeout)
        if not ok:
            raise RuntimeError(f"worker {index} 执行 {command} 失败：{reply}")
        return reply

    def _ensure_alive(self) -> list:
        """reset 前自检：worker 死亡时重启（长跑自愈；计入回收计数）。

        Returns:
            被重启的 worker 索引列表（无死亡则为空）；这些 worker 尚未 reset，
            调用方必须对它们重新 reset 后才能 step。
        """
        dead = [i for i, proc in enumerate(self._procs) if proc is None or not proc.is_alive()]
        if not dead:
            return []
        if not self.auto_restart_dead:
            raise RuntimeError(f"worker {dead} 已退出；请调用 restart() 或关闭 auto_restart_dead")
        for index in dead:
            logger.warning("worker %d 已退出，正在重启（§8.1 长跑自愈）", index)
            self._close_worker(index)
            self._launch(index)
        self.recycles += 1
        return dead

    def _close_worker(self, index: int) -> None:
        conn, proc = self._conns[index], self._procs[index]
        if conn is not None and proc is not None and proc.is_alive():
            try:
                conn.send(("close", None))
                if conn.poll(5.0):
                    conn.recv()
            except Exception:  # noqa: BLE001
                pass
        if proc is not None:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        if conn is not None:
            conn.close()
        self._conns[index] = None
        self._procs[index] = None

    def restart(self) -> None:
        """池级回收：关闭全部 worker（释放 RSS）后按同配置重建。"""
        self._accumulate_stats()
        for index in range(self.num_workers):
            self._close_worker(index)
        self.recycles += 1
        self.generation += 1
        self._specs_since_recycle = 0
        self._launch_all()
        logger.info("VectorEnvPool 重建完成：generation=%d recycles=%d", self.generation, self.recycles)

    def close(self) -> None:
        """关闭池（幂等）。"""
        if self._closed:
            return
        self._accumulate_stats()
        for index in range(self.num_workers):
            self._close_worker(index)
        self._closed = True
        self._active = 0

    def __enter__(self) -> "VectorEnvPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ 数据面
    def _prepare_specs(self, specs: Any) -> list:
        """规范 spec 列表：单个 spec = 广播；序列长度 ≤ num_workers（长于则报错）。"""
        if isinstance(specs, (list, tuple)):
            seq = list(specs)
            if len(seq) > self.num_workers:
                raise ValueError(
                    f"reset 收到 {len(seq)} 条 spec > num_workers={self.num_workers}；"
                    "请用 SpecAssigner 分批（按 env_key 分组可避免重复建 env）"
                )
        else:
            seq = [specs] * self.num_workers
        if not seq:
            raise ValueError("reset 需要至少一条 spec")
        if self.seed_pool is not None:
            low, high = self.seed_pool
            for spec in seq:
                seed = _spec_seed(spec)
                if not low <= seed < high:
                    raise ValueError(
                        f"spec.seed={seed} 不在 seed_pool=({low}, {high}) 内；常驻复用要求池覆盖所有 seed"
                    )
        return seq

    @property
    def recycle_due(self) -> bool:
        """``recycle_every_specs`` 阈值已到（需在整池 reset 边界回收 RSS）；未配置为 False。

        逐 worker reset 不会在 :meth:`reset_workers` 内重启进程池（否则会清掉未指定
        worker 的 episode 状态）；计数照常累计，达到阈值后本属性为 True，由调用方
        在安全边界（整池 :meth:`reset`）触发真正的回收。
        """
        if self.recycle_every_specs is None:
            return False
        return self._specs_since_recycle >= self.recycle_every_specs * self.num_workers

    def _maybe_recycle(self) -> None:
        if not self.recycle_due:
            return
        logger.info(
            "达到 recycle_every_specs=%d（累计 %d 条 spec），重建进程池以回收 RSS",
            self.recycle_every_specs,
            self._specs_since_recycle,
        )
        self.restart()

    def reset(self, specs: Any) -> list:
        """reset 一批 spec（每 worker 一条；单个 spec = 广播给全部 worker）。

        Returns:
            与生效 worker 等长的记录列表（``obs/info/reward/terminated/truncated/...``）。
        """
        if self._closed:
            raise RuntimeError("VectorEnvPool 已关闭")
        seq = self._prepare_specs(specs)
        self._ensure_alive()
        self._maybe_recycle()
        for index, spec in enumerate(seq):
            self._conns[index].send(("reset", spec))  # type: ignore[union-attr]
            self._worker_specs[index] = spec
        records = []
        for index in range(len(seq)):
            ok, reply = self._recv_from(self._conns[index], self._procs[index], self.recv_timeout)  # type: ignore[arg-type]
            if not ok:
                raise RuntimeError(f"worker {index} reset 失败：{reply}")
            records.append(reply)
        self._active = len(seq)
        self._specs_since_recycle += len(seq)
        self._cum_specs += len(seq)
        return records

    def reset_workers(self, worker_ids: Iterable[int], specs: Any = None) -> list:
        """只 reset 指定 worker（终局回收）；其余 worker 的 env/episode 状态原样继续。

        Args:
            worker_ids: worker 索引集合（0..num_workers-1 的非空子集，不允许重复）。
            specs: ``None`` = 复用各 worker 上次 reset 的 spec（要求该 worker reset 过）；
                单个 spec = 广播给全部指定 worker；序列 = 与 ``worker_ids`` 等长逐 worker。

        池级回收（``recycle_every_specs``）**不会**在本方法内重启进程池——否则未指定
        worker 的 episode 状态会被清掉。spec 计数照常累计，阈值到达后
        :attr:`recycle_due` 为 True，调用方应在下一个整池 :meth:`reset` 边界回收。

        死亡自愈：``auto_restart_dead`` 重启的 worker 若不在 ``worker_ids`` 内，会用其
        上次 spec 强制 reset，记录放入 :attr:`last_forced_resets`（``{index: record}``）；
        调用方必须把它当作该 worker 的 reset 记录消费（否则其 step 会因未 reset 报错）。

        Returns:
            与 ``worker_ids`` 等长的记录列表（顺序同入参）。
        """
        if self._closed:
            raise RuntimeError("VectorEnvPool 已关闭")
        ids = [int(index) for index in worker_ids]
        if not ids:
            raise ValueError("reset_workers 需要至少一个 worker id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"reset_workers 收到重复 worker id：{ids}")
        for index in ids:
            if not 0 <= index < self.num_workers:
                raise ValueError(f"worker id={index} 越界（num_workers={self.num_workers}）")
        given = specs is not None
        if specs is None:
            seq = [self._worker_specs[index] for index in ids]
            for index, spec in zip(ids, seq):
                if spec is None:
                    raise ValueError(f"worker {index} 尚未 reset 过，specs=None 无法复用其 spec")
        elif isinstance(specs, (list, tuple)):
            seq = list(specs)
            if len(seq) != len(ids):
                raise ValueError(f"specs 长度 {len(seq)} != worker_ids 长度 {len(ids)}")
        else:
            seq = [specs] * len(ids)
        if given and self.seed_pool is not None:
            low, high = self.seed_pool
            for spec in seq:
                seed = _spec_seed(spec)
                if not low <= seed < high:
                    raise ValueError(
                        f"spec.seed={seed} 不在 seed_pool=({low}, {high}) 内；常驻复用要求池覆盖所有 seed"
                    )

        restarted = self._ensure_alive()
        self.last_forced_resets = {}
        forced_ids = [
            index for index in restarted if index not in ids and self._worker_specs[index] is not None
        ]
        for index, spec in zip(ids, seq):
            self._conns[index].send(("reset", spec))  # type: ignore[union-attr]
            self._worker_specs[index] = spec
        records = []
        for index in ids:
            ok, reply = self._recv_from(self._conns[index], self._procs[index], self.recv_timeout)  # type: ignore[arg-type]
            if not ok:
                raise RuntimeError(f"worker {index} reset 失败：{reply}")
            records.append(reply)
        for index in forced_ids:  # 死进程重启但未被请求 reset：用上次 spec 强制 reset
            self._conns[index].send(("reset", self._worker_specs[index]))  # type: ignore[union-attr]
        for index in forced_ids:
            ok, reply = self._recv_from(self._conns[index], self._procs[index], self.recv_timeout)  # type: ignore[arg-type]
            if not ok:
                raise RuntimeError(f"worker {index} 强制 reset 失败：{reply}")
            self.last_forced_resets[index] = reply
            logger.warning("worker %d 自愈后已按上次 spec 强制 reset（§8.1）", index)
        if given:
            self._specs_since_recycle += len(ids)
            self._cum_specs += len(ids)
        return records

    def _normalize_action_batch(self, actions: Any) -> list:
        """把 step 输入规范成 per-worker 列表（长度 = active）。

        判别规则（写死以避免歧义）：
        - ``(active, k, 2)`` ndarray → per-worker 多步动作；
        - list/tuple 且长度 = active、元素为 array-like → per-worker（每项可为单步或序列）；
        - 其它（单个 ``(2,)``、单个 worker 的 ``(k,2)``、单个 ``(k,2)`` list）→ 广播给所有 worker。
        """
        if isinstance(actions, np.ndarray) and actions.ndim == 3:
            batch = list(actions)
        elif (
            isinstance(actions, (list, tuple))
            and len(actions) == self._active
            and all(np.ndim(item) >= 1 for item in actions)
        ):
            batch = list(actions)
        else:
            batch = [actions] * self._active
        if len(batch) != self._active:
            raise ValueError(f"step 需要 {self._active} 个动作（或广播单个），收到 {len(batch)}")
        return batch

    def _normalize_policy_batch(self, policy_actions: Any) -> list:
        """规范策略动作：per-worker 的 float64 ``(2,)`` 列表；``None`` → 全 None（不注入）。

        判别规则与 :meth:`_normalize_action_batch` 一致：``(active,2)`` / 长度 = active 的
        数组列表 → 逐 worker；单个 ``(2,)`` → 广播。
        """
        if policy_actions is None:
            return [None] * self._active
        if isinstance(policy_actions, np.ndarray) and policy_actions.ndim == 2:
            batch = [np.asarray(row, dtype=np.float64).reshape(-1) for row in policy_actions]
        elif (
            isinstance(policy_actions, (list, tuple))
            and len(policy_actions) == self._active
            and all(np.ndim(item) >= 1 for item in policy_actions)
        ):
            batch = [np.asarray(item, dtype=np.float64).reshape(-1) for item in policy_actions]
        else:
            batch = [np.asarray(policy_actions, dtype=np.float64).reshape(-1)] * self._active
        if len(batch) != self._active:
            raise ValueError(f"policy_actions 需要 {self._active} 项（或广播单个），收到 {len(batch)}")
        for item in batch:
            if item.shape[0] != 2:
                raise ValueError(f"policy_actions 每项需要 2 维 (ds, dtheta)，收到 shape={item.shape}")
        return batch

    def step(self, actions: Any, policy_actions: Any = None) -> list:
        """执行一批动作（每 worker 一项；若无法判定为批量则广播）。

        每项可为 ``(2,)`` 单步动作或 ``(k,2)`` 动作序列（worker 内顺序执行）。
        ``policy_actions``：本策略步的 ``(ds, dθ)``（``(2,)`` 广播或 ``(active,2)`` 逐
        worker；``None`` = 不注入，worker 保持现值）。worker 在构建步后观测**之前**写入
        ``env.prev_policy_action``，故 record 的 ``prev_action`` 与 ``obs["ego"][0, 6:8]``
        一致（§8.4）。
        """
        if self._closed:
            raise RuntimeError("VectorEnvPool 已关闭")
        if self._active <= 0:
            raise RuntimeError("step 前必须先 reset(specs)")
        seq = self._normalize_action_batch(actions)
        policy_seq = self._normalize_policy_batch(policy_actions)
        for index, (action, policy_action) in enumerate(zip(seq, policy_seq)):
            self._conns[index].send(("step", (action, policy_action)))  # type: ignore[union-attr]
        records = []
        for index in range(len(seq)):
            ok, reply = self._recv_from(self._conns[index], self._procs[index], self.recv_timeout)  # type: ignore[arg-type]
            if not ok:
                raise RuntimeError(f"worker {index} step 失败：{reply}")
            records.append(reply)
        return records

    # ------------------------------------------------------------------ 监控
    def _accumulate_stats(self) -> None:
        for index, proc in enumerate(self._procs):
            if proc is None or not proc.is_alive():
                continue
            try:
                stats = self._request(index, "stats")
            except Exception:  # noqa: BLE001
                continue
            self._cum_builds += int(stats.get("env_builds", 0))

    def recycle_stats(self) -> dict:
        """池级监控快照：generation / 累计 spec / env 重建数 / 各 worker RSS。"""
        workers = []
        rss_total = 0.0
        builds_total = 0
        for index, proc in enumerate(self._procs):
            alive = proc is not None and proc.is_alive()
            stats: dict = {"worker": index, "pid": None if proc is None else proc.pid, "alive": alive}
            if alive:
                try:
                    stats.update(self._request(index, "stats"))
                    rss_total += float(stats.get("rss_mb") or 0.0)
                    builds_total += int(stats.get("env_builds", 0))
                except Exception as exc:  # noqa: BLE001
                    stats["error"] = repr(exc)
            workers.append(stats)
        return {
            "generation": int(self.generation),
            "num_workers": int(self.num_workers),
            "recycles": int(self.recycles),
            "recycle_every_specs": self.recycle_every_specs,
            "specs_since_recycle": int(self._specs_since_recycle),
            "specs_total": int(self._cum_specs),
            "env_builds_total": int(self._cum_builds + builds_total),
            "rss_mb_total": float(rss_total),
            "rss_mb_parent": float(process_rss_mb()),
            "mem_available_mb": float(mem_available_mb()),
            "workers": workers,
        }

    def soak(
        self,
        specs: Iterable[Any],
        *,
        steps: int = 2,
        action: Sequence[float] = (0.0, 0.0),
        report: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """soak 测试：把一批 spec 依次压进池，返回"RSS vs spec 数"曲线。

        每个 batch = ``num_workers`` 条 spec；每条 spec 做 reset + ``steps`` 个 env step。
        常驻路径（同 env_key + seed_pool 覆盖）下应看到 ``env_builds`` 几乎不涨、
        RSS 平直；池级回收处 RSS 回落（锯齿 = 回收生效）。
        """
        spec_list = list(specs)
        report = report if report is not None else (lambda message: logger.info("[soak] %s", message))
        samples: list = []
        for start in range(0, len(spec_list), self.num_workers):
            batch = spec_list[start:start + self.num_workers]
            self.reset(batch)
            for _ in range(max(int(steps), 0)):
                self.step([action] * len(batch))
            stats = self.recycle_stats()
            samples.append(
                {
                    "n_specs": min(start + self.num_workers, len(spec_list)),
                    "specs_total": stats["specs_total"],
                    "env_builds_total": stats["env_builds_total"],
                    "rss_mb_total": stats["rss_mb_total"],
                    "generation": stats["generation"],
                    "recycles": stats["recycles"],
                }
            )
            report(
                f"{samples[-1]['n_specs']}/{len(spec_list)} specs, "
                f"rss={samples[-1]['rss_mb_total']:.0f}MB, builds={samples[-1]['env_builds_total']}, "
                f"gen={samples[-1]['generation']}"
            )
        first = samples[0] if samples else {"rss_mb_total": float("nan")}
        last = samples[-1] if samples else {"rss_mb_total": float("nan")}
        return {
            "n_specs": len(spec_list),
            "num_workers": self.num_workers,
            "recycle_every_specs": self.recycle_every_specs,
            "samples": samples,
            "rss_start_mb": float(first["rss_mb_total"]),
            "rss_end_mb": float(last["rss_mb_total"]),
            "delta_mb": float(last["rss_mb_total"] - first["rss_mb_total"]),
            "env_builds_total": int(last.get("env_builds_total", 0)),
            "recycles": int(last.get("recycles", 0)),
        }


# ======================================================================================
# spec 分配
# ======================================================================================
class SpecAssigner:
    """按 :func:`env_key` 分组的确定性 spec 分配器（最小化 env 重建）。

    分配规则：按 key 的分组顺序（首次出现顺序）依次处理；同一组内按 ``(seed, id)``
    排序后**轮转**分给各 worker（``worker w`` 拿 ``group[w::num_workers]``）。
    这样每个 worker 只在组边界处重建 env（重建数 ≈ 组数 × worker 数，而非 spec 数）；
    全程无随机数（确定性优先）。

    ``repeat=True`` 时 :meth:`next_batch` 用完后重开一轮（同样顺序）。
    """

    def __init__(
        self,
        specs: Iterable[Any],
        num_workers: int,
        *,
        seed_pool: Optional[Sequence[int]] = None,
        repeat: bool = False,
    ):
        if int(num_workers) < 1:
            raise ValueError(f"num_workers 必须 >= 1，收到 {num_workers}")
        self.specs = list(specs)
        self.num_workers = int(num_workers)
        self.seed_pool = None if seed_pool is None else (int(seed_pool[0]), int(seed_pool[1]))
        self.repeat = bool(repeat)
        self.epoch = 0
        self._batches: list = []
        self._cursor = 0
        self._build_epoch()

    def _build_epoch(self) -> None:
        groups: dict = {}
        for spec in self.specs:
            groups.setdefault(env_key(spec, seed_pool=self.seed_pool), []).append(spec)
        self._batches = []
        for group in groups.values():
            ordered = sorted(
                group,
                key=lambda spec: (int(getattr(spec, "seed", 0) or 0), int(getattr(spec, "id", 0) or 0)),
            )
            chunks = [ordered[w::self.num_workers] for w in range(self.num_workers)]
            for round_index in range(max((len(chunk) for chunk in chunks), default=0)):
                batch = [chunk[round_index] for chunk in chunks if round_index < len(chunk)]
                if batch:
                    self._batches.append(batch)
        self._cursor = 0

    @property
    def n_batches(self) -> int:
        return len(self._batches)

    def next_batch(self, *, advance_epoch: bool = True) -> list:
        """取下一批（长度 ≤ num_workers）；耗尽后 ``repeat=True`` 时开新 epoch。"""
        if self._cursor >= len(self._batches):
            if not self.repeat:
                raise StopIteration("SpecAssigner 已耗尽（repeat=False）")
            if advance_epoch:
                self.epoch += 1
            self._build_epoch()
        batch = list(self._batches[self._cursor])
        self._cursor += 1
        return batch

    def __iter__(self):
        while True:
            try:
                yield self.next_batch()
            except StopIteration:
                return
