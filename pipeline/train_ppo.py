"""Short PPO canary for the P1 action-distribution pipeline (not a performance claim)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.distributions import Normal

from env.metadrive_env import build_env
from env.obs import build_observation, stack_history
from env.scenario.spec import load_specs
from env.tracking import ExactTracker
from net.model import StructuredPlanner
from pipeline.rollout import batch_from_obs, denorm_action, first_segment_points

STEPS_PER_POLICY = 5


def collect_episode(env, model, spec, *, max_steps: int, device: str) -> Dict[str, torch.Tensor]:
    env.reset(); ego = env.agent; tracker = ExactTracker()
    frames: List[Dict[str, np.ndarray]] = []; poses: List[np.ndarray] = []
    prev_action = np.zeros(4); prev_speed = prev_heading = None
    obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []
    n = 0
    while n < max_steps:
        obs = build_observation(env, prev_action=prev_action, prev_speed=prev_speed, prev_heading=prev_heading, dt=0.5)
        frames.append(obs); poses.append(obs["pose"].copy())
        if len(frames) > 6: frames = frames[-6:]; poses = poses[-6:]
        hist = stack_history(frames, poses)
        batch = batch_from_obs(hist, device=device)
        with torch.no_grad():
            out = model(batch)
            mu = out["action_mu"][0]; logstd = out["action_logstd"][0].clamp(-5, 0)
            std = logstd.exp(); dist = Normal(mu, std)
            a_norm = dist.sample()
            logp = dist.log_prob(a_norm).sum()
            value = out["value"][0, 0]
        a = denorm_action(a_norm[None])[0].detach().cpu().numpy()
        tracker.arm(env, points=first_segment_points(a))
        rc0 = float(getattr(ego.navigation, "route_completion", 0.0))
        done = False; succ = coll = off = False
        for _ in range(STEPS_PER_POLICY):
            _, _, term, trunc, info = env.step([0.0, 0.0]); tracker.apply(env); n += 1
            if info:
                succ = succ or bool(info.get("arrive_dest", False))
                coll = coll or bool(info.get("crash", False))
                off = off or bool(info.get("out_of_road", False))
            if term or trunc: done = True; break
        rc1 = float(getattr(ego.navigation, "route_completion", rc0))
        reward = 10.0 * (rc1 - rc0) + (5.0 if succ else 0.0) - (5.0 if coll else 0.0) - (5.0 if off else 0.0)
        reward -= 0.02 * abs(float(ego.speed) - float(prev_speed if prev_speed is not None else ego.speed))
        obs_list.append(batch); act_list.append(a_norm.detach()); logp_list.append(logp.detach())
        val_list.append(value.detach()); rew_list.append(reward); done_list.append(done)
        prev_action = a; prev_speed = float(ego.speed); prev_heading = float(ego.heading_theta)
        if done: break
    return {
        "batch": obs_list, "actions": act_list, "logp": logp_list, "values": val_list,
        "rewards": rew_list, "dones": done_list,
    }


def gae(rewards, values, dones, gamma=0.95, lam=0.95):
    adv = []; last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_v * (1 - float(dones[t])) - values[t]
        last = delta + gamma * lam * (1 - float(dones[t])) * last
        adv.append(last)
    adv.reverse()
    ret = [a + v for a, v in zip(adv, values)]
    return np.array(adv, np.float32), np.array(ret, np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = StructuredPlanner(hidden=int(ck.get("config", {}).get("hidden", 128)))
    model.load_state_dict(ck["model"]); model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    specs = load_specs(args.specs)[: int(args.episodes)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = []
    for u in range(int(args.updates)):
        data = []
        for spec in specs:
            env = build_env(spec, use_render=False)
            try:
                data.append(collect_episode(env, model, spec, max_steps=int(args.max_steps), device=args.device))
            finally:
                env.close()
        # flatten
        batches, actions, logps, values, rewards, dones = [], [], [], [], [], []
        for d in data:
            batches += d["batch"]; actions += d["actions"]; logps += d["logp"]
            values += d["values"]; rewards += d["rewards"]; dones += d["dones"]
        if not actions:
            continue
        adv, ret = gae(rewards, [float(v) for v in values], dones)
        adv_t = torch.from_numpy(adv).to(args.device); ret_t = torch.from_numpy(ret).to(args.device)
        model.train()
        for _ in range(4):
            out_m = model(batches[0] if len(batches) == 1 else {
                k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]
            })
            mu = out_m["action_mu"]; std = out_m["action_logstd"].clamp(-5, 0).exp()
            dist = Normal(mu, std)
            newlogp = dist.log_prob(torch.stack(actions)).sum(-1)
            oldlogp = torch.stack(logps)
            ratio = (newlogp - oldlogp).exp()
            s1 = ratio * adv_t; s2 = ratio.clamp(0.8, 1.2) * adv_t
            ploss = -torch.min(s1, s2).mean()
            vloss = ((out_m["value"].squeeze(-1) - ret_t) ** 2).mean()
            loss = ploss + 0.5 * vloss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        rec = {"update": u + 1, "transitions": len(actions), "reward_sum": float(np.sum(rewards)),
               "policy_loss": float(ploss), "value_loss": float(vloss), "mean_reward": float(np.mean(rewards))}
        log.append(rec); print("[ppo]", rec)
    torch.save({"model": model.state_dict(), "config": {"hidden": model.hidden}}, out / "ppo_canary.pt")
    (out / "metrics.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[ppo] wrote {out/'ppo_canary.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
