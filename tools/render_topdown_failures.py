#!/usr/bin/env python
"""Render top-down MP4s for selected closed-loop episodes.

The renderer follows the historic ``r3_topdown_*`` videos: one 640x640 frame
per 0.1 s env step (10 fps real time), white map background, faded object
history (``num_stack=15``) and the ego trajectory in ego colour.

The policy loop is exactly :func:`pipeline.eval.closed_loop.run_episode`; the
only difference is a wrapper around ``env.step`` that grabs a frame before
each physics step.  Episode metrics are re-checked against the eval JSON so a
video cannot silently show a different rollout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import pygame

from env.metadrive_env import build_env
from env.scenario.spec import load_specs
from pipeline.eval.closed_loop import load_model, run_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--specs", default="data/specs/test_500.json")
    parser.add_argument(
        "--selection",
        required=True,
        help='JSON mapping difficulty -> [spec_id, ...], e.g. {"medium": [1, 2], "hard": [3]}',
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="optional eval metrics.json; observed episodes are matched against its rows",
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--prefix", default="r4b_test500_topdown")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="render only the first N (smoke)")
    return parser.parse_args()


def _render_frames(env: Any, size: int) -> List[Any]:
    frames: List[Any] = []
    original = env.step

    def step(action, *args, **kwargs):
        frame = env.render(
            text=None,
            mode="top_down",
            film_size=(2000, 2000),
            screen_size=(size, size),
            num_stack=15,
            draw_target_vehicle_trajectory=True,
            window=False,
        )
        frames.append(frame)
        return original(action, *args, **kwargs)

    env.step = step
    return frames


def _write_video(path: Path, frames: List[Any], fps: float) -> None:
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {int(spec.id): spec for spec in load_specs(args.specs)}
    selection: Dict[str, List[int]] = json.loads(
        Path(args.selection).read_text(encoding="utf-8")
    )
    expected: Dict[int, Dict[str, Any]] = {}
    if args.metrics:
        rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["episodes"]
        expected = {int(row["spec_id"]): row for row in rows}

    model = load_model(args.ckpt, args.device, strict=True)
    episodes: List[Dict[str, Any]] = []
    manifest = {
        "ckpt": str(args.ckpt),
        "specs": str(args.specs),
        "metrics": str(args.metrics) if args.metrics else None,
        "size": int(args.size),
        "fps": float(args.fps),
        "frames": "one per 0.1 s env step",
        "selection": selection,
        "episodes": episodes,
    }
    manifest_path = out_dir / f"{args.prefix}_manifest.json"

    order = 0
    total = min(
        sum(len(ids) for ids in selection.values()),
        int(args.limit) if args.limit > 0 else 10**9,
    )
    for difficulty, ids in selection.items():
        for spec_id in ids:
            if order >= total:
                break
            spec = specs[int(spec_id)]
            pygame.init()
            env = build_env(spec, use_render=False, lru_size=0)
            frames = _render_frames(env, int(args.size))
            try:
                row = run_episode(
                    env,
                    model,
                    spec,
                    controller="exact",
                    max_steps=int(args.max_steps),
                    device=str(args.device),
                )
            finally:
                env.close()

            status = "success" if row["strict_success"] else "fail"
            name = (
                f"{args.prefix}_{order:02d}_id{spec.id}_seed{spec.seed}_"
                f"{row['difficulty']}_{row['termination']}_{status}.mp4"
            )
            video_path = out_dir / name
            _write_video(video_path, frames, float(args.fps))

            exp = expected.get(int(spec.id))
            match = None
            if exp is not None:
                match = bool(
                    row["strict_success"] == exp["strict_success"]
                    and row["termination"] == exp["termination"]
                    and int(row["steps"]) == int(exp["steps"])
                )
            record = {
                "order": order,
                "spec_id": int(spec.id),
                "seed": int(spec.seed),
                "difficulty": row["difficulty"],
                "geometry": row["geometry"],
                "termination": row["termination"],
                "success": bool(row["success"]),
                "strict_success": bool(row["strict_success"]),
                "collision": bool(row["collision"]),
                "off_road": bool(row["off_road"]),
                "timeout": bool(row["timeout"]),
                "steps": int(row["steps"]),
                "policy_steps": int(row["policy_steps"]),
                "route_completion": float(row["route_completion"]),
                "frames": len(frames),
                "video": str(video_path.resolve()),
                "matches_eval": match,
            }
            episodes.append(record)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[{order + 1}/{total}] id={spec.id} seed={spec.seed} "
                f"{row['difficulty']} {row['termination']} steps={row['steps']} "
                f"frames={len(frames)} match={match} -> {video_path.name}",
                flush=True,
            )
            order += 1
            del frames

    print(f"manifest: {manifest_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
