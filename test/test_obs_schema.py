import numpy as np
from env.scenario.spec import load_specs
from env.metadrive_env import build_env
from env.obs import (OD_SLOTS, OD_DIM, LD_SLOTS, LD_POINTS, LD_DIM, NAV_DIM, EGO_DIM,
                     HISTORY_FRAMES, build_observation, stack_history)

def test_obs_schema_smoke():
    spec = load_specs("env/specs/scenarios_train_slice200.json")[0]
    env = build_env(spec, use_render=False)
    try:
        env.reset()
        obs = build_observation(env)
        assert obs["od"].shape == (OD_SLOTS, OD_DIM)
        assert obs["ld"].shape == (LD_SLOTS, LD_POINTS, LD_DIM)
        assert obs["nav"].shape == (NAV_DIM,)
        assert obs["ego"].shape == (EGO_DIM,)
        for k in ("od", "ld", "nav", "ego"):
            assert np.isfinite(obs[k]).all(), k
        hist = stack_history([obs] * HISTORY_FRAMES, [obs["pose"]] * HISTORY_FRAMES)
        assert hist["od"].shape == (HISTORY_FRAMES, OD_SLOTS, OD_DIM)
        assert hist["ld"].shape == (HISTORY_FRAMES, LD_SLOTS, LD_POINTS, LD_DIM)
        assert hist["hist_valid"].sum() == HISTORY_FRAMES
    finally:
        env.close()
