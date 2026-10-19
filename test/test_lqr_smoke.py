import numpy as np
from env.scenario.spec import load_specs
from env.metadrive_env import build_env
from env.tracking import LqrTracker

def test_lqr_straight_reference_smoke():
    spec = load_specs("env/specs/scenarios_train_slice200.json")[0]
    env = build_env(spec, use_render=False)
    try:
        env.reset(); ego = env.agent
        tr = env.engine.add_policy(ego.id, LqrTracker, ego, int(spec.seed))
        ref = np.zeros((30, 3)); ref[:, 0] = 0.6 * np.arange(1, 31)
        tr.set_reference(ref, base_pose=np.array([ego.position[0], ego.position[1], ego.heading_theta]))
        off = False
        for _ in range(30):
            _, _, term, trunc, info = env.step([0.0, 0.0])
            off = off or bool(info.get("out_of_road", False))
            if term or trunc: break
        assert not off
    finally:
        env.close()
