import numpy as np
from pipeline.rollout import segment_points, compose_plan_poses

def test_segment_endpoint_and_shapes():
    for a in (np.array([3.0, 0.0, 0.0, 6.0]), np.array([3.0, 1.0, 0.1, 6.0]), np.array([2.0, -1.0, -0.2, 5.0])):
        p = segment_points(a, n=5)
        assert p.shape == (5, 3)
        assert np.isfinite(p).all()
        assert np.allclose(p[-1, :3], a[:3], atol=1e-6)

def test_compose_plan():
    p = compose_plan_poses(np.array([[3.0, 0.0, 0.0, 6.0]] * 6), np.zeros(3))
    assert p.shape == (30, 3)
    assert np.isfinite(p).all()
    assert np.allclose(p[-1, :2], [18.0, 0.0], atol=1e-5)
