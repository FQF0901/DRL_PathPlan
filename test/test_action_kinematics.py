import math

import numpy as np
import torch

from env.tracking import ExactTracker
from env.action.action_space import CLEAN_BOUNDS
from pipeline.rollout import (
    denorm_action,
    denorm_action_with_bounds,
    first_segment_trajectory,
    segment_trajectory,
)


class _FakeVehicle:
    FRONT_WHEELBASE = 1.2
    REAR_WHEELBASE = 1.3
    max_steering = 40.0

    def __init__(self):
        self.position = np.zeros(3)
        self.heading_theta = 0.0
        self.velocity = np.zeros(2)
        self.angular_velocity = 0.0

    def set_position(self, position):
        self.position = np.array([position[0], position[1], 0.0], dtype=np.float64)

    def set_heading_theta(self, heading):
        self.heading_theta = float(heading)

    def set_velocity(self, velocity):
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()

    def set_angular_velocity(self, angular_velocity):
        self.angular_velocity = float(angular_velocity)


class _FakeManager:
    def __init__(self, vehicle):
        self.active_agents = {"ego": vehicle}
        self.episode_created_agents = {"ego": vehicle}


class _FakeEngine:
    def __init__(self, vehicle):
        self.agent_manager = _FakeManager(vehicle)


class _FakeEnv:
    def __init__(self, vehicle):
        self.engine = _FakeEngine(vehicle)



def test_segment_trajectory_endpoint_and_speed():
    a = np.array([3.0, 0.0, 0.0, 6.0])
    tr = segment_trajectory(a, 6.0, bounds=CLEAN_BOUNDS)
    assert tr.positions.shape == (5, 3)
    assert tr.speeds.shape == (5,)
    assert np.allclose(tr.positions[-1, :2], a[:2], atol=1e-12)
    assert abs(float(tr.positions[-1, 2]) - float(a[2])) < 1e-12
    assert abs(float(tr.speeds[-1]) - float(a[3])) < 1e-12
    assert tr.metrics["start_speed_mps"] == 6.0
    assert tr.feasible

    same = first_segment_trajectory(a, 6.0, bounds=CLEAN_BOUNDS)
    assert np.allclose(same.positions, tr.positions)
    assert np.allclose(same.speeds, tr.speeds)
    # dict-style interface for downstream consumers
    assert np.allclose(same["points"], tr.positions)
    assert np.allclose(same.as_dict()["speeds"], tr.speeds)



def test_denorm_action_with_bounds():
    normalized = torch.tensor([0.4, 0.5, 0.5, 0.4])
    canonical = denorm_action(normalized)
    assert torch.allclose(canonical, denorm_action_with_bounds(normalized, CLEAN_BOUNDS))
    assert torch.allclose(canonical, torch.tensor([3.0, 0.0, 0.0, 6.0]))
    arr = denorm_action_with_bounds(np.array([0.4, 0.5, 0.5, 0.4]), CLEAN_BOUNDS)
    assert np.allclose(arr, [3.0, 0.0, 0.0, 6.0])


def test_exact_tracker_uses_explicit_speed_profile_for_velocity():
    vehicle = _FakeVehicle()
    env = _FakeEnv(vehicle)
    traj = segment_trajectory(np.array([3.0, 0.0, 0.0, 6.0]), 6.0, bounds=CLEAN_BOUNDS)
    tracker = ExactTracker(dt=0.5, hz=10)
    tracker.arm(env, trajectory=traj)

    assert tracker.speeds is not None
    assert np.allclose(tracker.speeds, traj.speeds)

    for i in range(traj.n_points):
        pose = tracker.apply(env)
        assert pose.shape == (3,)
        assert np.allclose(pose, traj.positions[i], atol=1e-12)
        speed = float(np.linalg.norm(vehicle.velocity))
        assert abs(speed - traj.speeds[i]) < 1e-9
        heading = math.atan2(vehicle.velocity[1], vehicle.velocity[0])
        assert abs(float((heading - traj.positions[i, 2] + math.pi) % (2 * math.pi) - math.pi)) < 1e-12
        # trajectory carries kappa -> explicit yaw rate = kappa * speed
        assert abs(vehicle.angular_velocity - traj.kappa[i] * traj.speeds[i]) < 1e-12


def test_exact_tracker_accepts_points_and_speeds():
    vehicle = _FakeVehicle()
    env = _FakeEnv(vehicle)
    traj = segment_trajectory(np.array([3.0, 0.0, 0.0, 6.0]), 6.0, bounds=CLEAN_BOUNDS)

    tracker = ExactTracker(dt=0.5, hz=10)
    tracker.arm(env, points=traj.positions, speeds=traj.speeds)
    for i in range(traj.n_points):
        tracker.apply(env)
        assert abs(float(np.linalg.norm(vehicle.velocity)) - traj.speeds[i]) < 1e-9

    # dict interface carries positions + explicit speeds without a live env
    dict_tracker = ExactTracker(dt=0.5, hz=10)
    dict_tracker.arm(
        env,
        trajectory={"positions": traj.positions, "speeds": traj.speeds, "kappa": traj.kappa},
    )
    for i in range(traj.n_points):
        dict_tracker.apply(env)
        assert abs(float(np.linalg.norm(vehicle.velocity)) - traj.speeds[i]) < 1e-9

    # points-only path derives velocity from chord differences
    points_only = ExactTracker(dt=0.5, hz=10)
    points_only.arm(env, points=traj.positions)
    assert points_only.speeds is None
    points_only.apply(env)
    assert np.isfinite(vehicle.velocity).all()


def test_exact_tracker_sets_steering_from_path_curvature():
    vehicle = _FakeVehicle()
    env = _FakeEnv(vehicle)
    traj = segment_trajectory(np.array([3.0, 0.5, 0.2, 6.0]), 6.0)
    tracker = ExactTracker(dt=0.5, hz=10)
    tracker.arm(env, trajectory=traj)
    for _ in range(traj.n_points):
        tracker.apply(env)
    assert abs(float(vehicle.steering)) > 1e-6
    assert -1.0 <= float(vehicle.steering) <= 1.0


def test_exact_tracker_steering_matches_curvature_analytic():
    vehicle = _FakeVehicle()
    env = _FakeEnv(vehicle)
    traj = segment_trajectory(np.array([3.0, 0.5, 0.2, 6.0]), 6.0)
    tracker = ExactTracker(dt=0.5, hz=10)
    tracker.arm(env, trajectory=traj)
    tracker.apply(env, index=0)
    wheelbase = _FakeVehicle.FRONT_WHEELBASE + _FakeVehicle.REAR_WHEELBASE
    max_steer = np.deg2rad(_FakeVehicle.max_steering)
    expected = np.sign(traj.kappa[0]) * np.arctan(wheelbase * abs(traj.kappa[0])) / max_steer
    assert np.isclose(float(vehicle.steering), float(expected), atol=1e-12)
    assert np.sign(float(vehicle.steering)) == np.sign(float(traj.kappa[0]))
    assert -1.0 <= float(vehicle.steering) <= 1.0
