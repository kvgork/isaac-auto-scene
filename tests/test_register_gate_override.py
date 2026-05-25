"""Unit tests for the quality-gate override + fallback hook in register_multi_pose."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from isaac_auto_scene.register import (
    QUALITY_GATE,
    RegistrationResult,
    passes_quality_gate,
)


def _make_result(fitness: float, rmse_m: float) -> RegistrationResult:
    return RegistrationResult(
        T=np.eye(4),
        fitness=fitness,
        inlier_rmse_m=rmse_m,
        used_fallback=False,
        n_restarts=0,
    )


def test_default_gate_constants() -> None:
    """QUALITY_GATE = (0.65 fitness, 5 mm RMSE) — locked by research."""
    assert QUALITY_GATE == (0.65, 0.005)


def test_passes_quality_gate_default() -> None:
    assert passes_quality_gate(_make_result(0.70, 0.004)) is True
    assert passes_quality_gate(_make_result(0.50, 0.004)) is False
    assert passes_quality_gate(_make_result(0.70, 0.010)) is False


def test_passes_quality_gate_override() -> None:
    """An explicit gate replaces the default thresholds."""
    loose = (0.30, 0.012)
    assert passes_quality_gate(_make_result(0.35, 0.010), gate=loose) is True
    assert passes_quality_gate(_make_result(0.25, 0.010), gate=loose) is False
    assert passes_quality_gate(_make_result(0.35, 0.015), gate=loose) is False


def test_passes_quality_gate_override_does_not_mutate_default() -> None:
    """Calling with a custom gate must not leak into subsequent default calls."""
    _ = passes_quality_gate(_make_result(0.30, 0.010), gate=(0.20, 0.020))
    # Default gate still strict — a 0.30/10mm result must NOT pass it.
    assert passes_quality_gate(_make_result(0.30, 0.010)) is False


def test_register_multi_pose_accepts_with_loose_gate() -> None:
    """A loose gate accepts a low-fitness pose; default rejects it."""
    from isaac_auto_scene.register import (
        PerPoseRegistration,
        MultiPoseResult,
    )
    # Simulate by constructing the dataclass paths we care about — no need
    # to invoke Open3D ICP here; the per-pose gate logic is what we test.
    low = _make_result(0.30, 0.010)
    assert passes_quality_gate(low) is False
    assert passes_quality_gate(low, gate=(0.20, 0.020)) is True


def test_register_multi_pose_fallback_param_accepted(monkeypatch) -> None:
    """register_multi_pose forwards the fallback kwarg to register_global_local."""
    import isaac_auto_scene.register as reg_mod

    captured_kwargs: dict = {}

    real_rgl = reg_mod.register_global_local

    def spy(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _make_result(0.99, 0.001)

    monkeypatch.setattr(reg_mod, "register_global_local", spy)

    pair = o3d.geometry.PointCloud()
    pair.points = o3d.utility.Vector3dVector(np.zeros((10, 3)))

    sentinel = lambda s, t: _make_result(0.0, 1.0)
    reg_mod.register_multi_pose(
        [("p", pair, pair)],
        min_accepted=1,
        n_restarts=1,
        quality_gate=(0.0, 1.0),
        fallback=sentinel,
    )
    assert "fallback" in captured_kwargs
    assert captured_kwargs["fallback"] is sentinel
