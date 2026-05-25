"""Joint-pose validation + ArmDriver abstraction for multi-pose capture.

Public API
----------
JointPose          — frozen dataclass: name + joint dict + settle_s
ArmDriver          — Protocol for command/readback
MockArmDriver      — deterministic mock with optional servo-noise injection
load_poses         — YAML loader -> list[JointPose]
validate_pose      — check pose against URDF joint limits
validate_pose_set  — batch validate + dedup names

Design
------
Pose capture is decoupled from registration so it can run in the hardware
pixi env without pulling Open3D into the loop.  Real LeRobot driver
integration lives in a subclass that satisfies :class:`ArmDriver`.

URDF self-collision check is intentionally optional: it requires `fcl`
which is not in the default pixi env.  When the import fails we skip
collision and still enforce joint-limit validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import yaml
import yourdfpy


@dataclass(frozen=True)
class JointPose:
    """One named pose: joint-name -> radians + optional settle time.

    Attributes
    ----------
    name:
        Unique pose identifier.  Used as subdirectory name in capture output.
    joints:
        Mapping joint-name -> radians.  Joints missing from this dict default
        to zero when commanded.
    settle_s:
        Seconds to wait after commanding the pose before reading back.
    """

    name: str
    joints: dict[str, float]
    settle_s: float = 1.0


@runtime_checkable
class ArmDriver(Protocol):
    """Minimal interface a hardware arm driver must satisfy for capture.

    The real implementation wraps the LeRobot Feetech driver; tests use
    :class:`MockArmDriver`.  Keep this surface narrow — pose validation
    happens *outside* the driver, in :func:`validate_pose`.
    """

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def command_joints(self, joints: dict[str, float]) -> None:
        """Send a joint-angle target (radians).  Non-blocking on the wire."""
        ...

    def read_joints(self) -> dict[str, float]:
        """Return the most recent servo readback (radians)."""
        ...

    def __enter__(self) -> "ArmDriver": ...

    def __exit__(self, *args: object) -> None: ...


class MockArmDriver:
    """Deterministic mock for the multi-pose capture pipeline.

    The mock pretends to be a 6-DOF SO-101 by default and tracks the last
    commanded pose.  Readback can be perturbed by Gaussian noise to mimic
    servo repeatability (~0.5°-1° at joint level).
    """

    def __init__(
        self,
        joint_names: Iterable[str] | None = None,
        *,
        readback_noise_rad: float = 0.0,
        seed: int = 0,
    ) -> None:
        if joint_names is None:
            joint_names = (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        self._joint_names = tuple(joint_names)
        self._readback_noise_rad = float(readback_noise_rad)
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._last_commanded: dict[str, float] = {n: 0.0 for n in self._joint_names}

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def command_joints(self, joints: dict[str, float]) -> None:
        if not self._connected:
            raise RuntimeError("MockArmDriver: call connect() first")
        for name, value in joints.items():
            if name not in self._last_commanded:
                raise KeyError(f"unknown joint {name!r}")
            self._last_commanded[name] = float(value)

    def read_joints(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("MockArmDriver: call connect() first")
        out: dict[str, float] = {}
        for name, value in self._last_commanded.items():
            if self._readback_noise_rad > 0.0:
                noise = float(self._rng.normal(0.0, self._readback_noise_rad))
                out[name] = value + noise
            else:
                out[name] = value
        return out

    def __enter__(self) -> "MockArmDriver":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_poses(path: Path | str) -> list[JointPose]:
    """Load a YAML pose set.

    YAML schema (list of mappings):

    ```yaml
    poses:
      - name: home
        joints: {shoulder_pan: 0.0, shoulder_lift: 0.0, ...}
        settle_s: 1.0
      - name: pose_a
        joints: {shoulder_pan: 0.5, ...}
    ```
    """
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "poses" not in data:
        raise ValueError(f"{path}: missing top-level 'poses' key")
    raw = data["poses"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: 'poses' must be a non-empty list")

    out: list[JointPose] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: pose #{i} is not a mapping")
        try:
            name = str(entry["name"])
            joints_raw = entry["joints"]
        except KeyError as exc:
            raise ValueError(f"{path}: pose #{i} missing key {exc}") from exc
        if not isinstance(joints_raw, dict):
            raise ValueError(f"{path}: pose {name!r} joints must be a mapping")
        joints = {str(k): float(v) for k, v in joints_raw.items()}
        settle_s = float(entry.get("settle_s", 1.0))
        out.append(JointPose(name=name, joints=joints, settle_s=settle_s))
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseValidationError:
    """One validation failure on a pose."""

    pose_name: str
    reason: str


@dataclass(frozen=True)
class PoseValidationReport:
    """Result of validating a set of poses against a URDF."""

    ok: bool
    errors: tuple[PoseValidationError, ...] = field(default=())

    def __bool__(self) -> bool:
        return self.ok


def validate_pose(
    pose: JointPose,
    urdf: yourdfpy.URDF,
    *,
    check_self_collision: bool = False,
) -> list[PoseValidationError]:
    """Validate one pose against the URDF joint limits.

    Self-collision check is only attempted when ``check_self_collision`` is
    True AND a collision backend (``fcl`` via trimesh) is importable.  When
    the backend is missing we silently skip — joint-limit validation is the
    only mandatory check.
    """
    errors: list[PoseValidationError] = []

    actuated = set(urdf.actuated_joint_names)
    extra = set(pose.joints) - actuated
    if extra:
        errors.append(
            PoseValidationError(
                pose_name=pose.name,
                reason=f"unknown joint(s) {sorted(extra)} (not in URDF actuated joints)",
            )
        )

    for jname, value in pose.joints.items():
        joint = urdf.joint_map.get(jname)
        if joint is None:
            continue  # already flagged above
        limit = getattr(joint, "limit", None)
        if limit is None:
            continue
        lo = getattr(limit, "lower", None)
        hi = getattr(limit, "upper", None)
        if lo is not None and value < float(lo):
            errors.append(
                PoseValidationError(
                    pose_name=pose.name,
                    reason=f"{jname}={value:.4f} < lower {float(lo):.4f}",
                )
            )
        if hi is not None and value > float(hi):
            errors.append(
                PoseValidationError(
                    pose_name=pose.name,
                    reason=f"{jname}={value:.4f} > upper {float(hi):.4f}",
                )
            )

    if check_self_collision:
        try:
            _self_collision_check(urdf, pose)
        except _CollisionBackendUnavailable:
            pass
        except _PoseInSelfCollision as exc:
            errors.append(
                PoseValidationError(pose_name=pose.name, reason=str(exc))
            )

    return errors


def validate_pose_set(
    poses: list[JointPose],
    urdf: yourdfpy.URDF,
    *,
    check_self_collision: bool = False,
) -> PoseValidationReport:
    """Validate a list of poses; collect errors and dedup names."""
    errors: list[PoseValidationError] = []
    seen: set[str] = set()
    for pose in poses:
        if pose.name in seen:
            errors.append(
                PoseValidationError(
                    pose_name=pose.name, reason="duplicate pose name"
                )
            )
        seen.add(pose.name)
        errors.extend(
            validate_pose(pose, urdf, check_self_collision=check_self_collision)
        )
    return PoseValidationReport(ok=not errors, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Optional self-collision check (best-effort)
# ---------------------------------------------------------------------------


class _CollisionBackendUnavailable(RuntimeError):
    """Raised when fcl/trimesh collision backend is missing."""


class _PoseInSelfCollision(RuntimeError):
    """Raised when a pose has a self-collision pair."""


def _self_collision_check(urdf: yourdfpy.URDF, pose: JointPose) -> None:
    try:
        import trimesh  # noqa: F401
        from trimesh.collision import CollisionManager
    except ImportError as exc:  # pragma: no cover - depends on env
        raise _CollisionBackendUnavailable(str(exc)) from exc

    urdf.update_cfg(dict(pose.joints))
    manager = CollisionManager()
    try:
        for link_name, link in urdf.link_map.items():
            if not link.collisions:
                continue
            T = np.asarray(urdf.get_transform(link_name), dtype=np.float64)
            for i, col in enumerate(link.collisions):
                geom = col.geometry
                if geom is None or geom.mesh is None:
                    continue
                mesh = urdf.scene.geometry.get(col.name) if col.name else None
                if mesh is None:
                    continue
                manager.add_object(f"{link_name}_{i}", mesh, transform=T)
    except Exception as exc:  # pragma: no cover - backend quirks
        raise _CollisionBackendUnavailable(str(exc)) from exc

    in_col, pairs = manager.in_collision_internal(return_names=True)
    if in_col:
        raise _PoseInSelfCollision(f"self-collision: {sorted(pairs)[:3]}")
