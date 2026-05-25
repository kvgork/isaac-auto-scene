"""Manipulation-task asset spawner for the Isaac auto-scene (Phase 7).

The calibration scene (camera + table + light + arm) is fixed at calibration
time and lives in :mod:`isaac_auto_scene.scene_gen`.  Task assets (cup, block,
goal markers) are layered on **at runtime** so their poses can be randomized
per episode reset without mutating the calibration.

Public API
----------
TaskAssetSpec        — one manipulation asset: kind + geometry + pose ranges
TaskSceneSpec        — frozen container: list of TaskAssetSpec
sample_task_poses()  — draw a randomized concrete pose set from a TaskSceneSpec
spawn_task_assets()  — lazy Isaac-Lab spawner; spawns per spec into the scene

Isaac Sim 6.0 / PhysX 6.0 pitfall
---------------------------------
A kinematic ``RigidObjectCfg`` whose spawn is ``CuboidCfg`` hangs
``sim.reset()`` with ``Failed to get a valid attached USD stage id for
kinematic bodies``.  For dynamic pickup objects use ``RigidObjectCfg``
with a ``UsdFileCfg`` (real asset).  For static destination markers use
``AssetBaseCfg`` + plain ``CuboidCfg`` (no rigid/mass/collision props).
:func:`spawn_task_assets` routes by ``TaskAssetSpec.kind``.

Out-of-scope per plan §3 in v0; gated behind explicit caller opt-in.  The
calibration pipeline never invokes this module — only train/eval loops do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Asset kinds:
#   "cuboid_static"   — static destination marker (AssetBaseCfg + CuboidCfg)
#   "cuboid_dynamic"  — pickup block (RigidObjectCfg + UsdFileCfg required)
#   "cylinder_static" — static target ring (AssetBaseCfg + CylinderCfg)
#   "cylinder_dynamic"— pickup cup (RigidObjectCfg + UsdFileCfg required)
#   "usd"             — USD asset, dynamic if rigid=True, static otherwise
AssetKind = Literal[
    "cuboid_static",
    "cuboid_dynamic",
    "cylinder_static",
    "cylinder_dynamic",
    "usd",
]

XYZ = tuple[float, float, float]
QuatXYZW = tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskAssetSpec:
    """One manipulation asset with optional per-reset randomization ranges.

    Parameters
    ----------
    name:
        Stable prim suffix (becomes ``/World/Tasks/{name}``).
    kind:
        See :data:`AssetKind`.
    size_m:
        Primitive size — meaning depends on ``kind``:
        - cuboid_*: (sx, sy, sz) extents
        - cylinder_*: (radius, height, _ignored_)
        - usd: ignored
    usd_path:
        Required when ``kind == "usd"`` or when a dynamic primitive needs a
        real USD source (Isaac Sim 6.0 kinematic-CuboidCfg pitfall).
    pose_m / quat_xyzw:
        Default spawn pose (used when ``randomization`` is None or every
        component of randomization range is zero).
    randomization:
        ``((dx, dy, dz), (dyaw_rad,))`` half-ranges for uniform sampling.
        ``z`` is usually 0 (objects rest on the table); yaw is a single
        scalar so blocks/cups don't fly out of orientation.
    mass_kg:
        Rigid-body mass for dynamic objects.  Ignored for static.
    """

    name: str
    kind: AssetKind
    size_m: XYZ = (0.04, 0.04, 0.04)
    usd_path: str | None = None
    pose_m: XYZ = (0.0, 0.0, 0.05)
    quat_xyzw: QuatXYZW = (0.0, 0.0, 0.0, 1.0)
    randomization: tuple[XYZ, tuple[float]] | None = None
    mass_kg: float = 0.1

    def is_dynamic(self) -> bool:
        return self.kind.endswith("_dynamic") or (
            self.kind == "usd" and self.usd_path is not None
        )

    def requires_usd(self) -> bool:
        if self.kind == "usd":
            return True
        # Dynamic primitives need a real USD on Isaac Sim 6.0 due to the
        # kinematic-CuboidCfg pitfall — but the spawner falls back to
        # AssetBaseCfg + primitive when ``usd_path`` is None and the caller
        # accepts the static-only behaviour.
        return False


@dataclass(frozen=True)
class TaskSceneSpec:
    """Container for a randomizable set of task assets.

    Attributes
    ----------
    assets:
        Ordered list of asset specs.  Order is preserved in returned dicts
        so callers can index by position when convenient.
    parent_prim:
        Scene root for task assets.  Default ``/World/Tasks`` keeps task
        prims out of ``/World/{Table,DomeLight,D435,SO101}`` siblings.
    """

    assets: tuple[TaskAssetSpec, ...] = field(default_factory=tuple)
    parent_prim: str = "/World/Tasks"


def sample_task_poses(
    spec: TaskSceneSpec,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[XYZ, QuatXYZW]]:
    """Draw a concrete (position, quaternion) per asset from the spec.

    Returns a dict keyed by asset name so callers can write the sampled
    poses into ``init_state`` overrides or write them to a per-episode log.

    Yaw randomization composes with the asset's default ``quat_xyzw`` using
    a Z-axis-only delta (manipulation tasks typically only randomize yaw —
    full SO(3) randomization sends cups landing on their sides).
    """
    if rng is None:
        rng = np.random.default_rng()

    out: dict[str, tuple[XYZ, QuatXYZW]] = {}
    for asset in spec.assets:
        if asset.randomization is None:
            out[asset.name] = (asset.pose_m, asset.quat_xyzw)
            continue

        (dx_range, dy_range, dz_range), (dyaw_range,) = asset.randomization
        dx = rng.uniform(-dx_range, dx_range) if dx_range > 0 else 0.0
        dy = rng.uniform(-dy_range, dy_range) if dy_range > 0 else 0.0
        dz = rng.uniform(-dz_range, dz_range) if dz_range > 0 else 0.0
        dyaw = rng.uniform(-dyaw_range, dyaw_range) if dyaw_range > 0 else 0.0

        px = asset.pose_m[0] + dx
        py = asset.pose_m[1] + dy
        pz = asset.pose_m[2] + dz

        # Compose yaw delta with base quaternion (Z-axis half-angle math).
        bx, by, bz, bw = asset.quat_xyzw
        h = dyaw * 0.5
        dz_x, dz_y, dz_z, dz_w = 0.0, 0.0, float(np.sin(h)), float(np.cos(h))
        # q_new = q_delta * q_base  (rotate world-Z then apply base)
        nx = dz_w * bx + dz_x * bw + dz_y * bz - dz_z * by
        ny = dz_w * by - dz_x * bz + dz_y * bw + dz_z * bx
        nz = dz_w * bz + dz_x * by - dz_y * bx + dz_z * bw
        nw = dz_w * bw - dz_x * bx - dz_y * by - dz_z * bz

        out[asset.name] = ((px, py, pz), (nx, ny, nz, nw))
    return out


def spawn_task_assets(  # pragma: no cover - external Isaac Sim env
    spec: TaskSceneSpec,
    sampled_poses: dict[str, tuple[XYZ, QuatXYZW]] | None = None,
) -> dict[str, Any]:
    """Spawn task assets into the already-booted Isaac Sim app.

    **Precondition:** ``isaaclab.app.AppLauncher`` MUST have been booted
    before calling.  Returns a dict ``{name: prim_handle_or_None}``.

    When ``sampled_poses`` is None, defaults from each spec are used.
    Pass a draw from :func:`sample_task_poses` for randomized resets.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    out: dict[str, Any] = {}
    poses = sampled_poses or sample_task_poses(spec)

    for asset in spec.assets:
        prim_path = f"{spec.parent_prim}/{asset.name}"
        pos, quat = poses[asset.name]

        if asset.kind in ("cuboid_static", "cylinder_static"):
            shape_cfg = _primitive_shape(asset, sim_utils)
            cfg = AssetBaseCfg(
                prim_path=prim_path,
                spawn=shape_cfg,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=_xyzw_to_wxyz(quat)),
            )
            cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=pos, orientation=_xyzw_to_wxyz(quat))
            out[asset.name] = None  # static, no handle needed
            continue

        if asset.kind == "usd" or asset.kind.endswith("_dynamic"):
            if asset.usd_path is None:
                raise ValueError(
                    f"TaskAssetSpec(name={asset.name!r}, kind={asset.kind!r}) requires "
                    "usd_path: Isaac Sim 6.0 hangs sim.reset on kinematic primitive bodies. "
                    "Supply a real USD asset or change kind to *_static."
                )
            cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=asset.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=asset.mass_kg),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot=_xyzw_to_wxyz(quat),
                ),
            )
            from isaaclab.assets import RigidObject  # local import keeps soft-import contract
            out[asset.name] = RigidObject(cfg=cfg)
            continue

        raise ValueError(f"Unknown TaskAssetSpec kind: {asset.kind!r}")

    return out


def _primitive_shape(asset: TaskAssetSpec, sim_utils: Any) -> Any:  # pragma: no cover
    """Return the SpawnerCfg for a primitive asset kind."""
    if asset.kind.startswith("cuboid"):
        return sim_utils.CuboidCfg(size=asset.size_m)
    if asset.kind.startswith("cylinder"):
        radius, height, _ = asset.size_m
        return sim_utils.CylinderCfg(radius=radius, height=height)
    raise ValueError(f"_primitive_shape unsupported kind: {asset.kind!r}")


def _xyzw_to_wxyz(q: QuatXYZW) -> tuple[float, float, float, float]:
    """Isaac Lab uses (w, x, y, z) for ``init_state.rot``; convert."""
    x, y, z, w = q
    return (w, x, y, z)


__all__ = [
    "AssetKind",
    "TaskAssetSpec",
    "TaskSceneSpec",
    "sample_task_poses",
    "spawn_task_assets",
]
