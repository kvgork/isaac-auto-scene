"""Isaac Lab InteractiveSceneCfg builder from calibration JSON (Phase 6).

Public API
----------
SceneSpec            — frozen dataclass capturing all derived scene params
build_scene_spec()   — calib.json -> SceneSpec (no Isaac dependency)
write_usd_stub()     — minimal USDA file capturing camera + arm + table
build_isaac_scene()  — lazy-imports Isaac Lab; returns InteractiveSceneCfg
warm_up_render()     — wraps a render callable in the mandatory 30-frame warm-up

The SceneSpec layer carries enough information to fully describe the scene
without importing Isaac Sim, which is what the unit tests exercise.  The
Isaac-Lab integration path is gated behind a lazy import so the rest of
the package stays import-clean on machines without isaaclab installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from isaac_auto_scene.calibrate import CalibrationOutput, load_calibration
from isaac_auto_scene.utils.intrinsics import realsense_to_isaac


WARM_UP_FRAMES = 30  # IsaacLab#3250 texture-streaming bug — 30 frames mandatory


@dataclass(frozen=True)
class SceneSpec:
    """Derived parameters for the Isaac Lab scene, without any Isaac imports.

    Attributes
    ----------
    camera_position_m / camera_quat_xyzw:
        Camera pose in world frame.
    arm_position_m / arm_quat_xyzw:
        SO-101 root pose in world frame.
    arm_joint_angles_rad:
        Per-joint angles applied to the SO-101 articulation when spawning.
        Populated from ``CalibrationOutput.joint_angles_at_capture`` so the
        rendered arm matches the pose that was captured.  Empty dict
        defaults to URDF home (all zeros).
    table_position_m / table_quat_xyzw:
        Table centroid pose in world frame.  Defaults place the table at
        the world origin (legacy behaviour); pass the calibrated values
        from ``CalibrationOutput.T_cam_table`` to position the table
        where the segmenter actually found it relative to the camera.
    table_size_m:
        (sx, sy, sz) cuboid table extents.
    pinhole_cfg:
        Isaac PinholeCameraCfg kwargs derived from RealSense intrinsics.
    enable_ros2:
        Whether the ROS2 camera publisher OmniGraph should be attached.
    so101_usd_path:
        Optional path to the SO-101 USD asset.  When provided,
        :func:`build_isaac_scene` spawns an ArticulationCfg at
        ``/World/SO101``.  When ``None`` only the Xform stub appears in the
        USD stub output and ``build_isaac_scene`` skips articulation spawn.
    """

    camera_position_m: tuple[float, float, float]
    camera_quat_xyzw: tuple[float, float, float, float]
    arm_position_m: tuple[float, float, float]
    arm_quat_xyzw: tuple[float, float, float, float]
    table_size_m: tuple[float, float, float] = (0.6, 0.4, 0.02)
    table_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    arm_joint_angles_rad: dict[str, float] = field(default_factory=dict)
    pinhole_cfg: dict[str, float | int] = field(default_factory=dict)
    enable_ros2: bool = False
    so101_usd_path: str | None = None


SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
"""Canonical SO-101 joint order matching ``so101_new_calib.urdf``.

Borrowed from the sibling ``lerobot-isaac-training`` package
(see ``lerobot_isaac_env/so101_articulation.py``) — the legacy
CAD-style names ``(Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)``
are NOT compatible with the current canonical URDF/USD pair.
"""


def resolve_default_so101_usd() -> str | None:
    """Return the default SO-101 USD path from the sibling lerobot env, if present.

    The USD is not vendored in this repository to keep the package light;
    we borrow the asset converted by ``lerobot-isaac-training``.  Returns
    ``None`` when the asset is missing — callers must then pass an explicit
    path via ``SceneSpec.so101_usd_path``.
    """
    candidate = (
        Path.home()
        / "workspaces"
        / "lerobot-isaac-training"
        / "src"
        / "lerobot-isaac-env"
        / "assets"
        / "usd"
        / "so101.usd"
    )
    return str(candidate) if candidate.exists() else None


def build_scene_spec(
    calib: CalibrationOutput,
    *,
    table_size_m: tuple[float, float, float] = (0.6, 0.4, 0.02),
    enable_ros2: bool = False,
) -> SceneSpec:
    """Convert a CalibrationOutput into a SceneSpec.

    World convention: the table is at the origin, +Z up.  The camera and
    arm poses come directly from calib (already in world == table frame
    by upstream convention).
    """
    K = np.array(
        [
            [calib.intrinsics["fx"], 0.0, calib.intrinsics["cx"]],
            [0.0, calib.intrinsics["fy"], calib.intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pinhole_cfg = realsense_to_isaac(
        K, int(calib.intrinsics["width"]), int(calib.intrinsics["height"])
    )

    # The calibration provides arm-in-camera transform; world == camera
    # frame (camera at origin, identity rotation).  For richer scenes the
    # orchestrator would supply T_world_camera explicitly.
    arm_t = tuple(calib.translation_m)
    arm_q = tuple(calib.quat_xyzw)
    cam_t = (0.0, 0.0, 0.0)
    cam_q = (0.0, 0.0, 0.0, 1.0)

    # Table pose: prefer the segmentation-derived T_cam_table when it's in
    # the calib payload (new schema).  Legacy calibs fall back to placing
    # the table at the world origin.
    table_t = (0.0, 0.0, 0.0)
    table_q = (0.0, 0.0, 0.0, 1.0)
    if calib.T_cam_table is not None:
        T = np.asarray(calib.T_cam_table, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        from isaac_auto_scene.utils.transforms import rotation_matrix_to_quat_xyzw

        q = rotation_matrix_to_quat_xyzw(R)
        table_t = (float(t[0]), float(t[1]), float(t[2]))
        table_q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    return SceneSpec(
        camera_position_m=cam_t,
        camera_quat_xyzw=cam_q,
        arm_position_m=arm_t,  # type: ignore[arg-type]
        arm_quat_xyzw=arm_q,  # type: ignore[arg-type]
        arm_joint_angles_rad=dict(calib.joint_angles_at_capture or {}),
        table_size_m=table_size_m,
        table_position_m=table_t,
        table_quat_xyzw=table_q,
        pinhole_cfg=pinhole_cfg,
        enable_ros2=enable_ros2,
    )


def _q_to_R(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def write_usd_stub(spec: SceneSpec, out_path: Path) -> Path:
    """Write a minimal USDA file describing the scene.

    The output is **not** a full Isaac Sim asset graph — it captures enough
    of the camera/arm/table layout to satisfy the "non-empty USD" acceptance
    criterion in environments where Isaac Sim is not installed.  When Isaac
    Sim is available, use :func:`build_isaac_scene` instead.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _xform(name: str, t: tuple[float, float, float], q) -> str:
        R = _q_to_R(q)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        cells = ", ".join(f"{v:.6f}" for v in M.T.flatten())
        return (
            f"        def Xform \"{name}\"\n"
            f"        {{\n"
            f"            matrix4d xformOp:transform = ( ({cells[:0]}) )\n"
            f"            uniform token[] xformOpOrder = [\"xformOp:transform\"]\n"
            f"        }}\n"
        )

    def _matrix(t, q) -> str:
        R = _q_to_R(q)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        rows = []
        for i in range(4):
            row = ", ".join(f"{M[i, j]:.6f}" for j in range(4))
            rows.append(f"({row})")
        return "(" + ", ".join(rows) + ")"

    cam_m = _matrix(spec.camera_position_m, spec.camera_quat_xyzw)
    arm_m = _matrix(spec.arm_position_m, spec.arm_quat_xyzw)
    sx, sy, sz = spec.table_size_m
    # Sink the table by half its thickness so the top surface lies on
    # the calibrated plane — same convention as build_isaac_scene.
    table_t_sunken = (
        spec.table_position_m[0],
        spec.table_position_m[1],
        spec.table_position_m[2] - sz / 2.0,
    )
    table_m = _matrix(table_t_sunken, spec.table_quat_xyzw)

    so101_usd = spec.so101_usd_path or resolve_default_so101_usd()
    so101_ref_line = (
        f'        prepend references = @{so101_usd}@\n' if so101_usd else ""
    )

    contents = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def Camera "D435"
    {{
        matrix4d xformOp:transform = {cam_m}
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float focalLength = {spec.pinhole_cfg.get("focal_length", 1.93)}
        float horizontalAperture = {spec.pinhole_cfg.get("horizontal_aperture", 20.955)}
        int2 resolution = ({int(spec.pinhole_cfg.get("width", 640))}, {int(spec.pinhole_cfg.get("height", 480))})
    }}

    def Xform "SO101" (
{so101_ref_line}    )
    {{
        matrix4d xformOp:transform = {arm_m}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def Xform "Table"
    {{
        matrix4d xformOp:transform = {table_m}
        uniform token[] xformOpOrder = ["xformOp:transform"]

        def Cube "Geometry"
        {{
            double size = 1.0
            double3 xformOp:scale = ({sx}, {sy}, {sz})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}

    def DomeLight "DomeLight"
    {{
        float intensity = 1000.0
    }}
}}
"""
    out_path.write_text(contents)
    return out_path


def build_isaac_scene(spec: SceneSpec) -> dict[str, Any]:  # pragma: no cover - external sim env
    """Spawn the scene prims into an already-booted Isaac Sim app.

    **Precondition:** ``isaaclab.app.AppLauncher`` MUST have been booted with
    ``headless=True, enable_cameras=True`` before calling this — the
    ``isaacsim`` / ``omni`` runtime is only importable post-boot.

    Spawns:
      - ``/World/Table`` — CuboidCfg using ``spec.table_size_m``
      - ``/World/DomeLight`` — DomeLightCfg at default intensity
      - ``/World/D435`` — Camera at ``spec.camera_position_m`` /
        ``spec.camera_quat_xyzw`` with ``spec.pinhole_cfg`` intrinsics
      - ``/World/SO101`` — ArticulationCfg when ``spec.so101_usd_path`` is
        provided (or :func:`resolve_default_so101_usd` succeeds).  Omitted
        otherwise so headless smoke tests can still run without the USD.

    Returns a dict ``{"camera": Camera, "arm": Articulation | None}``.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    table_cfg = sim_utils.CuboidCfg(size=spec.table_size_m)
    # Place the table centroid at spec.table_position_m, sunk by half its
    # thickness so the top surface lies on the calibrated plane.  The
    # rotation comes from the segmenter's T_world_table fit (Z = plane
    # normal) so the cuboid lies flat along that surface.
    t = spec.table_position_m
    half_thick = spec.table_size_m[2] / 2.0
    sunken = (
        float(t[0]),
        float(t[1]),
        float(t[2]) - half_thick,
    )
    table_cfg.func(
        "/World/Table",
        table_cfg,
        translation=sunken,
        orientation=_quat_xyzw_to_wxyz(spec.table_quat_xyzw),
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/D435",
        update_period=0,
        height=int(spec.pinhole_cfg["height"]),
        width=int(spec.pinhole_cfg["width"]),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(spec.pinhole_cfg["focal_length"]),
            horizontal_aperture=float(spec.pinhole_cfg["horizontal_aperture"]),
            horizontal_aperture_offset=float(spec.pinhole_cfg["horizontal_aperture_offset"]),
            clipping_range=(0.05, 10.0),
        ),
    )
    camera = Camera(cfg=camera_cfg)

    arm = _build_so101_articulation(spec, sim_utils)
    return {"camera": camera, "arm": arm}


def _build_so101_articulation(spec: SceneSpec, sim_utils: Any) -> Any:  # pragma: no cover
    """Spawn the SO-101 USD at ``/World/SO101`` and return the Articulation.

    The arm's root pose comes from ``spec.arm_position_m`` /
    ``spec.arm_quat_xyzw``.  Joint angles default to URDF home; when the
    spec carries ``arm_joint_angles_rad`` (e.g. from
    ``CalibrationOutput.joint_angles_at_capture``) the returned
    Articulation must have its joints written via
    ``arm.write_joint_state_to_sim(...)`` AFTER ``sim.reset()``.  The
    caller is responsible for that — see ``scripts/render_isaac_scene.py``.

    Returns the Articulation runtime wrapper, or ``None`` when no USD is
    provided / resolved.
    """
    usd_path = spec.so101_usd_path or resolve_default_so101_usd()
    if usd_path is None:
        return None

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    joint_pos_init = {
        name: float(spec.arm_joint_angles_rad.get(name, 0.0))
        for name in SO101_JOINT_NAMES
    }

    cfg = ArticulationCfg(
        prim_path="/World/SO101",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.arm_position_m,
            rot=_quat_xyzw_to_wxyz(spec.arm_quat_xyzw),
            joint_pos=joint_pos_init,
            joint_vel={name: 0.0 for name in SO101_JOINT_NAMES},
        ),
        actuators={
            "so101_arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )
    # Spawn the USD prim explicitly (UsdFileCfg.func) — the Articulation
    # runtime wrapper that follows binds to this existing prim instead of
    # re-spawning.
    cfg.spawn.func(
        "/World/SO101",
        cfg.spawn,
        translation=spec.arm_position_m,
        orientation=_quat_xyzw_to_wxyz(spec.arm_quat_xyzw),
    )
    return Articulation(cfg=cfg)


def _quat_xyzw_to_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Isaac Lab ``init_state.rot`` uses (w, x, y, z) order."""
    x, y, z, w = q
    return (w, x, y, z)


def warm_up_render(render_step: Callable[[], None], n_frames: int = WARM_UP_FRAMES) -> None:
    """Step the renderer ``n_frames`` times before user code uses the image.

    Required by IsaacLab#3250 (texture streaming bug — first ~30 frames
    have incomplete/garbage textures).
    """
    for _ in range(n_frames):
        render_step()
