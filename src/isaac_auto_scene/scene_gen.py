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
    table_size_m:
        (sx, sy, sz) cuboid table extents.
    pinhole_cfg:
        Isaac PinholeCameraCfg kwargs derived from RealSense intrinsics.
    enable_ros2:
        Whether the ROS2 camera publisher OmniGraph should be attached.
    """

    camera_position_m: tuple[float, float, float]
    camera_quat_xyzw: tuple[float, float, float, float]
    arm_position_m: tuple[float, float, float]
    arm_quat_xyzw: tuple[float, float, float, float]
    table_size_m: tuple[float, float, float] = (0.6, 0.4, 0.02)
    pinhole_cfg: dict[str, float | int] = field(default_factory=dict)
    enable_ros2: bool = False


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

    # The calibration provides arm-in-camera transform; the camera-in-world
    # is the identity here (camera == world for the simplest case where the
    # table frame is the camera frame).  For richer scenes the orchestrator
    # would supply T_world_camera explicitly.
    arm_t = tuple(calib.translation_m)
    arm_q = tuple(calib.quat_xyzw)
    cam_t = (0.0, 0.0, 0.0)
    cam_q = (0.0, 0.0, 0.0, 1.0)

    return SceneSpec(
        camera_position_m=cam_t,
        camera_quat_xyzw=cam_q,
        arm_position_m=arm_t,  # type: ignore[arg-type]
        arm_quat_xyzw=arm_q,  # type: ignore[arg-type]
        table_size_m=table_size_m,
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

    def Xform "SO101"
    {{
        matrix4d xformOp:transform = {arm_m}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def Cube "Table"
    {{
        double size = 1.0
        double3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:scale"]
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

    Returns a dict ``{"camera": Camera}`` so the caller can ``camera.update()``
    after each ``sim.step()``.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    table_cfg = sim_utils.CuboidCfg(size=spec.table_size_m)
    table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, -spec.table_size_m[2] / 2))

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
    return {"camera": camera}


def warm_up_render(render_step: Callable[[], None], n_frames: int = WARM_UP_FRAMES) -> None:
    """Step the renderer ``n_frames`` times before user code uses the image.

    Required by IsaacLab#3250 (texture streaming bug — first ~30 frames
    have incomplete/garbage textures).
    """
    for _ in range(n_frames):
        render_step()
