"""Interactive viewer to manually align a CAD point cloud over a captured
arm cloud and emit the resulting SE(3) as calib.json (Phase 7+).

Public API
----------
ManualAlignSession   — view + drive an alignment session.
run_manual_align()   — top-level: load capture + CAD, open window, return final T.

Controls (inside the viewer window)
-----------------------------------
Translation (camera-frame axes, ``--step`` metres per keypress):
    A / D       —  −X / +X
    W / S       —  −Y / +Y   (D435: +Y is down)
    Q / E       —  −Z / +Z   (depth axis)

Rotation (around current step centroid, ``--rot-step`` rad per keypress):
    J / L       —  yaw  ± (around camera Y)
    I / K       —  pitch ± (around camera X)
    U / O       —  roll  ± (around camera Z)

Step control:
    +           —  double translation step
    -           —  halve translation step
    [           —  halve rotation step
    ]           —  double rotation step

Action keys:
    SPACE       —  snap to local ICP from current pose
    R           —  reset to identity
    Z           —  reset to initial T (if --init-from was supplied)
    ENTER       —  print current T to stdout and close window

The viewer renders the captured arm cloud in green and the CAD in red.
Closing the window with ENTER returns the current 4x4 SE(3) matrix to
the caller; closing with the window manager (ESC / X button) returns
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


@dataclass
class ManualAlignState:
    T: np.ndarray
    T_init: np.ndarray
    step_m: float
    rot_step_rad: float


def _rot_axis(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"axis must be x|y|z, got {axis!r}")
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _translate(dx: float, dy: float, dz: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def run_manual_align(
    cad_pts: np.ndarray,
    arm_cloud: o3d.geometry.PointCloud,
    *,
    T_init: np.ndarray | None = None,
    window_title: str = "isaac-auto-scene manual alignment",
    width: int = 1024,
    height: int = 768,
    step_m: float = 0.01,
    rot_step_deg: float = 5.0,
    icp_threshold_m: float = 0.02,
) -> np.ndarray | None:
    """Open an interactive window for manual SE(3) alignment.

    Returns the final 4x4 matrix when the user presses ENTER, or
    ``None`` if the window was closed without saving.
    """
    if T_init is None:
        T_init = np.eye(4)
    state = ManualAlignState(
        T=np.asarray(T_init, dtype=np.float64).copy(),
        T_init=np.asarray(T_init, dtype=np.float64).copy(),
        step_m=float(step_m),
        rot_step_rad=float(np.deg2rad(rot_step_deg)),
    )

    arm_geom = o3d.geometry.PointCloud(arm_cloud)
    arm_geom.paint_uniform_color((0.2, 0.85, 0.2))

    cad_local = np.asarray(cad_pts, dtype=np.float64)
    cad_geom = o3d.geometry.PointCloud()
    cad_geom.points = o3d.utility.Vector3dVector(cad_local)
    cad_geom.paint_uniform_color((0.95, 0.15, 0.15))
    cad_geom.transform(state.T)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title, width=width, height=height)
    vis.add_geometry(arm_geom)
    vis.add_geometry(cad_geom)
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
    vis.get_render_option().point_size = 2.5

    confirmed = {"value": False}
    last_T_holder: dict[str, np.ndarray] = {"T": state.T}

    def _apply_delta(delta: np.ndarray) -> None:
        state.T = delta @ state.T
        cad_geom.points = o3d.utility.Vector3dVector(cad_local)
        cad_geom.transform(state.T)
        cad_geom.paint_uniform_color((0.95, 0.15, 0.15))
        vis.update_geometry(cad_geom)
        last_T_holder["T"] = state.T

    def _reset(target: np.ndarray) -> None:
        state.T = target.copy()
        _apply_delta(np.eye(4))

    # Translation bindings (camera frame).
    def _make_t_cb(dx, dy, dz):
        def cb(vis_):  # type: ignore[unused-argument]
            _apply_delta(_translate(dx * state.step_m, dy * state.step_m, dz * state.step_m))
            return False
        return cb

    vis.register_key_callback(ord("A"), _make_t_cb(-1, 0, 0))
    vis.register_key_callback(ord("D"), _make_t_cb(+1, 0, 0))
    vis.register_key_callback(ord("W"), _make_t_cb(0, -1, 0))
    vis.register_key_callback(ord("S"), _make_t_cb(0, +1, 0))
    vis.register_key_callback(ord("Q"), _make_t_cb(0, 0, -1))
    vis.register_key_callback(ord("E"), _make_t_cb(0, 0, +1))

    # Rotation bindings around current centroid (camera frame axes).
    def _make_r_cb(axis: str, sign: int):
        def cb(vis_):  # type: ignore[unused-argument]
            angle = sign * state.rot_step_rad
            # Rotate about the CAD's current centroid to avoid translation drift.
            pts = np.asarray(cad_geom.points)
            c = pts.mean(axis=0) if len(pts) else np.zeros(3)
            T_to_origin = np.eye(4); T_to_origin[:3, 3] = -c
            T_back = np.eye(4); T_back[:3, 3] = +c
            R = _rot_axis(axis, angle)
            delta = T_back @ R @ T_to_origin
            _apply_delta(delta)
            return False
        return cb

    vis.register_key_callback(ord("J"), _make_r_cb("y", -1))  # yaw -
    vis.register_key_callback(ord("L"), _make_r_cb("y", +1))  # yaw +
    vis.register_key_callback(ord("I"), _make_r_cb("x", -1))  # pitch -
    vis.register_key_callback(ord("K"), _make_r_cb("x", +1))  # pitch +
    vis.register_key_callback(ord("U"), _make_r_cb("z", -1))  # roll -
    vis.register_key_callback(ord("O"), _make_r_cb("z", +1))  # roll +

    # Step control.
    def _bump_step(factor: float):
        def cb(vis_):  # type: ignore[unused-argument]
            state.step_m = max(1e-4, min(0.5, state.step_m * factor))
            print(f"[manual-align] step = {state.step_m*1000:.2f} mm")
            return False
        return cb

    def _bump_rot(factor: float):
        def cb(vis_):  # type: ignore[unused-argument]
            state.rot_step_rad = max(1e-4, min(np.pi / 2, state.rot_step_rad * factor))
            print(f"[manual-align] rot step = {np.degrees(state.rot_step_rad):.2f} deg")
            return False
        return cb

    vis.register_key_callback(ord("="), _bump_step(2.0))  # '+' without shift on most layouts
    vis.register_key_callback(ord("+"), _bump_step(2.0))
    vis.register_key_callback(ord("-"), _bump_step(0.5))
    vis.register_key_callback(ord("]"), _bump_rot(2.0))
    vis.register_key_callback(ord("["), _bump_rot(0.5))

    # Reset.
    vis.register_key_callback(ord("R"), lambda _v: (_reset(np.eye(4)), False)[1])
    vis.register_key_callback(ord("Z"), lambda _v: (_reset(state.T_init), False)[1])

    # ICP snap (SPACE = 32).
    def _icp_snap(_v):
        try:
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(cad_local)
            src.transform(state.T)
            src_t = o3d.t.geometry.PointCloud.from_legacy(src)
            tgt_t = o3d.t.geometry.PointCloud.from_legacy(arm_geom)
            tgt_t.estimate_normals(radius=0.02, max_nn=30)
            reg = o3d.t.pipelines.registration.icp(
                src_t,
                tgt_t,
                max_correspondence_distance=icp_threshold_m,
                init_source_to_target=o3d.core.Tensor(np.eye(4), dtype=o3d.core.Dtype.Float64),
                estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
            )
            delta = reg.transformation.numpy()
            print(
                f"[manual-align] ICP snap: fitness={reg.fitness:.3f} "
                f"rmse={reg.inlier_rmse*1000:.2f}mm"
            )
            _apply_delta(np.asarray(delta, dtype=np.float64))
        except Exception as exc:  # pragma: no cover - GUI runtime
            print(f"[manual-align] ICP snap failed: {exc}")
        return False

    vis.register_key_callback(32, _icp_snap)  # SPACE

    # Confirm (ENTER = 257 on Open3D / GLFW; also bind RIGHT-BRACKET fallback handled above).
    def _confirm(_v):
        confirmed["value"] = True
        vis.close()
        return False

    vis.register_key_callback(257, _confirm)  # ENTER
    vis.register_key_callback(ord("\r"), _confirm)

    print(
        "[manual-align] controls: WASDQE=translate, IJKL/UO=rotate, "
        "+/-/[]=steps, SPACE=ICP snap, R=reset to identity, Z=reset to "
        "init, ENTER=accept, close window=cancel"
    )
    vis.run()
    vis.destroy_window()

    if confirmed["value"]:
        return last_T_holder["T"]
    return None


__all__ = ["run_manual_align", "ManualAlignState"]
