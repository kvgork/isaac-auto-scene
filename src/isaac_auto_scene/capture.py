"""D435 RGB-D capture pipeline with temporal-median preprocessing.

Public API
----------
CaptureResult   — frozen dataclass holding RGB, depth, intrinsics, pcd
capture()       — collect N frames, apply temporal median, build point cloud
save_capture()  — write pcd.ply / depth_median.npy / rgb.png / intrinsics.json
load_capture()  — reload a saved capture from disk

Phase 2 — pyrealsense2 pipeline, 30-frame temporal median,
           spatial+temporal filters; hardware-mockable interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

import open3d as o3d

from isaac_auto_scene.utils.intrinsics import CameraIntrinsics

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# D435Source protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class D435Source(Protocol):
    """Protocol that any D435-compatible source must implement.

    Both ``RealSenseD435Source`` and ``MockD435Source`` satisfy this protocol.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Return (rgb_u8, depth_u16, intrinsics).

        rgb_u8   : (H, W, 3) uint8
        depth_u16: (H, W)    uint16  (raw sensor units; multiply by depth_unit
                                      to get metres)
        """
        ...

    def __enter__(self) -> "D435Source": ...

    def __exit__(self, *args: object) -> None: ...


# ---------------------------------------------------------------------------
# CaptureResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureResult:
    """Output of a single capture + preprocessing run.

    Attributes
    ----------
    rgb:
        (H, W, 3) uint8 colour image.
    depth:
        (H, W) float32 depth in **metres**.
    intrinsics:
        Camera K matrix + image dimensions.
    pcd:
        Coloured Open3D point cloud in the camera frame.  NaN/inf, zero-depth,
        and far (> 2.0 m) points have been removed.
    depth_unit:
        Metres per raw uint16 unit.  Default 1e-3 (D435: 1 unit = 1 mm).
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    pcd: o3d.geometry.PointCloud
    depth_unit: float = 1e-3


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------


def capture(
    *,
    source: D435Source,
    num_frames: int = 30,
    apply_spatial: bool = True,
    apply_temporal: bool = True,
) -> CaptureResult:
    """Capture and preprocess an RGB-D frame from *source*.

    Strategy
    --------
    1. Collect ``num_frames`` raw depth frames (hardware filters applied
       inside :class:`RealSenseD435Source`; skipped for mock sources).
    2. Stack depth arrays and compute per-pixel median → noise reduction.
    3. Take the colour frame from the *last* iteration (arm is static).
    4. Convert raw uint16 depth to float32 metres.
    5. Build a coloured Open3D point cloud; strip invalid points.

    Parameters
    ----------
    source:
        Any object satisfying the :class:`D435Source` protocol.
    num_frames:
        Number of frames to collect for temporal median (default 30).
    apply_spatial, apply_temporal:
        Flags passed for documentation purposes.  The actual RS2 filters are
        applied inside :class:`RealSenseD435Source`; they have no effect on
        :class:`MockD435Source`.

    Returns
    -------
    CaptureResult
    """
    depth_stack: list[np.ndarray] = []
    rgb_last: np.ndarray | None = None
    intrinsics_last: CameraIntrinsics | None = None

    for _ in range(num_frames):
        rgb, depth_u16, intr = source.read_frame()
        depth_stack.append(depth_u16.astype(np.float32))
        rgb_last = rgb
        intrinsics_last = intr

    assert rgb_last is not None and intrinsics_last is not None

    # Per-pixel temporal median over all collected frames
    depth_median_raw = np.median(np.stack(depth_stack, axis=0), axis=0).astype(np.float32)

    # Convert raw uint16 units → metres
    depth_unit = 1e-3  # D435 default: 1 count = 1 mm
    depth_m = depth_median_raw * depth_unit

    # Build coloured RGBD image and point cloud
    pcd = _build_pcd(rgb_last, depth_m, intrinsics_last)

    return CaptureResult(
        rgb=rgb_last,
        depth=depth_m,
        intrinsics=intrinsics_last,
        pcd=pcd,
        depth_unit=depth_unit,
    )


def _build_pcd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> o3d.geometry.PointCloud:
    """Build a coloured, filtered Open3D point cloud from RGB + depth arrays.

    Manual backprojection: x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    Avoids Open3D's create_from_rgbd_image (known segfaults on some
    Open3D/Python builds). Points with depth = 0, depth > 2.0 m,
    NaN, or inf are removed.
    """
    h, w = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= 2.0)

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_m.astype(np.float32)
    x = (uu - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
    y = (vv - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)

    pts = np.stack([x, y, z], axis=-1)[valid]
    colors = (rgb[valid].astype(np.float32) / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


# ---------------------------------------------------------------------------
# save_capture / load_capture
# ---------------------------------------------------------------------------


def save_capture(result: CaptureResult, out_dir: Path) -> None:
    """Persist a CaptureResult to *out_dir*.

    Written files
    -------------
    pcd.ply             — coloured point cloud (ASCII PLY)
    depth_median.npy    — float32 depth map in metres
    rgb.png             — colour image
    intrinsics.json     — camera K + dimensions + depth_unit
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Point cloud
    o3d.io.write_point_cloud(str(out_dir / "pcd.ply"), result.pcd)

    # Depth (metres, float32)
    np.save(str(out_dir / "depth_median.npy"), result.depth)

    # RGB image
    Image.fromarray(result.rgb).save(str(out_dir / "rgb.png"))

    # Intrinsics + meta
    meta = {
        "width": result.intrinsics.width,
        "height": result.intrinsics.height,
        "fx": result.intrinsics.fx,
        "fy": result.intrinsics.fy,
        "cx": result.intrinsics.cx,
        "cy": result.intrinsics.cy,
        "depth_unit": result.depth_unit,
    }
    (out_dir / "intrinsics.json").write_text(json.dumps(meta, indent=2))


def load_capture(in_dir: Path) -> CaptureResult:
    """Load a CaptureResult previously saved by :func:`save_capture`.

    Parameters
    ----------
    in_dir:
        Directory written by ``save_capture``.

    Returns
    -------
    CaptureResult
        The pcd is reconstructed from *pcd.ply*; all other fields are
        reloaded from the saved artefacts.
    """
    from PIL import Image

    in_dir = Path(in_dir)

    # Intrinsics + meta
    meta = json.loads((in_dir / "intrinsics.json").read_text())
    intrinsics = CameraIntrinsics(
        width=int(meta["width"]),
        height=int(meta["height"]),
        fx=float(meta["fx"]),
        fy=float(meta["fy"]),
        cx=float(meta["cx"]),
        cy=float(meta["cy"]),
    )
    depth_unit = float(meta.get("depth_unit", 1e-3))

    # Depth
    depth = np.load(str(in_dir / "depth_median.npy"))

    # RGB
    rgb = np.asarray(Image.open(str(in_dir / "rgb.png")))

    # Point cloud
    pcd = o3d.io.read_point_cloud(str(in_dir / "pcd.ply"))

    return CaptureResult(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        pcd=pcd,
        depth_unit=depth_unit,
    )


# ---------------------------------------------------------------------------
# Re-export MockD435Source for convenience  (acceptance criterion §4)
# ---------------------------------------------------------------------------

from isaac_auto_scene.utils.mocks import MockD435Source as MockD435Source  # noqa: E402, F401
