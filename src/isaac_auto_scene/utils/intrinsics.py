"""Camera intrinsics helpers: RealSense K matrix → Isaac Sim projection params.

CameraIntrinsics dataclass lives here so capture.py, mocks.py, and scene_gen.py
all import from the same location.

Phase 6 — realsense_to_isaac(K, w, h) conversion helper (research §8 formula).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Soft-import: only used in intrinsics_from_realsense_profile
try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    All values are in pixels except where noted.
    width/height are the sensor dimensions in pixels.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        """Return the 3×3 intrinsics matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_open3d(self) -> "o3d.camera.PinholeCameraIntrinsic":  # type: ignore[name-defined]  # noqa: F821
        """Return an Open3D PinholeCameraIntrinsic equivalent."""
        import open3d as o3d

        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )


def intrinsics_from_realsense_profile(profile: object) -> CameraIntrinsics:
    """Extract CameraIntrinsics from a pyrealsense2 stream profile.

    Only callable when pyrealsense2 is available.  The caller (RealSenseD435Source)
    must guard this itself.

    Parameters
    ----------
    profile:
        A ``pyrealsense2.video_stream_profile`` (color stream profile).
    """
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 not installed; pixi install -e hardware"
        )
    intr = profile.as_video_stream_profile().get_intrinsics()  # type: ignore[attr-defined]
    return CameraIntrinsics(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
    )


def realsense_to_isaac(
    K: np.ndarray,
    width: int,
    height: int,
    aperture_mm: float = 20.955,
) -> dict[str, float | int]:
    """Convert RealSense intrinsics K to Isaac Sim PinholeCameraCfg parameters.

    Formula (research §8):
        focal_length_mm = fx_px * aperture_mm / width_px
        horizontal_aperture_offset_mm = (cx - width/2) * aperture_mm / width_px

    Parameters
    ----------
    K:
        3×3 camera intrinsics matrix.
    width:
        Image width in pixels.
    height:
        Image height in pixels.
    aperture_mm:
        Horizontal aperture in mm (default D435 value 20.955 mm).

    Returns
    -------
    dict
        Keys: focal_length, horizontal_aperture, horizontal_aperture_offset,
        height, width.  Pass as ``**realsense_to_isaac(K, w, h)`` to
        ``sim_utils.PinholeCameraCfg``.
    """
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    focal_length_mm = fx * aperture_mm / width
    h_aperture_offset_mm = (cx - width / 2.0) * aperture_mm / width
    return {
        "focal_length": focal_length_mm,
        "horizontal_aperture": aperture_mm,
        "horizontal_aperture_offset": h_aperture_offset_mm,
        "height": height,
        "width": width,
    }
