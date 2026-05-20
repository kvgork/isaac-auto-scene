"""RealSense D435 source implementing the D435Source protocol.

Soft-import: pyrealsense2 is imported lazily so the default environment
(no RealSense) can import this module without error.  Instantiating
RealSenseD435Source without pyrealsense2 raises RuntimeError.

Phase 2 — hardware-gated implementation of the D435Source protocol.
"""

from __future__ import annotations

import numpy as np

from isaac_auto_scene.utils.intrinsics import CameraIntrinsics, intrinsics_from_realsense_profile

# Soft import ---------------------------------------------------------------
try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]
except ImportError:
    rs = None  # type: ignore[assignment]
# ---------------------------------------------------------------------------


class RealSenseD435Source:
    """D435Source implementation backed by a real RealSense D435 camera.

    Streams depth + colour at 640×480 @ 30 fps.  The depth stream is aligned to
    the colour frame via ``rs.align`` so intrinsics describe the colour sensor.

    Raises
    ------
    RuntimeError
        If pyrealsense2 is not installed (raised at instantiation time, not
        import time, so the module can always be imported).
    """

    def __init__(self) -> None:
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 not installed; pixi install -e hardware"
            )
        self._pipeline: object = None
        self._align: object = None
        self._spatial_filter: object = None
        self._temporal_filter: object = None
        self._intrinsics: CameraIntrinsics | None = None

    # ------------------------------------------------------------------
    # Protocol: start / stop / read_frame / context-manager
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the RealSense pipeline and filters."""
        pipeline = rs.pipeline()  # type: ignore[union-attr]
        cfg = rs.config()  # type: ignore[union-attr]
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # type: ignore[union-attr]
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)  # type: ignore[union-attr]
        profile = pipeline.start(cfg)

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)  # type: ignore[union-attr]
        self._spatial_filter = rs.spatial_filter()  # type: ignore[union-attr]
        self._temporal_filter = rs.temporal_filter()  # type: ignore[union-attr]

        # Extract intrinsics from the colour stream profile
        color_profile = (
            profile.get_stream(rs.stream.color)  # type: ignore[union-attr]
        )
        self._intrinsics = intrinsics_from_realsense_profile(color_profile)

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()  # type: ignore[union-attr]
            self._pipeline = None

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Return (rgb_u8, depth_u16, intrinsics) for one hardware frame.

        The depth frame has hardware spatial+temporal filters applied.

        RGB shape  : (H, W, 3) uint8
        Depth shape: (H, W)    uint16  (D435 default: 1 unit = 1 mm)
        """
        if self._pipeline is None or self._intrinsics is None:
            raise RuntimeError("RealSenseD435Source: call start() before read_frame()")

        frames = self._pipeline.wait_for_frames()  # type: ignore[union-attr]
        aligned = self._align.process(frames)  # type: ignore[union-attr]

        depth_frame = aligned.get_depth_frame()
        depth_frame = self._spatial_filter.process(depth_frame)  # type: ignore[union-attr]
        depth_frame = self._temporal_filter.process(depth_frame)  # type: ignore[union-attr]

        color_frame = aligned.get_color_frame()

        depth_u16 = np.asanyarray(depth_frame.get_data())
        rgb_u8 = np.asanyarray(color_frame.get_data())

        return rgb_u8, depth_u16, self._intrinsics

    def __enter__(self) -> "RealSenseD435Source":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
