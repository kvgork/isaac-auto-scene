"""Mock hardware interfaces for CI and unit testing without physical devices.

MockD435Source generates deterministic synthetic RGB-D frames: a flat table
plane (Z = z_plane) with a half-sphere "arm proxy" sitting on it.  Frame-to-
frame depth noise is Gaussian so the temporal-median path in capture.py has
real work to do.

Phase 2 — MockD435Stream (synthetic RGB-D frames), MockArm (zero-pose FK).
"""

from __future__ import annotations

import numpy as np

from isaac_auto_scene.utils.intrinsics import CameraIntrinsics

# Default synthetic intrinsics mirroring D435 @ 640×480
_DEFAULT_INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=385.0,
    fy=385.0,
    cx=320.0,
    cy=240.0,
)


class MockD435Source:
    """Deterministic synthetic D435 frames.

    Generates a flat table plane (Z = z_plane) with a half-sphere "arm proxy"
    sitting on it.  RGB is a fixed gradient.  Frame-to-frame noise is Gaussian
    on depth so temporal median has work to do.

    Parameters
    ----------
    width, height:
        Frame dimensions in pixels.
    z_plane:
        Depth of the flat table plane in metres.
    noise_std:
        Standard deviation of per-pixel Gaussian depth noise in metres.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        z_plane: float = 0.6,
        noise_std: float = 0.002,
        seed: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.z_plane = z_plane
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._started = False
        self._intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=385.0 * width / 640,
            fy=385.0 * height / 480,
            cx=width / 2.0,
            cy=height / 2.0,
        )
        # Build the noiseless depth and RGB base images once
        self._depth_base, self._rgb_base = self._build_base_images()

    # ------------------------------------------------------------------
    # Protocol: start / stop / read_frame / context-manager
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read_frame(self) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Return (rgb_u8, depth_u16, intrinsics) for one synthetic frame.

        RGB shape  : (H, W, 3) uint8
        Depth shape: (H, W)    uint16  (millimetres, matching D435 default)
        """
        if not self._started:
            raise RuntimeError("MockD435Source: call start() before read_frame()")

        # Gaussian depth noise in metres, then convert to uint16 millimetres
        noise_m = self._rng.normal(0.0, self.noise_std, (self.height, self.width)).astype(
            np.float32
        )
        depth_m = self._depth_base + noise_m
        depth_m = np.clip(depth_m, 0.0, 3.0)
        depth_u16 = (depth_m * 1000.0).astype(np.uint16)

        return self._rgb_base.copy(), depth_u16, self._intrinsics

    def __enter__(self) -> "MockD435Source":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_base_images(self) -> tuple[np.ndarray, np.ndarray]:
        """Build noiseless depth (float32 m) and RGB (uint8) images."""
        h, w = self.height, self.width

        # Pixel grid (image coordinates)
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # Default: entire image at z_plane depth
        depth = np.full((h, w), self.z_plane, dtype=np.float32)

        # Place a half-sphere "arm proxy" centred in the frame.
        # The sphere has a radius such that it protrudes ~0.15 m above the table.
        sphere_radius_px = 0.12 * self._intrinsics.fx  # ~46 px at default fx
        cx, cy = w / 2.0, h / 2.0
        sphere_r_m = 0.12  # metres

        dist_px = np.sqrt((uu - cx) ** 2 + (vv - cy) ** 2)
        mask = dist_px < sphere_radius_px
        # Z on the sphere surface (front hemisphere)
        z_sphere = self.z_plane - np.sqrt(
            np.maximum(0.0, sphere_r_m**2 - ((dist_px / sphere_radius_px) * sphere_r_m) ** 2)
        )
        depth = np.where(mask, z_sphere, depth)

        # Simple gradient RGB: red channel increases left-right,
        # green channel increases top-bottom.
        r_ch = (uu / w * 200 + 30).astype(np.uint8)
        g_ch = (vv / h * 200 + 30).astype(np.uint8)
        b_ch = np.full((h, w), 120, dtype=np.uint8)
        rgb = np.stack([r_ch, g_ch, b_ch], axis=2)

        return depth, rgb
