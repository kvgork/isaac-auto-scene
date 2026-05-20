"""Tests for isaac_auto_scene.capture (Phase 2).

Hardware test is gated with @pytest.mark.hardware and skipped in CI.
All other tests run without any physical device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_auto_scene.capture import CaptureResult, MockD435Source, capture, load_capture, save_capture
from isaac_auto_scene.utils.intrinsics import CameraIntrinsics


# ---------------------------------------------------------------------------
# test_mock_source_yields_frame_shape
# ---------------------------------------------------------------------------


def test_mock_source_yields_frame_shape() -> None:
    """MockD435Source.read_frame returns arrays with expected shapes and dtypes."""
    with MockD435Source(width=640, height=480) as src:
        rgb, depth_u16, intr = src.read_frame()

    assert rgb.shape == (480, 640, 3), f"unexpected rgb shape: {rgb.shape}"
    assert rgb.dtype == np.uint8, f"unexpected rgb dtype: {rgb.dtype}"

    assert depth_u16.shape == (480, 640), f"unexpected depth shape: {depth_u16.shape}"
    assert depth_u16.dtype == np.uint16, f"unexpected depth dtype: {depth_u16.dtype}"

    assert isinstance(intr, CameraIntrinsics)
    assert intr.width == 640
    assert intr.height == 480


def test_mock_source_custom_size() -> None:
    """MockD435Source respects custom width/height."""
    with MockD435Source(width=320, height=240) as src:
        rgb, depth_u16, intr = src.read_frame()

    assert rgb.shape == (240, 320, 3)
    assert depth_u16.shape == (240, 320)
    assert intr.width == 320
    assert intr.height == 240


def test_mock_source_raises_before_start() -> None:
    """read_frame() raises RuntimeError if called before start()."""
    src = MockD435Source()
    with pytest.raises(RuntimeError, match="start"):
        src.read_frame()


# ---------------------------------------------------------------------------
# test_capture_temporal_median_reduces_noise
# ---------------------------------------------------------------------------


def test_capture_temporal_median_reduces_noise() -> None:
    """Temporal median over 30 noisy frames must reduce depth std by > 3×.

    The flat-table region (centre patch) has known Gaussian noise.  A single
    frame's std should be near noise_std; after median it should drop to
    roughly noise_std / sqrt(30) * correction.
    """
    noise_std = 0.003  # 3 mm — detectable with 30 frames
    src = MockD435Source(noise_std=noise_std, seed=1)

    # Single frame std on the flat table region (away from the sphere)
    with src:
        rgb_single, depth_u16_single, intr = src.read_frame()

    depth_m_single = depth_u16_single.astype(np.float32) * 1e-3

    # Flat region: top-left quadrant (no sphere overlap)
    h, w = intr.height, intr.width
    row_end = h // 4
    col_end = w // 4
    patch_single = depth_m_single[:row_end, :col_end]
    std_single = float(np.std(patch_single))

    # Now run the full capture with 30-frame median
    with MockD435Source(noise_std=noise_std, seed=2) as src2:
        result = capture(source=src2, num_frames=30)

    patch_median = result.depth[:row_end, :col_end]
    std_median = float(np.std(patch_median))

    ratio = std_single / (std_median + 1e-9)
    assert ratio > 3.0, (
        f"Temporal median did not reduce noise sufficiently: "
        f"std_single={std_single:.5f} m, std_median={std_median:.5f} m, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# test_save_load_roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """save_capture → load_capture roundtrip preserves key fields."""
    with MockD435Source(seed=99) as src:
        result = capture(source=src, num_frames=5)

    save_capture(result, tmp_path)

    # Check expected files exist
    assert (tmp_path / "pcd.ply").exists()
    assert (tmp_path / "depth_median.npy").exists()
    assert (tmp_path / "rgb.png").exists()
    assert (tmp_path / "intrinsics.json").exists()

    loaded = load_capture(tmp_path)

    # Point count must match
    n_orig = len(result.pcd.points)
    n_loaded = len(loaded.pcd.points)
    assert n_orig == n_loaded, (
        f"pcd point count changed on roundtrip: {n_orig} → {n_loaded}"
    )
    assert n_orig > 0, "pcd is empty"

    # Intrinsics K must be equal
    np.testing.assert_array_almost_equal(
        result.intrinsics.K,
        loaded.intrinsics.K,
        decimal=6,
        err_msg="intrinsics K changed on roundtrip",
    )

    # depth_unit preserved
    assert result.depth_unit == loaded.depth_unit

    # depth array shape preserved
    assert result.depth.shape == loaded.depth.shape


# ---------------------------------------------------------------------------
# test_realsense_source_soft_import_error
# ---------------------------------------------------------------------------


def test_realsense_source_soft_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """RealSenseD435Source raises RuntimeError mentioning 'pyrealsense2' when rs=None."""
    import isaac_auto_scene.realsense_source as rs_mod

    monkeypatch.setattr(rs_mod, "rs", None)

    with pytest.raises(RuntimeError, match="pyrealsense2"):
        rs_mod.RealSenseD435Source()


# ---------------------------------------------------------------------------
# test_capture_pcd_nonempty
# ---------------------------------------------------------------------------


def test_capture_pcd_nonempty() -> None:
    """Captured point cloud must contain a substantial number of points."""
    with MockD435Source(seed=7) as src:
        result = capture(source=src, num_frames=5)

    n_pts = len(result.pcd.points)
    # MockD435Source at 640×480 with z_plane=0.6 m should yield ~100 k points
    # (many will be clipped at 2 m threshold; the table plane is within range)
    assert n_pts > 1000, f"pcd has too few points: {n_pts}"


# ---------------------------------------------------------------------------
# test_capture_depth_in_metres
# ---------------------------------------------------------------------------


def test_capture_depth_in_metres() -> None:
    """CaptureResult.depth must be float32 in metres, matching z_plane."""
    z_plane = 0.6
    with MockD435Source(z_plane=z_plane, noise_std=0.0, seed=0) as src:
        result = capture(source=src, num_frames=3)

    assert result.depth.dtype == np.float32
    # Flat table region (top-left quadrant, well away from the sphere)
    h, w = result.intrinsics.height, result.intrinsics.width
    patch = result.depth[: h // 4, : w // 4]
    assert patch.mean() == pytest.approx(z_plane, abs=0.01), (
        f"depth patch mean {patch.mean():.4f} m != expected z_plane {z_plane} m"
    )


# ---------------------------------------------------------------------------
# Hardware-gated test (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_real_d435_capture(tmp_path: Path) -> None:
    """Run a real capture + save with physical D435 hardware.

    Requires --run-hardware flag.
    """
    from isaac_auto_scene.realsense_source import RealSenseD435Source

    with RealSenseD435Source() as src:
        result = capture(source=src, num_frames=30)

    assert len(result.pcd.points) > 5000

    save_capture(result, tmp_path)
    loaded = load_capture(tmp_path)
    assert len(loaded.pcd.points) == len(result.pcd.points)
