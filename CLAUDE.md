# isaac-auto-scene — project notes

## Pixi environments
- `default`: `dev + geometry` (Open3D, trimesh, yourdfpy, scipy). No hardware. No Isaac Sim.
- `sim`: adds `isaac` feature (Isaac Sim + Isaac Lab via `pixi run install-isaac`).
- `hardware`: adds `realsense` feature (`pyrealsense2`).
- `full`: all features.

Run unit tests with `pixi run test` (default env). Hardware-gated tests via
`pixi run test-hw` (needs physical D435).

## Open3D 0.18 pitfalls observed on this build
- `o3d.geometry.PointCloud.create_from_rgbd_image(...)` — **segfaults**. Use
  manual back-projection (`(u - cx) * z / fx`, `(v - cy) * z / fy`).
- `o3d.pipelines.registration.registration_fgr_based_on_feature_matching` —
  **segfaults** on this build.
- `o3d.pipelines.registration.registration_ransac_based_on_feature_matching` —
  **segfaults**.
- `o3d.pipelines.registration.registration_icp` (legacy) — **segfaults** on
  small synthetic clouds, regardless of point-to-point vs point-to-plane.
- `o3d.pipelines.registration.registration_generalized_icp` — **segfaults**.
- `o3d.geometry.PointCloud(src).transform(T)` (copy-constructor then transform)
  **corrupts internal state**; subsequent `voxel_down_sample` segfaults. Build
  cloned point clouds from numpy directly instead.

Stable replacement: **`o3d.t.pipelines.registration.icp` (tensor API)**. We
use this for all ICP in `register.py` and run several random-rotation
restarts in lieu of FGR/RANSAC for the global init.

## Convention reminders (from the plan)
- Quaternions: **XYZW** order (Isaac Lab + SciPy default).
- Joint config: SO-101 midpoint-zero (URDF source
  `TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`).
- Isaac Sim 5.1 / Isaac Lab v2.3.2 only. Renderer needs `enable_cameras=True`
  and the mandatory 30-frame warm-up (`scene_gen.WARM_UP_FRAMES`).
- Quality gate: `fitness >= 0.65`, `inlier_rmse <= 5 mm`. Defined in
  `register.QUALITY_GATE`.

## Test fixtures
- `tests/fixtures/synthetic_pcd.py` — synthetic table + arm proxy PCD.
- `tests/fixtures/minimal_urdf.py` — 2-link revolute URDF (no STL deps).
Both keep CI free of hardware and external assets.
