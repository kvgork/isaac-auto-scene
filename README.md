# isaac-auto-scene

**Status: Phase 1 bootstrap**

A standalone Python package that captures one RealSense D435 RGB-D frame from a fixed mount, segments the table plane and isolates the SO-101 arm point cloud, runs FPFH+FGR global registration followed by GICP local refinement against the SO-101 CAD model (assembled via URDF forward kinematics at known encoder angles), and emits an Isaac Sim USD scene that mirrors the real setup — including D435 camera, SO-101 arm, and table at registered poses with dome lighting and an optional ROS2 camera publisher.

## Install

```bash
# Default environment (dev + geometry deps, no Isaac Sim)
pixi install

# Simulation environment (adds Isaac Sim feature)
pixi install -e sim

# Install Isaac Sim and Isaac Lab into the active sim/full environment
pixi run install-isaac

# Hardware environment (adds RealSense support)
pixi install -e hardware

# Full environment (all features)
pixi install -e full
```

## Development tasks

```bash
pixi run test          # Run all unit tests (no hardware required)
pixi run test-hw       # Run all tests including hardware-gated ones
pixi run lint          # ruff check
pixi run fmt           # ruff format
pixi run typecheck     # mypy
```

## CLI (Phase 6)

```bash
isaac-auto-scene calibrate   # capture → segment → register → calib.json
isaac-auto-scene generate    # calib.json → Isaac Sim USD scene
isaac-auto-scene render      # headless render to PNG frames
isaac-auto-scene validate    # forward-projection residual report
```

> The CLI is stubbed in Phase 1. Full implementation lands in Phase 6.

## Reference

- Plan: `01-Projects/isaac-auto-scene-package-plan.md` (vault)
- Research: `05-Wiki/research/2026-05-18-isaac-auto-scene-from-d435.md` (vault)
- Isaac Sim version: 5.1 | Isaac Lab version: v2.3.2 | Python: 3.11

## Phase roadmap

- [x] Phase 1 — Workspace bootstrap (this commit)
- [ ] Phase 2 — D435 capture + temporal-median preprocessing
- [ ] Phase 3 — Table-plane segmentation + arm cloud isolation
- [ ] Phase 4 — URDF FK assembly to point cloud (parallel with Phase 3)
- [ ] Phase 5 — FPFH+FGR→GICP registration + quality gate + calib.json
- [ ] Phase 6 — USD scene generation + full Click CLI + optional ROS2 bridge

## License

MIT — see [LICENSE](LICENSE).
