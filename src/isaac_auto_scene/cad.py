"""URDF -> assembled point cloud via forward kinematics (Phase 4).

Public API
----------
CADResult           — assembled trimesh, points (N,3), per-link transforms
load_urdf()         — yourdfpy.URDF wrapper that resolves mesh paths
assemble_pcd()      — FK at joint angles -> combined mesh -> Poisson-disk sample

Strategy
--------
1. Load URDF with yourdfpy; configure joint angles via ``update_cfg``.
2. For every link with a visual mesh, look up the world-frame transform from
   yourdfpy's scene graph; transform the trimesh into world.
3. Concatenate transformed trimeshes into one ``trimesh.Trimesh``.
4. Sample on the merged surface with Open3D's Poisson-disk sampler at the
   requested point spacing.

The merged trimesh + point array + per-link FK dict are returned together
so callers can cache by joint-angle hash without re-doing FK.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import yourdfpy


@dataclass(frozen=True)
class CADResult:
    """Output of one URDF -> point-cloud assembly run.

    Attributes
    ----------
    mesh:
        Merged ``trimesh.Trimesh`` of all visual links at the requested joint
        configuration, expressed in the URDF root frame.
    points:
        (N, 3) float64 array of Poisson-disk samples on ``mesh``.
    link_transforms:
        Mapping link name -> 4x4 homogeneous transform in the root frame.
    joint_angles:
        Joint-angle dict that produced this configuration (key -> radians).
    """

    mesh: trimesh.Trimesh
    points: np.ndarray
    link_transforms: dict[str, np.ndarray]
    joint_angles: dict[str, float]


def load_urdf(urdf_path: Path | str) -> yourdfpy.URDF:
    """Load a URDF with yourdfpy, resolving mesh paths relative to the file."""
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    return yourdfpy.URDF.load(str(urdf_path))


def _link_world_transform(urdf: yourdfpy.URDF, link_name: str) -> np.ndarray:
    """Return the 4x4 world transform of ``link_name`` from yourdfpy state."""
    return np.asarray(urdf.get_transform(link_name), dtype=np.float64)


def _link_visual_mesh(urdf: yourdfpy.URDF, link_name: str) -> trimesh.Trimesh | None:
    """Return a single ``trimesh.Trimesh`` for the link's visual, or None."""
    link = urdf.link_map.get(link_name)
    if link is None or not link.visuals:
        return None

    parts: list[trimesh.Trimesh] = []
    for visual in link.visuals:
        geom = visual.geometry
        if geom is None:
            continue
        local_mesh: trimesh.Trimesh | None = None

        if geom.mesh is not None:
            scene_or_mesh = urdf.scene.geometry.get(visual.name) if visual.name else None
            if scene_or_mesh is None:
                fname = geom.mesh.filename
                resolved = _resolve_mesh_path(urdf, fname)
                if resolved is None:
                    continue
                loaded = trimesh.load(str(resolved), force="mesh")
                if isinstance(loaded, trimesh.Trimesh):
                    local_mesh = loaded
            else:
                if isinstance(scene_or_mesh, trimesh.Trimesh):
                    local_mesh = scene_or_mesh
        elif geom.box is not None:
            local_mesh = trimesh.creation.box(extents=np.asarray(geom.box.size))
        elif geom.cylinder is not None:
            local_mesh = trimesh.creation.cylinder(
                radius=geom.cylinder.radius,
                height=geom.cylinder.length,
            )
        elif geom.sphere is not None:
            local_mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)

        if local_mesh is None:
            continue

        if visual.origin is not None:
            local_mesh = local_mesh.copy()
            local_mesh.apply_transform(np.asarray(visual.origin, dtype=np.float64))

        parts.append(local_mesh)

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)


def _resolve_mesh_path(urdf: yourdfpy.URDF, fname: str) -> Path | None:
    """Resolve a URDF mesh ``filename`` against the URDF directory."""
    if fname.startswith("package://"):
        return None
    candidate = Path(fname)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    urdf_dir = _urdf_directory(urdf)
    if urdf_dir is not None:
        rel = urdf_dir / fname
        if rel.exists():
            return rel
    return None


def _urdf_directory(urdf: yourdfpy.URDF) -> Path | None:
    """Return the directory the URDF was loaded from, or None.

    yourdfpy doesn't expose the source path directly on the URDF instance;
    it stashes a ``functools.partial`` filename handler with the directory
    baked in via ``keywords["dir"]``.  Older builds exposed ``_filename``
    on the URDF directly — check both.
    """
    direct = getattr(urdf, "_filename", None)
    if direct:
        return Path(direct).parent
    handler = getattr(urdf, "_filename_handler", None)
    if handler is not None and hasattr(handler, "keywords"):
        d = handler.keywords.get("dir")
        if d:
            return Path(d)
    return None


def assemble_pcd(
    urdf: yourdfpy.URDF,
    joint_angles: dict[str, float] | None = None,
    *,
    target_n_points: int = 15_000,
    init_factor: int = 5,
) -> CADResult:
    """Forward-kinematics assemble all visual links into one Poisson-disk PCD.

    Parameters
    ----------
    urdf:
        URDF loaded with :func:`load_urdf`.
    joint_angles:
        Mapping joint name -> radians.  Missing joints default to zero.
    target_n_points:
        Number of Poisson-disk samples in the output (default 15 k).
    init_factor:
        ``init_factor * target_n_points`` uniform samples are taken first;
        Poisson-disk is then run as a thinning pass.  5 matches the Open3D
        default.

    Returns
    -------
    CADResult
    """
    cfg: dict[str, float] = dict(joint_angles or {})
    if cfg:
        urdf.update_cfg(cfg)

    parts: list[trimesh.Trimesh] = []
    link_transforms: dict[str, np.ndarray] = {}

    for link_name in urdf.link_map.keys():
        T = _link_world_transform(urdf, link_name)
        link_transforms[link_name] = T

        mesh_local = _link_visual_mesh(urdf, link_name)
        if mesh_local is None:
            continue

        mesh_world = mesh_local.copy()
        mesh_world.apply_transform(T)
        parts.append(mesh_world)

    if not parts:
        raise ValueError("URDF has no visual meshes to assemble")

    merged = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    if not isinstance(merged, trimesh.Trimesh):
        raise TypeError(f"merged mesh has unexpected type: {type(merged)}")

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(merged.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(merged.faces, dtype=np.int32)),
    )
    o3d_mesh.compute_vertex_normals()

    pcd = o3d_mesh.sample_points_poisson_disk(
        number_of_points=target_n_points,
        init_factor=init_factor,
    )
    points = np.asarray(pcd.points, dtype=np.float64)

    full_cfg: dict[str, float] = {}
    for j_name in urdf.actuated_joint_names:
        full_cfg[j_name] = float(cfg.get(j_name, 0.0))

    return CADResult(
        mesh=merged,
        points=points,
        link_transforms=link_transforms,
        joint_angles=full_cfg,
    )
