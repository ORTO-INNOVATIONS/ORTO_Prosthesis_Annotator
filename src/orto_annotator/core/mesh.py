"""Source mesh loading and stable geometry fingerprinting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass
class SourceMesh:
    path: Path
    mesh: pv.PolyData
    fingerprint: str
    face_centers: np.ndarray

    @property
    def face_count(self) -> int:
        return self.mesh.n_cells


def geometry_fingerprint(mesh: pv.PolyData) -> str:
    """Hash ordered vertices and triangular connectivity, not the file container."""
    if not mesh.is_all_triangles:
        raise ValueError("STL mesh must contain triangles only.")
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(points.shape, dtype=np.int64).tobytes())
    digest.update(points.tobytes(order="C"))
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()


def load_stl(path: str | Path) -> SourceMesh:
    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() != ".stl":
        raise ValueError("Поддерживается только формат STL.")
    mesh = pv.read(source_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate(inplace=False)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("STL не содержит геометрии.")
    if not mesh.is_all_triangles:
        raise ValueError("Не удалось привести поверхность к треугольной сетке.")
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    return SourceMesh(
        path=source_path,
        mesh=mesh,
        fingerprint=geometry_fingerprint(mesh),
        face_centers=centers,
    )
