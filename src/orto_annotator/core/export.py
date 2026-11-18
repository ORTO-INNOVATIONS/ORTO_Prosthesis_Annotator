"""Training-data export with per-face labels and source identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orto_annotator.core.labels import LABEL_NAMES, Label
from orto_annotator.core.mesh import SourceMesh


def export_dataset_item(directory: str | Path, source: SourceMesh, labels: np.ndarray) -> tuple[Path, Path]:
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    values = np.asarray(labels, dtype=np.uint8)
    if values.shape != (source.face_count,):
        raise ValueError("Количество меток не совпадает с количеством граней.")
    stem = source.path.stem
    arrays_path = target / f"{stem}.labels.npz"
    metadata_path = target / f"{stem}.labels.json"
    np.savez_compressed(
        arrays_path,
        vertices=np.asarray(source.mesh.points, dtype=np.float32),
        faces=np.asarray(source.mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32),
        face_labels=values,
    )
    metadata = {
        "format": "orto-semantic-mesh-labels",
        "format_version": 1,
        "source_filename": source.path.name,
        "geometry_fingerprint_sha256": source.fingerprint,
        "point_count": int(source.mesh.n_points),
        "face_count": int(source.face_count),
        "label_encoding": {str(int(label)): LABEL_NAMES[label] for label in Label},
        "class_counts": {
            LABEL_NAMES[label]: int(np.count_nonzero(values == int(label)))
            for label in Label
        },
        "training_ready": bool(np.all(values != int(Label.UNLABELED))),
        "scope": "lower-leg prosthesis; complete or partially occluded connector",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays_path, metadata_path
