"""Versioned annotation project persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from orto_annotator.core.mesh import SourceMesh

PROJECT_FORMAT = "orto-prosthesis-annotation"
PROJECT_FORMAT_VERSION = 1


def save_project(path: str | Path, source: SourceMesh, labels: np.ndarray) -> None:
    target = Path(path).expanduser().resolve()
    values = np.asarray(labels, dtype=np.uint8)
    if values.shape != (source.face_count,):
        raise ValueError("Количество меток не совпадает с количеством граней исходной сетки.")
    if np.any(values > 4):
        raise ValueError("Обнаружены неизвестные значения классов.")
    payload = {
        "format": PROJECT_FORMAT,
        "format_version": PROJECT_FORMAT_VERSION,
        "application_version": "0.1.0",
        "created_or_saved_at": datetime.now(UTC).isoformat(),
        "source": {
            "filename": source.path.name,
            "geometry_fingerprint_sha256": source.fingerprint,
            "point_count": int(source.mesh.n_points),
            "face_count": int(source.face_count),
        },
        "labels": values.tolist(),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path, source: SourceMesh) -> np.ndarray:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("format") != PROJECT_FORMAT:
        raise ValueError("Это не проект ORTO Prosthesis Annotator.")
    if payload.get("format_version") != PROJECT_FORMAT_VERSION:
        raise ValueError("Версия формата проекта не поддерживается.")
    source_info = payload.get("source", {})
    if source_info.get("geometry_fingerprint_sha256") != source.fingerprint:
        raise ValueError("Проект относится к другой геометрии STL. Разметка не загружена.")
    labels = np.asarray(payload.get("labels", []), dtype=np.uint8)
    if labels.shape != (source.face_count,) or np.any(labels > 4):
        raise ValueError("Файл проекта содержит некорректную разметку.")
    return labels
