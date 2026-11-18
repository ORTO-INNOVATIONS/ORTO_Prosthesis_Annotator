"""Main annotation workstation window."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from orto_annotator.core.export import export_dataset_item
from orto_annotator.core.labels import LABEL_COLORS, LABEL_NAMES, Label
from orto_annotator.core.mesh import SourceMesh, load_stl
from orto_annotator.core.project import load_project, save_project


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ORTO Prosthesis Annotator 0.1.0 TEST")
        self.resize(1440, 900)
        self.source: SourceMesh | None = None
        self.labels = np.zeros(0, dtype=np.uint8)
        self.active_label = Label.SOCKET
        self.brush_radius = 12.0
        self.paint_enabled = False
        self._picker_observer = None

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#20252d")
        self.plotter.enable_trackball_style()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.plotter.interactor, 1)

        side = QVBoxLayout()
        side.setAlignment(Qt.AlignmentFlag.AlignTop)
        side.setSpacing(8)
        side.addWidget(QLabel("КЛАСС РАЗМЕТКИ"))
        self.class_buttons: dict[Label, QPushButton] = {}
        for label in (Label.SOCKET, Label.CONNECTOR, Label.FOOT, Label.UNCERTAIN, Label.UNLABELED):
            button = QPushButton(LABEL_NAMES[label])
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=label: self.set_active_label(value))
            button.setStyleSheet(
                f"QPushButton {{ text-align:left; padding:10px; border-left:8px solid "
                f"rgb{tuple(int(c * 255) for c in LABEL_COLORS[label])}; }}"
            )
            self.class_buttons[label] = button
            side.addWidget(button)
        self.class_buttons[self.active_label].setChecked(True)

        side.addSpacing(16)
        side.addWidget(QLabel("РАДИУС КИСТИ"))
        self.radius_label = QLabel("12 мм")
        side.addWidget(self.radius_label)
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(1, 100)
        self.radius_slider.setValue(12)
        self.radius_slider.valueChanged.connect(self._set_radius)
        side.addWidget(self.radius_slider)

        self.paint_button = QPushButton("Включить разметку")
        self.paint_button.setCheckable(True)
        self.paint_button.toggled.connect(self.set_paint_enabled)
        side.addWidget(self.paint_button)

        side.addSpacing(16)
        side.addWidget(QLabel("ПРОЕКТ"))
        for caption, callback in (
            ("Открыть STL", self.open_stl),
            ("Сохранить проект", self.save_current_project),
            ("Загрузить разметку", self.load_current_project),
            ("Экспорт датасета", self.export_current),
        ):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            side.addWidget(button)

        self.info_label = QLabel("Откройте STL для начала работы.")
        self.info_label.setWordWrap(True)
        side.addSpacing(16)
        side.addWidget(self.info_label)
        layout.addLayout(side, 0)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self._install_picker()

    def _install_picker(self) -> None:
        picker = pv._vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)

        def on_click(_obj, _event) -> None:
            if not self.paint_enabled or self.source is None:
                return
            x, y = self.plotter.interactor.GetEventPosition()
            if picker.Pick(x, y, 0, self.plotter.renderer):
                cell_id = picker.GetCellId()
                if cell_id >= 0:
                    self.paint_at(cell_id)

        self._picker_observer = self.plotter.interactor.AddObserver("LeftButtonPressEvent", on_click)

    def set_active_label(self, label: Label) -> None:
        self.active_label = label
        for key, button in self.class_buttons.items():
            button.setChecked(key == label)
        self.statusBar().showMessage(f"Активный класс: {LABEL_NAMES[label]}")

    def _set_radius(self, value: int) -> None:
        self.brush_radius = float(value)
        self.radius_label.setText(f"{value} мм")

    def set_paint_enabled(self, enabled: bool) -> None:
        self.paint_enabled = enabled
        self.paint_button.setText("Разметка включена — вращение отключено" if enabled else "Включить разметку")
        self.statusBar().showMessage("Щёлкните по поверхности для нанесения метки." if enabled else "Режим просмотра.")

    def open_stl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть скан протеза", "", "STL (*.stl)")
        if not path:
            return
        try:
            self.source = load_stl(path)
            self.labels = np.zeros(self.source.face_count, dtype=np.uint8)
            self._render_mesh()
            self.info_label.setText(
                f"{self.source.path.name}\n"
                f"Вершин: {self.source.mesh.n_points:,}\n"
                f"Треугольников: {self.source.face_count:,}\n"
                f"Размечено: 0%"
            )
            self.statusBar().showMessage("STL загружен. Геометрия исходной сетки сохранена.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки STL", str(exc))

    def _render_mesh(self) -> None:
        if self.source is None:
            return
        self.plotter.clear()
        mesh = self.source.mesh.copy(deep=True)
        mesh.cell_data["label"] = self.labels
        colors = np.asarray([LABEL_COLORS[Label(int(value))] for value in self.labels], dtype=float)
        mesh.cell_data["rgb"] = colors
        self.plotter.add_mesh(
            mesh,
            scalars="rgb",
            rgb=True,
            lighting=True,
            smooth_shading=True,
            show_edges=False,
            name="source_mesh",
        )
        self.plotter.reset_camera()
        self.plotter.render()

    def paint_at(self, cell_id: int) -> None:
        if self.source is None or not 0 <= cell_id < self.source.face_count:
            return
        center = self.source.face_centers[cell_id]
        distances = np.linalg.norm(self.source.face_centers - center, axis=1)
        affected = np.flatnonzero(distances <= self.brush_radius)
        self.labels[affected] = int(self.active_label)
        self._render_mesh()
        labeled = int(np.count_nonzero(self.labels != int(Label.UNLABELED)))
        percent = 100.0 * labeled / max(1, self.source.face_count)
        self.info_label.setText(
            f"{self.source.path.name}\n"
            f"Вершин: {self.source.mesh.n_points:,}\n"
            f"Треугольников: {self.source.face_count:,}\n"
            f"Размечено: {percent:.1f}%"
        )

    def save_current_project(self) -> None:
        if self.source is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "", "ORTO Project (*.ortoann.json)")
        if path:
            try:
                save_project(path, self.source, self.labels)
                self.statusBar().showMessage(f"Проект сохранён: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def load_current_project(self) -> None:
        if self.source is None:
            QMessageBox.information(self, "Нет STL", "Сначала откройте исходный STL.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить проект", "", "ORTO Project (*.json)")
        if path:
            try:
                self.labels = load_project(path, self.source)
                self._render_mesh()
                self.statusBar().showMessage("Разметка загружена.")
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка загрузки проекта", str(exc))

    def export_current(self) -> None:
        if self.source is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Каталог экспорта")
        if directory:
            try:
                paths = export_dataset_item(directory, self.source, self.labels)
                QMessageBox.information(self, "Экспорт завершён", "\n".join(str(path) for path in paths))
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка экспорта", str(exc))
