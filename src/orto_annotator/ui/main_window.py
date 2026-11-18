"""Main annotation workstation window."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, QTimer
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
from scipy.spatial import cKDTree

from orto_annotator.core.export import export_dataset_item
from orto_annotator.core.labels import LABEL_COLORS, LABEL_NAMES, Label
from orto_annotator.core.mesh import SourceMesh, load_stl
from orto_annotator.core.project import load_project, save_project


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ORTO Prosthesis Annotator 0.1.4 TEST")
        self.resize(1440, 900)
        self.source: SourceMesh | None = None
        self.labels = np.zeros(0, dtype=np.uint8)
        self.active_label = Label.SOCKET
        self.brush_radius = 12.0
        self.paint_enabled = False
        self._left_button_down = False
        self._picker = pv._vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._cell_locator = None
        self._last_screen_position: tuple[float, float] | None = None
        self._mesh_actor = None
        self._paint_render_pending = False
        self._last_pick_position: tuple[int, int] | None = None
        self._stroke_marks = np.zeros(0, dtype=np.uint32)
        self._stroke_id = 0
        self._cell_tree: cKDTree | None = None
        self._display_mesh: pv.PolyData | None = None
        self._display_rgb: np.ndarray | None = None
        self._last_painted_cell = -1

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#20252d")
        self.plotter.enable_trackball_style()
        self.plotter.interactor.installEventFilter(self)

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

    def eventFilter(self, watched, event) -> bool:
        if watched is not self.plotter.interactor or not self.paint_enabled:
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._left_button_down = True
            self._last_painted_cell = -1
            self._last_pick_position = None
            self._last_screen_position = None
            self._stroke_id += 1
            self._paint_from_mouse_event(event)
            return True

        if event_type == QEvent.Type.MouseMove:
            if self._left_button_down and event.buttons() & Qt.MouseButton.LeftButton:
                self._paint_from_mouse_event(event)
                return True
            self._left_button_down = False
            return True

        if event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self._left_button_down = False
            self._last_painted_cell = -1
            self._last_pick_position = None
            self._last_screen_position = None
            self._update_info()
            return True

        return super().eventFilter(watched, event)

    def _paint_from_mouse_event(self, event) -> None:
        if self.source is None:
            return
        position = event.position().toPoint()
        x, y = position.x(), self.plotter.interactor.height() - position.y()
        previous = self._last_screen_position
        self._last_screen_position = (float(x), float(y))
        if previous is None:
            self._pick_and_paint(x, y)
            return
        dx, dy = x - previous[0], y - previous[1]
        distance = float(np.hypot(dx, dy))
        steps = max(1, int(np.ceil(distance / 2.0)))
        for step in range(1, steps + 1):
            t = step / steps
            sample_x = int(round(previous[0] + dx * t))
            sample_y = int(round(previous[1] + dy * t))
            self._pick_and_paint(sample_x, sample_y)

    def _pick_and_paint(self, x: int, y: int) -> None:
        if self._picker.Pick(x, y, 0, self.plotter.renderer):
            cell_id = self._picker.GetCellId()
            if cell_id >= 0:
                self.paint_at(cell_id)

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
        self._left_button_down = False
        self._last_painted_cell = -1
        self._last_pick_position = None
        self._last_screen_position = None
        if enabled:
            # Base VTK style has no camera gestures; Qt events are handled by eventFilter.
            self.plotter.interactor.SetInteractorStyle(pv._vtk.vtkInteractorStyle())
            self.paint_button.setText("Разметка включена — ЛКМ рисует")
            self.statusBar().showMessage("Зажмите левую кнопку и ведите кисть по поверхности.")
        else:
            self.plotter.enable_trackball_style()
            self.paint_button.setText("Включить разметку")
            self.statusBar().showMessage("Режим просмотра: левая кнопка вращает модель.")

    def open_stl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть скан протеза", "", "STL (*.stl)")
        if not path:
            return
        try:
            self.source = load_stl(path)
            self.labels = np.zeros(self.source.face_count, dtype=np.uint8)
            self._cell_tree = cKDTree(self.source.face_centers, compact_nodes=True, balanced_tree=True)
            self._stroke_marks = np.zeros(self.source.face_count, dtype=np.uint32)
            self._stroke_id = 0
            self._render_mesh()
            self._update_info()
            self.statusBar().showMessage("STL загружен. Исходная геометрия сохранена.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки STL", str(exc))

    def _update_info(self) -> None:
        if self.source is None:
            return
        labeled = int(np.count_nonzero(self.labels != int(Label.UNLABELED)))
        percent = 100.0 * labeled / max(1, self.source.face_count)
        self.info_label.setText(
            f"{self.source.path.name}\n"
            f"Вершин: {self.source.mesh.n_points:,}\n"
            f"Треугольников: {self.source.face_count:,}\n"
            f"Размечено: {percent:.1f}%"
        )

    def _render_mesh(self) -> None:
        if self.source is None:
            return
        self.plotter.clear()
        # STL files often duplicate all three vertices per triangle. Merge the
        # display copy, then explicitly generate shared point normals with VTK.
        # The original source mesh and its face order remain the annotation authority.
        display_mesh = self.source.mesh.copy(deep=True).clean(
            point_merging=True,
            tolerance=1e-7,
            inplace=False,
        )
        if display_mesh.n_cells != self.source.face_count:
            raise RuntimeError("Объединение вершин изменило число граней STL.")
        normals = pv._vtk.vtkPolyDataNormals()
        normals.SetInputData(display_mesh)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.FlipNormalsOff()
        normals.SetFeatureAngle(180.0)
        normals.Update()
        self._display_mesh = pv.wrap(normals.GetOutput())
        self._display_mesh.GetPointData().SetActiveNormals("Normals")
        self._display_rgb = np.asarray(
            [
                tuple(round(channel * 255) for channel in LABEL_COLORS[Label(int(value))])
                for value in self.labels
            ],
            dtype=np.uint8,
        )
        self._display_mesh.cell_data["rgb"] = self._display_rgb
        self._mesh_actor = self.plotter.add_mesh(
            self._display_mesh,
            scalars="rgb",
            rgb=True,
            lighting=True,
            smooth_shading=True,
            show_edges=False,
            ambient=0.12,
            diffuse=0.88,
            specular=0.45,
            specular_power=32,
            name="source_mesh",
        )
        prop = self._mesh_actor.GetProperty()
        prop.SetLighting(True)
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.12)
        prop.SetDiffuse(0.88)
        prop.SetSpecular(0.45)
        prop.SetSpecularPower(32)
        prop.SetColor(0.82, 0.84, 0.88)
        prop.SetBackfaceCulling(False)
        renderer = self.plotter.renderer
        renderer.RemoveAllLights()
        key_light = pv._vtk.vtkLight()
        key_light.SetLightTypeToHeadlight()
        key_light.SetIntensity(1.15)
        renderer.AddLight(key_light)
        fill_light = pv._vtk.vtkLight()
        fill_light.SetLightTypeToCameraLight()
        fill_light.SetIntensity(0.35)
        renderer.AddLight(fill_light)
        self.plotter.reset_camera()
        self.plotter.render()

    def paint_at(self, cell_id: int) -> None:
        if (
            self.source is None
            or self._cell_tree is None
            or self._display_mesh is None
            or self._display_rgb is None
            or not 0 <= cell_id < self.source.face_count
        ):
            return
        center = self.source.face_centers[cell_id]
        affected_list = self._cell_tree.query_ball_point(
            center,
            self.brush_radius,
            workers=1,
            return_sorted=False,
        )
        if not affected_list:
            return
        affected = np.fromiter(affected_list, dtype=np.intp, count=len(affected_list))
        if self._left_button_down:
            fresh = self._stroke_marks[affected] != self._stroke_id
            affected = affected[fresh]
            if affected.size == 0:
                return
            self._stroke_marks[affected] = self._stroke_id
        if affected.size == 0:
            return
        new_value = int(self.active_label)
        changed = affected[self.labels[affected] != new_value]
        if changed.size == 0:
            return
        self.labels[changed] = new_value
        color = np.asarray(
            tuple(round(channel * 255) for channel in LABEL_COLORS[self.active_label]),
            dtype=np.uint8,
        )
        self._display_rgb[changed] = color
        vtk_colors = self._display_mesh.GetCellData().GetArray("rgb")
        if vtk_colors is not None:
            vtk_colors.Modified()
        self._display_mesh.Modified()
        if not self._paint_render_pending:
            self._paint_render_pending = True
            QTimer.singleShot(16, self._flush_paint_render)

    def _flush_paint_render(self) -> None:
        self._paint_render_pending = False
        self.plotter.render()

    def save_current_project(self) -> None:
        if self.source is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "", "ORTO Project (*.ortoann.json)")
        if path:
            try:
                save_project(path, self.source, self.labels)
                self.statusBar().showMessage(f"Проект сохранён: {path}")
            except Exception as exc:  # noqa: BLE001
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
                self._update_info()
                self.statusBar().showMessage("Разметка загружена.")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка загрузки проекта", str(exc))

    def export_current(self) -> None:
        if self.source is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Каталог экспорта")
        if directory:
            try:
                paths = export_dataset_item(directory, self.source, self.labels)
                QMessageBox.information(self, "Экспорт завершён", "\n".join(str(path) for path in paths))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Ошибка экспорта", str(exc))
