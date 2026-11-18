"""Main annotation workstation window."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
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
from vtkmodules.vtkFiltersExtraction import vtkExtractCells

from orto_annotator.core.export import export_dataset_item
from orto_annotator.core.labels import LABEL_COLORS, LABEL_NAMES, Label
from orto_annotator.core.mesh import SourceMesh, load_stl
from orto_annotator.core.project import load_project, save_project


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ORTO Prosthesis Annotator 0.2.6 TEST")
        self.resize(1440, 900)
        self.source: SourceMesh | None = None
        self.labels = np.zeros(0, dtype=np.uint8)
        self.active_label = Label.SOCKET
        self.brush_radius = 12.0
        self.paint_enabled = False
        self._left_button_down = False
        self._picker = pv._vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._face_tree: cKDTree | None = None
        self._face_normals = np.zeros((0, 3), dtype=np.float64)
        self._cell_locator = None
        self._last_screen_position: tuple[float, float] | None = None
        self._mesh_actor = None
        self._paint_render_pending = False
        self._last_pick_position: tuple[int, int] | None = None
        self._stroke_marks = np.zeros(0, dtype=np.uint32)
        self._stroke_id = 0
        self._display_mesh: pv.PolyData | None = None
        self._display_rgb: np.ndarray | None = None
        self._display_blocks: list[tuple[int, int, pv.PolyData, object, np.ndarray]] = []
        self._last_painted_cell = -1
        self._last_world_position: np.ndarray | None = None
        self._last_world_normal: np.ndarray | None = None
        self._brush_outline_source = None
        self._brush_outline_actor = None
        self._brush_cursor_render_pending = False

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
        if watched is not self.plotter.interactor:
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if self.paint_enabled:
                self._left_button_down = True
                self._last_painted_cell = -1
                self._last_pick_position = None
                self._last_screen_position = None
                self._last_world_position = None
                self._last_world_normal = None
                self._stroke_id += 1
                self._paint_from_mouse_event(event)
                return True

        if event_type == QEvent.Type.MouseMove:
            if self.paint_enabled and self._left_button_down and event.buttons() & Qt.MouseButton.LeftButton:
                self._paint_from_mouse_event(event)
                return True
            self._update_brush_cursor(event.position().toPoint().x(), event.position().toPoint().y())
            if self.paint_enabled:
                return True

        if event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self.paint_enabled:
                self._left_button_down = False
                self._last_painted_cell = -1
                self._last_pick_position = None
                self._last_screen_position = None
                self._last_world_position = None
                self._last_world_normal = None
                self._update_info()
                return True

        return super().eventFilter(watched, event)

    def _paint_from_mouse_event(self, event) -> None:
        if self.source is None:
            return
        position = event.position().toPoint()
        x = position.x()
        # Keep Qt top-left coordinates throughout the stroke; convert to VTK
        # bottom-left coordinates exactly once at the picker boundary.
        y = position.y()
        self._pick_and_paint(x, y)

    @staticmethod
    def _world_to_display(renderer, point: np.ndarray) -> np.ndarray:
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        return np.asarray(renderer.GetDisplayPoint(), dtype=np.float64)

    def _pick_and_paint(self, x: int, y: int) -> None:
        if self.source is None or self._mesh_actor is None or self._face_tree is None:
            return

        renderer = self.plotter.renderer
        height = renderer.GetSize()[1]
        if not self._picker.Pick(int(x), height - int(y), 0, renderer):
            self._hide_brush_cursor()
            self._last_screen_position = (int(x), int(y))
            self._last_world_position = None
            self._last_world_normal = None
            return

        cell_id = int(self._picker.GetCellId())
        if cell_id < 0 or cell_id >= self.source.face_count:
            self._hide_brush_cursor()
            return
        current_point = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
        current_normal = self._face_normals[cell_id]
        self._update_brush_outline(current_point, current_normal)
        previous_point = self._last_world_position
        previous_normal = self._last_world_normal
        self._last_screen_position = (int(x), int(y))
        self._last_world_position = current_point.copy()
        self._last_world_normal = current_normal.copy()

        # Interpolate surface-hit points in world space; only the actual cursor
        # position requires a VTK pick. Intermediate centres are NumPy-generated.
        hit_points = [current_point]
        hit_normals = [current_normal]
        if previous_point is not None and previous_normal is not None:
            distance = float(np.linalg.norm(current_point - previous_point))
            step_mm = max(1.0, self.brush_radius * 0.45)
            count = max(1, int(np.ceil(distance / step_mm)))
            if count > 1:
                fractions = np.arange(1, count, dtype=np.float64)[:, None] / count
                interpolated = previous_point[None, :] + fractions * (current_point - previous_point)[None, :]
                hit_points = [*interpolated, current_point]
                hit_normals = [*([previous_normal] * len(interpolated)), current_normal]

        groups = self._face_tree.query_ball_point(np.asarray(hit_points, dtype=np.float64), r=self.brush_radius, return_sorted=False)
        selected: list[np.ndarray] = []
        stroke_marks = self._stroke_marks
        stroke_id = self._stroke_id
        for ids, hit_normal in zip(groups, hit_normals, strict=True):
            if not ids:
                continue
            face_ids = np.asarray(ids, dtype=np.intp)
            face_ids = face_ids[self._face_normals[face_ids] @ hit_normal >= 0.0]
            if self._left_button_down and face_ids.size:
                # Deduplicate brush centres in O(n), without sorting large arrays.
                fresh = stroke_marks[face_ids] != stroke_id
                face_ids = face_ids[fresh]
                stroke_marks[face_ids] = stroke_id
            if face_ids.size:
                selected.append(face_ids)
        if selected:
            self.paint_at(np.concatenate(selected), already_stamped=True)

    def _update_brush_cursor(self, x: int, y: int) -> None:
        if self.source is None or self._mesh_actor is None:
            self._hide_brush_cursor()
            return
        height = self.plotter.renderer.GetSize()[1]
        if not self._picker.Pick(int(x), height - int(y), 0, self.plotter.renderer):
            self._hide_brush_cursor()
            return
        cell_id = int(self._picker.GetCellId())
        if cell_id < 0 or cell_id >= self.source.face_count:
            self._hide_brush_cursor()
            return
        point = np.asarray(self._picker.GetPickPosition(), dtype=np.float64)
        self._update_brush_outline(point, self._face_normals[cell_id])

    def _update_brush_outline(self, center: np.ndarray, normal: np.ndarray) -> None:
        if self._brush_outline_source is None:
            source = pv._vtk.vtkRegularPolygonSource()
            source.SetNumberOfSides(64)
            source.SetRadius(self.brush_radius)
            source.GeneratePolygonOff()
            mapper = pv._vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = pv._vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetProperty().SetLineWidth(2.0)
            actor.GetProperty().SetLighting(False)
            actor.PickableOff()
            self.plotter.renderer.AddActor(actor)
            self._brush_outline_source = source
            self._brush_outline_actor = actor
        source = self._brush_outline_source
        source.SetCenter(float(center[0]), float(center[1]), float(center[2]))
        source.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
        source.SetRadius(float(self.brush_radius))
        source.Modified()
        self._brush_outline_actor.SetVisibility(True)
        if not self._brush_cursor_render_pending:
            self._brush_cursor_render_pending = True
            QTimer.singleShot(16, self._flush_brush_cursor_render)

    def _hide_brush_cursor(self) -> None:
        if self._brush_outline_actor is not None and self._brush_outline_actor.GetVisibility():
            self._brush_outline_actor.SetVisibility(False)
            if not self._brush_cursor_render_pending:
                self._brush_cursor_render_pending = True
                QTimer.singleShot(16, self._flush_brush_cursor_render)

    def _flush_brush_cursor_render(self) -> None:
        self._brush_cursor_render_pending = False
        self.plotter.render()

    def set_active_label(self, label: Label) -> None:
        self.active_label = label
        for key, button in self.class_buttons.items():
            button.setChecked(key == label)
        self.statusBar().showMessage(f"Активный класс: {LABEL_NAMES[label]}")

    def _set_radius(self, value: int) -> None:
        self.brush_radius = float(value)
        self.radius_label.setText(f"{value} мм")
        if self._brush_outline_source is not None:
            self._brush_outline_source.SetRadius(self.brush_radius)
            self._brush_outline_source.Modified()
            if not self._brush_cursor_render_pending:
                self._brush_cursor_render_pending = True
                QTimer.singleShot(16, self._flush_brush_cursor_render)

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
            self._stroke_marks = np.zeros(self.source.face_count, dtype=np.uint32)
            self._face_tree = cKDTree(self.source.face_centers, compact_nodes=True, balanced_tree=True)
            faces = np.asarray(self.source.mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
            triangles = np.asarray(self.source.mesh.points, dtype=np.float64)[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            lengths[lengths == 0.0] = 1.0
            self._face_normals = normals / lengths[:, None]
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
        self._brush_outline_source = None
        self._brush_outline_actor = None
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
        self._picker.RemoveAllLocators()
        self._cell_locator = pv._vtk.vtkStaticCellLocator()
        self._cell_locator.SetDataSet(self._display_mesh)
        self._cell_locator.BuildLocator()
        self._picker.AddLocator(self._cell_locator)
        # Independent cell batches let a stroke update only touched VTK buffers.
        self._display_blocks = []
        self._display_rgb = None
        self._mesh_actor = None
        block_size = 25000
        face_count = self.source.face_count
        for start in range(0, face_count, block_size):
            end = min(start + block_size, face_count)
            ids = pv._vtk.vtkIdList()
            for cell_id in range(start, end):
                ids.InsertNextId(cell_id)
            extract = vtkExtractCells()
            extract.SetInputData(self._display_mesh)
            extract.SetCellList(ids)
            extract.Update()
            block_mesh = pv.wrap(extract.GetOutput())
            block_rgb = np.asarray(
                [
                    tuple(round(channel * 255) for channel in LABEL_COLORS[Label(int(value))])
                    for value in self.labels[start:end]
                ],
                dtype=np.uint8,
            )
            if block_mesh.n_cells != end - start:
                raise RuntimeError("Не удалось сохранить соответствие граней в блоке отображения.")
            block_mesh.cell_data["rgb"] = block_rgb
            actor = self.plotter.add_mesh(
                block_mesh,
                scalars="rgb",
                rgb=True,
                lighting=True,
                smooth_shading=True,
                show_edges=False,
                ambient=0.12,
                diffuse=0.88,
                specular=0.45,
                specular_power=32,
                name="source_mesh_" + str(start),
            )
            prop = actor.GetProperty()
            prop.SetLighting(True)
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.12)
            prop.SetDiffuse(0.88)
            prop.SetSpecular(0.45)
            prop.SetSpecularPower(32)
            prop.SetColor(0.82, 0.84, 0.88)
            prop.SetBackfaceCulling(False)
            self._display_blocks.append((start, end, block_mesh, actor, block_rgb))
            if self._mesh_actor is None:
                self._mesh_actor = actor
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

    def paint_at(self, cell_ids: int | np.ndarray, already_stamped: bool = False) -> None:
        if (
            self.source is None
            or self._display_mesh is None
            or not self._display_blocks
        ):
            return
        affected = np.atleast_1d(np.asarray(cell_ids, dtype=np.intp))
        affected = affected[(affected >= 0) & (affected < self.source.face_count)]
        if affected.size == 0:
            return
        if not already_stamped:
            affected = np.unique(affected)
            if self._left_button_down:
                fresh = self._stroke_marks[affected] != self._stroke_id
                affected = affected[fresh]
                if affected.size == 0:
                    return
                self._stroke_marks[affected] = self._stroke_id
        new_value = int(self.active_label)
        changed = affected[self.labels[affected] != new_value]
        if changed.size == 0:
            return
        self.labels[changed] = new_value
        color = np.asarray(
            tuple(round(channel * 255) for channel in LABEL_COLORS[self.active_label]),
            dtype=np.uint8,
        )
        # Source cell IDs remain global; only touched contiguous display batches
        # receive array writes and Modified notifications.
        first_block = int(changed.min()) // 25000
        last_block = int(changed.max()) // 25000
        for block_index in range(first_block, last_block + 1):
            start, end, block_mesh, _actor, block_rgb = self._display_blocks[block_index]
            local_mask = (changed >= start) & (changed < end)
            if not np.any(local_mask):
                continue
            local_ids = changed[local_mask] - start
            block_rgb[local_ids] = color
            vtk_colors = block_mesh.GetCellData().GetArray("rgb")
            if vtk_colors is not None:
                vtk_colors.Modified()
            block_mesh.Modified()
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
