"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from orto_annotator.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ORTO Prosthesis Annotator")
    app.setOrganizationName("ORTO-INNOVATIONS")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
