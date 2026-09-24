# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi>
#
# SPDX-License-Identifier: MIT

# https://doc.qt.io/qtforpython-6/tutorials/index.html
# https://doc.qt.io/qtforpython-6/tutorials/expenses/expenses.html

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QProgressDialog

from signal_app.signal_app_widget import SignalAppWidget
from signal_app.signal_converter import SignalConverter
from signal_app.worker import Worker

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SignalAppMainWindow(QMainWindow):
    signal_path: Path | None = None
    converted_signal_path: Path | None = None
    last_dir: Path = Path.home()  # Keep track of the last opened directory
    selected_open_filter = None
    selected_save_filter = None

    def __init__(self):
        super().__init__()

        self.file_menu = self.menuBar().addMenu("&File")
        file_open_action = self.file_menu.addAction("Open", self.file_open_action)
        file_open_action.setShortcut("Ctrl+O")
        file_convert_action = self.file_menu.addAction(
            "Convert", self.file_convert_action
        )
        file_convert_action.setShortcut("Ctrl+Alt+C")

        file_quit_action = self.file_menu.addAction("Quit", self.close)
        file_quit_action.setShortcut("Ctrl+Q")

        self.signal_app_widget = SignalAppWidget()
        self.setCentralWidget(self.signal_app_widget)
        self.setWindowTitle("Signal app template")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.signal_app_widget.stop_signal_analyser()
        event.accept()
        logger.debug("Closing application.")

    def file_open_action(self):
        signal_path, self.selected_open_filter = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Industrial Project signal file"),
            str(self.last_dir),  # ok, Qt seems to like path strings
            self.tr("Signal files (*.csv *.npy)"),
            self.selected_open_filter,
        )
        signal_path = Path(signal_path)  # but I like pathlib.Path
        self.last_dir = signal_path.parent
        self.signal_path = signal_path
        self.signal_app_widget.set_signal_path(signal_path)
        logger.debug("Opened file '%s'", signal_path)

    def file_convert_action(self):
        # Convert requires a CSV-file.
        if not self.signal_path or self.signal_path.suffix != ".csv":
            logger.debug(
                "File->Convert: loaded file '%s' is not a CSV-file.",
                self.signal_path.name,
            )
            return

        converted_signal_path, self.selected_save_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Industrial Project signal file"),
            str(self.signal_path.with_suffix(".npy")),
            self.tr("Signal files (*.npy)"),
            self.selected_save_filter,
        )
        if not converted_signal_path:
            logger.debug("File->Convert: user cancelled file selection.")
            return

        self.converted_signal_path = Path(converted_signal_path)

        # Prepare a modal progress dialog. QProgressDialog shows a label, progress bar, and a cancel button.
        # Modal dialogs block the application underneath, and we don't need to worry about user trying to
        # use the other actions within the user interface. The progress bar will close automatically when
        # progress reaches 100 % (i.e., when method call progress.setValue(100) is performed).
        progress = QProgressDialog(
            "Converting to binary format...", "Cancel", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModal)

        # Get a threadpool
        threadpool = QThreadPool.globalInstance()

        # Prepare a signal converter thread. The thread emit progress signals to update the progress dialog.
        # Start the thread after everything has been set up.
        converter = SignalConverter()
        worker = Worker(
            converter.start,
            source_signal_path=self.signal_path,
            target_signal_path=self.converted_signal_path,
        )
        worker.signals.progress.connect(progress.setValue)
        worker.signals.result.connect(self._converter_finished)
        progress.canceled.connect(converter.cancel)
        progress.setValue(0)
        threadpool.start(worker)
        logger.debug("File->Convert: conversion process launched")

    def _converter_finished(self, finished_successfully: bool):
        if finished_successfully:
            logger.debug("File->Convert: conversion finished successfully.")
            # Set the converted signal file as our active signal file.
            self.signal_path = self.converted_signal_path
            self.signal_app_widget.set_signal_path(self.signal_path)
        else:
            logger.debug("File->Convert: conversion was cancelled.")


def run():
    app = QApplication([])

    main_window = SignalAppMainWindow()
    main_window.resize(1024, 768)
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
