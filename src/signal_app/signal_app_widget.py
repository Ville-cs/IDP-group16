# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi>
# SPDX-FileContributor: 2019 Martin Fitzpatrick
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QGridLayout, QPushButton, QWidget

from signal_app.signal_analyzer import SignalAnalyzer
from signal_app.signal_window_chart_widget import SignalWindowChartWidget
from signal_app.worker import Worker


class SignalAppWidget(QWidget):
    # File path for the signal data
    signal_path: Path | None = None

    # Signals for SignalAnalyzer callbacks
    chart_set_axis_y = Signal(float, float)
    chart_update_data = Signal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()

        # Create a signal chart
        self.signal_window_chart = SignalWindowChartWidget()

        # Connect the chart to the chart handling signals.
        # Make these blocking queued connections. Signal analyzer thread will stop and wait until the emitted signal
        # finishes. This prevents problems arising from signal analyzer analyzing faster than the chart can show the
        # signal.
        self.chart_set_axis_y.connect(
            self.signal_window_chart.set_axis_y, type=Qt.BlockingQueuedConnection
        )
        self.chart_update_data.connect(
            self.signal_window_chart.replace_array, type=Qt.BlockingQueuedConnection
        )

        # Add buttons to start and stop the signal analyzer.
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.previous_window_button = QPushButton("Previous window")
        self.next_window_button = QPushButton("Next window")

        # Connect the buttons' pressed-signal to the appropriate methods
        self.start_button.pressed.connect(self.start_signal_analyser)
        self.stop_button.pressed.connect(self.stop_signal_analyser)
        self.previous_window_button.pressed.connect(
            self.previous_window_signal_analyzer
        )
        self.next_window_button.pressed.connect(self.next_window_signal_analyzer)

        # Create a simple layout for the widgets
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.start_button, 0, 0)
        self.layout.addWidget(self.stop_button, 0, 1)
        self.layout.addWidget(self.previous_window_button, 1, 0)
        self.layout.addWidget(self.next_window_button, 1, 1)
        self.layout.addWidget(self.signal_window_chart, 2, 0, 1, 2)

        # Prepare a thread pool and an instance of the signal analyzer.
        self.threadpool = QThreadPool.globalInstance()
        self.signal_analyzer = SignalAnalyzer()

    def set_signal_path(self, signal_path: Path):
        self.signal_path = signal_path
        if self.signal_path and self.signal_path.suffix == ".npy":
            self.previous_window_button.setEnabled(True)
            self.next_window_button.setEnabled(True)

            self.current_window_signal_analyzer()
        else:
            self.previous_window_button.setEnabled(False)
            self.next_window_button.setEnabled(False)

    @Slot()
    def start_signal_analyser(self):
        """
        Starts a signal analyzer in a worker thread, passes the signal file path
        and connects the appropriate signals.
        """

        # Create an instance of Worker that will run the signal analyzer's main
        # loop when the thread is started. Pass the signal callbacks.
        worker = Worker(
            self.signal_analyzer.start,
            signal_path=self.signal_path,
            set_chart_axis_y=self.chart_set_axis_y,
            update_chart=self.chart_update_data,
        )

        # Worker-class defines a "signals"-field that is a WorkerSignal-instance.
        # Worker instances will emit following signals in response the
        # runner-function (see Worker and WorkerSignals for their meanings).
        #
        # worker.signals.result.connect(self.print_output)
        # worker.signals.finished.connect(self.thread_complete)
        # worker.signals.progress.connect(self.progress_fn)

        # Start the worker thread in the thread pool
        self.threadpool.start(worker)

    @Slot()
    def stop_signal_analyser(self):
        self.signal_analyzer.stop()

    @Slot()
    def current_window_signal_analyzer(self):
        worker = Worker(
            self.signal_analyzer.current_window,
            signal_path=self.signal_path,
            set_chart_axis_y=self.chart_set_axis_y,
            update_chart=self.chart_update_data,
        )
        worker.signals.finished.connect(self._updated_window_signal_analyzer)
        self.threadpool.start(worker)

    @Slot()
    def previous_window_signal_analyzer(self):
        worker = Worker(
            self.signal_analyzer.previous_window,
            signal_path=self.signal_path,
            set_chart_axis_y=self.chart_set_axis_y,
            update_chart=self.chart_update_data,
        )
        worker.signals.finished.connect(self._updated_window_signal_analyzer)
        self.threadpool.start(worker)

    @Slot()
    def next_window_signal_analyzer(self):
        worker = Worker(
            self.signal_analyzer.next_window,
            signal_path=self.signal_path,
            set_chart_axis_y=self.chart_set_axis_y,
            update_chart=self.chart_update_data,
        )
        worker.signals.finished.connect(self._updated_window_signal_analyzer)
        self.threadpool.start(worker)

    @Slot()
    def _updated_window_signal_analyzer(self):
        self.previous_window_button.setEnabled(
            self.signal_analyzer.has_previous_window()
        )
        self.next_window_button.setEnabled(self.signal_analyzer.has_next_window())
