# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi>
#
# SPDX-License-Identifier: MIT

# A signal widget for showing n points of a signal.
#
# Based on Qt for Python 6 data visualization tutorial [1]
#
# [1] https://doc.qt.io/qtforpython-6/tutorials/datavisualize/plot_datapoints.html
import logging
import time

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

logger = logging.getLogger(__name__)


class SignalWindowChartWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Keep linters happy by setting class fields in the __init__
        self.series: QLineSeries | None = None
        self.axis_x: QValueAxis | None = None
        self.axis_y: QValueAxis | None = None

        self.chart = QChart()
        # self.chart.setAnimationOptions(QChart.AllAnimations)
        # self.chart.setAnimationDuration(50)
        self.add_series("Amplitude")

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        self.main_layout = QHBoxLayout()
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.chart_view.setSizePolicy(size)
        self.main_layout.addWidget(self.chart_view)

        self.setLayout(self.main_layout)

    @Slot(float, float)
    def set_axis_y(self, min_y: float, max_y: float):
        self.axis_y.setMin(min_y)
        self.axis_y.setMax(max_y)

    def add_series(self, name: str):
        self.series = QLineSeries()
        self.series.setName(name)

        self.chart.addSeries(self.series)

        self.axis_x = QValueAxis()
        self.axis_x.setTickCount(10)
        self.axis_x.setLabelFormat("%.2f")
        self.axis_x.setTitleText("Time")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setTickCount(10)
        self.axis_y.setLabelFormat("%.2f")
        self.axis_y.setTitleText("Amplitude")
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series.attachAxis(self.axis_y)

    @Slot(np.ndarray, np.ndarray)
    def replace_array(
        self,
        x: np.ndarray,
        y: np.ndarray,
        do_not_try_fix_what_is_not_broken: bool = False,
    ):
        start_time = time.time()

        if not do_not_try_fix_what_is_not_broken:
            if (
                x.base is not None
                or y.base is not None
                or x.dtype != np.float64
                or y.dtype != np.float64
            ):
                # It looks like some condition is not being handled within replaceNp and the graph will not be updated.
                # Make float64 copies of the provided arrays. That *seems* to work.
                logger.debug(
                    "Trying to fix non-broken input arrays: x %s %s",
                    x.base is None,
                    x.dtype,
                )
                logger.debug(
                    "Trying to fix non-broken input arrays: y %s %s",
                    y.base is None,
                    y.dtype,
                )
                x = x if x.base is None else x.astype(np.float64).copy()
                y = y if y.base is None else y.astype(np.float64).copy()

        self.series.replaceNp(x, y)
        self.axis_x.setMin(x.min())
        self.axis_x.setMax(x.max())
        end_time = time.time()
        logger.debug(
            "Finished updating chart in %.2f ms", (end_time - start_time) * 1000
        )
