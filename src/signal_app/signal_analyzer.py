# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi>
#
# SPDX-License-Identifier: MIT
import csv
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from signal_app.signal_converter import SOURCE_SAMPLE_RATE, sample_rate_from_path

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """
    A "signal analyzer" that just generates an array of cosine function values.

    Instances of this class are intended to be run from a worker thread. Data
    is emitted through callbacks back to the parent thread.
    """

    WINDOW_SECONDS = 600  # Let's take 10 minutes of the signal
    CHUNK_SECONDS = 10  # ten seconds at a time
    PLOT_SAMPLE_RATE = 500  # Plotting every sample is unfeasible, so the chart is shown at most at this rate

    # Peak detection parameters
    PEAK_HEIGHT = 70  # minimum peak height above the baseline (ADC units)
    PEAK_MIN_DISTANCE_SECONDS = 0.3  # peaks closer than this are counted as one
    BASELINE_SECONDS = 1.0  # the baseline is the signal median over blocks of this length

    def __init__(self):
        self.signal_path = None
        self.running = False
        self._set_sample_rate(SOURCE_SAMPLE_RATE)

    def _set_sample_rate(self, sample_rate: int):
        """
        Set the sample rate and reset everything that is measured in samples.
        """
        self.sample_rate = sample_rate
        self.window_size = self.WINDOW_SECONDS * sample_rate
        self.chunk_size = self.CHUNK_SECONDS * sample_rate
        self.plot_step = max(1, sample_rate // self.PLOT_SAMPLE_RATE)
        self.x = 0
        self.max_x = 0
        self.x_array = np.zeros((self.window_size,))
        self.y_array = np.zeros_like(self.x_array) * np.nan
        self.y_max = -np.inf  # Use ridiculous values to force y-scale change
        self.y_min = np.inf

    def _set_signal_path(self, signal_path: Path):
        """
        Set the signal file. Opening a different file resets the window state to the file's sample rate.
        """
        if signal_path != self.signal_path:
            self._set_sample_rate(sample_rate_from_path(signal_path))
            logger.debug("Signal '%s' sample rate: %d Hz", signal_path, self.sample_rate)
        self.signal_path = signal_path

    def detect_peaks(self, y: np.ndarray) -> np.ndarray:
        """
        Detect peaks rising more than `PEAK_HEIGHT` above the signal baseline.

        The baseline follows slow drifts in the signal: it is the median of each `BASELINE_SECONDS` block. Every
        stretch of the signal above the threshold yields its highest point, and peaks closer than
        `PEAK_MIN_DISTANCE_SECONDS` to a higher one are dropped.

        :param y: the signal. NaN values at either end (an unfilled window) are ignored.
        :return: indices of the detected peaks in `y`.
        """
        finite = np.flatnonzero(np.isfinite(y))
        if finite.size == 0:
            return np.array([], dtype=int)
        offset = finite[0]
        y = y[offset : finite[-1] + 1]

        # Pad the signal with NaNs to a whole number of blocks, so it can be reshaped into one row per block.
        block = int(self.BASELINE_SECONDS * self.sample_rate)
        n_blocks = -(-len(y) // block)
        padded = np.full(n_blocks * block, np.nan)
        padded[: len(y)] = y
        baseline = np.repeat(np.nanmedian(padded.reshape(n_blocks, block), axis=1), block)[: len(y)]
        heights = y - baseline

        # Find where the signal crosses the threshold up (starts) and back down (ends).
        above = np.concatenate(([False], heights > self.PEAK_HEIGHT, [False]))
        crossings = np.flatnonzero(np.diff(above.astype(np.int8)))
        starts, ends = crossings[::2], crossings[1::2]

        min_distance = self.PEAK_MIN_DISTANCE_SECONDS * self.sample_rate
        peaks = []
        for start, end in zip(starts, ends):
            peak = start + int(np.argmax(heights[start:end]))
            if peaks and peak - peaks[-1] < min_distance:
                if heights[peak] > heights[peaks[-1]]:
                    peaks[-1] = peak
            else:
                peaks.append(peak)
        return np.array(peaks, dtype=int) + offset

    def _read_signal(self, chunk_size: int = 1) -> Iterator[np.ndarray]:
        """
        Read the signal file in chunks of `chunk_size` rows.

        This method is a generator, and it should be called iteratively, for
        example in a `while` or a `for` loop. A generator method generates
        values as they are needed. Consequently, this method does not read
        the whole file at once, but only `count` rows that then yielded
        to the outer loop for processing. On the next loop iteration, the
        execution in this method continues after the yield statement.

        :param chunk_size: the number of rows in a chunk
        :return: a numpy array of shape (count, 2). The final array may be shorter.
        """
        if self.signal_path is None or not self.signal_path.is_file():
            # This could raise an exception, but then we would have to handle that somewhere.
            print(f"""signal path "{self.signal_path}" is not a valid file.""")
            return

        match self.signal_path.suffix.lower():
            case ".csv":
                with self.signal_path.open("r") as signal_file:
                    reader = csv.reader(signal_file)
                    # skip column headers ("adc1", "adc2")
                    next(reader)
                    data = np.zeros((chunk_size, 2))
                    i = 0
                    for i, row in enumerate(reader):
                        data[i % chunk_size] = np.array(row, np.uint16)
                        if i > 0 and i % chunk_size == 0:
                            yield data
                    if i % chunk_size != 0:
                        yield data[: i % chunk_size]
            case ".npy":
                # Open once to get array length.
                # The array in the file is a flat, single-dimensional array of n elements.
                signal = np.memmap(self.signal_path, dtype=np.int16, order="C")
                # Reopen the file with correct (n // 2, 2) shape.
                signal = np.memmap(
                    self.signal_path,
                    dtype=np.int16,
                    shape=(signal.shape[0] // 2, 2),
                    order="C",
                )
                self.max_x = signal.shape[0]
                # Throw out chunks of requested size
                for chunk_start in range(0, self.max_x, chunk_size):
                    yield signal[chunk_start : chunk_start + chunk_size, :].astype(
                        np.float64
                    )

    def _read_signal_at(self, start_x, end_x) -> tuple[np.ndarray, np.ndarray]:
        logger.debug("Read signal from %s to %s", start_x, end_x)
        x_array = np.linspace(start_x, end_x - 1, end_x - start_x)
        if self.signal_path is None or not self.signal_path.is_file():
            logger.error("Signal path '%s' is not a valid file.", self.signal_path)
            return x_array, np.ones((end_x - start_x, 2)) * np.nan

        match self.signal_path.suffix.lower():
            case ".csv":
                logger.error("This function does not support CSV-files.")
                return x_array, np.ones((end_x - start_x, 2)) * np.nan

            case ".npy":
                # Open once to get array length.
                # The array in the file is a flat, single-dimensional array of n elements.
                signal = np.memmap(self.signal_path, dtype=np.int16, order="C")
                # Reopen the file with correct (n // 2, 2) shape.
                signal = np.memmap(
                    self.signal_path,
                    dtype=np.int16,
                    shape=(signal.shape[0] // 2, 2),
                    order="C",
                )

                # Update maximum x value
                self.max_x = signal.shape[0]

                if end_x < self.max_x:
                    # Return a slice from the signal array.
                    return x_array, signal[start_x:end_x, :].astype(np.float64)
                elif start_x < self.max_x:
                    # The signal array has less data than requested.
                    # Return an array of requested size with all data that is still available, rest of it are NaN.
                    y_array = np.ones((end_x - start_x, 2)) * np.nan
                    y_array[: self.max_x - start_x] = signal[start_x:, :]
                    return x_array, y_array
                else:
                    # The request is completely out of bounds
                    # Return an array of requested size, but fill it with NaNs.
                    return x_array, np.ones((end_x - start_x, 2)) * np.nan

            case _:
                # Ok, what did you do or didn't do to get here?
                logger.error("Completely invalid input file?!")
                return x_array, np.ones((end_x - start_x, 2)) * np.nan

    def _start(self, set_axis_y=None):
        """
        Tasks that should be done before starting the main loop of SignalAnalyzer.
        """
        self.running = True
        if set_axis_y:
            set_axis_y.emit(self.y_min, self.y_max)

    def stop(self):
        """
        Stops the main loop.
        """
        self.running = False

    def start(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        set_peaks: Signal | None = None,
        progress_callback: Signal | None = None,
    ):
        """
        Starts the SignalAnalyzer main loop.

        :param signal_path: the path to the signal data file.
        :param set_chart_axis_y: Signal to update chart y-axis.
        :param update_chart: Signal to update a chart.
        :param set_peaks: Signal to mark the detected peaks (time and amplitude arrays) on the chart.
        :param progress_callback: Signal to update progress meters.
        """

        self._set_signal_path(signal_path)
        self._start(set_chart_axis_y)

        while self.running:
            self.x = -self.window_size

            end_time = time.time()
            for chunk in self._read_signal(self.chunk_size):
                # Record iteration start time
                start_time = time.time()
                logger.debug("Chunk read in %.2f ms", (start_time - end_time) * 1000)

                # Break out from the for-loop if `self.running` has changed
                if not self.running:
                    break

                chunk_size = chunk.shape[
                    0
                ]  # No guarantee that `chunk` is always of the requested size
                self.x_array = (
                    np.linspace(self.x, self.x + self.window_size - 1, self.window_size)
                    / self.sample_rate
                )
                self.y_array = np.roll(
                    self.y_array, -chunk_size
                )  # roll `self.y_array` back by `chunk_size`
                self.y_array[-chunk_size:] = chunk[
                    :, 1
                ]  # replace the last `chunk_size` entries with `chunk`
                self.x += chunk_size

                # Update y-axis min/max if need be:
                self.y_min = (
                    y_min
                    if (y_min := np.nanmin(self.y_array)) < self.y_min
                    else self.y_min
                )
                self.y_max = (
                    y_max
                    if (y_max := np.nanmax(self.y_array)) > self.y_max
                    else self.y_max
                )

                # Uh oh, we quickly notice that plotting this many points is downright unfeasible. Let's reduce the
                # load for visualization by selecting only every `plot_step`th value in the arrays.
                vis_x = self.x_array[
                    :: self.plot_step
                ].copy()  # these array slice views seem to go missing if not copied
                vis_y = self.y_array[:: self.plot_step].copy()

                # Detect peaks in the visible window
                peaks = self.detect_peaks(self.y_array)
                peak_x = self.x_array[peaks]
                peak_y = self.y_array[peaks]

                logger.debug(
                    "Computation done in %.2f ms", (time.time() - start_time) * 1000
                )

                # This signal will now block the thread until chart update is finished
                if set_chart_axis_y:
                    set_chart_axis_y.emit(self.y_min, self.y_max)

                # This signal will now block the thread until chart update is finished
                if update_chart:
                    update_chart.emit(vis_x, vis_y)

                if set_peaks:
                    set_peaks.emit(peak_x, peak_y)

                if progress_callback:
                    # This loop has no meaningful progress.
                    # Just report the current left-most x-coordinate.
                    progress_callback.emit(self.x)

                end_time = time.time()
                logger.debug(
                    "Finished iteration in %.2f ms", (end_time - start_time) * 1000
                )

            # Stop while-loop when the file ends
            self.running = False

    def load_window(
        self,
        start_x: int,
        end_x: int,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        set_peaks: Signal | None = None,
        progress_callback: Signal | None = None,
    ):
        """Loads a signal window.
        :param start_x: the start of the window
        :param end_x: the end of the window
        :param signal_path: the path to the signal data file.
        :param set_chart_axis_y: Signal to update chart y-axis.
        :param update_chart: Signal to update a chart.
        :param set_peaks: Signal to mark the detected peaks (time and amplitude arrays) on the chart.
        :param progress_callback: Signal to update progress meters."""
        self._set_signal_path(signal_path)
        if set_chart_axis_y:
            set_chart_axis_y.emit(self.y_min, self.y_max)

        self.x_array, self.y_array = self._read_signal_at(start_x, end_x)
        self.x_array = self.x_array / self.sample_rate
        self.y_array = self.y_array[:, 1]

        # DETECTION ALGORITHMS
        peaks = self.detect_peaks(self.y_array)

        # Update y-axis min/max if need be:
        self.y_min = (
            y_min if (y_min := np.nanmin(self.y_array)) < self.y_min else self.y_min
        )
        self.y_max = (
            y_max if (y_max := np.nanmax(self.y_array)) > self.y_max else self.y_max
        )

        vis_x = self.x_array[:: self.plot_step]
        vis_y = self.y_array[:: self.plot_step]

        if set_chart_axis_y:
            set_chart_axis_y.emit(self.y_min, self.y_max)

        if update_chart:
            update_chart.emit(vis_x, vis_y)

        if set_peaks:
            set_peaks.emit(self.x_array[peaks], self.y_array[peaks])

    def current_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        set_peaks: Signal | None = None,
        progress_callback: Signal | None = None,
    ):
        self._set_signal_path(signal_path)
        self.load_window(
            self.x,
            self.x + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            set_peaks,
            progress_callback,
        )

    def previous_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        set_peaks: Signal | None = None,
        progress_callback: Signal | None = None,
    ):
        self._set_signal_path(signal_path)
        self.x = self.x - self.window_size if self.x > self.window_size else 0
        self.load_window(
            self.x,
            self.x + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            set_peaks,
            progress_callback,
        )

    def next_window(
        self,
        signal_path: Path,
        set_chart_axis_y: Signal | None = None,
        update_chart: Signal | None = None,
        set_peaks: Signal | None = None,
        progress_callback: Signal | None = None,
    ):
        self._set_signal_path(signal_path)
        self.x = (
            self.x + self.window_size
            if self.max_x > self.x + self.window_size
            else self.x
        )
        self.load_window(
            self.x,
            self.x + self.window_size,
            signal_path,
            set_chart_axis_y,
            update_chart,
            set_peaks,
            progress_callback,
        )

    def has_previous_window(self):
        return self.x > 0

    def has_next_window(self):
        return self.x < self.max_x - self.window_size
