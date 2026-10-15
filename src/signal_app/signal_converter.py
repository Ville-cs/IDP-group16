import csv
from pathlib import Path
import os

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal


class SignalConverter:
    """Convert a CSV-signal file into a binary data file.

    This SignalConverter is intended to be run as a thread within an instance of the `Worker`-class.
    Usage example:
         # Get a thread pool
         threadpool = QThreadPool.globalInstance()

         # Create a SignalConverter instance
         converter = SignalConverter()

         # Prepare a Worker instance with `start`-method as the thread entry point
         worker = Worker(converter.start, source_signal_path=self.signal_path, target_signal_path=signal_path)

         # Connect worker's progress signal into a progress bar or dialog
         worker.signals.progress.connect(progress.setValue)

         # Start the thread
         threadpool.start(worker)
    """

    # Flag to indicate the conversion process should be cancelled.
    cancelled: bool = False

    def cancel(self):
        self.cancelled = True

    def start(
        self,
        source_signal_path: Path,
        target_signal_path: Path,
        progress_callback: Signal | None = None,
    ) -> bool:
        self.cancelled = False

        # Make sure the progress is at 0 %
        if progress_callback:
            progress_callback.emit(0)

        # a faster way to count rows by using binary chunks
        def count_data_rows(path, chunk_size=1 << 24): # 16 MB chunks
            n_lines = 0
            last_byte = b"\n"
            with path.open("rb") as f:
                while chunk := f.read(chunk_size):
                    n_lines += chunk.count(b"\n")
                    last_byte = chunk[-1:]
            if last_byte != b"\n":  # last line has no trailing newline
                n_lines += 1
            return n_lines - 1  # subtract header

        n_rows = count_data_rows(source_signal_path)

        # Was the process cancelled during our reading?
        if self.cancelled:
            return False

        # Open a memory mapped binary file for writing. Set data type to 16-bit integers. Save both adc1 and adc2
        # in their own columns as in the original file. Order refers to C and Fortran conventions of multidimensional
        # array layouts: C uses row-major and Fortran column-major layouts (see f.ex. Wikipedia article on the topic
        # https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        # NOTE: the binary file will not contain dtype, shape, or order information. They must be supplied manually.

        SAMPLE_RATE = 50_000
        STEP = 100  # downsampling to every 100th data point
        CHUNK_ROWS = SAMPLE_RATE

        n_out = (n_rows + STEP - 1) // STEP

        data = np.memmap(
            target_signal_path,
            dtype=np.int16,
            mode="w+",
            offset=0,
            shape=(n_out, 2),
            order="C",
        )

        CHUNK_ROWS = 500_000

        # A faster way to read CSVs in chunks using pandas
        written = 0
        reader = pd.read_csv(
            source_signal_path,
            header=0,
            dtype=np.int16,
            chunksize=CHUNK_ROWS,
            skip_blank_lines=True,
        )

        written = 0
        source_rows = 0

        for chunk in reader:
            values = chunk.to_numpy()[::STEP]
            data[written:written + len(values), :] = values
            written += len(values)
            source_rows += len(chunk)

            # Push changes to disk after each chunk
            data.flush()

            if progress_callback:
                progress_callback.emit((source_rows / n_rows) * 100)

            if self.cancelled:
                del data
                target_signal_path.unlink()  # Remove the binary data file
                return False

        data.flush()  # again, once more force the array to write changes to disk
        if not self.cancelled and progress_callback:
            # Finish the progress bar.
            progress_callback.emit(100)
        if self.cancelled and progress_callback:
            del data
            target_signal_path.unlink()
            progress_callback.emit(0)

        return not self.cancelled
