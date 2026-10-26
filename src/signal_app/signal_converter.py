from pathlib import Path

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

    SOURCE_SAMPLE_RATE = 50_000  # all source CSVs are assumed to be sampled at 50 kHz
    TARGET_SAMPLE_RATE = 1_000
    STEP = SOURCE_SAMPLE_RATE // TARGET_SAMPLE_RATE  # keep every 50th sample
    CHUNK_ROWS = 10 * SOURCE_SAMPLE_RATE  # read ten seconds of the signal at a time

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

        n_out = (n_rows + self.STEP - 1) // self.STEP

        data = np.memmap(
            target_signal_path,
            dtype=np.int16,
            mode="w+",
            offset=0,
            shape=(n_out, 2),
            order="C",
        )

        # A faster way to read CSVs in chunks using pandas. CHUNK_ROWS must be a multiple of STEP so that
        # every chunk starts on a kept sample.
        reader = pd.read_csv(
            source_signal_path,
            header=0,
            dtype=np.int16,
            chunksize=self.CHUNK_ROWS,
            skip_blank_lines=True,
        )

        written = 0
        source_rows = 0

        for chunk in reader:
            values = chunk.to_numpy()[:: self.STEP]
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
                if progress_callback:
                    progress_callback.emit(0)
                return False

        del data
        # Blank lines are counted by count_data_rows but skipped by pandas; drop the unused rows at the end.
        if written < n_out:
            with target_signal_path.open("r+b") as target_file:
                target_file.truncate(written * 2 * np.dtype(np.int16).itemsize)

        if progress_callback:
            # Finish the progress bar.
            progress_callback.emit(100)

        return True
