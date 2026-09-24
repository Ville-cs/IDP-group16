import csv
from pathlib import Path

import numpy as np
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

        # This is terrible: read each row from the file to count the number of rows
        row_count = 0
        with source_signal_path.open() as source_signal_file:
            reader = csv.reader(source_signal_file)
            next(reader)
            for row in csv.reader(source_signal_file):
                if not row:
                    continue
                row_count += 1

                # Was the process cancelled during our reading?
                if self.cancelled:
                    return False

        # Open a memory mapped binary file for writing. Set data type to 16-bit integers. Save both adc1 and adc2
        # in their own columns as in the original file. Order refers to C and Fortran conventions of multidimensional
        # array layouts: C uses row-major and Fortran column-major layouts (see f.ex. Wikipedia article on the topic
        # https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        # NOTE: the binary file will not contain dtype, shape, or order information. They must be supplied manually.
        data = np.memmap(
            target_signal_path,
            dtype=np.int16,
            mode="w+",
            offset=0,
            shape=(row_count, 2),
            order="C",
        )

        # Read the CSV-file row-by-row writing it into the binary data file. This approach is slow, but it allows full
        # control of the reading process - every 50000 points or 1 second of data, the progress bar is updated.
        # Using some other reader that allows larger chunks could be more efficient than this.
        with source_signal_path.open() as source_signal_file:
            reader = csv.reader(source_signal_file)
            next(reader)
            for index, row in enumerate(reader):
                if not row:
                    continue
                data[index, :] = row
                if index % (50000 * 60) == 0:
                    # Force the data array to write changes to disk. This presumably happens automatically too at
                    # implementation-defined points, but this the API provided to the numpy users.
                    data.flush()
                if progress_callback and index % 50000 == 0:
                    progress_callback.emit((index / row_count) * 100)

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
