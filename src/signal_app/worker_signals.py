# SPDX-FileCopyrightText: 2019 Martin Fitzpatrick
# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi
#
# SPDX-License-Identifier: MIT

from PySide6.QtCore import QObject, Signal


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished:
        No data

    error:
        tuple (exctype, value, traceback.format_exc())

    result:
        object data returned from processing, anything
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)
