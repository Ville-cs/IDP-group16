<!--
SPDX-FileCopyrightText: 2025 Joni Hyttinen <joni.hyttinen@uef.fi>

SPDX-License-Identifier: CC-BY-NC-SA-4.0
-->

# Team members

- Ville Poikela
- Joni Niiranen
- Viet Nguyen
- Tejaswini Shekharaswamy

# Signal analyzer template

This repository contains a project template that can be used as a base for your
own solution. The application is written in [Python](https://www.python.org) and
[Qt for Python](https://doc.qt.io/qtforpython-6/), and utilizes [numpy](https://numpy.org/)
for computational tasks.

## Python and virtual environments

### Setting up a uv virtual environment

This approach uses the system Python. We recommend that you first install a
tool to manage Python applications: [pipx](https://pypa.github.io/pipx/). The **pipx**-tool is used to
install Python applications in their own dedicated virtual environments.

- Install **pipx**

  `python3.14 -m pip install --user pipx`

The project is described in **pyproject.toml**, which includes descriptions of
the needed dependencies and scripts to run the project for [uv](https://docs.astral.sh/uv/)
packaging and dependency management tool.

- Install **uv**

  `pipx install uv`

- Download or `git clone` the project on your system
- In the project folder, create the project virtual environment

  `uv sync`

- Open the project in your chosen development environment and use uv integration,
  if such is available, to manage the virtual environment.

## Running the application

If **uv** was used to create the virtual environment, the application can
be run with command

`uv run signal_app`

from the project directory.

Other solutions, like **Anaconda**, should be able to run the program in the
designated virtual environment simply by running

    conda activate idp2026
    python -m singal_app.signal_app_main_window

## File tree

- signal_app

  The source code for the project.
  - \_\_init\_\_.py

    Package initialization code. Empty file.

  - signal_analyzer.py

    A signal analyzer class whose instances are run in worker threads.
    Communicates with the user interface code through signals.

  - signal_app_main_window.py

    The main class of the application. Composes the main window and contains
    program main methods. The main widget is in a separate class.

  - signal_app_widget.py

    The main widget of the application. The instantiated main class of the
    application creates an instance of this class as its center widget.

  - signal_window_chart_widget.py

    A simple chart widget for showing a signal.

  - worker.py

    A helper class to construct worker threads.

  - worker_signals.py

    A helper class' helper class. Defines the signals that a worker thread
    may emit after a successful completion, an exception, or for progress
    indication.

- pyproject.toml

  Project file for `uv`-build system.

- uv.lock

  The exact Python packages that are or should be installed in a virtual
  environment managed by `uv`.

- README.md

  This file.

## Extending the template

Tasks to solve the industrial problem by extending the template could include:

- Investigate the suitability of [SciPy](https://scipy.org/) for signal processing tasks?

  | Environment manager | Command        |
  | ------------------- | -------------- |
  | uv                  | `uv add scipy` |

- Add code to open a saved signal file?
- Add code read a saved signal file?
  - Perhaps it is not necessary to read the whole file into memory at once..?
    - windowing functions?
    - memory mapping?
- Edit **signal_analyzer.py** to do something useful
- Extend the user interface to show positive detections?
- Extend the application to count positive/false detections using a ground
  truth file?
- Switch **QtCharts** to something more capable?
