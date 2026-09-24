<!--
SPDX-FileCopyrightText: 2025 Joni Hyttinen <joni.hyttinen@uef.fi>

SPDX-License-Identifier: CC-BY-NC-SA-4.0
-->

# Signal analyzer template
This repository contains a project template that can be used as a base for your
own solution. The application is written in [Python](https://www.python.org) and
[Qt for Python](https://doc.qt.io/qtforpython-6/), and utilizes [numpy](https://numpy.org/) 
for computational tasks.

## Python and virtual environments
Python ecosystem has multiple ways to install and use packages. Especially
for Windows and macOS users, a fairly simple to install Python distribution
is [Anaconda](https://anaconda.com). Linux-users may find Anaconda or its
package manager **conda** in their distribution's package repository.

Another option is to use the system Python.
- On Windows, one can find **Python Install Manager** in *Microsoft Store*.
  In a Terminal session, a copy of **Python 3.14** can be installed with command 
  `pymanager install 3.14`;
- On Linux, the distribution should provide **Python 3.14** in their package
  repository;
- On macOS, a third-party package repository, [Homebrew](https://brew.sh/), can
  be used to install the current **Python 3.14**.

### Setting up a Conda virtual environment

- Create a minimal new virtual environment:

  ``conda create -n idp2026 python=3.14``

- Activate the new virtual environment

  ``conda activate idp2026``

- Install **PySide6**

  ``pip install pyside6``

  > We specifically avoid using Anaconda's repositories here. PySide6 is 
  > available in **conda-forge**, but their build does not include QtCharts,
  > which is needed for this application.

- Install the other dependencies

  ``conda install numpy``

- Updating the environment can be done with

  ``conda update --all``

- Download or `git clone` the project on your system.
- Open it the development environment of your choice.
- Choose the newly created **idp2026** virtual environment as your project's
  Python environment.

### Setting up a uv virtual environment
This approach uses the system Python. We recommend that you first install a
tool to manage Python applications: [pipx](https://pypa.github.io/pipx/). The **pipx**-tool is used to
install Python applications in their own dedicated virtual environments.

- Install **pipx**
  
  ``python3.14 -m pip install --user pipx``

The project is described in **pyproject.toml**, which includes descriptions of
the needed dependencies and scripts to run the project for [uv](https://docs.astral.sh/uv/)
packaging and dependency management tool.

- Install **uv**

  ``pipx install uv``

- Download or `git clone` the project on your system
- In the project folder, create the project virtual environment
  
  `uv sync`

- Open the project in your chosen development environment and use uv integration,
  if such is available, to manage the virtual environment.

## Running the application
If **uv** was used to create the virtual environment, the application can
be run with command

``uv run signal_app``

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

  - signal\_analyzer.py

    A signal analyzer class whose instances are run in worker threads.
    Communicates with the user interface code through signals.
  
  - signal\_app\_main\_window.py
  
    The main class of the application. Composes the main window and contains
    program main methods. The main widget is in a separate class.

  - signal\_app\_widget.py
  
    The main widget of the application. The instantiated main class of the
    application creates an instance of this class as its center widget.

  - signal\_window\_chart\_widget.py
    
    A simple chart widget for showing a signal.

  - worker.py
  
    A helper class to construct worker threads.

  - worker\_signals.py
  
    A helper class' helper class. Defines the signals that a worker thread
    may emit after a successful completion, an exception, or for progress
    indication.
  
- pyproject.toml

  Project file for ``uv``-build system.

- uv.lock

  The exact Python packages that are or should be installed in a virtual
  environment managed by ``uv``.

- README.md
  
  This file.

## Extending the template

Tasks to solve the industrial problem by extending the template could include:
- Investigate the suitability of [SciPy](https://scipy.org/) for signal processing tasks?
  
  | Environment manager | Command                          |
  |---------------------|----------------------------------|
  | Anaconda            | `conda -n idp2026 install scipy` |
  | uv                  | `uv add scipy`                   |

- Add code to open a saved signal file?
- Add code read a saved signal file?
  - Perhaps it is not necessary to read the whole file into memory at once..?
    - windowing functions?
    - memory mapping?
- Edit **signal\_analyzer.py** to do something useful
- Extend the user interface to show positive detections?
- Extend the application to count positive/false detections using a ground
  truth file?
- Switch **QtCharts** to something more capable?

