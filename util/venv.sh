#!/bin/bash

SYS_PYTHON_BIN="/usr/tce/packages/python/python-3.7.2/bin/python3"
PIP_ARGS="--no-cache --find-links=https://www-lc.llnl.gov/python/wheelhouse"
VENV_NAME="venv-mili-python-3.7.2"

module load python/3.7.2
python3 --version
${SYS_PYTHON_BIN} -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate
pip3 install --upgrade pip # need newer pip to build numpy on rz
pip3 install ${PIP_ARGS} -e .