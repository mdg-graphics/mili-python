#!/bin/bash
# this is just a source-able file to get developers a working
#  edit venv quickly, not to be released

PYVS=3.10.8

module load python/${PYVS}
python3 -m venv .venv-mili-python-${PYVS}
source .venv-mili-python-${PYVS}/bin/activate
pip3 install --timeout=100 --upgrade pip
pip3 install --timeout=100 --find-links=https://wci-repo.llnl.gov/repository/pypi-group/simple -e .[dev]
