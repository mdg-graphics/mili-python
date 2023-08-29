#!/bin/bash
# this is just a source-able file to get developers a working
#  edit venv quickly, not to be released

if [[ $SYS_TYPE == "toss_3_x86_64_ib" ]]; then
    PYVS=3.8.2
elif [[ $SYS_TYPE == "toss_4_x86_64_ib" ]]; then
    PYVS=3.9.12
fi

module load python/${PYVS}
python3 -m venv .venv-mili-python-${PYVS}
source .venv-mili-python-${PYVS}/bin/activate
pip3 install --upgrade pip
pip3 install --find-links=https://wci-repo.llnl.gov/repository/pypi-group/simple -e .[dev]
