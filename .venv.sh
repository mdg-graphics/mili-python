# this is just a source-able file to get developers a working
#  edit venv quickly, not to be released
module load python/3.7.2
python3 -m venv .venv-mili-python-3.7.2
source .venv-mili-python-3.7.2/bin/activate
pip3 install --upgrade pip
pip3 install --find-links=https://www-lc.llnl.gov/python/wheelhouse -e .
