# MDG Mili Python Reader
[![pipeline status](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/pipeline.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)
[![coverage report](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/coverage.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)

# Changelog

A history of the changes made to mili-python can be found ![here](doc/changelog.md)

# Manual

The Mili Python manual can be found ![here](doc/source/manual.md)

# Developer Guide

### Style Guide
The current style guide for mili-python is [PEP8](https://peps.python.org/pep-0008/), except instead of indentation being 4 spaces we use 2 spaces.

### Virtual Environment

You will need to create a local virtual environment for testing and development. To do this run:
```
cd mili-python
source .venv.sh
```
This will create a virtual environment called `.venv-mili-python-3.8.2` on Toss3 and `.venv-mili-python-3.9.12` on Toss4 with all the required dependencies for mili-python and activate it. To deactivate the virtual environment, run the command `deactivate` and to activate the environment run `source .venv-mili-python-3.8.2/bin/activate`.

### Testing

To run the test suite locally, cd into the directory `mili-python/src` and run: `python3 -m unittest discover tests`

### Deployment

The mili-python reader is distributed through the WCI Nexus repository.

Nexus documentation:
- Link to nexus docs: https://wci-svc-doc.llnl.gov/repo/nexus/
- Setup for python: https://wci-svc-doc.llnl.gov/repo/setup_proxy/#python-pypi
- Publishing python packages: https://wci-svc-doc.llnl.gov/repo/publishing/#python-pypi

Before continuing take a look at the above documentation and perform the steps described in the `Setup for python` link.

To generate the python `whl` file that is distributed run the following:
```
cd mili-python/
version=$(cat src/mili/__init__.py | grep 'version' | grep -Eo "[[:digit:]]+,[[:digit:]]+,[[:digit:]]+" | tr , . )
git tag -a v${version} -m "version $version"
git push origin v${version}
python3 -m build .
```

This will generate a `./dist` directory in the top level of the `mili-python` repository that contains the `whl` file, as well as the `sdist` (not currently distributed).

To upload the `whl` file to the nexus repository you will need to use twine as shown below:
```
pip3 install twine
python3 -m twine upload -r pypi-wci <whl-file>
```
> **NOTE**: The password requested is your AD password.

### PDF Manual

To re-generate the pdf version of the manual:
```
source .venv-mili-python-3.10.8/bin/activate
cd doc
./gen_pdf_manual.sh
```

# License
----------------

Mili Python Reader is distributed under the terms of both the MIT license.

All new contributions must be made under both the MIT license.

See [LICENSE-MIT](https://github.com/mdg/mili-python/LICENSE)

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-838121
