# MDG Mili Python Reader
[![pipeline status](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/pipeline.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)
[![coverage report](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/coverage.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)

# Changelog

A history of the changes made to mili-python can be found ![here](doc/changelog.md)

# Manual / Docs

The Mili Python manual and documentation can be found at the following links:
- CZ: https://lc.llnl.gov/mili-python/
- RZ: https://rzlc.llnl.gov/mili-python/

# Developer Guide

### Style Guide

The current style guide for mili-python is [PEP8](https://peps.python.org/pep-0008/), except instead of indentation being 4 spaces we use 2 spaces.

### Docstring Style

Sphinx is used to generate documentation for the API. This documentation is built from the docstrings in the code. The end user will see this documentation, so care should be taken both in writing and checking the docstrings. The CI tests the docstring using `pydocstyle` and will fail if any errors are found.

An example docstring:

```python
"""Summary line ending with a period.

Some other information that goes into it.

Args:
    arg1 (int): Description
    arg2 (Dict[str, Object]): A very longgggggggggggggggggggggggggggggggggggg
        ggggggggggggggg description that wraps around at 100 characters. Lines
        for the same argument should be indented.

Returns:
    int:
        The return type is optional and specified in the beginning
        followed by a colon. Following the colon is the desired description.
        Notice when there are multiple lines, the following line must be
        indented to match the first indented line.
"""
```

### Virtual Environment

You will need to create a local virtual environment for testing and development. To do this run:
```
cd mili-python
source .venv.sh
```
This will create a virtual environment called `.venv-mili-python-3.10.8` on Toss4 with all the required dependencies for mili-python and activate it. To deactivate the virtual environment, run the command `deactivate` and to activate the environment run `source .venv-mili-python-3.10.8/bin/activate`.

# Mypy

Mili-python uses mypy to perform type checking as a part of its CI. To run mypy locally, run: `mypy src/mili`.

### Testing

To run the test suite locally, cd into the directory `mili-python/src` and run: `python3 -m unittest discover tests`.

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

### Documentation

To generate and deploy new documentation do the following:

```bash
# This needs to be done on both CZ and RZ
source .venv-mili-python-3.10.8/bin/activate
cd doc/
make html
python3 deploy_docs.py
```

**NOTE**: You can build the documentation once on RZ and then transfer the `html` directory and the `deploy_docs.py` script to CZ and run the script there to deploy the documentation.

# License
----------------

Mili Python Reader is distributed under the terms of both the MIT license.

All new contributions must be made under both the MIT license.

See [LICENSE-MIT](https://github.com/mdg/mili-python/LICENSE)

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-838121
