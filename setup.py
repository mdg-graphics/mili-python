#!/usr/bin/env python
import os
import re
import toml

from git import Repo 
from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

with open('pyproject.toml', 'r') as fh:
  requirements = toml.loads(fh.read())

prod = requirements['dependencies']

repo = Repo(os.getcwd())
tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
latest_tag = str(tags[-1])
version_match = re.search(r'v*(\d\.\d\.\d)',latest_tag)
if version_match:
  version = version_match.group(1)
else:
  version = "indev"

setup(
  name="mili",
  version=version,
  description="modules for interacting with mili database files",
  long_description=long_description,
  author="William R Tobin, Kevin Durrenberger",
  author_email="tobin6@llnl.gov, durrenberger1@llnl.gov",
  packages=find_packages('src',exclude=['tests']),
  package_dir={'': 'src'},
  install_requires=[x + prod[x] if prod[x] != "*" else x for x in prod],
  python_requires=requirements['project']['requires-python']
)
