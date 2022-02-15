#!/usr/bin/env python

from setuptools import setup, find_packages
import toml

with open('README.md', 'r') as fh:
  long_description = fh.read()

with open('pyproject.toml', 'r') as fh:
  requirements = toml.loads(fh.read())

prod = requirements['dependencies']

setup(
  name="mili",
  version="0.2.0",
  description="modules for interacting with mili database files",
  long_description=long_description,
  author="William R Tobin, Kevin Durrenberger",
  author_email="tobin6@llnl.gov, durrenberger1@llnl.gov",
  packages=find_packages(),
  scripts=["scripts/mili-query.py"],
  install_requires=[x + prod[x] if prod[x] != "*" else x for x in prod],
  python_requires=requirements['project']['requires-python']
)
