#!/usr/bin/env python3
import re
import toml
from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

with open('pyproject.toml', 'r') as fh:
  requirements = toml.loads(fh.read())

prod = requirements['dependencies']

def get_property(prop, package):
    result = re.search(r'{}\s*=\s*(.*)\s*'.format(prop), open(f'src/{package}/__init__.py').read())
    return result.group(1).strip('"\'')

setup(
  name="mili",
  version=get_property('__version__','mili').strip('()').replace(',','.'),
  description="modules for interacting with mili database files",
  long_description=long_description,
  author="William R Tobin, Kevin Durrenberger, Ryan Hathaway",
  author_email="tobin6@llnl.gov, durrenberger1@llnl.gov, hathaway6@llnl.gov",
  packages=find_packages('src',exclude=['tests']),
  package_dir={'': 'src'},
  package_data={ 'mili' : ["src/mili/py.typed"] },
  install_requires=[x + prod[x] if prod[x] != "*" else x for x in prod],
  python_requires=requirements['project']['requires-python']
)
