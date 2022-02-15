#!/usr/bin/env python

from clize import run
from pprint import pprint
from mili import reader

def add_docstring_from(original):
  def wrapper(target):
    target.__doc__ += original.__doc__
    return target
  return wrapper

@add_docstring_from(reader.MiliDatabase.query)
def open_query( base_filename, svar_names, class_sname, *, material = None, labels = None, states = None, ips = None ):
  """
  Open a mili database and immediately query the database with the specified input. Pretty-printing the result.
  : param base_filename : The base filename of the mili A-file, exluding processor-specifying numerals and the A suffix
  """
  pprint( reader.open_database( base_filename ).query( svar_names, class_sname, material = material, labels = labels, states = states, ips = ips ) )

def main()  :
  run(open_query)

if __name__ == "__main__":
  main()