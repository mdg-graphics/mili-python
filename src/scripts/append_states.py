#!/usr/bin/env python3
"""
SPDX-License-Identifier: (MIT)
"""

import argparse
import importlib
from mili.append_states import AppendStatesTool
from typing import Dict

def main():
  """
  This block is here to allow users to run this tool from the command line instead of from a script.
  Usage: python3 -m mili.append_states -i <input_file.py>
  """
  # Parse input arguments
  description = "Tool to append states and data to Mili databases."
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-i', action='store', dest='input_file', default=None, required=True,
                      help='The input file containing append states specificition.')
  args = parser.parse_args()
  input_file = args.input_file

  # Verify input file is python file
  if not input_file.endswith(".py"):
    raise ValueError(f"input_file must be a python source file. The file '{input_file}' does not end with .py")
  input_file = input_file.replace(".py", "")

  # Attempt to import input file
  try:
    input = importlib.import_module( input_file )
  except:
    print(f"An error occurred while trying to import '{input_file}'. Please check that there are no errors in your input file.")
    # Re-raise exception
    raise

  # Verify that append_states_spec dictionary exists
  if not hasattr(input, "append_states_spec"):
    raise ValueError("The input file must contain the variable 'append_states_spec'.")
  if not isinstance( input.append_states_spec, Dict ):
    raise ValueError(f"The variable 'append_states_spec' must be a dictionary. Type is {type(input.append_states_spec)}")

  tool = AppendStatesTool(input.append_states_spec)
  tool.write_states()