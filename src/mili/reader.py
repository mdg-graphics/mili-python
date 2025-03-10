"""Mili-python reader module.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import os

from typing import *

from mili.parallel import ServerWrapper, LoopWrapper
from mili.afileIO import afiles_by_base, MiliFileNotFoundError
from mili.milidatabase import MiliDatabase
from mili.miliinternal import _MiliInternal
from mili.utils import *
from mili.reductions import *


def open_database(base : Union[str,os.PathLike],
                  procs: List[int] = [],
                  suppress_parallel: bool = False,
                  experimental: bool = False,
                  merge_results: bool = True,
                  **kwargs: Any) -> MiliDatabase:
  """Open a database for querying.

  This opens the database metadata files and does additional processing to optimize query
  construction and execution. Don't use this to perform database verification, instead prefer AFileIO.parse_database().

  Args:
   base (Union[str,os.PathLike]): the base filename of the mili database (e.g. for 'pltA', just 'plt', for parallel
      databases like 'dblplt00A', also exclude the rank-digits, giving 'dblplt').
   procs ([List[int], default=[]) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Bool, default=False) : optionally return a serial database reader object if possible (for serial databases).
      Note: if the database is parallel, suppress_parallel==True will return a reader that will
      query each processes database files in series.
   experimental (Bool, default=False) : optional developer-only argument to try experimental parallel features
   merge_results (Bool, default=True): Merge parallel results into the serial format.

  WARNING: Reading a database in parallel on LLNL machines should not be done on Login nodes as this can use a lot of resources and negatively
  affect machine performance for other users.

  NOTE: While users can set the argument 'merge_results' to False, that is not recommended.
  """
  # ensure dir_name is the containing dir and base is only the file name
  dir_name = os.path.dirname( base )
  if dir_name == '':
    dir_name = os.getcwd()
  if not os.path.isdir( dir_name ):
    raise MiliFileNotFoundError( f"Cannot locate mili file directory {dir_name}.")

  base = os.path.basename( base )
  afiles = afiles_by_base( dir_name, base, procs )
  if len(afiles) == 0:
    raise MiliFileNotFoundError( f"No A files with the basename '{base}' were found in the directory '{dir_name}'")

  proc_bases = [ afile[:-1] for afile in afiles ] # drop the A to get each processes base filename for A,T, and S files

  parallel_handler: Union[type[_MiliInternal],type[LoopWrapper],type[ServerWrapper]]
  if suppress_parallel or len(proc_bases) == 1:
    if len(proc_bases) == 1:
      parallel_handler = _MiliInternal
    else:
      parallel_handler = LoopWrapper
  else:
    parallel_handler = ServerWrapper

  # Open Mili Database.
  mili_database = MiliDatabase(dir_name, proc_bases, parallel_handler, merge_results, **kwargs)

  return mili_database