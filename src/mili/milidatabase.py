"""
SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", UserWarning) # Pandas NUMEXPR warning

import numpy as np
import os

from typing import *

from mili.parallel import ServerWrapper, LoopWrapper
from mili.adjacency import GeometricMeshInfo
from mili.miliinternal import _MiliInternal, ReturnCode
from mili.reductions import *

class MiliPythonError(Exception):
  pass

def parse_return_codes(return_codes: Union[Tuple[ReturnCode,str], List[Tuple[ReturnCode,str]]]) -> None:
  if isinstance(return_codes, tuple):
    return_codes = [return_codes]
  return_codes = np.array(return_codes)
  if not np.all([rcode_tup[0] == ReturnCode.OK for rcode_tup in return_codes]):
    # An error has occurred. Need to determine severity.
    num_ret_codes = len(return_codes)
    error_types = np.array(return_codes)[:,0]
    errors = np.where(np.isin(error_types, ReturnCode.ERROR))[0]
    critical = np.where(np.isin(error_types, ReturnCode.CRITICAL))[0]
    if len(critical) > 0 or len(errors) == num_ret_codes:
      all_errors = np.concatenate((return_codes[errors], return_codes[critical]))
      error_msgs = list(set([f"{retcode[0].str_repr()}: {retcode[1]}" for retcode in all_errors]))
      raise MiliPythonError(", ".join(error_msgs))


class MiliDatabase:
  def __init__(self,
               dir_name: Union[str,bytes,os.PathLike],
               base_files: List[Union[str,bytes,os.PathLike]],
               parallel_handler: Union[_MiliInternal,LoopWrapper,ServerWrapper],
               merge_results: bool,
               **kwargs):
    """MiliDatabase class supports querying and other function on Serial and Parallel Mili databases."""
    self.merge_results = merge_results
    dir_names = [dir_name] * len(base_files)
    proc_pargs = [ [dir_name,base_file] for dir_name, base_file in zip(dir_names, base_files)]
    proc_kwargs = [kwargs.copy() for _ in proc_pargs]

    self.serial = len(proc_pargs) == 1

    if parallel_handler == _MiliInternal:
      self._mili = _MiliInternal( *proc_pargs[0], **proc_kwargs[0] )
    else:
      shared_mem = kwargs.get("shared_mem", False)
      if shared_mem:
        self._mili = parallel_handler( _MiliInternal, proc_pargs, proc_kwargs, use_shared_memory=True )
      else:
        self._mili = parallel_handler( _MiliInternal, proc_pargs, proc_kwargs )

    # black list functions to not be exposed to users
    black_list = ["clear_return_code", "returncode", "connectivity_ids",
                  "subrecords", "state_variables", "mesh_object_classes",
                  "int_points", "parameter", "parameters"]
    # These functions are manually wrapped
    skip_functions = ["nodes"]
    # Functions to ignore for the reduce functions
    ignore_reduce = ["close"]

    # Add member functions to this object based on the member functions of the wrapped objects
    call_lambda = lambda attr, wrapped_function : lambda *pargs, **kwargs: self.__postprocess(attr, wrapped_function(*pargs, **kwargs) )
    wrapped_type = type(self._mili)
    for func in ( func for func in dir(self._mili) if not func.startswith('_') ):
      if func not in black_list and func not in skip_functions:
        # Class Methods
        attr = getattr(self._mili, func)
        if callable(attr):
          if func not in ignore_reduce and func not in reduce_functions:
            # This exception is intended to be caught in testing so we can ensure that all new function are added
            # to the reduce_functions dictionary.
            raise Exception(f"The function {func} must be added to the reduce_functions dictionary in reductions.py")
          setattr(self, func, call_lambda(func, attr ))
        elif isinstance(attr, (GeometricMeshInfo,LoopWrapper)):
          setattr(self, func, attr)

  def __postprocess(self, function_name: str, results: Any) -> Any:
    return_codes = self._mili.returncode()
    self._mili.clear_return_code()
    parse_return_codes(return_codes)
    if self.serial or not self.merge_results:
      return results
    else:
      reduce_func = reduce_functions.get(function_name, lambda x: x)
      return reduce_func(results)

  def nodes(self) -> np.ndarray:
    """Getter for initial nodal coordinates.

    Returns:
      np.ndarray: A numpy array (num_nodes by mesh_dimensions) containing the initial coordinates for each node.
    """
    if self.serial or not self.merge_results:
      return self._mili.nodes()
    else:
      # Concatenate list of node coordinates which will contain duplicates
      nodes = np.concatenate( self._mili.nodes() )
      # Get index of first appearance of each node
      _, indexes = np.unique(nodes, axis=0, return_index=True)
      # Sort to perserve relative ordering of nodes (have to do this because np.unique sorts)
      indexes.sort()
      return nodes[indexes]