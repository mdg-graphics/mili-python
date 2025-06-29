"""
SPDX-License-Identifier: (MIT)
"""
# mypy: ignore-errors

from typing import *
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from numpy import iterable

# utility classes to allow users to interact with the results data structures in ways that might be more natural for them

class ResultWrapper:
  """
    Provides a simple interface for a user to pull data out of results for specific labels, states, or both.
      This does not enforce parallel reductions since the databases have no concept of ownership of a shared node,
      so the data will still come out as a single result for each processor in a list.
    If the user wants a parallel reduction to get all the results as condensed as possible, they need to use a
      ReductionWrapper and supply a reduction heuristic (or use the default of lower-rank owns conflicting labels
      which is used in both diablo and paradyn IIRC).
  """

  def __init__(self, result : Union[List,Dict], times : Optional[List] = None):
    if issubclass(dict, result):
      result = [result]
    self.__result = result
    self.__times = times if times is not None else defaultdict(lambda : 0.0)

  def states( self, svar : Type[str] ) :
    states = set()
    for proc_result in self.__result:
      svar_result = proc_result.get(svar)
      if svar_result is not None:
        [ states.add( state ) for state in svar_result['layout']['state'] ]
    return np.array( list(states), dtype=np.int32 )

  def labels( self, svar : Type[str] ):
    labels = set()
    for proc_result in self.__result:
      svar_result = proc_result.get(svar)
      if svar_result is not None:
        for srec_name, srec_result in svar_result['layout'].items():
          if srec_name == 'state':
            continue
          [ labels.add( label ) for label in srec_result.labels( ) ]
    return np.array( list(labels), dtype=np.int64 )

  def for_labels( self, svar : str, labels : Union[List[int],Type[int],npt.ArrayLike]):
    if iterable( labels ):
      labels = np.array( labels, dtype = np.int32 )
    else:
      labels = np.array( [ labels ], dtype = np.int32 )
    return self.for_labels_for_states( svar, labels, self.states( svar ) )

  def for_states( self, svar : str, states : Union[List[int],Type[int],npt.ArrayLike]):
    if iterable( states ):
      states = np.array( states, dtype = np.int32 )
    else:
      states = np.array( [ states ], dtype = np.int32 )
    return self.for_labels_for_states( svar, self.labels( svar ), states )

  def for_labels_for_states( self,
                             svar : str,
                             labels : Union[List[int],int,npt.ArrayLike],
                             states : Union[List[int],int,npt.ArrayLike] ):
    filtered_result = [ ]
    for proc_result in self.__result:
      svar_result = proc_result.get(svar)
      proc_filtered = { }
      if svar_result is not None:
        sidx = np.searchsorted( states, svar_result['layout']['state'] )
        for srec_name, srec_result in svar_result['data'].keys():
          lidx = np.searchsorted( labels, svar_result['layout'][srec_name] )
          proc_filtered[ srec_name ] = srec_result[ sidx, lidx, : ], self.__times[ sidx ]
    return filtered_result

class ReductionWrapper(ResultWrapper):
  """
    This forwards extraction operations to the wrapped ResultWrapper and then performs reductions on the data
      to get the user a final result when processing queries from a parallel database.
    The default reduction heuristic is that a process that comes sooner in the parallel database (had a lower
      MPI rank in the original simulation) owns the data, so when conflicts are discovered, the lower-rank data
      is chosen over any higher-rank data.
  """
  def __init__(self, result : Union[List,Dict]):
    super(ReductionWrapper).__init__(result)
