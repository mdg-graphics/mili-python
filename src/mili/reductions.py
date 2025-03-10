"""
SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations
from typing import List, Dict, Union, Any, overload, TypeVar, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mili.datatypes import QueryDict, QueryLayout

# Dictionary Key and Value type vars
KT = TypeVar('KT')
VT = TypeVar('VT')

def dictionary_merge_no_concat(dictionaries: List[Dict[KT,VT]]) -> Dict[KT,VT]:
  """Merge dictionaries. When same key appears, just overwrite."""
  merged = {}
  for dictionary in dictionaries:
    merged.update(dictionary)
  return merged

def dictionary_merge_concat(dictionaries: List[Dict[KT,NDArray[Any]]], axis: Optional[int] = None) -> Dict[KT,NDArray[Any]]:
  """Merge dictionaries. When same key appears in multiple dictionaries, concatenate results."""
  combined_dictionary = {}
  for dictionary in dictionaries:
    for key, value in dictionary.items():
      if key not in combined_dictionary:
        combined_dictionary[key] = value
      else:
        combined_dictionary[key] = np.append(combined_dictionary[key], value, axis=axis)
  return combined_dictionary

def dictionary_merge_concat_unique(dictionaries: List[Dict[KT,NDArray[Any]]], axis: Optional[int] = None) -> Dict[KT,NDArray[Any]]:
  """Merge dictionaries. When same key appears in multiple dictionaries, concatenate unique results."""
  combined_dictionary = {}
  for dictionary in dictionaries:
    for key, value in dictionary.items():
      if key not in combined_dictionary:
        combined_dictionary[key] = value
      else:
        combined_dictionary[key] = np.append(combined_dictionary[key], value, axis=axis)
  for key in combined_dictionary:
    # We use pandas unique because it is faster than numpy and maintains order by default.
    combined_dictionary[key] = pd.unique(combined_dictionary[key])
  return combined_dictionary

def merge_result_dictionaries( result_dicts: List[Dict[str,QueryDict]] ) -> Dict[str,QueryDict]:
  """Combine parallel results dictionaries into a single dictionary."""
  merged_results: dict[str,QueryDict] = {}

  svars = list(set(np.concatenate([list(res.keys()) for res in result_dicts])))
  svars = [str(sv) for sv in svars]
  for svar in svars:
    # Get list of processors that contain data
    processors_with_data = [ result_dicts[i] for i in np.where( [d[svar]['data'].size > 0 for d in result_dicts] )[0] ]
    if processors_with_data:
      merged_results[svar] = QueryDict(
        source = processors_with_data[0][svar]['source'],
        title = processors_with_data[0][svar]['title'],
        class_name = processors_with_data[0][svar]['class_name'],
        layout = QueryLayout(
          states = processors_with_data[0][svar]['layout']['states'],
          times = processors_with_data[0][svar]['layout']['times'],
          components = processors_with_data[0][svar]['layout']['components'],
          labels = np.concatenate( [d[svar]['layout']['labels'] for d in processors_with_data] ),
        ),
        data = np.concatenate( [d[svar]['data'][:,:,:] for d in processors_with_data], axis=1 ),
        modifier = processors_with_data[0][svar].get('modifier', ''),
      )

      _, indexes, counts = np.unique(merged_results[svar]['layout']['labels'], return_counts=True, return_index=True)
      if np.any(counts > 1):
        merged_results[svar]['data'] = merged_results[svar]['data'][:,indexes,:]
        merged_results[svar]['layout']['labels'] = merged_results[svar]['layout']['labels'][indexes]

  return merged_results

def merge_dataframes(dataframes: List[Dict[str,pd.DataFrame]]) -> Dict[str,pd.DataFrame]:
  """Merge a list of DataFrames from the MiliDatabase.query method into a single Pandas DataFrame."""
  merged_dataframes = {}
  svars = list(set(np.concatenate([list(res.keys()) for res in dataframes])))
  svars = [str(sv) for sv in svars]

  for svar in svars:
    svar_dfs = [df_dict[svar] for df_dict in dataframes]
    svar_dfs = [df for df in svar_dfs if not df.empty]
    df_merged = pd.concat(svar_dfs, axis=1, copy=False )
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    merged_dataframes[svar] = df_merged

  return merged_dataframes

@overload
def combine( parallel_results: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]] ) -> Dict[str,QueryDict]:
  ...
@overload
def combine( parallel_results: Union[Dict[str,pd.DataFrame],List[Dict[str,pd.DataFrame]]] ) -> Dict[str,pd.DataFrame]:
  ...

def combine( parallel_results: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]],Dict[str,pd.DataFrame],List[Dict[str,pd.DataFrame]]] ) -> Union[Dict[str,QueryDict],Dict[str,pd.DataFrame]]:
  """Given a List of parallel result dictionaries or Pandas DataFrames, Merge data into a single dictionary or DataFrame."""
  if isinstance( parallel_results, dict ) or isinstance( parallel_results, pd.DataFrame ):
    return parallel_results

  if isinstance(parallel_results, list) and len(parallel_results) > 0:
    # Need to figure out if this is dictionaries vs dataframes
    if isinstance(list(parallel_results[0].values())[0], dict):
      return merge_result_dictionaries( parallel_results )  # type: ignore  # mypy can't tell that it's the correct datatype.
    if isinstance(list(parallel_results[0].values())[0], pd.DataFrame):
      return merge_dataframes( parallel_results )  # type: ignore  # mypy can't tell that it's the correct datatype.

  raise ValueError(f"An error occurred while trying to combine")

def list_concatenate_unique_str(l: List[List[str]]) -> List[str]:
  """Concatenate list and remove duplicate values"""
  l = [i for i in l if i is not None]
  return [str(i) for i in pd.unique(np.concatenate(l))]

def list_concatenate_unique(l: Union[List[List[Any]],NDArray[Any]]) -> NDArray[Any]:
  """Concatenate list and remove duplicate values"""
  l = [i for i in l if i is not None and len(i) > 0]
  if len(l) == 0:
    return np.empty([0])
  else:
    return pd.unique(np.concatenate(l))  # type: ignore  # Ignoring because pd.unique can return Any

def list_concatenate(l: Union[List[List[Any]],NDArray[Any]]) -> NDArray[Any]:
  """Concatenate list"""
  l = [i for i in l if i is not None and len(i) > 0]
  if len(l) == 0:
    return np.empty([0])
  else:
    return np.concatenate(l)

def zeroth_entry(l: List[Any]) -> Any:
  """Every value in the list is the same, just return the zero-th entry"""
  return l[0]

def reduce_nodes_of_elems(parallel_results: List[Tuple[NDArray[np.int32],NDArray[np.int32]]]) -> Tuple[NDArray[np.int32],NDArray[np.int32]]:
  """Merge the results from the function nodes_of_elems"""
  combined_nodes = []
  combined_elems = []
  for result in parallel_results:
    nodes, elems = result
    if nodes.size > 0 and elems.size > 0:
      combined_nodes.append(nodes)
      combined_elems.append(elems)
  return np.concatenate(combined_nodes, dtype=np.int32), np.concatenate(combined_elems, dtype=np.int32)

def reduce_labels(parallel_results: Union[List[NDArray[np.int32]],List[Dict[str,NDArray[np.int32]]]]) -> Union[NDArray[np.int32],Dict[str,NDArray[np.int32]]]:
  """Merge the results of the labels method into the serial format"""
  if isinstance(parallel_results[0], dict):
    return dictionary_merge_concat_unique(parallel_results)  # type: ignore  # Mypy can't differentiate between the two possible types of parallel_results
  else:
    return list_concatenate_unique(parallel_results)  # type: ignore  # Mypy can't differentiate between the two possible types of parallel_results

def reduce_connectivity(parallel_results: Union[List[NDArray[np.int32]],List[Dict[str,NDArray[np.int32]]]]) -> Union[NDArray[np.int32],Dict[str,NDArray[np.int32]]]:
  """Merge the results of the connectivity method into the serial format"""
  if isinstance(parallel_results[0], dict):
    return dictionary_merge_concat(parallel_results, axis=0)  # type: ignore  # Mypy can't differentiate between the two possible types of parallel_results
  else:
    return list_concatenate(parallel_results)  # type: ignore  # Mypy can't differentiate between the two possible types of parallel_results