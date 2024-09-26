"""
SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations
from typing import List, Dict, Union, Any
import numpy as np
import pandas as pd

def dictionary_merge_no_concat(dictionaries: List[dict]) -> dict:
  """Merge dictionaries. When same key appears, just overwrite."""
  merged = {}
  for dictionary in dictionaries:
    merged.update(dictionary)
  return merged

def dictionary_merge_concat(dictionaries: List[dict], axis=None) -> dict:
  """Merge dictionaries. When same key appears in multiple dictionaries, concatenate results."""
  combined_dictionary = {}
  for dictionary in dictionaries:
    for key, value in dictionary.items():
      if key not in combined_dictionary:
        combined_dictionary[key] = value
      else:
        combined_dictionary[key] = np.append(combined_dictionary[key], value, axis=axis)
  return combined_dictionary

def dictionary_merge_concat_unique(dictionaries: List[dict], axis=None) -> dict:
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

def merge_result_dictionaries( result_dicts: List[dict] ) -> Dict:
  """Combine parallel results dictionaries into a single dictionary."""
  merged_results = {}

  svars = list(set(np.concatenate([list(res.keys()) for res in result_dicts])))
  svars = [str(sv) for sv in svars]
  for svar in svars:
    merged_results[svar] = {}
    # Get list of processors that contain data
    processors_with_data = [ result_dicts[i] for i in np.where( [d[svar]['data'].size > 0 for d in result_dicts] )[0] ]
    if processors_with_data:
      merged_results[svar]['layout'] = {}
      merged_results[svar]['source'] = processors_with_data[0][svar]['source']
      merged_results[svar]['layout']['states'] = processors_with_data[0][svar]['layout']['states']
      merged_results[svar]['data'] = np.concatenate( [d[svar]['data'][:,:,:] for d in processors_with_data], axis=1 )
      merged_results[svar]['layout']['labels'] = np.concatenate( [d[svar]['layout']['labels'] for d in processors_with_data] )

      _, indexes, counts = np.unique(merged_results[svar]['layout']['labels'], return_counts=True, return_index=True)
      if np.any(counts > 1):
        merged_results[svar]['data'] = merged_results[svar]['data'][:,indexes,:]
        merged_results[svar]['layout']['labels'] = merged_results[svar]['layout']['labels'][indexes]

  return merged_results

def merge_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
  """Merge a list of DataFrames from the MiliDatabase.query method into a single Pandas DataFrame."""
  merged_dataframes = {}
  svars = list(set(np.concatenate([list(res.keys()) for res in dataframes])))
  svars = [str(sv) for sv in svars]

  for svar in svars:
    svar_dfs = [df_dict.get(svar) for df_dict in dataframes]
    svar_dfs = [df for df in svar_dfs if not df.empty]
    df_merged = pd.concat(svar_dfs, axis=1, copy=False )
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    merged_dataframes[svar] = df_merged

  return merged_dataframes

def combine( parallel_results: List[Union[Dict,pd.DataFrame]] ) -> Union[Dict,pd.DataFrame]:
  """Given a List of parallel result dictionaries or Pandas DataFrames, Merge data into a single dictionary or DataFrame."""
  if isinstance( parallel_results, dict ) or isinstance( parallel_results, pd.DataFrame ):
    return parallel_results

  if isinstance(parallel_results, list) and len(parallel_results) > 0:
    # Need to figure out if this is dictionaries vs dataframes
    if isinstance(list(parallel_results[0].values())[0], dict):
      return merge_result_dictionaries( parallel_results )
    if isinstance(list(parallel_results[0].values())[0], pd.DataFrame):
      return merge_dataframes( parallel_results )
  return parallel_results

def list_concatenate_unique_str(l: List[List[str]]) -> List[str]:
  """Concatenate list and remove duplicate values"""
  l = [i for i in l if i is not None]
  return list(pd.unique(np.concatenate(l)))

def list_concatenate_unique(l: List[List[Any]]) -> np.ndarray:
  """Concatenate list and remove duplicate values"""
  l = [i for i in l if i is not None]
  return pd.unique(np.concatenate(l))

def list_concatenate(l: List[List[Any]]) -> np.ndarray:
  """Concatenate list"""
  l = [i for i in l if i is not None]
  return np.concatenate(l)

def zeroth_entry(l: List[Any]) -> Any:
  """Every value in the list is the same, just return the zero-th entry"""
  return l[0]

def reduce_nodes_of_elems(parallel_results):
  """Merge the results from the function nodes_of_elems"""
  combined_nodes = []
  combined_elems = []
  for result in parallel_results:
    nodes, elems = result
    if nodes.size > 0 and elems.size > 0:
      combined_nodes.append(nodes)
      combined_elems.append(elems)
  combined_nodes = np.concatenate(combined_nodes)
  combined_elems = np.concatenate(combined_elems)
  return combined_nodes, combined_elems

def reduce_labels(parallel_results) -> Union[np.ndarray,Dict[str,np.ndarray]]:
  """Merge the results of the labels method into the serial format"""
  if isinstance(parallel_results[0], dict):
    return dictionary_merge_concat_unique(parallel_results)
  else:
    return list_concatenate_unique(parallel_results)

def reduce_connectivity(parallel_results) -> Union[np.ndarray,Dict[str,np.ndarray]]:
  """Merge the results of the connectivity method into the serial format"""
  if isinstance(parallel_results[0], dict):
    return dictionary_merge_concat(parallel_results, axis=0)
  else:
    return list_concatenate(parallel_results)