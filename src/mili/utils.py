"""
SPDX-License-Identifier: (MIT)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict
from functools import reduce

def dictionary_merge_concat(dictionaries):
  """Merge dictionaries. When same key appears in multiple dictionaries, concatenate results."""
  combined_dictionary = {}
  for dictionary in dictionaries:
    for key, value in dictionary.items():
      if key not in combined_dictionary:
        combined_dictionary[key] = value
      else:
        combined_dictionary[key] = np.append(combined_dictionary[key], value)
  return combined_dictionary

def dictionary_merge_concat_unique(dictionaries):
  """Merge dictionaries. When same key appears in multiple dictionaries, concatenate unique results."""
  combined_dictionary = {}
  for dictionary in dictionaries:
    for key, value in dictionary.items():
      if key not in combined_dictionary:
        combined_dictionary[key] = value
      else:
        combined_dictionary[key] = np.unique(np.append(combined_dictionary[key], value))
  return combined_dictionary

def merge_result_dictionaries( result_dicts: List[dict] ) -> Dict:
  """Combine parallel results dictionaries into a single dictionary."""
  merged_results = {}

  svars = list(set(np.concatenate([list(res.keys()) for res in result_dicts])))
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

  for svar in svars:
    svar_dfs = [df_dict.get(svar) for df_dict in dataframes]
    svar_dfs = [df for df in svar_dfs if not df.empty]
    df_merged = pd.concat(svar_dfs, axis=1, copy=False )
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    merged_dataframes[svar] = df_merged

  return merged_dataframes

def combine( parallel_results: List[Union[Dict,pd.DataFrame]] ) -> Union[Dict,pd.DataFrame]:
  """Given a List of parallel result dictionaries or Pandas DataFrames, Merge data into a single dictionary or DataFrame.

  NOTE: This does not take an insignificant amount of time
  """
  if isinstance( parallel_results, dict ) or isinstance( parallel_results, pd.DataFrame ):
    return parallel_results

  if isinstance(parallel_results, list) and len(parallel_results) > 0:
    # Need to figure out if this is dictionaries vs dataframes
    if isinstance(list(parallel_results[0].values())[0], dict):
      return merge_result_dictionaries( parallel_results )
    if isinstance(list(parallel_results[0].values())[0], pd.DataFrame):
      return merge_dataframes( parallel_results )
  return parallel_results

def results_by_element( result_dict: Union[Dict,List[Dict]] ) -> dict:
  """Reorganize result data in a new dictionary with the form { svar: { element: <list_of_results> } }"""
  if isinstance( result_dict, dict ):
    result_dict = [result_dict]

  reorganized_data = {}
  for processor_result in result_dict:
    for svar in processor_result:
      if svar not in reorganized_data:
        reorganized_data[svar] = {}

      for elem_idx, element in enumerate(processor_result[svar]['layout']['labels']):
        if element not in reorganized_data:
          reorganized_data[svar][element] = processor_result[svar]['data'][:,elem_idx,:]

  return reorganized_data

def writeable_from_results_by_element( results_dict: dict, results_by_element: dict ) -> dict:
  """Given the query result dictionary and the results_by_element dictionary generate writeable dictionary."""
  results_dict = combine(results_dict)
  for result in results_by_element:
    if result in results_dict:
      for idx, element in enumerate(list(results_by_element[result].keys())):
          # NOTE: We are able to just use idx (0-N) because we assume results_dict and results_by_element
          # are from the same query which means that the element order in results_dict and results_by_element
          # will be the same.
          results_dict[result]['data'][:,idx] = results_by_element[result][element]
  return results_dict

def query_data_to_dataframe(data: np.ndarray, states: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
  """Creates a Pandas DataFrame from a 3d NumPy array returned by MiliDatabase.Query

  Args:
    data (numpy.ndarray): A 3-dimensional NumPy array
    states (numpy.ndarray): The states in the query
    labels (numpy.ndarray): The labels in the query

  Returns:
    A Pandas DataFrame.
  """
  if data.ndim != 3:
    raise ValueError("Arry must be 3-dimensional")

  if data.shape[2] == 1:
    df = pd.DataFrame(data.reshape(data.shape[:-1]), columns=labels, index=states)
  else:
    df = pd.DataFrame.from_records(data)
    df.index = states
    df.columns = labels

  return df

def result_dictionary_to_dataframe(result_dict: Union[List[dict],dict]) -> pd.DataFrame:
  """Convert dictionary from default format of MiliDatabase.query method to a Pandas DataFrame."""
  result_dataframes = {}

  if isinstance(result_dict, list):
    result_dict = combine(result_dict)

  for svar_name, svar_result_dict in result_dict.items():
    df = pd.DataFrame()
    if svar_result_dict['data'].size > 0:
      data = svar_result_dict['data']
      states = svar_result_dict['layout']['states']
      labels = svar_result_dict['layout']['labels']
      df: pd.DataFrame = query_data_to_dataframe( data, states, labels )
    result_dataframes[svar_name] = df

  return result_dataframes

def dataframe_to_result_dictionary(result_df: Dict[str,pd.DataFrame]) -> dict:
  """Convert dataframe to MiliDatabase.query result dictionary format."""
  result_dict = {}
  for svar_name, svar_df in result_df.items():
    if not svar_df.empty:
      result_dict[svar_name] = {}
      result_dict[svar_name]['layout'] = {
        "states": np.array(svar_df.index, dtype=np.int32),
        "labels": np.array(svar_df.columns, dtype=np.int32),
      }
      data_shape = svar_df.shape + ( svar_df.iloc[0,0].shape if svar_df.iloc[0,0].ndim else (1,) )
      data_type = svar_df.iloc[0,0].dtype
      result_dict[svar_name]['data'] = np.empty(data_shape, dtype=data_type)
      for i,ii in enumerate(svar_df.index):
        for j,jj in enumerate(svar_df.columns):
          result_dict[svar_name]['data'][i,j,:] = svar_df.loc[ii,jj]

  return result_dict