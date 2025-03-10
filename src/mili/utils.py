"""Various mili-python utilities.

SPDX-License-Identifier: (MIT)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional, Any
from numpy.typing import NDArray, ArrayLike

from mili.reductions import combine
from mili.datatypes import QueryDict, QueryLayout

def results_by_element(result_dict: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]]) -> Dict[str,Dict[np.integer,NDArray[np.floating]]]:
  """Reorganize result data in a new dictionary with the form { svar: { element: <list_of_results> } }.

  Args:
    result_dict (Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]]): Result dictionary(s) from MiliDatabase.query method.

  NOTE: This transformation loses some of the information stored in the result_dictionary. The
        times, class_name, source, title and components in the original result dictionary are
        not transferred.
  """
  if not isinstance(result_dict, dict):
    result_dict = combine(result_dict)

  reorganized_data: Dict[str,Dict[np.integer,NDArray[np.floating]]] = {}
  for svar in result_dict:
    if svar not in reorganized_data:
      reorganized_data[svar] = {}

    if isinstance( result_dict[svar], pd.DataFrame ):
      raise ValueError("The results_by_element function does not support Pandas Dataframes.")

    for elem_idx, element in enumerate(result_dict[svar]['layout']['labels']):
      if element not in reorganized_data:
        reorganized_data[svar][element] = result_dict[svar]['data'][:,elem_idx,:]

  return reorganized_data

def writeable_from_results_by_element(results_dict: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]],
                                      results_by_element: Dict[str,Dict[np.integer,NDArray[np.floating]]]) -> Dict[str,QueryDict]:
  """Given the query result dictionary and the results_by_element dictionary generate writeable dictionary.

  Args:
    result_dict (Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]]): Result dictionary(s) from MiliDatabase.query method.
    results_by_element (Dict[str,Dict[np.integer,NDArray[np.floating]]]): The results_by_element format of result_dict. Must be the same query.

  Returns:
    Dict[str,QueryDict]: A dictionary that is writable using the Milidatabase.query method.
  """
  if not isinstance(results_dict, dict):
    results_dict = combine(results_dict)
  for result in results_by_element:
    if result in results_dict and results_dict[result]['data'].size > 0:
      result_shape = results_dict[result]['data'][:,0].shape
      for idx, element in enumerate(list(results_by_element[result].keys())):
          # NOTE: We are able to just use idx (0-N) because we assume results_dict and results_by_element
          # are from the same query which means that the element order in results_dict and results_by_element
          # will be the same.
          write_data = np.array(results_by_element[result][element])
          if write_data.shape != result_shape:
            # Try to be less strict about data shape.
            write_data = np.reshape(write_data, result_shape)
          results_dict[result]['data'][:,idx] = write_data
  return results_dict

def query_data_to_dataframe(data: NDArray[np.floating], states: NDArray[np.int32], labels: NDArray[np.int32]) -> pd.DataFrame:
  """Creates a Pandas DataFrame from a 3d NumPy array returned by MiliDatabase.Query.

  Args:
    data (NDArray[np.floating]): A 3-dimensional NumPy array
    states (NDArray[np.int32]): The states in the query
    labels (NDArray[np.int32]): The labels in the query

  Returns:
    A Pandas DataFrame.
  """
  if data.ndim != 3:
    raise ValueError("'data' Array must be 3-dimensional")
  if states.ndim != 1:
    raise ValueError("'states' Array must be 1-dimensional")
  if labels.ndim != 1:
    raise ValueError("'labels' Array must be 1-dimensional")
  if data.shape[0] != states.shape[0]:
    raise ValueError("Mismatch between shape of states and data.")

  if data.shape[2] == 1:
    df = pd.DataFrame(data.reshape(data.shape[:-1]), columns=labels, index=states)
  else:
    df = pd.DataFrame.from_records(data)
    df.index = states  # type: ignore
    df.columns = labels  # type: ignore

  return df

def result_dictionary_to_dataframe(result_dict: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]]) -> Dict[str,pd.DataFrame]:
  """Convert dictionary from default format of MiliDatabase.query method to a Pandas DataFrame.

  Args:
    result_dict (Union[List[dict],dict]): Result dictionary(s) from MiliDatabase.query methd.

  Returns:
    A Pandas DataFrame.

  NOTE: This transformation loses some of the information stored in the result_dictionary. The
        times, class_name, source, title and components in the original result dictionary are
        not transferred.
  """
  result_dataframes = {}

  if isinstance(result_dict, list):
    result_dict = combine(result_dict)

  for svar_name, svar_result_dict in result_dict.items():
    if isinstance(svar_result_dict, pd.DataFrame):
      result_dataframes[svar_name] = svar_result_dict
    else:
      df = pd.DataFrame()
      if svar_result_dict['data'].size > 0:
        data = svar_result_dict['data']
        states = svar_result_dict['layout']['states']
        labels = svar_result_dict['layout']['labels']
        df = query_data_to_dataframe( data, states, labels )
      result_dataframes[svar_name] = df

  return result_dataframes

def dataframe_to_result_dictionary(result_df: Dict[str,pd.DataFrame]) -> Dict[str,QueryDict]:
  """Convert dataframe to MiliDatabase.query result dictionary format.

  Args:
    result_df (Dict[str,pd.DataFrame]): Result dictionary of dataframes.
  """
  result_dict = {}
  for svar_name, svar_df in result_df.items():
    if not svar_df.empty:
      data_shape = svar_df.shape + ( svar_df.iloc[0,0].shape if svar_df.iloc[0,0].ndim else (1,) )  # type: ignore
      data_type = svar_df.iloc[0,0].dtype  # type: ignore
      result_dict[svar_name] = QueryDict(
        source = "unknown",
        class_name = "unknown",
        title = "unknown",
        layout = QueryLayout(
          states = np.array(svar_df.index, dtype=np.int32),
          labels = np.array(svar_df.columns, dtype=np.int32),
          times = np.empty([0], dtype=np.float32),
          components = [],
        ),
        data = np.empty(data_shape, dtype=data_type),
      )
      for i,ii in enumerate(svar_df.index):
        for j,jj in enumerate(svar_df.columns):
          result_dict[svar_name]['data'][i,j,:] = svar_df.loc[ii,jj]

  return result_dict

def argument_to_ndarray(argument: ArrayLike, dtype: Any) -> Optional[NDArray[Any]]:
  """Convert an ArrayLike object into a numpy array of the specified dtype.

  Args:
    argument (ArrayLike): An array like object to be converted to a numpy array.
    dtype: The numpy dtype for the array

  Returns:
    Optional[NDArray[Any]]: A numpy array of type dtype. If an exception occurs trying to convert
      argument to the numpy array, then None is returned.
  """
  try:
    if np.isscalar(argument):
      as_array = np.array([argument], dtype=dtype)
    else:
      as_array = np.array(argument, dtype=dtype)
  except:
    as_array = None
  return as_array