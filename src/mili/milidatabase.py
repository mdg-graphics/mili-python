"""MiliDatabase module.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", UserWarning) # Pandas NUMEXPR warning

import numpy as np
import numpy.typing as npt
from numpy import iterable
import os

from typing import *
import pandas as pd

import mili.reductions as reductions
from mili.parallel import ServerWrapper, LoopWrapper
from mili.miliinternal import _MiliInternal, ReturnCode
from mili.datatypes import StateMap
from mili.utils import result_dictionary_to_dataframe

class MiliPythonError(Exception):
  """Mili Python Exception object."""
  pass

def parse_return_codes(return_codes: Union[Tuple[ReturnCode,str], List[Tuple[ReturnCode,str]]]) -> None:
  """Processes return codes from MiliDatabase and check for errors or exceptions."""
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
  """MiliDatabase class that supports querying and other functions on Serial and Parallel Mili databases."""
  def __init__(self,
               dir_name: Union[str,bytes,os.PathLike],
               base_files: List[Union[str,bytes,os.PathLike]],
               parallel_handler: Type[Union[_MiliInternal,LoopWrapper,ServerWrapper]],
               merge_results: bool,
               **kwargs):
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

  def __postprocess(self, results: Any, reduce_function: Any) -> Any:
    return_codes = self._mili.returncode()
    self._mili.clear_return_code()
    parse_return_codes(return_codes)
    self.__check_for_exceptions(results)
    if self.serial or not self.merge_results:
      return results
    else:
      return reduce_function(results)

  def __check_for_exceptions(self, results) -> None:
    """Catch any unhandled exception from parallel wrappers."""
    if self.serial and isinstance(results, Exception):
      raise results
    elif iterable(results):
      for res in results:
        if isinstance(res, Exception):
          raise res

  # We define the enter and exit methods to support using MiliDatabase as a context manager
  def __enter__(self):
    """__enter__ method to support context manager protocol."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """__exit__ method to support context manager protocol."""
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  def close(self):
    """Close the mili database and shutdown any subprocesses being used."""
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  @property
  def geometry(self):
    """Getter for internal geometry object."""
    return self._mili.geometry

  def reload_state_maps(self) -> None:
    """Reload the state maps."""
    return self.__postprocess(
      results = self._mili.reload_state_maps(),
      reduce_function = reductions.zeroth_entry)

  def nodes(self) -> np.ndarray:
    """Getter for initial nodal coordinates.

    Returns:
      np.ndarray: A numpy array (num_nodes by mesh_dimensions) containing the initial coordinates for each node.
    """
    if self.serial or not self.merge_results:
      return self._mili.nodes()
    else:
      # Concatenate list of node labels which will contain duplicates
      nlabels = reductions.list_concatenate( self._mili.labels("node") )
      # Get index of first appearance of each node
      _, indexes = np.unique(nlabels, axis=0, return_index=True)
      # Sort to perserve relative ordering of nodes (have to do this because np.unique sorts)
      indexes.sort()
      # Return nodal coordinates for unique node labels
      nodes = np.concatenate( self._mili.nodes() )
      return nodes[indexes]

  def state_maps(self) -> List[StateMap]:
    """Getter for internal list of StateMaps.

    Returns:
      List[StateMap]: A list of StateMap objects.
    """
    return self.__postprocess(
      results = self._mili.state_maps(),
      reduce_function = reductions.zeroth_entry)

  def srec_fmt_qty(self) -> int:
    """Getter for State record format quantity.

    Returns:
      int: The number of state record formats.
    """
    return self.__postprocess(
      results = self._mili.srec_fmt_qty(),
      reduce_function = reductions.zeroth_entry)

  def mesh_dimensions(self) -> int:
    """Getter for Mesh Dimensions.

    Returns:
      int: Either 2 or 3 for the number of dimensions.
    """
    return self.__postprocess(
      results = self._mili.mesh_dimensions(),
      reduce_function = reductions.zeroth_entry)

  def class_names(self) -> List[str]:
    """Getter for all class names in the database.

    Returns:
      List[str]: List of element class names in the database.
    """
    return self.__postprocess(
      results = self._mili.class_names(),
      reduce_function = reductions.list_concatenate_unique_str)

  def int_points_of_state_variable(self, svar_name: str, class_name: str) -> np.ndarray:
    """Get the available integration points for a state variable + class_name.

    Args:
      svar_name (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      np.ndarray: Array of integration points.
    """
    return self.__postprocess(
      results = self._mili.int_points_of_state_variable(svar_name, class_name),
      reduce_function = reductions.list_concatenate_unique)

  def element_sets(self) -> Dict[str,List[int]]:
    """Getter for the element sets.

    Returns:
      Dict[str,List[int]]: Keys are element set names, values are list of integers
    """
    return self.__postprocess(
      results = self._mili.element_sets(),
      reduce_function = reductions.dictionary_merge_no_concat)

  def integration_points(self) -> Dict[str,List[int]]:
    """Get the available integration points for each material.

    Returns:
      Dict[str,List[int]]: Keys are material numbers, values are a list of integration points.
    """
    return self.__postprocess(
      results = self._mili.integration_points(),
      reduce_function = reductions.dictionary_merge_no_concat)

  def times( self, states : Optional[Union[List[int],int]] = None ) -> np.ndarray:
    """Get the times for each state in the database.

    Args:
      states (Optional[Union[List[int],int]]): If provided, only return the times for the
        specified state numbers.

    Returns:
      np.ndarray: numpy array of times.
    """
    return self.__postprocess(
      results = self._mili.times(states),
      reduce_function = reductions.zeroth_entry)

  def queriable_svars(self, vector_only = False, show_ips = False) -> List[str]:
    """Get a list of state variable names that can be queried.

    Args:
      vector_only (bool, default=False): Return on vector state variables.
      show_ips (bool, default=False): Show the available integration points.

    Returns:
      List[str]: A list of state variable names that can be queried.
    """
    return self.__postprocess(
      results = self._mili.queriable_svars(vector_only, show_ips),
      reduce_function = reductions.list_concatenate_unique_str)

  def supported_derived_variables(self) -> List[str]:
    """Get the derived variables that mili-python currently supports.

    Returns:
      List[str]: A List of derived variable names that can be calculated by mili-python.
    """
    return self.__postprocess(
      results = self._mili.supported_derived_variables(),
      reduce_function = reductions.zeroth_entry)

  def derived_variables_of_class(self, class_name: str) -> List[str]:
    """Get the derived variables that can be calculated for the specified class.

    Args:
      class_name (str): The element class name.

    Returns:
      List[str]: List of derived variables that can be calculated for the specified class.
    """
    return self.__postprocess(
      results = self._mili.derived_variables_of_class(class_name),
      reduce_function = reductions.list_concatenate_unique_str)

  def classes_of_derived_variable(self, var_name: str) -> List[str]:
    """Get the classes for which a derived variable can be calculated.

    Args:
      var_name (str): The derived variable name:

    Returns:
      List[str]: List of element class names for which var_name can be calculated.
    """
    return self.__postprocess(
      results = self._mili.classes_of_derived_variable(var_name),
      reduce_function = reductions.list_concatenate_unique_str)

  @overload
  def labels(self, class_name : str) -> np.ndarray: ...
  @overload
  def labels(self, class_name : None = ...) -> Dict[str,np.ndarray]: ...

  def labels(self, class_name: Optional[str] = None) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element labels.

    Args:
      class_name (Optional[str]): If provided, only return labels for specifid element class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the labels for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the labels for the specified element class.
    """
    return self.__postprocess(
      results = self._mili.labels(class_name),
      reduce_function = reductions.reduce_labels)

  def materials(self) -> Dict[str,List[int]]:
    """Get materials dictionary from the database.

    Returns:
      Dict[str,List[int]]: Keys are the material names, Values are a list of material numbers.
    """
    return self.__postprocess(
      results = self._mili.materials(),
      reduce_function = reductions.zeroth_entry)

  def material_numbers(self) -> np.ndarray:
    """Get a List of material numbers in the database.

    Returns:
      np.ndarray: A numpy array of the material numbers.
    """
    return self.__postprocess(
      results = self._mili.material_numbers(),
      reduce_function = reductions.list_concatenate_unique)

  @overload
  def connectivity(self, class_name : str) -> np.ndarray: ...
  @overload
  def connectivity(self, class_name : None = ...) -> Dict[str,np.ndarray]: ...

  def connectivity(self, class_name : Optional[str] = None) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element connectivity as element LABELS.

    Args:
      class_name (str): An element class name. If provided only return connectivty for the specified class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the connectivity for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the connectivity for the specified element class. If the specified element
        class does not exists then None is returned.
    """
    return self.__postprocess(
      results = self._mili.connectivity(class_name),
      reduce_function = reductions.reduce_connectivity)

  def material_classes(self, mat: Union[str,int]) -> List[str]:
    """Get list of classes of a specified material.

    Args:
      mat (Union[str,int]): A material name or number.

    Returns:
      List[str]: List of element classes associated with the material.
    """
    return self.__postprocess(
      results = self._mili.material_classes(mat),
      reduce_function = reductions.list_concatenate_unique_str)

  def classes_of_state_variable(self, svar: str) -> List[str]:
    """Get list of element classes for a state variable.

    Args:
      svar (str): The state variable name.

    Returns:
      List[str]: List of element classes the state variable exists for.
    """
    return self.__postprocess(
      results = self._mili.classes_of_state_variable(svar),
      reduce_function = reductions.list_concatenate_unique_str)

  def state_variables_of_class(self, class_name: str) -> List[str]:
    """Get list of primal state variables for a given element class.

    Args:
      class_name (str): The element class name.

    Returns:
      List[str]: A list of primal state variables that can be queried for the specified element class.
    """
    return self.__postprocess(
      results = self._mili.state_variables_of_class(class_name),
      reduce_function = reductions.list_concatenate_unique_str)

  def state_variable_titles(self) -> dict[str,str]:
    """Get dictionary of state variable titles for each state variable.

    Returns:
      dict[str,str]: Dictionary where keys are svar names and values are svar titles.
    """
    return self.__postprocess(
      results = self._mili.state_variable_titles(),
      reduce_function = reductions.dictionary_merge_no_concat
    )

  def containing_state_variables_of_class(self, svar: str, class_name: str) -> List[str]:
    """Get List of state variables that contain the specific state variable + class_name.

    Args:
      svar (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      List[str]: List of containing state variables.
    """
    return self.__postprocess(
      results = self._mili.containing_state_variables_of_class(svar, class_name),
      reduce_function = reductions.list_concatenate_unique_str)

  def components_of_vector_svar(self, svar: str) -> List[str]:
    """Get a list of component state variables of a vector state variable.

    Args:
      svar (str): The name of a vector state variable.

    Returns:
      List[str]: The components of the vector state variable.
    """
    return self.__postprocess(
      results = self._mili.components_of_vector_svar(svar),
      reduce_function = reductions.list_concatenate_unique_str)

  def parts_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of part numbers for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of part numbers for each element of the class name.
    """
    return self.__postprocess(
      results = self._mili.parts_of_class_name(class_name),
      reduce_function = reductions.list_concatenate)

  def materials_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of materials for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of material numbers for each element of the class name.
    """
    return self.__postprocess(
      results = self._mili.materials_of_class_name(class_name),
      reduce_function = reductions.list_concatenate)

  def class_labels_of_material( self, mat: Union[str,int], class_name: str ) -> np.ndarray:
    """Convert a material name into labels of the specified class (if any).

    Args:
      mat (Union[str,int]): The material name or number.
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of labels of the specified class name and material.
    """
    return self.__postprocess(
      results = self._mili.class_labels_of_material(mat, class_name),
      reduce_function = reductions.list_concatenate)

  def all_labels_of_material( self, mat: Union[str,int] ) -> Dict[str,np.ndarray]:
    """Given a specific material. Find all labels with that material and return their values.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      Dict[str,np.ndarray]: Keys are element class names. Values are numpy arrays of element labels.
    """
    return self.__postprocess(
      results = self._mili.all_labels_of_material(mat),
      reduce_function = reductions.dictionary_merge_concat)

  def nodes_of_elems(self, class_sname: str, elem_labels: Union[int,List[int]]) -> Tuple[np.ndarray,np.ndarray]:
    """Find nodes associated with elements by label.

    Args:
      class_sname (str): The element class name.
      elem_labels (List[int]): List of element labels.

    Returns:
      Tuple[np.ndarray,np.ndarray]: (The nodal connectivity, The element labels)
    """
    return self.__postprocess(
      results = self._mili.nodes_of_elems(class_sname, elem_labels),
      reduce_function = reductions.reduce_nodes_of_elems)

  def nodes_of_material(self, mat: Union[str,int] ) -> np.ndarray:
    """Find nodes associated with a material number.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      numpy.array: A list of all nodes associated with the material number.
    """
    return self.__postprocess(
      results = self._mili.nodes_of_material(mat),
      reduce_function = reductions.list_concatenate_unique)

  def __process_query_modifier(self, modifier: str, results: dict, as_dataframe: bool ):
    if modifier not in ("min", "max", "average", "cummin", "cummax"):
      raise ValueError("Invalide modifier. Must be one of (min, max, average, cummin, cummax).")
    results = reductions.combine(results)

    for svar in results:
      results[svar]["modifier"] = modifier

    ##### Minimum #####
    if modifier == "min":
      for svar in results:
        labels = results[svar]['layout']['labels']
        min_indexes = np.argmin( results[svar]["data"], axis=1, keepdims=True )
        results[svar]["data"] = np.take_along_axis( results[svar]["data"], min_indexes, axis=1 )
        results[svar]["layout"]["labels"] = labels[min_indexes.flatten()]

      if as_dataframe:
        # Special handing for min/max dataframes
        for svar in results:
          states = results[svar]["layout"]["states"]
          labels = results[svar]["layout"]["labels"]
          if len(states) == len(labels):
            data = results[svar]["data"].flatten()
          else:
            data = np.reshape( results[svar]["data"], (len(states),-1))
            labels = np.reshape( labels, (len(states),-1))
          results[svar] = pd.DataFrame( zip(data,labels), index=states, columns=["min", "label"])

    ##### Maximum #####
    elif modifier == "max":
      for svar in results:
        labels = results[svar]['layout']['labels']
        max_indexes = np.argmax( results[svar]["data"], axis=1, keepdims=True )
        results[svar]["data"] = np.take_along_axis( results[svar]["data"], max_indexes, axis=1 )
        results[svar]["layout"]["labels"] = labels[max_indexes.flatten()]

      if as_dataframe:
        # Special handing for min/max dataframes
        for svar in results:
          states = results[svar]["layout"]["states"]
          labels = results[svar]["layout"]["labels"]
          if len(states) == len(labels):
            data = results[svar]["data"].flatten()
          else:
            data = np.reshape( results[svar]["data"], (len(states),-1))
            labels = np.reshape( labels, (len(states),-1))
          results[svar] = pd.DataFrame( zip(data,labels), index=states, columns=["max", "label"])

    ##### Average #####
    elif modifier == "average":
      for svar in results:
        labels = results[svar]['layout']['labels']
        results[svar]["data"] = np.average( results[svar]["data"], axis=1, keepdims=True )

      if as_dataframe:
        # Special handing for average dataframes
        for svar in results:
          data = results[svar]["data"].flatten()
          states = results[svar]["layout"]["states"]
          if len(data) != len(states):
            data = results[svar]["data"]
            results[svar] = pd.DataFrame.from_records(data, index=states, columns=["average"])
          else:
            results[svar] = pd.DataFrame( data, index=states, columns=["average"])

    ##### Cumulative Min #####
    elif modifier == "cummin":
      for svar in results:
        states = results[svar]["layout"]["states"]
        for i in range(1,len(states)):
          results[svar]["data"][i] = np.minimum( results[svar]["data"][i], results[svar]["data"][i-1])

    ##### Cumulative Max #####
    elif modifier == "cummax":
      for svar in results:
        states = results[svar]["layout"]["states"]
        for i in range(1,len(states)):
          results[svar]["data"][i] = np.maximum( results[svar]["data"][i], results[svar]["data"][i-1])

    return results

  def query( self,
             svar_names : Union[List[str],str],
             class_sname : str,
             material : Optional[Union[str,int]] = None,
             labels : Optional[Union[List[int],int]] = None,
             states : Optional[Union[List[int],int]] = None,
             ips : Optional[Union[List[int],int]] = None,
             write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None,
             as_dataframe: Optional[bool] = False,
             modifier: Optional[str] = None,
             **kwargs ) -> dict:
    """Query the database for state variables or derived variables, returning data for the specified parameters, optionally writing data to the database.

    Args:
      svar_names (Union[List[str],str]): The names of the state variables to be queried.
      class_sname (str): The element class name being queried (e.g. brick. shell, node).
      material (Optional[Union[str,int]], default=None): Optional material name or number to select labels from.
      labels (Optional[Union[List[int],int]], default=None): Optional labels to query data for, filtered by material
        if material if material is supplied, default is all.
      states (Optional[Union[List[int],int]], default=None): Optional state numbers from which to query data, default is all.
      ips (Optional[Union[List[int],int]], default=None): Optional integration point to query for vector array state variables, default is all available.
      write_data (Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]], default=None): Optional the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format,
        then modify the values desired, then query again with the modified result in this param
      as_dataframe (Optional[bool]): If True the result is returned as a Pandas DataFrame.
      modifier (Optional[str]): Optional string specifying modifer to apply to results. Valid modifiers are min, max, average, cummin, cummax.
    """
    results = self.__postprocess(
      results = self._mili.query(svar_names, class_sname, material, labels, states, ips, write_data, **kwargs),
      reduce_function = reductions.combine)

    if modifier:
      results = self.__process_query_modifier(modifier, results, as_dataframe)

    if as_dataframe:
      return result_dictionary_to_dataframe(results)
    return results

  def append_state(self,
                   new_state_time: float,
                   zero_out: bool = True,
                   limit_states_per_file: Optional[int] = None,
                   limit_bytes_per_file: Optional[int] = None) -> int:
    """Appends a new state to the end of an existing Mili database.

    Args:
      new_state_time (float): The time of the new state.
      zero_out (bool, default=True): If True, all results for the new timestep are set to zero (With the exception of
        nodal positions and sand flags. nodal_positions are copied from the previous state and sand flags are set to 1).
        If false, the data from the previous state (if available) will be copied to the new state.
      limit_states_per_file (Optional[int]): If provided limits the number of states per state file.
      limit_bytes_per_file (Optional[int]): If provided limits the files size of state files.

    Returns:
      int: The updated number of states in the database.
    """
    return self.__postprocess(
      results = self._mili.append_state(new_state_time, zero_out, limit_states_per_file, limit_bytes_per_file),
      reduce_function = reductions.zeroth_entry)

  def copy_non_state_data(self, new_base_name: str) -> None:
    """Copy non state data from an existing Mili database into a new database without any states (Just the A file).

    Non state data include the geometry, states variables, subrecords and parameters.

    Args:
      new_base_name (str): The base name of the new database.
    """
    return self.__postprocess(
      results = self._mili.copy_non_state_data(new_base_name),
      reduce_function = reductions.zeroth_entry)