"""
SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", UserWarning) # Pandas NUMEXPR warning

import numpy as np
import numpy.typing as npt
from numpy.lib.function_base import iterable
import os

from typing import *

from mili.parallel import ServerWrapper, LoopWrapper
from mili.miliinternal import _MiliInternal, ReturnCode
from mili.reductions import *
from mili.datatypes import StateMap

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
               parallel_handler: Type[Union[_MiliInternal,LoopWrapper,ServerWrapper]],
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

  def __postprocess(self, function_name: str, results: Any) -> Any:
    return_codes = self._mili.returncode()
    self._mili.clear_return_code()
    parse_return_codes(return_codes)
    self.__check_for_exceptions(results)
    if self.serial or not self.merge_results:
      return results
    else:
      reduce_func = reduce_functions.get(function_name, lambda x: x)
      return reduce_func(results)

  def __check_for_exceptions(self, results) -> None:
    """Catch any unhandled exception from parallel wrappers"""
    if self.serial and isinstance(results, Exception):
      raise results
    elif iterable(results):
      for res in results:
        if isinstance(res, Exception):
          raise res

  # We define the enter and exit methods to support using MiliDatabase as a context manager
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  def close(self):
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  @property
  def geometry(self):
    return self._mili.geometry

  def reload_state_maps(self) -> None:
    """Reload the state maps."""
    return self.__postprocess("reload_state_maps", self._mili.reload_state_maps())

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

  def state_maps(self) -> List[StateMap]:
    """Getter for internal list of StateMaps.

    Returns:
      List[StateMap]: A list of StateMap objects.
    """
    return self.__postprocess("state_maps", self._mili.state_maps())

  def srec_fmt_qty(self) -> int:
    """Getter for State record format quantity.

    Returns:
      int: The number of state record formats.
    """
    return self.__postprocess("srec_fmt_qty", self._mili.srec_fmt_qty())

  def mesh_dimensions(self) -> int:
    """Getter for Mesh Dimensions.

    Returns:
      int: Either 2 or 3 for the number of dimensions.
    """
    return self.__postprocess("mesh_dimensions", self._mili.mesh_dimensions())

  def class_names(self) -> List[str]:
    """Getter for all class names in the database.

    Returns:
      List[str]: List of element class names in the database.
    """
    return self.__postprocess("class_names", self._mili.class_names())

  def int_points_of_state_variable(self, svar_name: str, class_name: str) -> np.ndarray:
    """Get the available integration points for a state variable + class_name.

    Args:
      svar_name (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      np.ndarray: Array of integration points.
    """
    return self.__postprocess("int_points_of_state_variable",
                              self._mili.int_points_of_state_variable(svar_name, class_name))

  def element_sets(self) -> Dict[str,List[int]]:
    """Getter for the element sets.

    Returns:
      Dict[str,List[int]]: Keys are element set names, values are list of integers
    """
    return self.__postprocess("element_sets", self._mili.element_sets())

  def integration_points(self) -> Dict[str,List[int]]:
    """Get the available integration points for each material.

    Returns:
      Dict[str,List[int]]: Keys are material numbers, values are a list of integration points.
    """
    return self.__postprocess("integration_points", self._mili.integration_points())

  def times( self, states : Optional[Union[List[int],int]] = None ) -> np.ndarray:
    """Get the times for each state in the database.

    Args:
      states (Optional[Union[List[int],int]]): If provided, only return the times for the
        specified state numbers.

    Returns:
      np.ndarray: numpy array of times.
    """
    return self.__postprocess("times", self._mili.times(states))

  def queriable_svars(self, vector_only = False, show_ips = False) -> List[str]:
    """Get a list of state variable names that can be queried.

    Args:
      vector_only (bool, default=False): Return on vector state variables.
      show_ips (bool, default=False): Show the available integration points.

    Returns:
      List[str]: A list of state variable names that can be queried.
    """
    return self.__postprocess("queriable_svars", self._mili.queriable_svars(vector_only, show_ips))

  def supported_derived_variables(self) -> List[str]:
    """Get the derived variables that mili-python currently supports.

    Returns:
      List[str]: A List of derived variable names that can be calculated by mili-python.
    """
    return self.__postprocess("supported_derived_variables", self._mili.supported_derived_variables())

  def derived_variables_of_class(self, class_name: str) -> List[str]:
    """Get the derived variables that can be calculated for the specified class.

    Args:
      class_name (str): The element class name.

    Returns:
      List[str]: List of derived variables that can be calculated for the specified class.
    """
    return self.__postprocess("derived_variables_of_class", self._mili.derived_variables_of_class(class_name))

  def classes_of_derived_variable(self, var_name: str) -> List[str]:
    """Get the classes for which a derived variable can be calculated.

    Args:
      var_name (str): The derived variable name:

    Returns:
      List[str]: List of element class names for which var_name can be calculated.
    """
    return self.__postprocess("classes_of_derived_variable", self._mili.classes_of_derived_variable(var_name))

  def labels(self, class_name: Optional[str] = None) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element labels.

    Args:
      class_name (Optional[str]): If provided, only return labels for specifid element class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the labels for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the labels for the specified element class.
    """
    return self.__postprocess("labels", self._mili.labels(class_name))

  def materials(self) -> Dict[str,List[int]]:
    """Get materials dictionary from the database.

    Returns:
      Dict[str,List[int]]: Keys are the material names, Values are a list of material numbers.
    """
    return self.__postprocess("materials", self._mili.materials())

  def material_numbers(self) -> np.ndarray:
    """Get a List of material numbers in the database.

    Returns:
      np.ndarray: A numpy array of the material numbers.
    """
    return self.__postprocess("material_numbers", self._mili.material_numbers())

  def connectivity( self, class_name : Optional[str] = None ) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element connectivity as element LABELS

    Args:
      class_name (str): An element class name. If provided only return connectivty for the specified class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the connectivity for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the connectivity for the specified element class. If the specified element
        class does not exists then None is returned.
    """
    return self.__postprocess("connectivity", self._mili.connectivity(class_name))

  def material_classes(self, mat: Union[str,int]) -> List[str]:
    """Get list of classes of a specified material.

    Args:
      mat (Union[str,int]): A material name or number.

    Returns:
      List[str]: List of element classes associated with the material.
    """
    return self.__postprocess("material_classes", self._mili.material_classes(mat))

  def classes_of_state_variable(self, svar: str) -> List[str]:
    """Get list of element classes for a state variable.

    Args:
      svar (str): The state variable name.

    Returns:
      List[str]: List of element classes the state variable exists for.
    """
    return self.__postprocess("classes_of_state_variable", self._mili.classes_of_state_variable(svar))

  def containing_state_variables_of_class(self, svar: str, class_name: str) -> List[str]:
    """Get List of state variables that contain the specific state variable + class_name

    Args:
      svar (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      List[str]: List of containing state variables.
    """
    return self.__postprocess("containing_state_variables_of_class",
                              self._mili.containing_state_variables_of_class(svar, class_name))

  def components_of_vector_svar(self, svar: str) -> List[str]:
    """Get a list of component state variables of a vector state variable.

    Args:
      svar (str): The name of a vector state variable.

    Returns:
      List[str]: The components of the vector state variable.
    """
    return self.__postprocess("components_of_vector_svar", self._mili.components_of_vector_svar(svar))

  def parts_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of part numbers for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of part numbers for each element of the class name.
    """
    return self.__postprocess("parts_of_class_name", self._mili.parts_of_class_name(class_name))

  def materials_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of materials for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of material numbers for each element of the class name.
    """
    return self.__postprocess("materials_of_class_name", self._mili.materials_of_class_name(class_name))

  def class_labels_of_material( self, mat: Union[str,int], class_name: str ) -> np.ndarray:
    """Convert a material name into labels of the specified class (if any).

    Args:
      mat (Union[str,int]): The material name or number.
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of labels of the specified class name and material.
    """
    return self.__postprocess("class_labels_of_material", self._mili.class_labels_of_material(mat, class_name))

  def all_labels_of_material( self, mat: Union[str,int] ) -> Dict[str,np.ndarray]:
    """Given a specific material. Find all labels with that material and return their values.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      Dict[str,np.ndarray]: Keys are element class names. Values are numpy arrays of element labels.
    """
    return self.__postprocess("all_labels_of_material", self._mili.all_labels_of_material(mat))

  def nodes_of_elems(self, class_sname: str, elem_labels: Union[int,List[int]]) -> Tuple[np.ndarray,np.ndarray]:
    """Find nodes associated with elements by label.

    Args:
      class_sname (str): The element class name.
      elem_labels (List[int]): List of element labels.

    Returns:
      Tuple[np.ndarray,np.ndarray]: (The nodal connectivity, The element labels)
    """
    return self.__postprocess("nodes_of_elems", self._mili.nodes_of_elems(class_sname, elem_labels))

  def nodes_of_material(self, mat: Union[str,int] ) -> np.ndarray:
    """Find nodes associated with a material number.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      numpy.array: A list of all nodes associated with the material number.
    """
    return self.__postprocess("nodes_of_material", self._mili.nodes_of_material(mat))

  def query( self,
             svar_names : Union[List[str],str],
             class_sname : str,
             material : Optional[Union[str,int]] = None,
             labels : Optional[Union[List[int],int]] = None,
             states : Optional[Union[List[int],int]] = None,
             ips : Optional[Union[List[int],int]] = None,
             write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None,
             as_dataframe: bool = False,
             **kwargs ) -> dict:
    '''
    Query the database for svars, returning data for the specified parameters, optionally writing data to the database.
    The parameters passed to query can be identical across a parallel invocation of query, since each individual
      database query object will filter only for the subset of the query it has available.

    : param svar_names : short names for state variables being queried
    : param class_sname : mesh class name being queried
    : param material : optional sname or material number to select labels from
    : param labels : optional labels to query data about, filtered by material if material if material is supplied, default is all
    : param states: optional state numbers from which to query data, default is all
    : param ips : optional for svars with array or vec_array aggregation query just these components, default is all available
    : param write_data : optional the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format,
                   then modify the values desired, then query again with the modified result in this param
    : param as_dataframe : optional. If True the result is returned as a Pandas DataFrame
    '''
    return self.__postprocess("query", self._mili.query(svar_names, class_sname, material, labels, states, ips, write_data, as_dataframe, **kwargs))

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
    return self.__postprocess("append_state", self._mili.append_state(new_state_time, zero_out, limit_states_per_file, limit_bytes_per_file))

  def copy_non_state_data(self, new_base_name: str) -> None:
    """Copy the geometry, states variables, subrecords and parameters from an existing Mili database
       into a new database without any states (Just the A file).

    Args:
      new_base_name (str): The base name of the new database.
    """
    return self.__postprocess("copy_non_state_data", self._mili.copy_non_state_data(new_base_name))