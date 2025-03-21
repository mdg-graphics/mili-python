"""MiliDatabase module.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpy import iterable
import os

from typing import (Union, List, Tuple, Dict, Literal, Optional, overload,
                    Literal, Type, Any, Callable, TYPE_CHECKING)

import pandas as pd
from enum import Enum

import mili.reductions as reductions
from mili.parallel import ServerWrapper, LoopWrapper
from mili.miliinternal import _MiliInternal, ReturnCode
from mili.datatypes import StateMap, QueryDict, ReturnCode, ReturnCodeTuple, Metadata
from mili.mdg_defines import mdg_enum_to_string
from mili.utils import result_dictionary_to_dataframe
from mili.geometric_mesh_info import GeometricMeshInfo

if TYPE_CHECKING:
  # NOTE: We only import these when Type checking. These enums should not
  # be used internally, outside of typing and in the derived variable
  # specifications
  from mili.mdg_defines import EntityType, StateVariableName


class MiliPythonError(Exception):
  """Mili Python Exception object."""
  pass

def parse_return_codes(return_codes: Union[ReturnCodeTuple, List[ReturnCodeTuple]]) -> None:
  """Processes return codes from MiliDatabase and check for errors or exceptions."""
  if isinstance(return_codes, tuple):
    return_codes = [return_codes]
  if not all([rcode_tup[0] == ReturnCode.OK for rcode_tup in return_codes]):
    # An error has occurred. Need to determine severity.
    num_rc = len(return_codes)
    errors = [rc for rc in return_codes if rc[0] == ReturnCode.ERROR]
    critical = [rc for rc in return_codes if rc[0] == ReturnCode.CRITICAL]
    if len(critical) > 0 or len(errors) == num_rc:
      error_msgs = list(set([f"{rc[0].str_repr()}: {rc[1]}" for rc in return_codes if rc[0] in (ReturnCode.ERROR, ReturnCode.CRITICAL)]))
      raise MiliPythonError(", ".join(error_msgs))


class ResultModifier(Enum):
  """Functions available for modifying results from the query method."""
  CUMMIN = "cummin"
  CUMMAX = "cummax"
  MIN = "min"
  MAX = "max"
  AVERAGE = "average"
  MEDIAN = "median"
  STDDEV = "stddev"


class MiliDatabase:
  """MiliDatabase class that supports querying and other functions on Serial and Parallel Mili databases."""
  def __init__(self,
               dir_name: Union[str,os.PathLike],
               base_files: Union[List[str],List[os.PathLike]],
               parallel_handler: Type[Union[_MiliInternal,LoopWrapper,ServerWrapper]],
               merge_results: bool,
               **kwargs: Any) -> None:
    self.merge_results = merge_results
    dir_names = [dir_name] * len(base_files)
    proc_pargs = [ [dir_name,base_file] for dir_name, base_file in zip(dir_names, base_files)]
    proc_kwargs = [kwargs.copy() for _ in proc_pargs]

    self.serial = len(proc_pargs) == 1

    self.mili: Union[_MiliInternal,LoopWrapper,ServerWrapper]
    if parallel_handler == _MiliInternal:
      self._mili = _MiliInternal( *proc_pargs[0], **proc_kwargs[0] )  # type: ignore  # mypy doesn't like the list/dict unpacking for _MiliInternal args.
    else:
      shared_mem = kwargs.get("shared_mem", False)
      if shared_mem and parallel_handler == ServerWrapper:
        self._mili = parallel_handler( _MiliInternal, proc_pargs, proc_kwargs, use_shared_memory=True )  # type: ignore  # mypy thinks parallel_handler could be _MiliInternal, but it can't be.
      else:
        self._mili = parallel_handler( _MiliInternal, proc_pargs, proc_kwargs )  # type: ignore  # mypy thinks parallel_handler could be _MiliInternal, but it can't be.

  def __postprocess(self, results: Optional[Any], reduce_function: Callable[[Any], Any]) -> Any:
    return_codes = self._mili.returncode()
    self._mili.clear_return_code()
    parse_return_codes(return_codes)
    self.__check_for_exceptions(results)
    if self.serial or not self.merge_results:
      return results
    else:
      return reduce_function(results)

  def __check_for_exceptions(self, results: Any) -> None:
    """Catch any unhandled exception from parallel wrappers."""
    if self.serial and isinstance(results, Exception):
      raise results
    elif iterable(results):
      for res in results:
        if isinstance(res, Exception):
          raise res

  # We define the enter and exit methods to support using MiliDatabase as a context manager
  def __enter__(self) -> MiliDatabase:
    """__enter__ method to support context manager protocol."""
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """__exit__ method to support context manager protocol."""
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  def close(self) -> None:
    """Close the mili database and shutdown any subprocesses being used."""
    if hasattr(self._mili, "close") and callable(getattr(self._mili, "close")):
      # close is a ServerWrapper method that ensures all subprocesses are killed.
      self._mili.close()

  @property
  def geometry(self) -> GeometricMeshInfo:
    """Getter for internal geometry object."""
    return self._mili.geometry

  def reload_state_maps(self) -> bool:
    """Reload the state maps."""
    result: bool
    result = self.__postprocess(results = self._mili.reload_state_maps(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def metadata(self) -> Metadata:
    """Getter for Mili file metadata dictionary.

    Returns:
      Metadata: The meta data dictionary.
    """
    result: Metadata
    result = self.__postprocess(results = self._mili.metadata(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def nodes(self) -> NDArray[np.floating]:
    """Getter for initial nodal coordinates.

    Returns:
      NDArray[np.float32]: A numpy array (num_nodes by mesh_dimensions) containing the initial coordinates for each node.
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
    result: List[StateMap]
    result = self.__postprocess(results = self._mili.state_maps(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def srec_fmt_qty(self) -> int:
    """Getter for State record format quantity.

    Returns:
      int: The number of state record formats.
    """
    result: int
    result = self.__postprocess(results = self._mili.srec_fmt_qty(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def mesh_dimensions(self) -> int:
    """Getter for Mesh Dimensions.

    Returns:
      int: Either 2 or 3 for the number of dimensions.
    """
    result: int
    result = self.__postprocess(results = self._mili.mesh_dimensions(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def class_names(self) -> List[str]:
    """Getter for all class names (entity types) in the database.

    Returns:
      List[str]: List of element class names (entity types) in the database.
    """
    result: List[str]
    result = self.__postprocess(results = self._mili.class_names(),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def int_points_of_state_variable(self, svar_name: Union[str,StateVariableName], entity_type: Union[str,EntityType]) -> NDArray[np.int32]:
    """Get the available integration points for a state variable + entity type.

    Args:
      svar_name (Union[str,StateVariableName]): The state variable name.
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      NDArray[np.int32]: Array of integration points.
    """
    result: NDArray[np.int32]
    entity_type_str = mdg_enum_to_string(entity_type)
    svar_name_str = mdg_enum_to_string(svar_name)
    result = self.__postprocess(results = self._mili.int_points_of_state_variable(svar_name_str, entity_type_str),
                                reduce_function = reductions.list_concatenate_unique)
    return result

  def element_sets(self) -> Dict[str,List[int]]:
    """Getter for the element sets.

    Returns:
      Dict[str,List[int]]: Keys are element set names, values are list of integers
    """
    result: Dict[str,List[int]]
    result = self.__postprocess(results = self._mili.element_sets(),
                                reduce_function = reductions.dictionary_merge_no_concat)
    return result

  def integration_points(self) -> Dict[str,List[int]]:
    """Get the available integration points for each material.

    Returns:
      Dict[str,List[int]]: Keys are material numbers, values are a list of integration points.
    """
    result: Dict[str,List[int]]
    result = self.__postprocess(results = self._mili.integration_points(),
                                reduce_function = reductions.dictionary_merge_no_concat)
    return result

  def times(self, states: Optional[ArrayLike] = None) -> NDArray[np.float64]:
    """Get the times for each state in the database.

    Args:
      states (Optional[ArrayLike]): If provided, only return the times for the
        specified state numbers.

    Returns:
      NDArray[np.float64]: numpy array of times.
    """
    result: NDArray[np.float64]
    result = self.__postprocess(results = self._mili.times(states),
                                reduce_function = reductions.zeroth_entry)
    return result

  def queriable_svars(self, vector_only: bool = False, show_ips: bool = False) -> List[str]:
    """Get a list of state variable names that can be queried.

    Args:
      vector_only (bool, default=False): Return on vector state variables.
      show_ips (bool, default=False): Show the available integration points.

    Returns:
      List[str]: A list of state variable names that can be queried.
    """
    result: List[str]
    result = self.__postprocess(results = self._mili.queriable_svars(vector_only, show_ips),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def supported_derived_variables(self) -> List[str]:
    """Get the derived variables that mili-python currently supports.

    Returns:
      List[str]: A List of derived variable names that can be calculated by mili-python.
    """
    result: List[str]
    result =  self.__postprocess(results = self._mili.supported_derived_variables(),
                                 reduce_function = reductions.zeroth_entry)
    return result

  def derived_variables_of_class(self, entity_type: Union[str,EntityType]) -> List[str]:
    """Get the derived variables that can be calculated for the specified entity type.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      List[str]: List of derived variables that can be calculated for the specified entity type.
    """
    result: List[str]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.derived_variables_of_class(entity_type_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def classes_of_derived_variable(self, var_name: Union[str,StateVariableName]) -> List[str]:
    """Get the classes (entity types) for which a derived variable can be calculated.

    Args:
      var_name (Union[str,StateVariableName]): The derived variable name:

    Returns:
      List[str]: List of element class names (entity types) for which var_name can be calculated.
    """
    result: List[str]
    var_name_str = mdg_enum_to_string(var_name)
    result = self.__postprocess(results = self._mili.classes_of_derived_variable(var_name_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  @overload
  def labels(self, entity_type: Union[str,EntityType]) -> NDArray[np.int32]: ...
  @overload
  def labels(self, entity_type: None = ...) -> Dict[str,NDArray[np.int32]]: ...

  def labels(self, entity_type: Optional[Union[str,EntityType]] = None) -> Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]:
    """Getter for the element labels.

    Args:
      entity_type (Optional[Union[str,EntityType]], default=None): If provided, only return labels for the
        specified entity type ("brick", "node", etc.).

    Returns:
      Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]: If entity_type is None then a dictionary containing
      the labels for each entity type/class is returned. If entity_type is not None, then a numpy array
      is returned containing the labels for the specified entity type/class.
    """
    result: Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]
    entity_type_str: Optional[str] = None
    if entity_type is not None:
      entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.labels(entity_type_str),
                                reduce_function = reductions.reduce_labels)
    return result

  def materials(self) -> Dict[str,List[int]]:
    """Get materials dictionary from the database.

    Returns:
      Dict[str,List[int]]: Keys are the material names, Values are a list of material numbers.
    """
    result: Dict[str,List[int]]
    result = self.__postprocess(results = self._mili.materials(),
                                reduce_function = reductions.zeroth_entry)
    return result

  def material_numbers(self) -> NDArray[np.int32]:
    """Get a List of material numbers in the database.

    Returns:
      NDArray[np.int32]: A numpy array of the material numbers.
    """
    result: NDArray[np.int32] = self.__postprocess(results = self._mili.material_numbers(),
                                                   reduce_function = reductions.list_concatenate_unique)
    return result

  @overload
  def connectivity(self, entity_type: Union[str,EntityType]) -> NDArray[np.int32]: ...
  @overload
  def connectivity(self, entity_type: None = ...) -> Dict[str,NDArray[np.int32]]: ...

  def connectivity(self, entity_type: Optional[Union[str,EntityType]] = None) -> Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]:
    """Getter for the element connectivity as element LABELS.

    Args:
      entity_type (Optional[Union[str,EntityType]], default=None): If provided, only return connectivity for the
        specified entity type ("brick", "node", etc.).

    Returns:
      Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]: If entity_type is None then a dictionary containing
      the connectivity for each entity type/class is returned. If entity_type is not None, then a numpy array
      is returned containing the connectivity for the specified entity type/class.
    """
    result: Union[Dict[str,NDArray[np.int32]],NDArray[np.int32]]
    entity_type_str: Optional[str] = None
    if entity_type is not None:
      entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.connectivity(entity_type_str),
                                reduce_function = reductions.reduce_connectivity)
    return result

  def faces(self, entity_type: Union[str,EntityType], label: int) -> Dict[int,NDArray[np.int32]]:
    """Getter for the faces of an element of a specified entity type/class.

    NOTE: Currently only supports HEX elements.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).
      label (int): The element label.

    Returns:
      Dict[int,NDArray[np.int32]]: A dictionary with the keys 1-6 for each face of the hex element. The value for
      each key is a numpy array of 4 intergers specifying the nodes that make up that face.
    """
    result: Dict[int,NDArray[np.int32]]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.faces(entity_type_str, label),
                                reduce_function = reductions.dictionary_merge_no_concat)
    return result

  def material_classes(self, mat: Union[str,int,np.integer]) -> List[str]:
    """Get list of classes (entity types) of a specified material.

    Args:
      mat (Union[str,int,np.integer]): A material name or number.

    Returns:
      List[str]: List of element classes (entity types) associated with the material.
    """
    result: List[str]
    result = self.__postprocess(results = self._mili.material_classes(mat),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def classes_of_state_variable(self, svar: Union[str,StateVariableName]) -> List[str]:
    """Get list of element classes (entity types) for a state variable.

    Args:
      svar (Union[str,StateVariableName]): The state variable name.

    Returns:
      List[str]: List of element classes (entity types) the state variable exists for.
    """
    result: List[str]
    svar_str = mdg_enum_to_string(svar)
    result = self.__postprocess(results = self._mili.classes_of_state_variable(svar_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def state_variables_of_class(self, entity_type: Union[str,EntityType]) -> List[str]:
    """Get list of primal state variables for a given entity type/class.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      List[str]: A list of primal state variables that can be queried for the specified entity type/class.
    """
    result: List[str]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.state_variables_of_class(entity_type_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def state_variable_titles(self) -> dict[str,str]:
    """Get dictionary of state variable titles for each state variable.

    Returns:
      dict[str,str]: Dictionary where keys are svar names and values are svar titles.
    """
    result: dict[str, str]
    result = self.__postprocess(results = self._mili.state_variable_titles(),
                                reduce_function = reductions.dictionary_merge_no_concat)
    return result

  def containing_state_variables_of_class(self, svar: Union[str,StateVariableName], entity_type: Union[str,EntityType]) -> List[str]:
    """Get List of state variables that contain the specific state variable + entity type.

    Args:
      svar (Union[str,StateVariableName]): The state variable name.
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      List[str]: List of containing state variables.
    """
    result: List[str]
    entity_type_str = mdg_enum_to_string(entity_type)
    svar_str = mdg_enum_to_string(svar)
    result = self.__postprocess(results = self._mili.containing_state_variables_of_class(svar_str, entity_type_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def components_of_vector_svar(self, svar: Union[str,StateVariableName]) -> List[str]:
    """Get a list of component state variables of a vector state variable.

    Args:
      svar (Union[str,StateVariableName]): The name of a vector state variable.

    Returns:
      List[str]: The components of the vector state variable.
    """
    result: List[str]
    svar_str = mdg_enum_to_string(svar)
    result = self.__postprocess(results = self._mili.components_of_vector_svar(svar_str),
                                reduce_function = reductions.list_concatenate_unique_str)
    return result

  def parts_of_class_name(self, entity_type: Union[str,EntityType]) -> NDArray[np.int32]:
    """Get List of part numbers for all elements of a given entity type/class.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      NDArray[np.int32]: array of part numbers for each element of the entity type.
    """
    result: NDArray[np.int32]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.parts_of_class_name(entity_type_str),
                                reduce_function = reductions.list_concatenate)
    return result

  def materials_of_class_name(self, entity_type: Union[str,EntityType]) -> NDArray[np.int32]:
    """Get List of materials for all elements of a given entity type/class.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      NDArray[np.int32]: array of material numbers for each element of the entity type.
    """
    result: NDArray[np.int32]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.materials_of_class_name(entity_type_str),
                                reduce_function = reductions.list_concatenate)
    return result

  def class_labels_of_material(self, mat: Union[str,int,np.integer], entity_type: Union[str,EntityType]) -> NDArray[np.int32]:
    """Convert a material name into labels of the specified entity type/class (if any).

    Args:
      mat (Union[str,int,np.integer]): The material name or number.
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).

    Returns:
      NDArray[np.int32]: array of labels of the specified class name (entity type) and material.
    """
    result: NDArray[np.int32]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.class_labels_of_material(mat, entity_type_str),
                                reduce_function = reductions.list_concatenate)
    return result

  def all_labels_of_material( self, mat: Union[str,int,np.integer] ) -> Dict[str,NDArray[np.int32]]:
    """Given a specific material. Find all labels with that material and return their values.

    Args:
      mat (Union[str,int,np.integer]): The material name or number.

    Returns:
      Dict[str,NDArray[np.int32]]: Keys are element entity types. Values are numpy arrays of element labels.
    """
    result: Dict[str,NDArray[np.int32]]
    result = self.__postprocess(results = self._mili.all_labels_of_material(mat),
                                reduce_function = reductions.dictionary_merge_concat)
    return result

  def nodes_of_elems(self, entity_type: Union[str,EntityType], elem_labels: ArrayLike) -> Tuple[NDArray[np.int32],NDArray[np.int32]]:
    """Find nodes associated with elements by label.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).
      elem_labels (ArrayLike): List of element labels.

    Returns:
      Tuple[NDArray[np.int32],NDArray[np.int32]]: (The nodal connectivity, The element labels)
    """
    result: Tuple[NDArray[np.int32],NDArray[np.int32]]
    entity_type_str = mdg_enum_to_string(entity_type)
    result = self.__postprocess(results = self._mili.nodes_of_elems(entity_type_str, elem_labels),
                                reduce_function = reductions.reduce_nodes_of_elems)
    return result

  def nodes_of_material(self, mat: Union[str,int,np.integer] ) -> NDArray[np.int32]:
    """Find nodes associated with a material number.

    Args:
      mat (Union[str,int,np.integer]): The material name or number.

    Returns:
      NDArray[np.int32]: A list of all nodes associated with the material number.
    """
    result: NDArray[np.int32]
    result = self.__postprocess(results = self._mili.nodes_of_material(mat),
                                reduce_function = reductions.list_concatenate_unique)
    return result

  def __process_query_modifier(self, modifier: ResultModifier,
                               results: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]],
                               as_dataframe: bool ) -> Dict[str,QueryDict]:
    data: NDArray[np.floating]
    merged_results = reductions.combine(results)
    for svar in merged_results:
      merged_results[svar]["modifier"] = modifier.value

    ##### Minimum #####
    if modifier == ResultModifier.MIN:
      for svar in merged_results:
        labels = merged_results[svar]['layout']['labels']
        min_indexes = np.argmin( merged_results[svar]["data"], axis=1, keepdims=True )
        merged_results[svar]["data"] = np.take_along_axis( merged_results[svar]["data"], min_indexes, axis=1 )
        merged_results[svar]["layout"]["labels"] = labels[min_indexes.flatten()]

      if as_dataframe:
        # Special handing for min/max dataframes
        for svar in merged_results:
          states = merged_results[svar]["layout"]["states"]
          labels = merged_results[svar]["layout"]["labels"]
          if len(states) == len(labels):
            data = merged_results[svar]["data"].flatten()
          else:
            data = np.reshape( merged_results[svar]["data"], (len(states),-1))
            labels = np.reshape( labels, (len(states),-1))
          merged_results[svar] = pd.DataFrame( zip(data,labels), index=states, columns=[modifier.value, "label"])  # type: ignore  # mypy errors because it expects result to be QueryDict.

    ##### Maximum #####
    elif modifier == ResultModifier.MAX:
      for svar in merged_results:
        labels = merged_results[svar]['layout']['labels']
        max_indexes = np.argmax( merged_results[svar]["data"], axis=1, keepdims=True )
        merged_results[svar]["data"] = np.take_along_axis( merged_results[svar]["data"], max_indexes, axis=1 )
        merged_results[svar]["layout"]["labels"] = labels[max_indexes.flatten()]

      if as_dataframe:
        # Special handing for min/max dataframes
        for svar in merged_results:
          states = merged_results[svar]["layout"]["states"]
          labels = merged_results[svar]["layout"]["labels"]
          if len(states) == len(labels):
            data = merged_results[svar]["data"].flatten()
          else:
            data = np.reshape( merged_results[svar]["data"], (len(states),-1))
            labels = np.reshape( labels, (len(states),-1))
          merged_results[svar] = pd.DataFrame( zip(data,labels), index=states, columns=[modifier.value, "label"])  # type: ignore  # mypy errors because it expects result to be QueryDict.

    ##### Average #####
    elif modifier == ResultModifier.AVERAGE:
      for svar in merged_results:
        labels = merged_results[svar]['layout']['labels']
        merged_results[svar]["data"] = np.average( merged_results[svar]["data"], axis=1, keepdims=True )

      if as_dataframe:
        # Special handing for average dataframes
        for svar in merged_results:
          data = merged_results[svar]["data"]
          states = merged_results[svar]["layout"]["states"]
          if data.size != len(states):
            merged_results[svar] = pd.DataFrame.from_records(data, index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.
          else:
            merged_results[svar] = pd.DataFrame( data.flatten(), index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.

    ##### Cumulative Min #####
    elif modifier == ResultModifier.CUMMIN:
      for svar in merged_results:
        states = merged_results[svar]["layout"]["states"]
        for i in range(1,len(states)):
          merged_results[svar]["data"][i] = np.minimum( merged_results[svar]["data"][i], merged_results[svar]["data"][i-1])

    ##### Cumulative Max #####
    elif modifier == ResultModifier.CUMMAX:
      for svar in merged_results:
        states = merged_results[svar]["layout"]["states"]
        for i in range(1,len(states)):
          merged_results[svar]["data"][i] = np.maximum( merged_results[svar]["data"][i], merged_results[svar]["data"][i-1])

    ##### Median #####
    elif modifier == ResultModifier.MEDIAN:
      for svar in merged_results:
        merged_results[svar]["data"] = np.median( merged_results[svar]["data"], axis=1, keepdims=True )

      if as_dataframe:
        # Special handing for median dataframes
        for svar in merged_results:
          data = merged_results[svar]["data"].flatten()
          states = merged_results[svar]["layout"]["states"]
          if len(data) != len(states):
            data = merged_results[svar]["data"]
            merged_results[svar] = pd.DataFrame.from_records(data, index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.
          else:
            merged_results[svar] = pd.DataFrame( data, index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.

    ##### Standard Deviation #####
    elif modifier == ResultModifier.STDDEV:
      for svar in merged_results:
        merged_results[svar]["data"] = np.std( merged_results[svar]["data"], axis=1, keepdims=True )

      if as_dataframe:
        # Special handing for stddev dataframes
        for svar in merged_results:
          data = merged_results[svar]["data"].flatten()
          states = merged_results[svar]["layout"]["states"]
          if len(data) != len(states):
            data = merged_results[svar]["data"]
            merged_results[svar] = pd.DataFrame.from_records(data, index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.
          else:
            merged_results[svar] = pd.DataFrame( data, index=states, columns=[modifier.value])  # type: ignore  # mypy errors because it expects result to be QueryDict.

    return merged_results

  @overload
  def query(self, svar_names: Union[List[str],List[StateVariableName],str,StateVariableName],
            entity_type: Union[str,EntityType], material: Optional[Union[str,int]] = None,
            labels: Optional[Union[List[int],int]] = None, states: Optional[Union[List[int],int]] = None,
            ips: Optional[Union[List[int],int]] = None, write_data: Optional[Dict[str,QueryDict]] = None,
            as_dataframe: Literal[False] = False, modifier: Optional[ResultModifier] = None,
            **kwargs: Any) -> Dict[str,QueryDict]: ...

  @overload
  def query(self, svar_names: Union[List[str],List[StateVariableName],str,StateVariableName],
            entity_type: Union[str,EntityType], material: Optional[Union[str,int]] = None,
            labels: Optional[Union[List[int],int]] = None, states: Optional[Union[List[int],int]] = None,
            ips: Optional[Union[List[int],int]] = None, write_data: Optional[Dict[str,QueryDict]] = None,
            as_dataframe: Literal[True] = ..., modifier: Optional[ResultModifier] = None,
            **kwargs: Any) -> Dict[str,pd.DataFrame]: ...

  def query(self,
            svar_names: Union[List[str],List[StateVariableName],str,StateVariableName],
            entity_type: Union[str,EntityType],
            material: Optional[Union[str,int]] = None,
            labels: Optional[Union[List[int],int]] = None,
            states: Optional[Union[List[int],int]] = None,
            ips: Optional[Union[List[int],int]] = None,
            write_data: Optional[Dict[str,QueryDict]] = None,
            as_dataframe: bool = False,
            modifier: Optional[ResultModifier] = None,
            **kwargs: Any) -> Union[Dict[str,pd.DataFrame],Dict[str,QueryDict]]:
    """Query the database for state variables or derived variables, returning data for the specified parameters, optionally writing data to the database.

    Args:
      svar_names (Union[List[str],List[StateVariableName],str,StateVariableName]): The names of the state variables to be queried.
      entity_type (Union[str,EntityType]): The entity type/class being queried (e.g. brick. shell, node).
      material (Optional[Union[str,int]], default=None): Optional material name or number to select labels from.
      labels (Optional[Union[List[int],int]], default=None): Optional labels to query data for, filtered by material
        if material if material is supplied, default is all.
      states (Optional[Union[List[int],int]], default=None): Optional state numbers from which to query data, default is all. This does
        support negative indexing where -1 can be used to get the last state, -2 the second to last, etc.
      ips (Optional[Union[List[int],int]], default=None): Optional integration point to query for vector array state variables, default is all available.
      write_data (Optional[Dict[str,QueryDict]], default=None): Optional the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format,
        then modify the values desired, then query again with the modified result in this param
      as_dataframe (bool, default = False): If True the result is returned as a Pandas DataFrame.
      modifier (Optional[ResultModifier]): Optional modifer to apply to results.
    """
    if write_data and modifier:
      raise ValueError("Result modifiers may not be used when the write_data argument is passed.")

    result: Union[Dict[str,pd.DataFrame],Dict[str,QueryDict]]

    entity_type_str = mdg_enum_to_string(entity_type)
    if isinstance(svar_names, list):
      svar_names = [mdg_enum_to_string(svar) for svar in svar_names]
    else:
      svar_names = mdg_enum_to_string(svar_names)

    result = self.__postprocess(
      results = self._mili.query(svar_names, entity_type_str, material, labels, states, ips, write_data, **kwargs),
      reduce_function = reductions.combine)

    if modifier:
      result = self.__process_query_modifier(modifier, result, as_dataframe)  # type: ignore  # mypy errors because of pd.DataFrame

    if as_dataframe:
      return result_dictionary_to_dataframe(result)  # type: ignore  # mypy errors because of pd.DataFrame
    return result

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
    result: int
    result = self.__postprocess(
      results = self._mili.append_state(new_state_time, zero_out, limit_states_per_file, limit_bytes_per_file),
      reduce_function = reductions.zeroth_entry)
    return result

  def copy_non_state_data(self, new_base_name: str) -> None:
    """Copy non state data from an existing Mili database into a new database without any states (Just the A file).

    Non state data include the geometry, states variables, subrecords and parameters.

    Args:
      new_base_name (str): The base name of the new database.
    """
    self.__postprocess(
      results = self._mili.copy_non_state_data(new_base_name),  # type: ignore  # mypy errors because function returns None.
      reduce_function = reductions.zeroth_entry)