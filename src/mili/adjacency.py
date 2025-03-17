"""Adjacency Queries for Mili Databases.

SPDX-License-Identifier: (MIT)
"""

from __future__ import annotations
from typing import *
import numpy as np
from numpy.typing import NDArray, ArrayLike

from mili.reductions import dictionary_merge_concat_unique
from mili.milidatabase import MiliDatabase
from mili.mdg_defines import mdg_enum_to_string

if TYPE_CHECKING:
  # NOTE: We only import these when Type checking. These enums should not
  # be used internally, outside of typing and in the derived variable
  # specifications
  from mili.mdg_defines import EntityType

class AdjacencyMapping:
  """A wrapper around MiliDatabase that handles adjacency queries.

  Args:
      mili (MiliDatabase): The Mili database.
  """
  def __init__(self, mili: MiliDatabase):
    self.mili: MiliDatabase = mili
    self.serial = mili.serial

  def compute_centroid(self, entity_type: Union[str,EntityType], label: int, state: int) -> NDArray[np.float32]:
    """Compute the centroid of a given mesh entity at a given state.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).
      labels (int): The element label.
      state (int): The state at which to calculate the centroid

    Returns:
      NDArray[np.float32]. The coordinates of the centroid.
    """
    entity_type_str = mdg_enum_to_string(entity_type)
    centroid = self.mili.geometry.compute_centroid(entity_type_str, label, state)
    if not self.serial:
      centroid = np.unique([c for c in centroid if c is not None], axis=0)  # type: ignore   # mypy error caused by different results for serial vs parallel.
      centroid = centroid[0] if len(centroid) == 1 else None
    if centroid is None:
      raise ValueError((f"Could not calculate centroid for entity_type={entity_type_str}, label={label} at state {state}.\n"
                        f"Make sure that the specified entity type, label and state all exist."))
    return centroid

  def mesh_entities_within_radius(self,
                                  entity_type: Union[str,EntityType],
                                  label: int,
                                  state: int,
                                  radius: float,
                                  material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Dict[str,NDArray[np.int32]]:
    """Get all mesh entities within a specified radius from a specified mesh entity at a specified state.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).
      label (str): The element label.
      state (int): The state number.
      radius (float): The radius within which to search.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).
    """
    entity_type_str = mdg_enum_to_string(entity_type)
    centroid = self.compute_centroid(entity_type_str, label, state)
    return self.mesh_entities_near_coordinate(centroid, state, radius, material)

  def mesh_entities_near_coordinate(self,
                                    coordinate: Union[List[float],NDArray[np.floating]],
                                    state: int,
                                    radius: float,
                                    material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Dict[str,NDArray[np.int32]]:
    """Get all mesh entities within a specified radius from a specified coordinate at a given state.

    Args:
      coordinate (List[float]): The coordinate.
      state (int): The state number.
      radius (float): The radius within which to search.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).
    """
    nodes_in_radius = self.mili.geometry.nodes_within_radius( coordinate, radius, state, material )
    if not self.serial:
      nodes_in_radius = np.unique(np.concatenate(nodes_in_radius))
    elems_in_radius = self.elems_of_nodes(nodes_in_radius, material)
    elems_in_radius["node"] = nodes_in_radius
    return elems_in_radius

  def elems_of_nodes(self, node_labels: ArrayLike,
                     material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Dict[str,NDArray[np.int32]]:
    """Find elements associated with the specified nodes.

    Args:
      node_labels (ArrayLike): List of node labels.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).

    Returns:
      Dict[str,NDArray[np.int32]]: Keys are element entity types. Values are numpy arrays of element labels.
    """
    elems = self.mili.geometry.elems_of_nodes(node_labels, material)
    if not self.serial:
      elems = dictionary_merge_concat_unique(elems)  # type: ignore   # mypy error caused by different results for serial vs parallel.
    return elems

  def nearest_node(self, point: Union[List[float],NDArray[np.floating]], state: int,
                   material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Tuple[int,float]:
    """Get the nearest node to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]] = None): Limit gathered elements to a specific material(s).

    Returns:
      Tuple[int,float]: The node label and distance.
    """
    if isinstance(point, list):
      point = np.array(point)
    if isinstance(material, (str,int)):
      material = [material]

    nearest_node = self.mili.geometry.nearest_node(point, state, material)
    if not self.serial:
      nearest_node = min(nearest_node, key=lambda x: x[1])  # type: ignore   # mypy error caused by different results for serial vs parallel.
    return nearest_node

  def nearest_element(self, point: Union[List[float],NDArray[np.floating]], state: int,
                      material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Tuple[str,int,float]:
    """Get the nearest element to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]] = None): Limit gathered elements to a specific material(s).

    Returns:
      Tuple[str,int,float]: The element entity type, label, and distance.
    """
    if isinstance(point, list):
      point = np.array(point)
    if isinstance(material, (str,int)):
      material = [material]

    nearest_per_proc = self.mili.geometry.nearest_element(point, state, material)
    if not self.serial:
      nearest_per_proc = min(nearest_per_proc, key=lambda x: x[2])  # type: ignore   # mypy error caused by different results for serial vs parallel.
    return nearest_per_proc

  def neighbor_elements(self, entity_type: Union[str,EntityType], label: int,
                        material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None,
                        neighbor_radius: int = 1) -> Dict[str,NDArray[np.int32]]:
    """Gather all neighbor elements to a specified element.

    Args:
      entity_type (Union[str,EntityType]): The entity type ("brick", "node", etc.).
      label (int): The element label.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]] = None): Limit gathered elements to a specific material(s).
      neighbor_radius (int, default=1): The number of neighbors to go out from the specified element.

    Returns:
      Dict[str,NDArray[np.int32]]: Keys are element entity types. Values are numpy arrays of element labels.
    """
    entity_type_str = mdg_enum_to_string(entity_type)
    labels = self.mili.labels(entity_type_str)
    if labels is None:
      raise ValueError(f"No labels found for entity_type '{entity_type_str}'")
    if not self.serial and not self.mili.merge_results:
      labels = np.unique(np.concatenate(list(filter(lambda x : x is not None, labels))))
    if label not in labels:
      raise ValueError(f"The label '{label}' was not found for the entity type '{entity_type_str}'")

    def nodes_of_elems(entity_type: str, element_labels: ArrayLike) -> NDArray[np.int32]:
      """Wrap call to MiliDatabase.nodes_of_elems to handle serial vs parallel."""
      nodes_of_elems = self.mili.nodes_of_elems(entity_type, element_labels)
      if not self.serial and not self.mili.merge_results:
        nodes = np.concatenate([n[0] for n in nodes_of_elems if n[0].size > 0], dtype=np.int32).ravel()
      else:
        nodes = nodes_of_elems[0].ravel()
      return nodes

    def material_classes(material: Union[Union[str,int],List[Union[str,int]]]) -> List[str]:
      """Wrap call to MiliDatabase.material_classes to handle serial vs parallel."""
      if isinstance(material, (str,int)):
        material = [material]
      class_names = []
      for mat in material:
        classes_of_material = self.mili.material_classes(mat)
        if not self.serial and not self.mili.merge_results:
          classes_of_material = np.unique(np.concatenate(classes_of_material))
        class_names.extend( classes_of_material )
      class_names = list(set(class_names))
      return class_names

    def class_labels_of_material(material: Union[Union[str,int],List[Union[str,int]]], entity_type: str) -> NDArray[np.int32]:
      """Wrap call to MiliDatabase.class_labels_of_material to handle serial vs parallel."""
      if isinstance(material, (str,int)):
        material = [material]
      class_labels = np.empty([0], dtype=np.int32)
      for mat in material:
        elems_of_material = self.mili.class_labels_of_material(mat, entity_type)
        if not self.serial and not self.mili.merge_results:
          elems_of_material = np.unique(np.concatenate(elems_of_material))
        class_labels = np.unique(np.concatenate((class_labels, elems_of_material)))
      return class_labels

    elements: Dict[str,NDArray[np.int32]] = {}
    nodes = nodes_of_elems(entity_type_str, label)
    nodes_processed = set()
    nodes_to_process = set(nodes)
    steps_from_elem = 0
    while len(nodes_to_process) > 0 and steps_from_elem < neighbor_radius:
      # Get all elements associated with the nodes we are currently processing
      nodes_processed.update(nodes_to_process)
      elems = self.elems_of_nodes(list(nodes_to_process))
      elements = dictionary_merge_concat_unique([elements, elems])

      nodes_to_process.clear()
      steps_from_elem += 1

      # Get the nodes of the elements we just found and mark them as needing to be processed
      # filter out nodes that have already been processed
      if steps_from_elem < neighbor_radius:
        for elem_class, elem_labels in elems.items():
          nodes = nodes_of_elems(elem_class, elem_labels)
          nodes_to_process.update(nodes)
        nodes_to_process = nodes_to_process.difference(nodes_processed)

    if material:
      # Filter out elements not of the specified material
      classes_of_material = material_classes(material)
      elements = { k:v for k,v in elements.items() if k in classes_of_material }

      for elem_class in elements:
        elems_of_material = class_labels_of_material(material, elem_class)
        where = np.where(np.isin(elements[elem_class], elems_of_material))
        elements[elem_class] = elements[elem_class][where]

    return elements