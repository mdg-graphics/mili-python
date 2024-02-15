"""
Copyright (c) 2016-2022, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by
 William Tobin (tobin6@llnl.hov), Kevin Durrenberger (durrenberger1@llnl.gov),
 and Ryan Hathaway (hathaway6@llnl.gov).
 CODE-OCEC-16-056.
 All rights reserved.

 This file is part of Mili. For details, see TODO: <URL describing code
 and how to download source>.

 Please also read this link-- Our Notice and GNU Lesser General
 Public License.

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License (as published by
 the Free Software Foundation) version 2.1 dated February 1999.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 and conditions of the GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software Foundation,
 Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

from __future__ import annotations
from typing import *
from numpy.lib.function_base import iterable
from mili.parallel import LoopWrapper,PoolWrapper,ServerWrapper
import numpy as np

"""
Example Usage:

db = reader.open_database(...)
adjacency = AdjacencyMapping(db)

elems = adjacency.mesh_entities_within_radius("brick", 1, 100, radius=1.0)
"""

class AdjacencyMapping:
  """A wrapper around MiliDatabase or Parallel Wrappers that handles adjacency queries.

  Args:
      obj (Union[MiliDatabase,ParallelWrapper]): The Mili database or ParallelWrapper object.
  """
  def __init__(self, obj):
    self.obj = obj
    self.serial = True
    if isinstance(obj, (LoopWrapper,PoolWrapper,ServerWrapper)):
      self.serial = False

  def __compute_centroid_helper(self, class_name: str, label: int, state: int):
    centroid = self.obj.geometry.compute_centroid(class_name, label, state)
    if not self.serial:
      centroid = np.unique(list(filter(lambda x : x is not None, centroid)), axis=0)
      centroid = centroid[0] if len(centroid) == 1 else None
    if centroid is None:
      raise ValueError((f"Could not calculate centroid for class_name={class_name}, label={label} at state {state}.\n"
                        f"Make sure that the specified class name, label and state all exist."))
    return centroid

  def mesh_entities_within_radius(self,
                                  class_name: str,
                                  label: int,
                                  state: int,
                                  radius: float,
                                  material: Optional[Union[str,int]] = None):
    """Get all mesh entities within a specified radius from a specified mesh entity at a specified state.

    Args:
      class_name (str): The mesh entity class name.
      label (str): The element label.
      state (int): The state number.
      radius (float): The radius within which to search.
      material (Optional[Union[str,int]], default=None): Limit search to a specific material.
    """
    centroid = self.__compute_centroid_helper(class_name, label, state)
    return self.mesh_entities_near_coordinate(centroid, state, radius, material)

  def mesh_entities_near_coordinate(self,
                                    coordinate: List[float],
                                    state: int,
                                    radius: float,
                                    material: Optional[Union[str,int]] = None):
    """Get all mesh entities within a specified radius from a specified coordinate at a given state."""
    nodes_in_radius = self.obj.geometry.nodes_within_radius( coordinate, radius, state )
    if not self.serial:
      nodes_in_radius = np.unique(np.concatenate(nodes_in_radius))

    return self.obj.geometry.elems_of_nodes(nodes_in_radius, material)

  def elems_of_nodes(self, node_labels):
    """Find elements associated with the specified nodes."""
    return self.obj.geometry.elems_of_nodes(node_labels)

  def nearest_node(self, point: List[float], state: int) -> Tuple[int,float]:
    """Get the nearest node to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.
      max_search_iters (Optional[int]): The max number of search iterations.

    Returns:
      Tuple[int,float]: The node label and distance.
    """
    if isinstance(point, list):
      point = np.array(point)

    nearest_node = self.obj.geometry.nearest_node(point, state)
    if not self.serial:
      nearest_node = min(nearest_node, key=lambda x: x[1])
    return nearest_node

  def nearest_element(self, point: List[float], state: int) -> Tuple[str,int,float]:
    """Get the nearest element to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.

    Returns:
      Tuple[str,int,float]: The element class, label, and distance.
    """
    if isinstance(point, list):
      point = np.array(point)

    nearest_per_proc = self.obj.geometry.nearest_element(point, state)
    if not self.serial:
      nearest_per_proc = min(nearest_per_proc, key=lambda x: x[2])
    return nearest_per_proc

class GeometricMeshInfo:
  """A wrapper around MiliDatabase objects that handles Geometric mesh info and queries.

  Args:
      db (MiliDatabase): The Mili database object.
  """
  def __init__(self, db):
    self.db = db

  def nearest_node(self, point: List[float], state: int) -> Tuple[int,float]:
    """Get the nearest node to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.

    Returns:
      Tuple[int,float]: The node label and distance.
    """
    if isinstance(point, list):
      point = np.array(point)
    nodal_coordinates = self.db.query("nodpos", "node", states=[state])
    nodpos_data = nodal_coordinates["nodpos"]["data"][0]
    distances_from_point = np.linalg.norm(nodpos_data - point, axis=1)
    node_index = np.argmin(distances_from_point)
    return nodal_coordinates['nodpos']['layout']['labels'][node_index], distances_from_point[node_index]

  def nearest_element(self, point: List[float], state: int) -> Tuple[str,int,float]:
    """Get the nearest element to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.

    Returns:
      Tuple[str,int,float]: The element class, label, and distance.
    """
    if isinstance(point, list):
      point = np.array(point)

    nodal_coordinates = self.db.query("nodpos", "node", states=[state], output_object_labels=False)
    nodpos_data = nodal_coordinates['nodpos']['data']
    element_connectivity = self.db.connectivity()
    # Calculate centroids and distances for each element class
    minimums = []
    for elem_class, elem_conns in element_connectivity.items():
      conns = elem_conns[:,:-1]
      elem_centroids = np.sum(nodpos_data[0][conns], axis=1) / float(conns.shape[1])
      distances_from_point = np.linalg.norm(elem_centroids - point, axis=1)
      min_elem_index = np.argmin(distances_from_point)
      minimums.append([elem_class, min_elem_index, distances_from_point[min_elem_index]])
    nearest_element = min(minimums, key=lambda x: x[2])
    # covert elem id to label before returning
    nearest_element[1] = self.db.labels(nearest_element[0])[nearest_element[1]]
    return tuple(nearest_element)

  def compute_centroid(self, class_name: str, label: int, state: int):
    """Computes the centroid of a given mesh entity at a given state."""
    labels = self.db.labels(class_name)
    if labels is None or label not in labels:
      return None

    elem_conns = []
    if class_name == "node":
      elem_conns = np.array([label], dtype = np.int32)
    else:
      connectivity = self.db.connectivity(class_name)
      if connectivity is None:
        return None
      elem_idx = np.where( np.isin(labels, label) )[0][0]
      elem_conns = self.db.labels("node")[connectivity[elem_idx][:-1]]

    if len(elem_conns) != 0:
      centroid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
      nodal_coordinates = self.db.query("nodpos", "node", labels=elem_conns, states=[state])
      centroid = np.sum(nodal_coordinates["nodpos"]["data"][0], axis=0)
      centroid = centroid / float(len(elem_conns))
      return centroid

    return None

  def nodes_within_radius(self, center, radius, state):
    """Get all nodes within a radius of a given point at the specified state."""
    if isinstance(center, list):
      center = np.array(center)

    bounding_box_min = center - radius
    bounding_box_max = center + radius

    nodal_coordinates = self.db.query("nodpos", "node", states=[state])
    nodpos_data = nodal_coordinates["nodpos"]["data"][0]

    # Eliminate all nodes not within bounding box
    nodes_in_bounding_box = np.where( np.all(nodpos_data >= bounding_box_min, axis=1) & np.all(nodpos_data <= bounding_box_max, axis=1) )[0]
    # Get nodes that are actually within radius
    distances_from_center = np.linalg.norm(nodpos_data[nodes_in_bounding_box] - center, axis=1)
    node_ids = nodes_in_bounding_box[distances_from_center <= radius]
    nodes_in_radius = nodal_coordinates["nodpos"]["layout"]["labels"][node_ids]

    return nodes_in_radius

  def elems_of_nodes(self, node_labels, material: Optional[Union[int,str]] = None):
    """Find elements associated with the specified nodes."""
    if type(node_labels) is not list:
      if iterable(node_labels):
        node_labels = list(node_labels)
      else:
        node_labels = [ node_labels ]
    nodes = self.db.labels("node")
    if all( label not in nodes for label in node_labels ):
      return {}

    # Connectivity stores nodes by index, not label. So convert labels to indexes
    nlabels = np.where(np.isin(nodes, node_labels))[0]

    elems_of_nodes = {}
    elem_conns = self.db.connectivity()
    for class_name in elem_conns:
      class_conns = elem_conns[class_name]
      if material is not None:
        # Filter out element classes that do not have any elements of this material
        mats_of_class = np.unique(self.db.materials_of_class_name(class_name))
        if material not in mats_of_class:
          continue
        # Filter out elements not of the specified material
        elems_of_mat = np.where(class_conns[:,-1] == material)[0]
        class_conns = elem_conns[class_name][elems_of_mat,:]

      matches, _ = np.where(np.isin(class_conns[:,:-1], nlabels))
      if len(matches) != 0:
        matches = np.unique( matches )
        elems_of_nodes[class_name] = np.unique( self.db.labels(class_name)[matches] )

    return elems_of_nodes
