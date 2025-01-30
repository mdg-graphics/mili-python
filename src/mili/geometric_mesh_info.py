"""Geometric mesh info class used by _MiliInternal.

SPDX-License-Identifier: (MIT)
"""
from typing import *
import numpy as np
from numpy import iterable

class GeometricMeshInfo:
  """A wrapper around _MiliInternal objects that handles Geometric mesh info and queries.

  Args:
      db (_MiliInternal): The _MiliInternal object.
  """
  def __init__(self, db):
    self.db: _MiliInternal = db

  def nearest_node(self, point: List[float], state: int,
                   material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Tuple[int,float]:
    """Get the nearest node to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).

    Returns:
      Tuple[int,float]: The node label and distance.
    """
    if isinstance(point, list):
      point = np.array(point)
    if isinstance(material, (str,int)):
      material = [material]

    node_labels = None
    if material:
      node_labels = np.empty([0], dtype=np.int32)
      for mat in material:
        node_labels = np.concatenate((node_labels, self.db.nodes_of_material(mat)))

    nodal_coordinates = self.db.query("nodpos", "node", labels=node_labels, states=[state])
    if nodal_coordinates["nodpos"]["data"].size == 0:
      return -1, np.finfo(np.float32).max
    nodpos_data = nodal_coordinates["nodpos"]["data"][0]
    distances_from_point = np.linalg.norm(nodpos_data - point, axis=1)
    node_index = np.argmin(distances_from_point)
    return nodal_coordinates['nodpos']['layout']['labels'][node_index], distances_from_point[node_index]

  def nearest_element(self, point: List[float], state: int,
                      material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Tuple[str,int,float]:
    """Get the nearest element to a specified point.

    Args:
      point (List[float]): The coordinates of the point.
      state (int): The state number.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).

    Returns:
      Tuple[str,int,float]: The element class, label, and distance.
    """
    if isinstance(point, list):
      point = np.array(point)
    if isinstance(material, (str,int)):
      material = [material]
    # Need to convert any material names to numbers as that is how they are stored in the connectivity
    if material:
      material_dict = self.db.materials()
      material = [ material_dict.get(mat, mat) for mat in material ]

    nodal_coordinates = self.db.query("nodpos", "node", states=[state], output_object_labels=False)
    nodpos_data = nodal_coordinates['nodpos']['data']
    element_connectivity = self.db.connectivity_ids()
    minimums = []
    for elem_class, elem_conns in element_connectivity.items():
      # Calculate centroids and distances for each element class
      conns = elem_conns[:,:-1]
      elem_centroids = np.sum(nodpos_data[0][conns], axis=1) / float(conns.shape[1])
      distances_from_point = np.linalg.norm(elem_centroids - point, axis=1)
      # If filtering by material, mark any distance of an invalid material as max float32 value
      if material:
        mats = elem_conns[:,-1]
        invalid_mats = np.where( np.isin(mats, material, invert=True))
        distances_from_point[invalid_mats] = np.finfo(np.float32).max
      # Store minumum
      min_elem_index = np.argmin(distances_from_point)
      minimums.append([elem_class, min_elem_index, distances_from_point[min_elem_index]])
    nearest_element = min(minimums, key=lambda x: x[2])
    # covert elem id to label before returning
    nearest_element[1] = self.db.labels(nearest_element[0])[nearest_element[1]]
    return tuple(nearest_element)

  def compute_centroid(self, class_name: str, label: int, state: int):
    """Compute the centroid of a given mesh entity at a given state."""
    labels = self.db.labels(class_name)
    if labels is None or label not in labels:
      return None

    elem_conns = []
    if class_name == "node":
      elem_conns = np.array([label], dtype = np.int32)
    else:
      connectivity = self.db.connectivity_ids(class_name)
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

  def nodes_within_radius(self, center: List[float], radius: float, state: int,
                          material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> np.ndarray:
    """Get all nodes within a radius of a given point at the specified state.

    Args:
      center (List[float]): The center coordinate.
      radius (float): The radius within which to search.
      state (int): The state number.
      material (Optional[Union[Union[str,int],List[Union[str,int]]]], default=None): Limit search to specific material(s).
    """
    if isinstance(center, list):
      center = np.array(center)
    if isinstance(material, (str,int)):
      material = [material]

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

    if material:
      nodes_of_material = np.empty([0], dtype=np.int32)
      for mat in material:
        nodes_of_material = np.concatenate((nodes_of_material, self.db.nodes_of_material(mat)))
      nodes_in_radius = np.intersect1d(nodes_in_radius, nodes_of_material)

    return nodes_in_radius

  def elems_of_nodes(self, node_labels,
                     material: Optional[Union[Union[str,int],List[Union[str,int]]]] = None) -> Dict[str,np.ndarray]:
    """Find elements associated with the specified nodes."""
    if isinstance(material, (str,int)):
      material = [material]
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
    elem_conns = self.db.connectivity_ids()
    for class_name in elem_conns:
      matches, _ = np.where(np.isin(elem_conns[class_name][:,:-1], nlabels))
      if len(matches) != 0:
        matches = np.unique( matches )
        elems_of_nodes[class_name] = np.unique( self.db.labels(class_name)[matches] )

    if material:
      # Filter out elements not of the specified material
      classes_of_material = []
      for mat in material:
        classes_of_material.extend( self.db.material_classes(mat) )
      elems_of_nodes = { k:v for k,v in elems_of_nodes.items() if k in classes_of_material }

      for elem_class in elems_of_nodes:
        elems_of_material = np.empty([0], dtype=np.int32)
        for mat in material:
          elems_of_material = np.concatenate((elems_of_material, self.db.class_labels_of_material(mat, elem_class)))
        where = np.where(np.isin(elems_of_nodes[elem_class], elems_of_material))
        elems_of_nodes[elem_class] = elems_of_nodes[elem_class][where]

    return elems_of_nodes