"""Result Projection Module.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from mili.datatypes import QueryDict, Superclass
if TYPE_CHECKING:
    from mili.milidatabase import MiliDatabase

def __average_adjacent_projection(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result to nodes by averaging results for all adjacent elements in a given query.

    Args:
        milidatabase (MiliDatabase): The MiliDatabase object associated with the query.
        query (QueryDict): The query to project to the associated nodes.
    """
    # Initialize nodal result dictionary
    node_result: QueryDict = QueryDict(
        class_name = 'node',
        modifier = query['modifier'],
        title = query['title'],
        source = query['source'],
        data = np.empty([0], dtype=np.float32),
        layout = {
            'labels': np.empty([0], dtype=np.int32),
            'states': query['layout']['states'],
            'times': query['layout']['times'],
            'components': query['layout']['components']
        }
    )

    # Initialize nodal data and labels array
    elem_labels = query['layout']['labels']
    node_labels, associated_elems = milidatabase.nodes_of_elems(query['class_name'], elem_labels)
    if milidatabase.superclass_from_class_name(query['class_name']) == Superclass.M_BEAM:
        node_labels = node_labels[:,:-1]  # Ignore third node for beams
    unique_node_labels = np.unique(np.concatenate(node_labels))
    qty_nodes = len(unique_node_labels)
    incoming_data_shape = query['data'].shape
    outgoing_data_shape = (incoming_data_shape[0], qty_nodes, incoming_data_shape[2])
    outgoing_data_type = query['data'].dtype
    node_result['data'] = np.zeros(outgoing_data_shape, dtype=outgoing_data_type)
    node_result['layout']['labels'] = unique_node_labels

    # Indexing for each element in associated_elems into query data
    ae_flat = associated_elems.ravel()
    order = np.argsort(elem_labels)
    pos_in_sorted = np.searchsorted(elem_labels[order], ae_flat)
    indices_flat = order[pos_in_sorted]
    elem_indices = indices_flat.reshape(associated_elems.shape)

    # Indexing for each label in node_labels into unique_node_labels array
    order = np.argsort(unique_node_labels)
    sorted_unique = unique_node_labels[order]
    flat_nodes = node_labels.ravel()
    pos_in_sorted = np.searchsorted(sorted_unique, flat_nodes)
    indices_flat = order[pos_in_sorted]
    node_indices = indices_flat.reshape(node_labels.shape)

    # Count the number of elements adjacent to the node
    qty_states = len(query['layout']['states'])
    adj_elem_cnt = np.zeros((qty_states, qty_nodes), dtype=np.float32)
    for n_idxs in node_indices:
        adj_elem_cnt[:,n_idxs] += 1.0

    # Sum results at each node
    for elem_idxs, n_idxs in zip(elem_indices, node_indices):
        for comp_idx in range(incoming_data_shape[2]):
            node_result['data'][:,n_idxs,comp_idx] += query['data'][:,elem_idxs,comp_idx]

    # Divide by number of adjacent elements
    for comp_idx in range(incoming_data_shape[2]):
        node_result['data'][:,:,comp_idx] /= adj_elem_cnt[:,:]

    return node_result

def __volume_weighted_average_projection(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result to nodes using a volume weighted average of the adjacent elements in a given query.

    Args:
        milidatabase (MiliDatabase): The MiliDatabase object associated with the query.
        query (QueryDict): The query to project to the associated nodes.
    """
    # Initialize nodal result dictionary
    node_result: QueryDict = QueryDict(
        class_name = 'node',
        modifier = query['modifier'],
        title = query['title'],
        source = query['source'],
        data = np.empty([0], dtype=np.float32),
        layout = {
            'labels': np.empty([0], dtype=np.int32),
            'states': query['layout']['states'],
            'times': query['layout']['times'],
            'components': query['layout']['components']
        }
    )

    # Initialize nodal data and labels array
    elem_labels = query['layout']['labels']
    node_labels, associated_elems = milidatabase.nodes_of_elems(query['class_name'], elem_labels)
    unique_node_labels = np.unique(np.concatenate(node_labels))
    qty_nodes = len(unique_node_labels)
    incoming_data_shape = query['data'].shape
    outgoing_data_shape = (incoming_data_shape[0], qty_nodes, incoming_data_shape[2])
    outgoing_data_type = query['data'].dtype
    node_result['data'] = np.zeros(outgoing_data_shape, dtype=outgoing_data_type)
    node_result['layout']['labels'] = unique_node_labels

    # Query element volumes
    states = query['layout']['states']
    qty_states = len(states)
    element_volumes = milidatabase.query("element_volume", query['class_name'], labels=elem_labels, states=states)  # type: ignore
    element_volume_data = element_volumes['element_volume']['data']

    # Indexing for each element in associated_elems into element volume data
    ae_flat = associated_elems.ravel()
    order = np.argsort(elem_labels)
    pos_in_sorted = np.searchsorted(elem_labels[order], ae_flat)
    indices_flat = order[pos_in_sorted]
    element_volume_indices = indices_flat.reshape(associated_elems.shape)

    # Indexing for each label in node_labels into unique_node_labels array
    order = np.argsort(unique_node_labels)
    sorted_unique = unique_node_labels[order]
    flat_nodes = node_labels.ravel()
    pos_in_sorted = np.searchsorted(sorted_unique, flat_nodes)
    indices_flat = order[pos_in_sorted]
    node_indices = indices_flat.reshape(node_labels.shape)

    # Sum the volumes at each node
    node_volume_sums = np.zeros((qty_states, qty_nodes), dtype=np.float32)

    for ev_idx, n_idxs in zip(element_volume_indices, node_indices):
        node_volume_sums[:,n_idxs] += element_volume_data[:,ev_idx,0]

    for ev_idx, n_idxs in zip(element_volume_indices, node_indices):
        for comp_idx in range(incoming_data_shape[2]):
            node_result['data'][:,n_idxs,comp_idx] += query['data'][:,ev_idx,comp_idx] * (element_volume_data[:,ev_idx,0] / node_volume_sums[:,n_idxs])

    return node_result

def hex_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Hexes to Nodes."""
    return __volume_weighted_average_projection(milidatabase, query)

def quad_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Quads to Nodes."""
    return __average_adjacent_projection(milidatabase, query)

def tri_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Tris to Nodes."""
    return __average_adjacent_projection(milidatabase, query)

def beam_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Beams to Nodes."""
    return __average_adjacent_projection(milidatabase, query)

def truss_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Trusses to Nodes."""
    return __average_adjacent_projection(milidatabase, query)

def tet_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Tets to Nodes."""
    return __volume_weighted_average_projection(milidatabase, query)

def particle_to_nodal(milidatabase: MiliDatabase, query: QueryDict) -> QueryDict:
    """Project result from Tets to Nodes."""
    return __average_adjacent_projection(milidatabase, query)