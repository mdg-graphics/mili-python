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
import inspect
from typing import *
from mili.reader import *
import numpy as np

class DerivedExpressions:
  """A wrapper class for MiliDatabase that calculates derived results using the primal
  state variables found in a mili database.

  Args:
      db (MiliDatabase): The mili database object
  
  """
  def __init__(self, db: MiliDatabase):
    self.db = db

    self.__derived_expressions = {
      "disp_x": {
        "primals": ["ux"],  # The primals needed to compute the derived result
        "primals_class": [None],  # The element class of each primal, None = same as requested class_name.
        "compute_function": self.__compute_node_displacement
      },
      "disp_y": {
        "primals": ["uy"],
        "primals_class": [None],
        "compute_function": self.__compute_node_displacement
      },
      "disp_z": {
        "primals": ["uz"],
        "primals_class": [None],
        "compute_function": self.__compute_node_displacement
      },
      "disp_mag": {
        "primals": ["ux", "uy", "uz"],
        "primals_class": [None, None, None],
        "compute_function": self.__compute_node_displacement_magnitude
      },
      "disp_rad_mag_xy": {
        "primals": ["ux", "uy"],
        "primals_class": [None, None],
        "compute_function": self.__compute_node_radial_displacement
      },
      "vel_x": {
        "primals": ["ux"],
        "primals_class": [None],
        "compute_function": self.__compute_node_velocity
      },
      "vel_y": {
        "primals": ["uy"],
        "primals_class": [None],
        "compute_function": self.__compute_node_velocity
      },
      "vel_z": {
        "primals": ["uz"],
        "primals_class": [None],
        "compute_function": self.__compute_node_velocity
      },
      "acc_x": {
        "primals": ["ux"],
        "primals_class": [None],
        "compute_function": self.__compute_node_acceleration
      },
      "acc_y": {
        "primals": ["uy"],
        "primals_class": [None],
        "compute_function": self.__compute_node_acceleration
      },
      "acc_z": {
        "primals": ["uz"],
        "primals_class": [None],
        "compute_function": self.__compute_node_acceleration
      },
      "vol_strain": {
        "primals": ["ex","ey","ez"],
        "primals_class": [None, None, None],
        "compute_function": self.__compute_vol_strain
      },
      "prin_strain1": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain
      },
      "prin_strain2": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain
      },
      "prin_strain3": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain
      },
      "prin_dev_strain1": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain
      },
      "prin_dev_strain2": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain
      },
      "prin_dev_strain3": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain
      },
      # Alternate calculation methods used to check for possibility of errors.
      "prin_strain1_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain_alt
      },
      "prin_strain2_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain_alt
      },
      "prin_strain3_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_strain_alt
      },
      "prin_dev_strain1_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain_alt
      },
      "prin_dev_strain2_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain_alt
      },
      "prin_dev_strain3_alt": {
        "primals": ["ex","ey","ez","exy","eyz","ezx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_dev_principal_strain_alt
      },
      "prin_stress1": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_stress
      },
      "prin_stress2": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_stress
      },
      "prin_stress3": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_stress
      },
      "eff_stress": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_effective_stress
      },
      "pressure": {
        "primals": ['sx', 'sy', 'sz'],
        "primals_class": [None, None, None],
        "compute_function": self.__compute_pressure 
      },
      "prin_dev_stress1": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_dev_stress
      },
      "prin_dev_stress2": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_dev_stress
      },
      "prin_dev_stress3": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_principal_dev_stress
      },
      "max_shear_stress": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_max_shear_stress
      },
      "triaxiality": {
        "primals": ["sx","sy","sz","sxy","syz","szx"],
        "primals_class": [None, None, None, None, None, None],
        "compute_function": self.__compute_triaxiality
      },
      "eps_rate": {
        "primals": ["eps"],
        "primals_class": [None],
        "compute_function": self.__compute_plastic_strain_rate
      },
      # TODO: Add more primals here
    }

  def supported_variables(self):
    """Return a list of derived expressions that are supported.

    NOTE: This does not mean all derived variables can be calculated for a given simulation.
          Only that mili-python can caclulate them if all required inputs exist.
    """
    return list(self.__derived_expressions.keys())
    
  def __init_derived_query_parameters(self,
                              result_name : str,
                              class_name : str,
                              material : Optional[Union[str,int]] = None,
                              labels : Optional[Union[List[int],int]] = None,
                              states : Optional[Union[List[int],int]] = None,
                              ips : Optional[Union[List[int],int]] = None):
    """
    Parse the query parameters and normalize them to the types expected by the rest of the query operation,
        throws TypeException and/or ValueException as appropriate. Does not throw exceptions in cases where
        the result of the argument deviation from the norm would be expected to be encountered when operating in parallel.
    """
    # Check that derived result is supported
    if result_name not in self.__derived_expressions:
      raise ValueError(f"The derived result '{result_name}' is not supported.")

    # Check that class name exists in problem
    if class_name not in self.db.class_names():
      raise ValueError(f"The class name '{class_name}' does not exist.")

    # Check that states are valid
    min_st = 1
    max_st = len(self.db.state_maps())
    if states is None:
      states = np.arange( min_st, max_st+1, dtype = np.int32 )

    if type(states) is int:
      states = np.array( [ states ], dtype = np.int32 )
    elif iterable( states ) and not type( states ) == str:
      states = np.array( states, dtype = np.int32 )
    # Check for any states that are out of bounds
    if np.any( states < min_st ) or np.any( states > max_st ):
        raise ValueError((f"Attempting to query states that do not exist. "
                          f"Minimum state = {min_st}, Maximum state = {max_st}"))

    if not isinstance( states, np.ndarray ):
      raise TypeError( f"'states' must be None, an integer, or a list of integers" )

    if iterable( labels ):
      labels = np.array( labels, dtype = np.int32 )
    elif labels is not None:
      labels = np.array( [ labels ], dtype = np.int32 )

    # Filter labels queried by material if provided, or select all labels of given material if no other labels are given
    if material is not None:
      mat_labels = self.db.class_labels_of_material(material, class_name)
      if labels is not None:
        labels = np.intersect1d( labels, mat_labels )
      else:
        labels = mat_labels

    if labels is None:
      labels = self.db.labels().get( class_name, np.empty([0],np.int32) )
    
    if type(ips) is int:
      ips = [ips]
    if ips is None:
      ips = []

    if type(ips) is not list:
      raise TypeError( 'comp must be an integer or list of integers' )
    
    return result_name, class_name, material, labels, states, ips

  def __parse_derived_result_spec( self, result_name: str, class_name: str, kwargs: dict ):
    """Parse the derived result specification, perform error handling, set up kwargs (if necessary)."""
    required_primals = np.array( self.__derived_expressions[result_name]['primals'] )
    primal_classes = self.__derived_expressions[result_name]['primals_class']
    primal_classes = np.array( [ pclass if pclass is not None else class_name for pclass in primal_classes] )
    compute_function = self.__derived_expressions[result_name]['compute_function']

    # Check that the element classes exists in problem.
    for primal_class in primal_classes:
      if primal_class not in self.db.class_names():
        raise ValueError(f"The primal class name '{primal_class}' does not exist.")

    # Check that all required primals exist for the desired element class.
    primals_found_for_class = np.array([ pclass in self.db.classes_of_state_variable(primal) for primal,pclass in zip(required_primals,primal_classes) ])
    if np.any( primals_found_for_class == False ):
      raise ValueError((f"The required primals do not all exist for the required_classes\n"
                        f"required_primals = {required_primals}\n"
                        f"required_classes = {primal_classes}"))
    
    # Handle keyword arguments
    # Use function reflection to get list of arguments for the derived compute function
    function_signature = inspect.signature(compute_function)
    function_arguments = set(function_signature.parameters.keys())

    # Check that user has not provide any extra arguments.
    for keyword in kwargs:
      if keyword not in function_arguments:
        raise ValueError(f"Unexpected keyword argument '{keyword}' was provided.")

    # Check that any keyword arguments have been provided.
    consistant_arguments = set(['self', 'result_name', 'primal_data', 'query_args'])
    keyword_arguments = function_arguments - consistant_arguments
    if keyword_arguments:
      for keyword in keyword_arguments:
        if keyword not in kwargs:
          # Check for default value
          if function_signature.parameters[keyword].default == inspect.Parameter.empty:
            raise ValueError((f"For the derived result '{result_name}' you must "
                              f"provide the key word argument '{keyword}'"))
    
    return required_primals, primal_classes, kwargs, compute_function
  
  def __generate_elem_node_map(self, elem_node_association):
    """Generate masks for each element that maps to nodal data.

    Args:
      elem_node_association: The output of the nodes_to_elem MiliDatabase function.
    """
    elem_node_map = {}
    nodes_by_elem, elem_order = elem_node_association

    # Get list of unique nodes that need to be queried
    labels_to_query = np.unique( nodes_by_elem.flatten() )

    # Generate mask for each elements nodal results
    for elem_label, associated_nodes in zip( elem_order, nodes_by_elem ):
      elem_node_map[elem_label[0]] = np.where(np.isin(labels_to_query, associated_nodes))[0]

    return labels_to_query, elem_node_map

  def __query_required_primals(self,
                               required_primals: Union[List[str],str],
                               primal_classes: Union[List[str],str],
                               class_name: str,
                               labels : List[int],
                               states : List[int],
                               ips: Union[List[int],int]):
    """Query all the required primals for a derived calculation."""
    query_args = {
      'svar_names': required_primals,
      'class_sname': primal_classes,
      'labels': labels,
      'states': states,
      'ips': ips,
      'elem_node_map': None  # Only exists when calculating element result from nodal values
    }

    # Group required primals by class_name
    unique_classes = list(set(primal_classes))
    primals_grouped_by_class = []
    for cname in unique_classes:
      idxs_of_primals = np.where( np.isin(primal_classes, cname) )
      primals_grouped_by_class.append( (required_primals[idxs_of_primals], cname) )

    # query each group of states variables for each class
    primal_data = {}
    for primal_names, primal_class_name in primals_grouped_by_class:
      # By default use the labels requested for the derived result
      labels_to_query = labels

      # Special case:
      if primal_class_name != class_name:
        # Element class of derived result does not match element class of one or more required primal
        # so we need to map the labels from one super class to another. Most commonly this
        # will be getting all the nodes that make up each of the requested elements.
        if primal_class_name == 'node':
          associated_nodes = self.db.nodes_of_elems( class_name, labels )
          labels_to_query, elem_node_map = self.__generate_elem_node_map( associated_nodes )
          query_args['elem_node_map'] = elem_node_map
        else:
          # Don't know if there are any other cases, skip for now...
          continue

      # Do the query
      primal_data.update( self.db.query(primal_names, primal_class_name, None, labels_to_query, states, ips) )

    return primal_data, query_args
    
  def query(self,
            result_name: str,
            class_name: str,
            material : Optional[Union[str,int]] = None,
            labels : Optional[Union[List[int],int]] = None,
            states : Optional[Union[List[int],int]] = None,
            ips : Optional[Union[List[int],int]] = None,
            **kwargs):
    """General derived result function.
    
    : param result_name : The derived result to compute
    : param class_sname : mesh class name being queried
    : param material : optional sname or material number to select labels from
    : param labels : optional labels to query data about, filtered by material if material if material is supplied, default is all
    : param states: optional state numbers from which to query data, default is all
    : param ips : optional for svars with array or vec_array aggregation query just these components, default is all available
    """
    # Validate incoming data
    result_name, class_name, material, labels, states, ips = self.__init_derived_query_parameters(result_name, class_name, material, labels, states, ips)
    
    # Get specifications for this derived result
    required_primals, primal_classes, kwargs, compute_function = self.__parse_derived_result_spec( result_name, class_name, kwargs )

    # Gather the required primal data
    primal_data, query_args = self.__query_required_primals( required_primals, primal_classes, class_name, labels, states, ips )

    # Call function to derive result
    derived_result = compute_function( result_name, primal_data, query_args, **kwargs )
    
    return derived_result
  
  def __initialize_result_dictionary( self, result_name: str, primal_data: dict, states: List[int] ) -> dict:
    """Initialize the empty dictionary structure to store the derived results in."""
    primal = list(primal_data.keys())[0]
    derived_result = {}
    derived_result[result_name] = { 'source': 'derived' }
    derived_result[result_name]['data'] = np.empty_like( primal_data[primal]['data'])
    derived_result[result_name]['layout'] = primal_data[primal]['layout']
    return derived_result
  
  def __get_nodal_reference_positions( self, components: Union[List[str],str], reference_state: int, labels ):
    """Load the nodal reference positions for the specified state and nodes."""
    if isinstance(components, str):
      components = [components]
    qty_components: int = len(components)

    result_idxs = np.where( np.isin(['ux','uy','uz'], components) )[0]

    # Determine what labels we have
    labels_of_class = self.db.labels().get( 'node', np.empty([0], dtype=np.int32) )
    ordinals = np.where( np.isin( labels_of_class, labels ) )[0]
    qty_nodes = len(ordinals)

    if reference_state == 0:
      # Use initial nodal positions
      initial_positions = self.db.nodes()
      # Get x,y,z primal values for specified elements
      reference_data = (initial_positions[ordinals])[:,result_idxs]
    else:
      # Query specific state from reference data
      data_query = self.db.query( components, 'node', None, labels, reference_state)

      # Create array with the reference nodal coordinates for each node
      reference_data = np.zeros((qty_nodes,qty_components))
      for idx, comp in enumerate(components):
        reference_data[:,idx:idx+1] = data_query[comp]['data'][0]

    return reference_data

  def __compute_node_displacement(self,
                                  result_name: str,
                                  primal_data: dict,
                                  query_args: dict,
                                  reference_state: int = 0):
    """Calculate the derived result 'disp_x', 'disp_y', or 'disp_z'."""
    labels = query_args['labels']
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['disp_x', 'disp_y', 'disp_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( required_primal, reference_state, labels )

    # Compute the displacement
    disp = primal_data[required_primal]['data'] - reference_data
    derived_result[result_name]['data'] = disp

    return derived_result
  
  def __compute_node_displacement_magnitude(self,
                                  result_name: str,
                                  primal_data: dict,
                                  query_args: dict,
                                  reference_state: int = 0):
    """Calculate the derived result 'disp_mag'."""
    labels = query_args['labels']
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( ['ux', 'uy', 'uz'], reference_state, labels )

    # Compute the displacement components
    dx = primal_data['ux']['data'] - reference_data[:,:1]
    dy = primal_data['uy']['data'] - reference_data[:,1:2]
    dz = primal_data['uz']['data'] - reference_data[:,2:]
    
    derived_result[result_name]['data'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return derived_result

  def __compute_node_radial_displacement(self,
                                  result_name: str,
                                  primal_data: dict,
                                  query_args: dict,
                                  reference_state: int = 0):
    """Calculate the derived result 'disp_rad_mag_xy'."""
    labels = query_args['labels']
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( ['ux', 'uy', 'uz'], reference_state, labels )

    # Compute the displacement components
    dx = primal_data['ux']['data'] - reference_data[:,:1]
    dy = primal_data['uy']['data'] - reference_data[:,1:2]

    derived_result[result_name]['data'] = np.sqrt(dx**2 + dy**2)

    return derived_result

  def __compute_node_velocity(self,
                              result_name: str,
                              primal_data: dict,
                              query_args: dict):
    """Calculate the derived result 'vel_x', 'vel_y', or 'vel_z'."""
    labels = query_args['labels']
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['vel_x', 'vel_y', 'vel_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    mask = (states != 1)  # Boolean list used to exclude 1st state (if requested)
    states_prev = states[mask] - 1  # list of previous states

    # Query data for the previous states
    previous_data = self.db.query( required_primal, 'node', None, labels, states_prev)
    previous_data = previous_data[required_primal]['data']

    # Compute the displacements
    disp = primal_data[required_primal]['data'][mask] - previous_data

    # Compute dt
    time_prev = self.db.times()[states_prev-1]  # offset because state 1 = index 0 for these arrays
    time_curr = self.db.times()[states[mask]-1]
    dt_inv = 1.0 / (time_curr - time_prev)
    dt_inv = np.expand_dims(dt_inv, 1)  # add dummy dimension to convert to 2D

    derived_result[result_name]['data'][mask] = disp * dt_inv
    derived_result[result_name]['data'][~mask] = 0  # velocity at 1st state set to zero

    return derived_result
  
  def __compute_node_acceleration(self,
                                  result_name: str,
                                  primal_data: dict,
                                  query_args: dict,
                                  reference_state: int = 0):
    """Calculate the derived result 'acc_x', 'acc_y', or 'acc_z'."""
    # TODO: Finish up
    labels = query_args['labels']
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['acc_x', 'acc_y', 'acc_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    # Create array masks (boolean lists) for operating on parts of the request
    max_st = len(self.db.state_maps())  # last state in the database
    mask_cent = (states != 1) & (states != max_st)  # mask for central diff. calculations
    mask_fwd = (states == 1)  # mask for forward diff. calculations
    mask_back = (states == max_st)  # mask for backward diff. calculations
    
    states_prev = states[mask_cent] - 1  # list of previous states
    states_next = states[mask_cent] + 1  # list of next states
    
    # Query data for the previous states (center diff. only)
    u_p_cent = self.db.query( required_primal, 'node', None, labels, states_prev)
    u_p_cent = u_p_cent[required_primal]['data']

    # Query data for the next states (center diff. only)
    u_n_cent = self.db.query( required_primal, 'node', None, labels, states_next)
    u_n_cent = u_n_cent[required_primal]['data']

    # Get the data for the current states  (center diff. only)
    u_c_cent = primal_data[required_primal]['data'][mask_cent]

    times = self.db.times()

    # Central Difference Calculation
    if any(mask_cent==True):
      delta_t = 0.5 * (times[states_next-1] - times[states_prev-1])  # Calculate average dt
      one_over_tsqr = 1 / (delta_t**2)
      
      accel = (u_n_cent - 2*u_c_cent + u_p_cent) * one_over_tsqr
      derived_result[result_name]['data'][mask_cent] = accel

    # Use backward differnce if the last state is requested
    # Requires 2 previous states, and assumes no duplicate states (i.e. 1 state)
    if any(mask_back==True):
      u_p = self.db.query( required_primal, 'node', None, labels, max_st-1)
      u_pp = self.db.query( required_primal, 'node', None, labels, max_st-2)
      u_p = u_p[required_primal]['data']  # previous state coordinates
      u_pp = u_pp[required_primal]['data']  # previous-previous state coordinates
      u_c = primal_data[required_primal]['data'][mask_back]  # current state coordinates

      delta_t = 0.5 * (times[-1] - times[-3])  # average time step
      one_over_tsqr = 1 / (delta_t**2)
      accel = (u_c - 2*u_p + u_pp) * one_over_tsqr
      
      derived_result[result_name]['data'][mask_back] = accel

    # Use forward differnce if the first state is requested
    # Requires 2 forward states, and assumes no duplicate states (i.e. 1 state)
    if any(mask_fwd==True):
      u_n = self.db.query( required_primal, 'node', None, labels, 2)
      u_nn = self.db.query( required_primal, 'node', None, labels, 3)
      u_n = u_n[required_primal]['data']  # previous state coordinates
      u_nn = u_nn[required_primal]['data']  # previous-previous state coordinates
      u_c = primal_data[required_primal]['data'][mask_fwd]  # current state coordinates

      delta_t = 0.5 * (times[2] - times[0])  # average time step
      one_over_tsqr = 1 / (delta_t**2)
      accel = (u_nn - 2*u_n + u_c) * one_over_tsqr
      
      derived_result[result_name]['data'][mask_fwd] = accel

    return derived_result
  
  def __compute_vol_strain(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'vol_strain'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Perform principal stress computation
    ex: np.ndarray = primal_data['ex']['data']
    ey: np.ndarray = primal_data['ey']['data']
    ez: np.ndarray = primal_data['ez']['data']
    derived_result[result_name]['data'] = ex + ey + ez
            
    return derived_result

  def __compute_principal_strain(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_strain1', 'prin_strain2', 'prin_strain3'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Perform principal stress computation
    ex: np.ndarray = primal_data['ex']['data']
    ey: np.ndarray = primal_data['ey']['data']
    ez: np.ndarray = primal_data['ez']['data']
    exy: np.ndarray = primal_data['exy']['data']
    eyz: np.ndarray = primal_data['eyz']['data']
    ezx: np.ndarray = primal_data['ezx']['data']

    num_states = len(ex[:,0,0])
    num_labels = len(ex[0,:,0])
    num_ipt = len(ex[0,0,:])  # number of integration points
    
    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [ex, exy, ezx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [exy, ey, eyz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [ezx, eyz, ez])  # 3 x num_states x num_labels x 1 array
    strain = np.stack( [x_col, y_col, z_col])  # full 5-d stress array
    
    result_matrix = np.empty_like(ex)
    for i in range(num_states):
      for j in range(num_labels):
        for k in range(num_ipt):
          eigen_values, _ = np.linalg.eig(strain[:, :, i, j, k,])  # all three principal stresses
          if result_name=='prin_strain1':
            result_matrix[i,j,k] = max(eigen_values)
          elif result_name =='prin_strain3':
            result_matrix[i,j,k] = min(eigen_values)
          elif result_name =='prin_strain2':
            eigen_values.sort()  # results aren't sorted by default
            result_matrix[i,j,k] = eigen_values[1]  # get the 2nd value
          else:
            raise ValueError
    
    derived_result[result_name]['data'] = result_matrix
            
    return derived_result

  def __compute_principal_strain_alt(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt'."""
    """
    Uses an alternate calculation method based on the griz algorithm. This was used to help debug
    the code by checking to see if the methods matched.
    """
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    ex: np.ndarray = primal_data['ex']['data']
    ey: np.ndarray = primal_data['ey']['data']
    ez: np.ndarray = primal_data['ez']['data']
    exy: np.ndarray = primal_data['exy']['data']
    eyz: np.ndarray = primal_data['eyz']['data']
    ezx: np.ndarray = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain
    
    # Create 6 x num_states x num_labels x 1 array
    dev_strain = np.stack( [ex-e_hyd, ey-e_hyd, ez-e_hyd, exy, eyz, ezx] )
    # Calc J2
    J2 = -(dev_strain[0,:,:,:]*dev_strain[1,:,:,:] \
          + dev_strain[1,:,:,:]*dev_strain[2,:,:,:] \
          + dev_strain[0,:,:,:]*dev_strain[2,:,:,:]) \
            + dev_strain[3,:,:,:]**2 \
            + dev_strain[4,:,:,:]**2 \
            + dev_strain[5,:,:,:]**2
    
    # Calc J3
    J3 = -dev_strain[0,:,:,:] * dev_strain[1,:,:,:] * dev_strain[2,:,:,:] \
        - 2.0 * dev_strain[3,:,:,:] * dev_strain[4,:,:,:] * dev_strain[5,:,:,:] \
        + dev_strain[0,:,:,:] * dev_strain[4,:,:,:]**2 \
        + dev_strain[1,:,:,:] * dev_strain[5,:,:,:]**2 \
        + dev_strain[2,:,:,:] * dev_strain[3,:,:,:]**2

    # Calculate a limit check value for all J2!=0.
    nz_mask = J2 > 0.0  # mask for non-zero J2 values (J2 is never negative)
    limit_check = np.zeros_like(J2)  # default value is zero (for J2<=0)
    limit_check[nz_mask] = J2[nz_mask]
    
    # Break out elements that pass the limit check (non-zero J2)
    limit_mask = limit_check >= 1e-12  # elements that pass the limit check
    J2_slice = J2[limit_mask]
    J3_slice = J3[limit_mask]

    alpha: np.ndarray = -0.5 * np.sqrt(27/J2_slice) * J3_slice/J2_slice
    
    # Limit alpha to the range -1 to 1
    alpha[alpha<0] = np.maximum(alpha[alpha<0], -1.0)
    alpha[alpha>0] = np.minimum(alpha[alpha>0], 1.0)
        
    # Calculate the load angle (in rad)
    angle: np.ndarray = np.arccos(alpha) * (1/3)
    
    # Calculate an intermediate value
    value: np.ndarray = 2 * np.sqrt(J2_slice * (1/3))
    
    # Modify the load angle for 2nd and 3rd principal stresses
    if result_name=='prin_strain2_alt':
        angle = angle - 2*np.pi * (1/3)
    elif result_name=='prin_strain3_alt':
        angle = angle + 2*np.pi * (1/3)

    # Calculate the requested principal stress (zero if limit_check failed)
    princ_strain: np.ndarray = np.zeros_like(J2)
    princ_strain[limit_mask] = value * np.cos(angle)
    
    princ_strain[limit_mask] = princ_strain[limit_mask] + e_hyd[limit_mask]  # convert to principal strain
    
    derived_result[result_name]['data'] = princ_strain

    return derived_result

  def __compute_dev_principal_strain(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    ex: np.ndarray = primal_data['ex']['data']
    ey: np.ndarray = primal_data['ey']['data']
    ez: np.ndarray = primal_data['ez']['data']
    exy: np.ndarray = primal_data['exy']['data']
    eyz: np.ndarray = primal_data['eyz']['data']
    ezx: np.ndarray = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain

    num_states = len(ex[:,0,0])
    num_labels = len(ex[0,:,0])
    num_ipt = len(ex[0,0,:])
    
    # Create 3 x 3 x num_states x num_labels x num_ipt array
    x_col = np.stack( [ex-e_hyd, exy, ezx])  # 3 x num_states x num_labels x num_ipt array
    y_col = np.stack( [exy, ey-e_hyd, eyz])  # 3 x num_states x num_labels x num_ipt array
    z_col = np.stack( [ezx, eyz, ez-e_hyd])  # 3 x num_states x num_labels x num_ipt array
    dev_strain = np.stack( [x_col, y_col, z_col])  # full 5-d stress array
    
    result_matrix = np.empty_like(ex)
    for i in range(num_states):
      for j in range(num_labels):
        for k in range(num_ipt):
          eigen_values, _ = np.linalg.eig(dev_strain[:, :, i, j, k,])  # all three principal stresses
          if result_name=='prin_dev_strain1':
            result_matrix[i,j,k] = max(eigen_values)
          elif result_name =='prin_dev_strain3':
            result_matrix[i,j,k] = min(eigen_values)
          elif result_name =='prin_dev_strain2':
            eigen_values.sort()  # results aren't sorted by default
            result_matrix[i,j,k] = eigen_values[1]  # get the 2nd value
          else:
            raise ValueError
    
    derived_result[result_name]['data'] = result_matrix
            
    return derived_result

  def __compute_dev_principal_strain_alt(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt'."""
    """
    Uses an alternate calculation method based on the griz algorithm. This was used to help debug
    the code by checking to see if the methods matched.
    Can also calculate the derived result 'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt'.
    """
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    ex: np.ndarray = primal_data['ex']['data']
    ey: np.ndarray = primal_data['ey']['data']
    ez: np.ndarray = primal_data['ez']['data']
    exy: np.ndarray = primal_data['exy']['data']
    eyz: np.ndarray = primal_data['eyz']['data']
    ezx: np.ndarray = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain
    
    # Create 6 x num_states x num_labels x 1 array
    dev_strain = np.stack( [ex-e_hyd, ey-e_hyd, ez-e_hyd, exy, eyz, ezx] )
    
    # Calc J2
    J2 = -(dev_strain[0,:,:,:]*dev_strain[1,:,:,:] \
          + dev_strain[1,:,:,:]*dev_strain[2,:,:,:] \
          + dev_strain[0,:,:,:]*dev_strain[2,:,:,:]) \
            + dev_strain[3,:,:,:]**2 \
            + dev_strain[4,:,:,:]**2 \
            + dev_strain[5,:,:,:]**2
    
    # Calc J3
    J3 = -dev_strain[0,:,:,:] * dev_strain[1,:,:,:] * dev_strain[2,:,:,:] \
        - 2.0 * dev_strain[3,:,:,:] * dev_strain[4,:,:,:] * dev_strain[5,:,:,:] \
        + dev_strain[0,:,:,:] * dev_strain[4,:,:,:]**2 \
        + dev_strain[1,:,:,:] * dev_strain[5,:,:,:]**2 \
        + dev_strain[2,:,:,:] * dev_strain[3,:,:,:]**2

    # Calculate a limit check value for all J2!=0.
    nz_mask = J2 > 0.0  # mask for non-zero J2 values (J2 is never negative)
    limit_check = np.zeros_like(J2)  # default value is zero (for J2<=0)
    limit_check[nz_mask] = J2[nz_mask]
    
    # Break out elements that pass the limit check (non-zero J2)
    limit_mask = limit_check >= 1e-12  # elements that pass the limit check
    J2_slice = J2[limit_mask]
    J3_slice = J3[limit_mask]

    alpha: np.ndarray = -0.5 * np.sqrt(27/J2_slice) * J3_slice/J2_slice
    
    # Limit alpha to the range -1 to 1
    alpha[alpha<0] = np.maximum(alpha[alpha<0], -1.0)
    alpha[alpha>0] = np.minimum(alpha[alpha>0], 1.0)
        
    # Calculate the load angle (in rad)
    angle: np.ndarray = np.arccos(alpha) * (1/3)
    
    # Calculate an intermediate value
    value: np.ndarray = 2 * np.sqrt(J2_slice * (1/3))
    
    # Modify the load angle for 2nd and 3rd principal stresses
    if result_name=='prin_strain2_alt' or result_name=='prin_dev_strain2_alt':
        angle = angle - 2*np.pi * (1/3)
    elif result_name=='prin_strain3_alt' or result_name=='prin_dev_strain3_alt':
        angle = angle + 2*np.pi * (1/3)
        
    # Calculate the requested principal stress (zero if limit_check failed)
    princ_strain: np.ndarray = np.zeros_like(J2)
    princ_strain[limit_mask] = value * np.cos(angle)
    
    if 'dev' not in result_name:
      princ_strain[limit_mask] = princ_strain[limit_mask] + e_hyd  # convert to principal strain
    
    derived_result[result_name]['data'] = princ_strain
            
    return derived_result

  def __compute_principal_stress(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_stress1', 'prin_stress2', 'prin_stress3'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    sx: np.ndarray = primal_data['sx']['data']
    sy: np.ndarray = primal_data['sy']['data']
    sz: np.ndarray = primal_data['sz']['data']
    sxy: np.ndarray = primal_data['sxy']['data']
    syz: np.ndarray = primal_data['syz']['data']
    szx: np.ndarray = primal_data['szx']['data']

    num_labels = len(sx[0,:,0])
    num_states = len(sx[:,0,0])
    num_ips = len(sx[0,0,:])
    
    # Create 3 x 3 x l x m x 1 array
    x_col = np.stack( [sx, sxy, szx])  # 3 x l x m x 1 array
    y_col = np.stack( [sxy, sy, syz])  # 3 x l x m x 1 array
    z_col = np.stack( [szx, syz, sz])  # 3 x l x m x 1 array
    stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array
    
    result_matrix = np.empty_like(sx)
    for i in range(num_states):
      for j in range(num_labels):
        for k in range(num_ips):
          eigen_values, _ = np.linalg.eig(stress[:, :, i, j, k,])  # all three principal stresses
          if result_name=='prin_stress1':
            result_matrix[i,j,k] = max(eigen_values)
          elif result_name =='prin_stress3':
            result_matrix[i,j,k] = min(eigen_values)
          elif result_name =='prin_stress2':
            eigen_values.sort()  # results aren't sorted by default
            result_matrix[i,j,k] = eigen_values[1]  # get the 2nd value
          else:
            raise ValueError
      
      derived_result[result_name]['data'] = result_matrix
            
    return derived_result

  def __compute_effective_stress(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'eff_stress'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)
    dev_stress_x = sx+pressure
    dev_stress_y = sy+pressure
    dev_stress_z = sz+pressure
    
    # Calculate the 2nd deviatoric stress invariant, J2
    J2 = 0.5 * (dev_stress_x**2 + dev_stress_y**2 + dev_stress_z**2) \
              + sxy*sxy + syz*syz + szx*szx

    derived_result[result_name]['data'] = np.sqrt(3*J2)
            
    return derived_result

  def __compute_pressure(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'pressure'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Perform pressure computation
    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    derived_result[result_name]['data'] = (-1/3) * (sx + sy + sz)

    return derived_result

  def __compute_principal_dev_stress(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    num_states = len(sx[:,0,0])
    num_labels = len(sx[0,:,0])
    num_ips = len(sx[0,0,:])
    
    pressure = -(1/3) * (sx + sy + sz)
    
    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [sx+pressure, sxy, szx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [sxy, sy+pressure, syz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [szx, syz, sz+pressure])  # 3 x num_states x num_labels x 1 array
    dev_stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array

    result_matrix = np.empty_like(sx)
    for i in range(num_states):
      for j in range(num_labels):
        for k in range(num_ips):
          eigen_values, _ = np.linalg.eig(dev_stress[:, :, i, j, k,])  # all three principal stresses
          if result_name=='prin_dev_stress1':
            result_matrix[i,j,k] = max(eigen_values)
          elif result_name =='prin_dev_stress3':
            result_matrix[i,j,k] = min(eigen_values)
          elif result_name =='prin_dev_stress2':
            eigen_values.sort()  # results aren't sorted by default
            result_matrix[i,j,k] = eigen_values[1]  # get the 2nd value
          else:
            raise ValueError
      
      derived_result[result_name]['data'] = result_matrix
            
    return derived_result

  def __compute_max_shear_stress(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'max_shear_stress'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    num_states = len(sx[:,0,0])
    num_labels = len(sx[0,:,0])
    num_ips = len(sx[0,0,:])
    
    pressure = -(1/3) * (sx + sy + sz)
    
    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [sx+pressure, sxy, szx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [sxy, sy+pressure, syz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [szx, syz, sz+pressure])  # 3 x num_states x num_labels x 1 array
    stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array

    result_matrix = np.empty_like(sx)
    for i in range(num_states):
      for j in range(num_labels):
        for k in range(num_ips):
          eigen_values, _ = np.linalg.eig(stress[:, :, i, j, k,])  # all three principal stresses
          result_matrix[i,j,k] = 0.5 * (max(eigen_values) - min(eigen_values))
    
    derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_triaxiality(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'triaxiality'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)
    dev_stress_x = sx+pressure
    dev_stress_y = sy+pressure
    dev_stress_z = sz+pressure
    
    # Calculate the 2nd deviatoric stress invariant, J2
    J2 = 0.5 * (dev_stress_x**2 + dev_stress_y**2 + dev_stress_z**2) \
              + sxy*sxy + syz*syz + szx*szx
              
    # Calculate the effective stress
    seff = np.sqrt(3*J2)

    derived_result[result_name]['data'] = -pressure/seff

    return derived_result

  def __compute_plastic_strain_rate(self,
                         result_name: str,
                         primal_data: dict,
                         query_args: dict):
    """Calculate the derived result 'eps_rate'."""
    states = query_args['states']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, states )

    # Get the additional query arguments
    labels = query_args['labels']
    ips = query_args['ips']
    primal_classes = query_args['class_sname'][0]
    
    # Create mask that excludes first and last states (only states compatable with center difference)
    max_st = len(self.db.state_maps())  # last state in the database
    mask_cent = (states != 1) & (states != max_st)
    mask_first = states == 1  # mask that includes only first state (if requested)
    mask_last = states == max_st  # mask that includes only last state (if requested)
    
    # Create lists of states n-1 and n+1
    states_prev = states[mask_cent] - 1
    states_next = states[mask_cent] + 1

    # Query data from states before and after desired states
    eps_prev_q = self.db.query('eps', primal_classes, None, labels, states_prev, ips)
    eps_next_q = self.db.query('eps', primal_classes, None, labels, states_next, ips)
    
    state_times = self.db.times()

    eps_curr = primal_data['eps']['data'][mask_cent]  # requested state eps
    eps_prev = eps_prev_q['eps']['data']
    eps_next = eps_next_q['eps']['data']
    time_prev = state_times[states_prev-1]  # offset because state 1 = index 0 for these arrays
    time_curr = state_times[states[mask_cent]-1]
    time_next = state_times[states_next-1]
    
    dt1_inv = 1.0 / (time_curr - time_prev)
    dt2_inv = 1.0 / (time_next - time_curr)
    
    # Add dummy dimensions so these 1D arrays can be multiplied by the 3D arrays.
    dt1_inv = np.expand_dims(dt1_inv, (1,2))
    dt2_inv = np.expand_dims(dt2_inv, (1,2))
    
    e0 = (eps_curr - eps_prev) * dt1_inv  # backward difference calculation
    e1 = (e0 + (eps_next - eps_curr) * dt2_inv) * 0.5  # average with forward diff. calc.

    derived_result[result_name]['data'][mask_cent] = e1
    derived_result[result_name]['data'][mask_first] = 0  # set first state to zero
    
    # Calculate last state separately using backward difference.  This probably isn't very common.
    if any(mask_last==True):
      states_last_prev = states[mask_last]-1  # 2nd to last state (includes duplicates)
      eps_last_prev_q = self.db.query('eps', primal_classes, None, labels, states_last_prev, ips)
      eps_curr = primal_data['eps']['data'][mask_last]
      eps_prev = eps_last_prev_q['eps']['data']
      time_prev = self.db.times()[states_last_prev-1]  # time arrays are all 1D
      time_curr = self.db.times()[states[mask_last]-1]
      
      dt1_inv = 1.0 / (time_curr - time_prev)
      dt1_inv = np.expand_dims(dt1_inv, (1,2))  # add dummy dimensions to convert to 3D
      e0 = (eps_curr - eps_prev) * dt1_inv  # backward difference calculation

      derived_result[result_name]['data'][mask_last] = e0

    return derived_result

