"""Append States to MiliDatabase.

SPDX-License-Identifier: (MIT)
"""

import argparse
import importlib
import numpy as np
from mili import reader, datatypes
from typing import *
from numpy.core.fromnumeric import prod

class AppendStatesTool:
  """Tool to append additional states + data to an existing Mili database.

  Args:
    input_dictionary (dict): Dictionary specifying data + states to be appended.
  """
  VALID_OUTPUT_TYPES = ["mili"]
  VALID_OUTPUT_MODES = ["write", "append"]

  def __init__(self, input_dictionary: Dict):
    self.append_states_spec = input_dictionary
    self.database = self.__check_append_states_spec()

  def __get_spec_var(self, key, type_ok, required=False, err_msg="" ):
    # Check key exists
    variable = self.append_states_spec.get( key, None )
    if variable is None:
      if required:
        raise KeyError( f"Key '{key}' is missing from append_states_spec dictionary. {err_msg}" )
    # Check values valid
    if variable is not None:
      variable = self.__check_valid_type(key,variable,type_ok,required,err_msg)
    return variable

  def __check_valid_type( self, key, variable, type_ok, required, err_msg ):
    # Try to accept and check both single value or array of values
    if isinstance(variable,(list,np.ndarray)):
      variable = [ self.__check_valid_type(key,var,type_ok,required,err_msg) for var in variable ]
      if len(variable) == 0:
        variable = None
        if required:
          raise AssertionError( f"Missing values in required array for append_states_spec[{key}].{err_msg}" )
      return variable

    if isinstance(variable,str) and type_ok != str:
      variable = type_ok(variable)
    if isinstance(variable,(np.float32,np.int32,int)) and type_ok == float:
      variable = type_ok(variable)

    # Check type of values specified
    if not isinstance(variable, type_ok):
        raise TypeError( f"Invalid value type specified for append_states_spec[{key}]: {variable}. Expected type: {type_ok}." )

    # If required string, check not empty
    if required:
      if type(variable) == str and variable.strip() == "":
        raise AssertionError( f"Missing required value for append_states_spec[{key}].{err_msg}" )

    return variable

  def __parse_svar_name( self, svar_query_input ):
    """Handle svar names of form vector[component]."""
    comp_start_idx = svar_query_input.find('[')
    if comp_start_idx != -1:
      vector = svar_query_input[:comp_start_idx]
      svar = svar_query_input[comp_start_idx+1:-1]
    else:
      svar = svar_query_input
      vector = None
    return svar, vector

  def __check_append_states_spec(self):
    """Check append_states_spec dictionary has required info and info is compatible with original database."""
    # Check database_basename specified (required)
    in_database_name = self.__get_spec_var("database_basename", str, True)

    # Check output_type specified (required)
    found_valid = False
    output_types = self.__get_spec_var("output_type", str, required=True)
    for output_type in output_types:
      if output_type.lower() not in AppendStatesTool.VALID_OUTPUT_TYPES:
        continue
      found_valid = True
    if not found_valid:
        raise ValueError( f"Invalid value(s) specified for output_type: [', '.join({output_types})]. " \
                          f"Valid output types are {AppendStatesTool.VALID_OUTPUT_TYPES}" )

    # Check output_mode specified (required)
    output_mode = self.__get_spec_var("output_mode", str, required=True)
    if output_mode not in AppendStatesTool.VALID_OUTPUT_MODES:
        raise ValueError( f"Invalid value(s) specified for output_mode: {output_mode}." )

    # Check for output_basename (required if output_mode=="write")
    req = True if output_mode == "write" else False
    out_database_name = self.__get_spec_var("output_basename", str, required=req,
                                            err_msg=" Required if output_mode == 'write'.")

    # Check for state file state count / size limits
    limit_states = self.__get_spec_var("limit_states_per_file", int, required=False)
    if limit_states is not None:
      if limit_states < 1:
        raise ValueError(f"The value of 'limit_states_per_file' must be >= 1. Current value is {limit_states}")
    limit_bytes = self.__get_spec_var("limit_bytes_per_file", int, required=False)
    if limit_bytes is not None:
      if limit_bytes < 1:
        raise ValueError(f"The value of 'limit_bytes_per_file' must be >= 1. Current value is {limit_bytes}")

    # Check states is valid int
    states = self.__get_spec_var("states", int, required=True)
    if not isinstance(states,(int,np.int32)) or states < 0:
        raise ValueError( f"Invalid value(s) specified for states: {states}." )

    # Check time_inc or state_times provided (at least one required)
    state_times = self.__get_spec_var("state_times", float)
    time_inc = self.__get_spec_var("time_inc", float)
    if state_times is None and time_inc is None:
      raise KeyError( f"Could not find state_times or time_inc in append_states_spec. At least one is required." )

    # Try to open original mili database from database_basename from user input
    orig_database = reader.open_database( in_database_name, suppress_parallel=False, merge_results=True )
    self.serial_database = orig_database.serial

    orig_state_times = orig_database.times()
    orig_last_time = orig_state_times[-1]
    # Get state_times from time_inc and last state time from original db
    if state_times == None and time_inc != None:
        # In case time_inc was given as an array
        if isinstance(time_inc,(list,np.ndarray)):
          time_inc = time_inc[0]
        if time_inc <= 0: # kind of redundant bc we check that state_times are ascending and first new timestep is greater than last old
          raise ValueError( f"Invalid value specified for time_inc: {time_inc}." )
        state_times = [ orig_last_time+(time_inc*(i+1)) for i in range(states) ]
    elif state_times != None and time_inc != None:
        print( f"Warning: state_times and time_inc both specified. Time_inc will be ignored." )

    # Don't need to check this if we created state_times
    if states != len(state_times):
        raise ValueError( f"Mismatching value specified for states ({states}) and size of state_times." )

    # Check state_times to append are later than existing state_times
    if output_mode == "append" and len(orig_state_times) > 0:
      if orig_last_time > state_times[0]:
        raise ValueError( f"Invalid value(s) specified for state_times: {state_times}. " \
                          "State times to append must come after original state times." )

    # Check state_times are in ascending order # append_state() checks that each new state_time is greater than last
    # (?) If not in asc order do we want to put into ascending order and not error out?
    ascending = all(state_times[i] < state_times[i+1] for i in range(states-1))
    if not ascending:
      raise ValueError( f"Invalid values specified for state_times: {state_times}. New state times must be in ascending order." )


    # Check state variables
    for class_name, modified_svar_dict in self.append_states_spec["state_variables"].items():
      for svar, input_svar_dict in modified_svar_dict.items():
        svar_class = class_name
        svar_labels = input_svar_dict.get("labels")
        svar_data = input_svar_dict.get("data")
        svar_int_point = input_svar_dict.get("int_point", None)
        if svar_class is None or svar_labels is None or svar_data is None:
          raise ValueError( f"Missing values for state variable {svar}. Required: 'class', 'labels', 'data'." )
        svar_class = svar_class.lower()

        svar_labels = np.array(svar_labels, dtype=np.int32)

        svar, svar_vector_name = self.__parse_svar_name( svar )
        # Look up svar component quantity to allow overwriting arrays
        all_svars = orig_database._mili.state_variables()
        if self.serial_database:
          svar_def = all_svars.get(svar, None)
          if svar_def is None:
            raise ValueError(f"Unable to find {svar} state variable in original database.")
        else:
          for proc_svars in all_svars:
            svar_def = proc_svars.get(svar, None)
            if svar_def is None:
              continue
          if svar_def is None:
            raise ValueError(f"Unable to find {svar} state variable in original database.")

        if svar_vector_name is not None:
          if self.serial_database:
            vector_def = all_svars.get(svar_vector_name, None)
            if vector_def is None:
              raise ValueError(f"Unable to find {svar_vector_name} state variable in original database.")
          else:
            for proc_svars in all_svars:
              vector_def = proc_svars.get(svar_vector_name, None)
              if vector_def is None:
                continue
            if vector_def is None:
              raise ValueError(f"Unable to find {svar_vector_name} state variable in original database.")
        else:
          # Check for special cases of stress_in|mid|out or strain_in|out
          containing_svars = np.unique(orig_database.containing_state_variables_of_class(svar, svar_class))
          if len(containing_svars) > 1:
            raise ValueError(f"The state variable '{svar}' exists in multiple vectors ({containing_svars}) for " \
                             f"the class '{svar_class}'. You will need to specify the svar as vector-name[comp-name] in append_states_spec.")


        svar_atom_qty = svar_def.atom_qty
        if svar_def.agg_type == datatypes.StateVariable.Aggregation.VEC_ARRAY:
          svar_atom_qty = int(svar_atom_qty / prod(svar_def.dims) )

        # Check svar and svar_class exist in original database
        orig_svar_classes = orig_database.classes_of_state_variable(svar)
        if svar_class not in orig_svar_classes:
          raise ValueError( f"Unable to find {svar_class} {svar} data in original database." )

        # Check labels match
        orig_labels = orig_database.labels(svar_class)
        if not np.all(np.isin(svar_labels, orig_labels)):
            raise ValueError( f"Value(s) specified for {svar_class} {svar} labels [{svar_labels}] do not match values in original database." )

        # Check if integration points exist for this state variable + class_name
        available_int_points = orig_database.int_points_of_state_variable(svar, svar_class)

        # If there are multiple integration points
        if len(available_int_points) > 0:
          if svar_int_point is None:
            svar_int_point = available_int_points
          elif isinstance(svar_int_point, int) and svar_int_point in available_int_points:
            svar_int_point = [svar_int_point]
          else:
            raise ValueError(f"Invalid value for {svar_class} {svar}['int_point']: {svar_int_point}. " \
                             f"Availabe integration points: {available_int_points}")
        if len(available_int_points) == 0 and svar_int_point is not None:
          raise ValueError(f"Invalid value for {svar_class} {svar}['int_point']: {svar_int_point}. " \
                           f"No integration points exists for this state variable/class.")
        n_int_points = 1 if svar_int_point is None else len(svar_int_point)

        values_per_element = n_int_points * svar_atom_qty

        # Check size and shape of data (1D or 2D array for single new state or 2D array for multiple where len(svar_data) = # new states, len(svar_data[0]) = # new labels)
        np_svar_data = np.array(svar_data)
        data_dims = np_svar_data.ndim
        data_shape = np_svar_data.shape
        num_labels = len(set(svar_labels))
        expected_data_size = num_labels * values_per_element
        expected_data_all_states = expected_data_size * states
        reshaped = False

        if states == 1:
          # Can be 1D or 2D array
          if not (data_dims == 1 and data_shape[0] == expected_data_size) and not (data_dims == 2 and data_shape[0] == 1 and data_shape[1] == expected_data_size):
            raise ValueError( f"Invalid shape or size of data array specified for {svar_class} {svar}['data']: dims = {data_dims}, shape = {data_shape}. " \
                              f"Number of states = {states}. Expected data shape = ({states}, {expected_data_size})." )
          # Change to 2D array so all data formatted same way no matter number of states
          if data_dims == 1:
              svar_data = [svar_data]
        elif states > 1:
          if not data_dims == 2:
            # Check data is right length if 1D array and transform
            if not isinstance(svar_data[0],(list,np.ndarray)):
              len_data_1d = len(svar_data)
              if len_data_1d != expected_data_size*states:
                raise ValueError( f"Invalid size of array specified for {svar_class} len({svar}['data']) = {len_data_1d}. Expected length of 1D array = {expected_data_all_states}." )
              svar_data = np_svar_data.reshape(states,expected_data_size)
              reshaped = True

            # Check data is not a jagged array
            check_for_jagged = [ len(inner_arr) for inner_arr in svar_data ]
            jagged = any(inner_len != expected_data_size for inner_len in check_for_jagged)
            if jagged:
              raise ValueError( f"Invalid shape or size of data array specified for {svar_class} {svar}['data']. " \
                                f"Lengths of arrays specified for each state: {check_for_jagged}. " \
                                f"Expected lengths of arrays for each state: %s" % ([expected_data_size]*states) )
            elif not jagged and not reshaped: # Should not reach
              raise ValueError( f"Invalid shape or size of data array specified for {svar_class} {svar}['data']." )

          # Check if 2D array but data for all states combined into single array
          if data_shape[0] == 1:
            if data_shape[1] == expected_data_size*states:
              svar_data = np_svar_data.reshape(states,expected_data_size)
              reshaped = True
            else:
              raise ValueError( f"Invalid size of array specified for {svar_class} len({svar}['data'][0]) = {data_shape[1]}. Expected length of data[0] = {expected_data_all_states}." )

          # Get dims and shape again if reshaped (or just don't check data_shape if reshaped == True)
          if reshaped:
            np_svar_data = np.array(svar_data)
            data_dims = np_svar_data.ndim
            data_shape = np_svar_data.shape

          # Check has data for each new state (states > 1)
          if data_shape[0] != states:
            raise ValueError( f"Invalid size of 2D array specified for {svar_class} {svar} (len({svar}['data'] = {data_shape[0]}). " \
                              f"Expected length of {svar}['data'] = number of states ({states})." )

          # Check size of data[state_num] == number of labels (states > 1)
          if data_shape[1] != expected_data_size:
            raise ValueError( f"Invalid 2D array specified for {svar_class} {svar} data (len({svar}['data'][state_num]) = {data_shape[1]}). " \
                              f"Expected length of {svar}['data'][state_num] = number of new labels ({num_labels}) " \
                              f"* number of integration points ({n_int_points}) * number of svar components ({svar_atom_qty})." )


        # Save to format append_states() is expecting
        input_svar_dict["data"] = np.array(svar_data)
        input_svar_dict["int_point"] = svar_int_point

    # Update append_states_spec
    self.append_states_spec["database_basename"] = in_database_name
    self.append_states_spec["output_type"] = output_types
    self.append_states_spec["output_mode"] = output_mode
    if out_database_name != None:
      self.append_states_spec["output_basename"] = out_database_name # (?) Check if it would be better to include key with empty string
    self.append_states_spec["states"] = states
    self.append_states_spec["state_times"] = state_times

    return orig_database

  def write_states(self):
    """Write out the new states to the database."""
    if "mili" in self.append_states_spec['output_type']:
      self.__write_mili_output()

  def __get_new_state_times(self, database):
    """Generate new state times and numbers to add to the database."""
    new_state_times = []
    new_state_numbers = []
    state_times = database.times()
    if "state_times" in self.append_states_spec:
      new_state_times = self.append_states_spec['state_times']
    elif "time_inc" in self.append_states_spec:
      time_inc = self.append_states_spec['time_inc']
      final_state_time = 0.0 if len(state_times) == 0  else state_times[-1]
      new_state_times = self.append_states_spec['states'] * [final_state_time]
      new_state_times = [ time + (time_inc*idx) for idx, time in enumerate(new_state_times, start=1) ]

    final_state_number = len(state_times)
    new_state_numbers = self.append_states_spec['states'] * [final_state_number]
    new_state_numbers = [st_num+idx for idx,st_num in enumerate(new_state_numbers, start=1)]

    return new_state_times, new_state_numbers

  def __write_mili_output(self):
    """Write out new states when output_type is mili."""
    output_database = None
    if self.append_states_spec['output_mode'] == "append":
      output_database = self.database
    elif self.append_states_spec['output_mode'] == "write":
      self.database.copy_non_state_data( self.append_states_spec['output_basename'] )
      output_database = reader.open_database( self.append_states_spec['output_basename'], experimental=True, merge_results=True )

    new_state_times, new_state_numbers = self.__get_new_state_times(output_database)

    state_limit = self.append_states_spec.get("limit_states_per_file", None)
    byte_limit = self.append_states_spec.get("limit_bytes_per_file", None)

    for new_time in new_state_times:
      output_database.append_state( new_time, limit_states_per_file=state_limit, limit_bytes_per_file=byte_limit)

    for class_name, modified_svars in self.append_states_spec['state_variables'].items():
      for svar_name, spec in modified_svars.items():
        labels = spec['labels']
        int_point = spec.get('int_point', None)
        new_data = spec['data']
        svar_result = output_database.query(svar_name, class_name, labels=labels, states=new_state_numbers, ips=int_point)
        current_shape = svar_result[svar_name]['data'].shape
        new_data = np.reshape(new_data, current_shape)
        svar_result[svar_name]['data'] = new_data
        svar_result[svar_name]['layout']['labels'] = labels
        output_database.query(svar_name, class_name, labels=labels, states=new_state_numbers, ips=int_point, write_data=svar_result)