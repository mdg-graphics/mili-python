"""Internal Mili file object.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
import os
import re
import io

from collections import defaultdict
from numpy import iterable
from typing import *

from mili.datatypes import *
from mili.afileIO import *
from mili.parallel import *
from mili.derived import DerivedExpressions
from mili.adjacency import GeometricMeshInfo
from mili.reductions import *

def np_empty(dtype):
  """Create empty array of specified data type."""
  return np.empty([0],dtype=dtype)


class ReturnCode(Enum):
  """Return code enum for _MiliInternal."""
  OK = 0
  ERROR = 1
  CRITICAL = 2

  def str_repr(self) -> str:
    """Get string representation of ReturnCode."""
    return ["Success", "Error", "Critical"][self.value]


class _MiliInternal:
  """The internal Mili object representing a single A-file and its associated state files.

  Args:
    dir_name (os.PathLike) : an os.PathLike denoting the file sytem directory to search for all state files described by the A-file
    base_filename (os.PathLike) : the base of the files for this database (usually for one process):
                                    - {base_filename}A is the A-file
                                    - {base_filename}T is the T-file in mili format >= v3
                                    - {base_filename}### are state file(s) containing state data for this database
  """
  def __init__( self, dir_name : Union[str,bytes,os.PathLike], base_filename : Union[str,bytes,os.PathLike], **kwargs ):
    self.__return_code = (ReturnCode.OK, "")
    self.__state_files = []
    self.__smaps = []

    self.__svars : Mapping[ str, StateVariable ] = {}
    self.__srec_fmt_qty = 0
    self.__srecs : List[ Subrecord ] = []

    # indexing / index discovery
    self.__labels = {}
    self.__mats = defaultdict(list)
    self.__int_points = defaultdict( lambda : defaultdict( list ) )

    # mostly used for input validation
    self.__class_to_sclass = {}

    # allow query by material
    self.__elems_of_mat = defaultdict(lambda : defaultdict( lambda : np_empty(np.int32)) )  # map from material number to dict of class to elems
    self.__elems_of_part = defaultdict(lambda : defaultdict( lambda : np_empty(np.int32)) )  # map from part number to dict of class to elems

    # mesh data
    self.__MO_class_data = {}
    self.__conns_ids = {}
    self.__conns_labels = {}
    self.__nodes = None
    self.__mesh_dim = 0

    self.__params = defaultdict( lambda : defaultdict( list ) )

    self.__afile = AFile()
    self.__base_filename = base_filename
    self.__dir_name = dir_name
    log_validator = kwargs.get("log_validator", False)
    parse_success = AFileParser(log_validator=log_validator).parse( self.__afile, os.path.join( dir_name, base_filename ) )
    if not parse_success:
      raise MiliAParseError("AFile parsing validation failure!")

    self.__sfile_suf_len = self.__afile.sfile_suffix_length
    sfile_re = re.compile(re.escape(base_filename) + f"(\d{{{self.__sfile_suf_len},}})$")

    # load and sort the state files numerically (NOT alphabetically)
    self.__state_files = list(map(sfile_re.match,os.listdir(dir_name)))
    self.__state_files = list(filter(lambda match: match != None,self.__state_files))
    self.__state_files = list(sorted(self.__state_files, key = lambda match: int(match.group(1))))
    self.__state_files = [ os.path.join(dir_name,match.group(0)) for match in self.__state_files ]

    # pull values out of the afile and compute utility values/arrays to speed up query operations
    self.__smaps = self.__afile.smaps
    self.__mesh_dim = self.__afile.dirs[DirectoryDecl.Type.MILI_PARAM].get( "mesh dimensions", 3 )
    for sname, param in self.__afile.dirs[DirectoryDecl.Type.TI_PARAM].items():
      if sname.startswith("Node Labels"):
        self.__labels[ "node" ] = param
      elif sname.startswith("Element Labels") and not "ElemIds" in sname:
        class_sname = re.search( r'Sname-(\w*)', sname ).group(1)
        if class_sname not in self.__labels.keys():
          self.__labels[ class_sname ] = np_empty( np.int32 )
        self.__labels[ class_sname ] = np.concatenate( ( self.__labels[ class_sname ], param ) )
      elif sname.startswith("MAT_NAME"):
        mat_num = int( re.match(r"MAT_NAME_(\d+)", sname).group(1) )
        self.__mats[ param ].append( mat_num )
        self.__params[f"MAT_NAME_{mat_num}"] = param
      elif sname.startswith('IntLabel_es_'):
        ip_name = sname[ sname.find('es_'): ]
        self.__int_points[ip_name] = param.tolist()
      elif sname.startswith("SetRGB"):
        self.__params[sname] = param
      elif sname == "particles_on":
        self.__params[sname] = param

    # Add all mili parameters and application parameters to parameters dictionary
    for sname, param in self.__afile.dirs[DirectoryDecl.Type.MILI_PARAM].items():
      self.__params[sname] = param
    for sname, param in self.__afile.dirs[DirectoryDecl.Type.APPLICATION_PARAM].items():
      self.__params[sname] = param

    # setup svar.svars for easier VECTOR svar traversal
    self.__svars = self.__afile.dirs[DirectoryDecl.Type.STATE_VAR_DICT]
    def addComps( svar ):
      """Small recursive function to populate svar.svars[] all the way down."""
      if svar.agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ] and len(svar.svars) == 0:
        for comp_name in svar.comp_names:
          comp = self.__svars[comp_name]
          svar.svars.append( comp )
          addComps( comp )

    def addContaining( svar ):
      """Small recursive function to populate svar.containing_svars[] all the way down."""
      if svar.agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
        for comp_name in svar.comp_names:
          comp = self.__svars[comp_name]
          comp.containing_svar_names.append( svar.name )
          addContaining( comp )

    def addIntPoints( es_name, svar_comp_names ):
      """Small recursive function to populate self.__int_points all the way down."""
      for svar_comp_name in svar_comp_names:
        if not svar_comp_name in self.__int_points.keys():
          self.__int_points[svar_comp_name] = {}
        self.__int_points[svar_comp_name][es_name] = self.__int_points[es_name[:-1]]
        comp = self.__svars[svar_comp_name]
        if comp.agg_type in [StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
          addIntPoints( es_name, comp.comp_names )

    # add component svar objects to vec/vec_array aggregate svars instead of just names
    for svar in self.__svars.values():
      addComps( svar )
      addContaining( svar )

      if svar.name[:-1] in self.__int_points.keys():
        svar_comp_names = svar.comp_names
        stress = self.__svars.get('stress')
        stress_comps = [ svar.name for svar in stress.svars ] if stress is not None else []
        strain = self.__svars.get('strain')
        strain_comps = [ svar.name for svar in strain.svars ] if strain is not None else []

        if len( list(set(svar_comp_names) & set(stress_comps)) ) == 6:
          if 'stress' not in self.__int_points.keys():
            self.__int_points['stress'] = {}
          self.__int_points['stress'][svar.name] = self.__int_points[svar.name[:-1]]

        if len( list(set(svar_comp_names) & set(strain_comps)) ) == 6:
          if 'strain' not in self.__int_points.keys():
            self.__int_points['strain'] = {}
            self.__int_points['strain'][svar.name] = self.__int_points[svar.name[:-1]]

        addIntPoints( svar.name, svar_comp_names )
        # This allows us to query by element set name
        self.__int_points[svar.name][svar.name] = self.__int_points[svar.name[:-1]]

    # the class def info is stored in the DirectoryDecl, not in a parsed directory data structure
    for class_def in self.__afile.dir_decls[DirectoryDecl.Type.CLASS_DEF]:
      self.__class_to_sclass[ class_def.strings[0] ] = Superclass( class_def.modifier_idx2 )

    for sname, class_ident_data in self.__afile.dirs[DirectoryDecl.Type.CLASS_IDENTS].items():
      mesh_ident = class_ident_data['idents']
      if sname not in self.__labels.keys():
        self.__labels[ sname ] = np_empty( np.int32 )
      self.__labels[ sname ] = np.append( self.__labels[sname], np.arange( mesh_ident[0], mesh_ident[1] + 1, dtype = np.int32 ) )

    self.__nodes = np.empty( [0, self.__mesh_dim], np.float32 )
    for _, node_data in self.__afile.dirs[DirectoryDecl.Type.NODES].items():
      self.__nodes = np.concatenate( (self.__nodes, node_data['nodes']) )

    for sname, mo_class_data in self.__afile.dirs[DirectoryDecl.Type.CLASS_DEF].items():
      self.__MO_class_data[sname] = mo_class_data
      self.__class_to_sclass[sname] = mo_class_data.sclass

    for sname, elem_conn_data in self.__afile.dirs[DirectoryDecl.Type.ELEM_CONNS].items():
      elem_conn = elem_conn_data.get('conns', np_empty(np.int32))
      # Connectivity IDS
      if sname not in self.__conns_ids.keys():
        self.__conns_ids[ sname ] = np.ndarray( (0,elem_conn.shape[1]-1), dtype = np.int32 )
      self.__conns_ids[ sname ] = elem_conn[:,:-1].copy()  # remove the part id, keep material number
      self.__conns_ids[ sname ][:,:-1] -= 1  # Account for fortran indexing (Except for material number)
      # Connectivity LABELS
      if sname not in self.__conns_labels.keys():
        self.__conns_labels[ sname ] = np.ndarray( (0,elem_conn.shape[1]-1), dtype = np.int32 )
      self.__conns_labels[ sname ] = elem_conn[:,:-1].copy()  # remove the part id, keep material number
      self.__conns_labels[ sname ][:,:-1] = self.__labels[ "node" ][ self.__conns_ids[sname][:][:,:-1] ] # Copy node labels

      mats = np.unique( elem_conn[:,-2] )
      for mat in mats:
        # this assumes the load order of the conn blocks is identical to their mo_ids... which is probably the case but also is there not a way to make that explicit?
        self.__elems_of_mat[ mat ][ sname ] = np.concatenate( ( self.__elems_of_mat[ mat ][ sname ], np.nonzero( elem_conn[:,-2] == mat )[0] ) )

      parts = np.unique( elem_conn[:,-1] )
      for part in parts:
        # this assumes the load order of the conn blocks is identical to their mo_ids... which is probably the case but also is there not a way to make that explicit?
        self.__elems_of_part[ part ][ sname ] = np.concatenate( (self.__elems_of_part[ part ][ sname ], np.nonzero( elem_conn[:,-1] == part)[0] ) )

    offset = 0
    self.__srec_fmt_qty = 1
    for sname, srec in self.__afile.dirs[DirectoryDecl.Type.SREC_DATA].get('subrecords',{}).items():
      srec : Subrecord
      # add all svars to srec
      srec.svars = [ self.__afile.dirs[DirectoryDecl.Type.STATE_VAR_DICT][svar_name] for svar_name in srec.svar_names ]

      # For each svar in this subrecord add all nested svars to subrecord and initialize layout of component svars
      srec.svar_comp_layout = []
      for svar in srec.svars:
        # nested svars
        nested_svars = svar.recursive_svars
        for nested_svar in nested_svars:
          nested_svar.srecs.append(srec)
        # component svars
        srec.svar_comp_layout += svar.comp_layout

      srec.svar_atom_lengths = np.fromiter( (svar.atom_qty for svar in srec.svars), dtype = np.int64 )
      srec.svar_atom_offsets = np.zeros( srec.svar_atom_lengths.size, dtype = np.int64 )
      srec.svar_atom_offsets[1:] = srec.svar_atom_lengths.cumsum()[:-1]
      srec.atoms_per_label = sum( srec.svar_atom_lengths )

      srec.svar_comp_offsets = []
      for svar_comp in srec.svar_comp_layout:
        svar_length = [ self.__svars[svar].list_size if self.__svars[svar].agg_type != StateVariable.Aggregation.SCALAR.value else 1 for svar in svar_comp ]
        svar_length = np.insert( svar_length, 0, 0 )
        srec.svar_comp_offsets.append( np.cumsum( svar_length ) )

      # the ordinal blocks are set during the parse, these are only precomputed to make the query easier so we don't deal with them during the parse, only the query setup
      srec.ordinal_block_counts = np.concatenate( ( [0], np.diff( srec.ordinal_blocks.reshape(-1,2), axis=1).flatten( ) ) )
      srec.svar_ordinal_offsets = np.cumsum( np.fromiter( ( svar.atom_qty * srec.total_ordinal_count for svar in srec.svars ), dtype = np.int64 ) )
      srec.svar_byte_offsets = np.cumsum( np.fromiter( ( svar.atom_qty * srec.total_ordinal_count * svar.data_type.byte_size() for svar in srec.svars ), dtype = np.int64 ) )
      # Insert 0 a front of arrays
      srec.svar_ordinal_offsets = np.insert( srec.svar_ordinal_offsets, 0, 0, axis=0 )
      srec.svar_byte_offsets = np.insert( srec.svar_byte_offsets, 0, 0, axis=0 )

      srec.byte_size = srec.total_ordinal_count * sum( svar.data_type.byte_size() * svar.atom_qty for svar in srec.svars )
      srec.state_byte_offset = offset
      offset += srec.byte_size

      self.__srecs.append( srec )

    # Handle case where mesh object class exists but no idents provided just number 1-N
    for class_sname in self.__MO_class_data:
      if class_sname not in self.__labels:
        self.__MO_class_data[class_sname].elem_qty = self.__conns_ids.get(class_sname,np_empty(np.int32)).shape[0]
        self.__MO_class_data[class_sname].idents_exist = False
        self.__labels[class_sname] = np.array( np.arange(1, self.__MO_class_data[class_sname].elem_qty + 1 ))
      else:
        self.__MO_class_data[class_sname].elem_qty = self.__labels[class_sname].size

    # Create instance of wrapper class to handle derived queries
    self.__derived = DerivedExpressions( self )
    # Create instance of wrapper class to handle Geometric Mesh information/queries
    self.__geometry = GeometricMeshInfo( self )

  @property
  def geometry(self) -> GeometricMeshInfo:
    return self.__geometry

  def clear_return_code(self):
    """Reset the value of the internal return code."""
    self.__return_code = (ReturnCode.OK, "")

  def returncode(self) -> Tuple[ReturnCode,str]:
    """Getter for the returncode.

    Returns:
      Tuple[ReturnCode,str]: The return code object and status string.
    """
    return self.__return_code

  def reload_state_maps(self) -> None:
    """Reload the state maps."""
    afile = AFile()
    parse_success = AFileParser().parse( afile, os.path.join( self.__dir_name, self.__base_filename ) )
    if not parse_success:
      self.__return_code = (ReturnCode.ERROR, "AFile parsing validation failure!")
    else:
      self.__smaps = afile.smaps

  def nodes(self) -> np.ndarray:
    """Getter for initial nodal coordinates.

    Returns:
      np.ndarray: A numpy array (num_nodes by mesh_dimensions) containing the initial coordinates for each node.
    """
    return self.__nodes

  def state_maps(self) -> List[StateMap]:
    """Getter for internal list of StateMaps.

    Returns:
      List[StateMap]: A list of StateMap objects.
    """
    return self.__smaps

  def subrecords(self) -> List[Subrecord]:
    """Getter for internal list of Subrecords.

    Returns:
      List[Subrecord]: A list of Subrecord objects.
    """
    return self.__srecs

  def parameters(self) -> Dict[str,Any]:
    """Getter for mili parameters dictionary.

    Returns:
      Dict[str,Any]: Keys are parameter names, Values are parameter values
    """
    return self.__params

  def parameter(self, name: str, default: Optional[Any] = None) -> Any:
    """Getter for single parameter value.

    Args:
      name (str): The parameter to retrieve.
      default (Optional[Any]): value to return if parameter is not found.

    Returns:
      Any: The value of the parameter
    """
    return self.__params.get(name, default)

  def srec_fmt_qty(self) -> int:
    """Getter for State record format quantity.

    Returns:
      int: The number of state record formats.
    """
    return self.__srec_fmt_qty

  def mesh_dimensions(self) -> int:
    """Getter for Mesh Dimensions.

    Returns:
      int: Either 2 or 3 for the number of dimensions.
    """
    return self.__mesh_dim

  def class_names(self) -> List[str]:
    """Getter for all class names in the database.

    Returns:
      List[str]: List of element class names in the database.
    """
    return list(self.__MO_class_data.keys())

  def mesh_object_classes(self) -> Dict[str,MeshObjectClass]:
    """Getter for internal Mesh Object class data.

    Returns:
      Dict[str,MeshObjectClass]: Keys are element class names. Values are MeshObjectClass objects.
    """
    return self.__MO_class_data

  def int_points(self) -> Dict[str,Union[List[int],Dict[str,List[int]]]]:
    """Getter for internal integration point dictionary."""
    return self.__int_points

  def int_points_of_state_variable(self, svar_name: str, class_name: str) -> np.ndarray:
    """Get the available integration points for a state variable + class_name.

    Args:
      svar_name (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      np.ndarray: Array of integration points.
    """
    int_points = []
    if svar_name in self.__int_points:
      for es_name in self.__int_points[svar_name]:
        if class_name in self.classes_of_state_variable(es_name):
          int_points = np.array(self.__int_points[svar_name][es_name][:-1])
    return int_points

  def element_sets(self) -> Dict[str,List[int]]:
    """Getter for the element sets.

    Returns:
      Dict[str,List[int]]: Keys are element set names, values are list of integers
    """
    return {k:v for k,v in self.__int_points.items() if k.startswith("es_") and k[-1].isdigit()}

  def integration_points(self) -> Dict[str,List[int]]:
    """Get the available integration points for each material.

    Returns:
      Dict[str,List[int]]: Keys are material numbers, values are a list of integration points.
    """
    elem_sets = self.element_sets()
    mat_int_points = {}
    for eset, int_points in elem_sets.items():
      mat = eset[-1:]
      mat_int_points[mat] = int_points[:-1]
    return mat_int_points

  def times( self, states : Optional[Union[List[int],int]] = None ) -> np.ndarray:
    """Get the times for each state in the database.

    Args:
      states (Optional[Union[List[int],int]]): If provided, only return the times for the
        specified state numbers.

    Returns:
      np.ndarray: numpy array of times.
    """
    if isinstance(states, (int, np.integer)):
      states = np.array( [ states ], dtype = np.int32 )
    elif iterable( states ) and not type( states ) == str:
      states = np.array( states, dtype = np.int32 )
    if not isinstance( states, np.ndarray ) and states is not None :
      self.return_code = (ReturnCode.ERROR, f"'states' must be None, an integer, or a list of integers")
      states = np_empty(np.int32)
    if states is None:
      result = np.array( [ smap.time for smap in self.__smaps ] )
    else:
      result = np.array( [ self.__smaps[state-1].time for state in states ] )
    return result

  def state_variables(self) -> Dict[str,StateVariable]:
    """Getter for internal dictionary of StateVariable objects.

    Returns:
      Dict[str,StateVariable]: Keys are state variable names, Values are StateVariable objects.
    """
    return self.__svars

  def queriable_svars(self, vector_only = False, show_ips = False) -> List[str]:
    """Get a list of state variable names that can be queried.

    Args:
      vector_only (bool, default=False): Return on vector state variables.
      show_ips (bool, default=False): Show the available integration points.

    Returns:
      List[str]: A list of state variable names that can be queried.
    """
    queriable = []
    for sname, svar in self.__svars.items():
      if svar.agg_type in ( StateVariable.Aggregation.VEC_ARRAY, StateVariable.Aggregation.VECTOR ):
        if svar.agg_type == StateVariable.Aggregation.VECTOR:
          queriable.append( sname )
        for subvar in svar.svars:
          name = sname
          if show_ips and svar.agg_type == StateVariable.Aggregation.VEC_ARRAY:
            name += f"[0-{svar.dims[0]}]"
          name += f"[{subvar.name}]"
          queriable.append( name )
      else:
        if not vector_only:
          queriable.append( sname )
    return queriable

  def supported_derived_variables(self) -> List[str]:
    """Get the derived variables that mili-python currently supports.

    Returns:
      List[str]: A List of derived variable names that can be calculated by mili-python.
    """
    return self.__derived.supported_variables()

  def derived_variables_of_class(self, class_name: str) -> List[str]:
    """Get the derived variables that can be calculated for the specified class.

    Args:
      class_name (str): The element class name.

    Returns:
      List[str]: List of derived variables that can be calculated for the specified class.
    """
    return self.__derived.derived_variables_of_class(class_name)

  def classes_of_derived_variable(self, var_name: str) -> List[str]:
    """Get the classes for which a derived variable can be calculated.

    Args:
      var_name (str): The derived variable name:

    Returns:
      List[str]: List of element class names for which var_name can be calculated.
    """
    return self.__derived.classes_of_derived_variable(var_name)

  def labels(self, class_name: Optional[str] = None) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element labels.

    Args:
      class_name (Optional[str]): If provided, only return labels for specifid element class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the labels for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the labels for the specified element class.
    """
    if class_name is not None:
      return self.__labels.get(class_name, np_empty(np.int32))
    return self.__labels

  def materials(self) -> Dict[str,List[int]]:
    """Get materials dictionary from the database.

    Returns:
      Dict[str,List[int]]: Keys are the material names, Values are a list of material numbers.
    """
    return self.__mats

  def material_numbers(self) -> np.ndarray:
    """Get a List of material numbers in the database.

    Returns:
      np.ndarray: A numpy array of the material numbers.
    """
    return np.array(list(self.__elems_of_mat.keys()))

  def connectivity( self, class_name : Optional[str] = None ) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element connectivity as element LABELS.

    Args:
      class_name (str): An element class name. If provided only return connectivty for the specified class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the connectivity for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the connectivity for the specified element class. If the specified element
        class does not exists then None is returned.
    """
    if class_name is not None:
      return self.__conns_labels.get(class_name, None)
    return self.__conns_labels

  def connectivity_ids( self, class_name : Optional[str] = None ) -> Union[Dict[str,np.ndarray],np.ndarray]:
    """Getter for the element connectivity as element IDS.

    Args:
      class_name (str): An element class name. If provided only return connectivty for the specified class.

    Returns:
      Union[Dict[str,np.ndarray],np.ndarray]: If class_name is None the a dictionary containing
        the connectivity for each element class is returned. If class_name is not None, then a numpy array
        is returned containing the connectivity for the specified element class. If the specified element
        class does not exists then None is returned.
    """
    if class_name is not None:
      return self.__conns_ids.get(class_name, None)
    return self.__conns_ids

  def __valid_material_type(self, mat: Any ) -> bool:
    """Check if type is valid for material number input."""
    return isinstance(mat, (str, int, np.integer))

  def material_classes(self, mat: Union[str,int]) -> List[str]:
    """Get list of classes of a specified material.

    Args:
      mat (Union[str,int]): A material name or number.

    Returns:
      List[str]: List of element classes associated with the material.
    """
    if not self.__valid_material_type(mat):
      self.__return_code = (ReturnCode.ERROR, 'material must be string or int')
      classes = []
    else:
      mat_nums = []
      if type(mat) is str and mat in self.__mats.keys():
        mat_nums = self.__mats.get( mat, [] )
      elif type(mat) is str and mat.isdigit():
        mat = int(mat)
      if isinstance(mat, (int, np.integer)) and mat in self.__elems_of_mat.keys():
        mat_nums = [ mat ]
      classes = []
      for mat in mat_nums:
        for class_name in self.__elems_of_mat.get(mat,{}).keys():
          classes.append(class_name)
    return classes

  def classes_of_state_variable(self, svar: str) -> List[str]:
    """Get list of element classes for a state variable.

    Args:
      svar (str): The state variable name.

    Returns:
      List[str]: List of element classes the state variable exists for.
    """
    classes = []
    state_variable = self.__svars.get(svar, None)
    if state_variable:
      classes = list(set([ srec.class_name for srec in state_variable.srecs ]))
    return classes

  def state_variables_of_class(self, class_name: str) -> List[str]:
    """Get list of primal state variables for a given element class.

    Args:
      class_name (str): The element class name.

    Returns:
      List[str]: A list of primal state variables that can be queried for the specified element class.
    """
    state_variables = []
    for svar_name, svar in self.__svars.items():
      for srec in svar.srecs:
        if srec.class_name == class_name and svar_name not in state_variables:
          state_variables.append(svar_name)
    return state_variables

  def state_variable_titles(self) -> dict[str,str]:
    """Get dictionary of state variable titles for each state variable.

    Returns:
      dict[str,str]: Dictionary where keys are svar names and values are svar titles.
    """
    svar_titles = {}
    svar_titles.update( self.__derived.derived_variable_titles() )
    for sname, svar in self.__svars.items():
      svar_titles[sname] = svar.title
    return svar_titles

  def containing_state_variables_of_class(self, svar: str, class_name: str) -> List[str]:
    """Get List of state variables that contain the specific state variable + class_name.

    Args:
      svar (str): The state variable name.
      class_name (str): The element class name.

    Returns:
      List[str]: List of containing state variables.
    """
    containing_svars = []
    if svar in self.__svars:
      potential_containing_svars = np.array(self.__svars[svar].containing_svar_names)
      of_same_class = [class_name in self.classes_of_state_variable(containing_svar) for containing_svar in potential_containing_svars]
      containing_svars = list(potential_containing_svars[np.nonzero(of_same_class)[0]])
    return containing_svars

  def components_of_vector_svar(self, svar: str) -> List[str]:
    """Get a list of component state variables of a vector state variable.

    Args:
      svar (str): The name of a vector state variable.

    Returns:
      List[str]: The components of the vector state variable.
    """
    comp_names = []
    svar_spec = self.__svars.get(svar, None)
    if svar_spec is None:
      self.__return_code = (ReturnCode.ERROR, f"The state variable {svar} does not exists")
    elif svar_spec.agg_type not in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
      self.__return_code = (ReturnCode.ERROR, f"The state variable {svar} is not a Vector")
    else:
      comp_names = svar_spec.comp_names
    return comp_names

  def parts_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of part numbers for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of part numbers for each element of the class name.
    """
    elem_parts = np.zeros( self.__labels.get( class_name, np_empty(np.int32) ).shape, dtype=np.int32 )
    elem_parts[:] = -1
    for part, labels in self.__elems_of_part.items():
      idxs = labels.get( class_name, np_empty(np.int32) )
      elem_parts[ idxs ] = part
    return elem_parts

  def materials_of_class_name( self, class_name: str ) -> np.ndarray:
    """Get List of materials for all elements of a given class name.

    Args:
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of material numbers for each element of the class name.
    """
    elem_mats = np.zeros( self.__labels.get( class_name, np_empty(np.int32) ).shape, dtype=np.int32 )
    elem_mats[:] = -1
    for mat, labels in self.__elems_of_mat.items():
      idxs = labels.get( class_name, np_empty(np.int32) )
      elem_mats[ idxs ] = mat
    return elem_mats

  def class_labels_of_material( self, mat: Union[str,int], class_name: str ) -> np.ndarray:
    """Convert a material name into labels of the specified class (if any).

    Args:
      mat (Union[str,int]): The material name or number.
      class_name (str): The element class name.

    Returns:
      np.ndarray: array of labels of the specified class name and material.
    """
    if class_name not in self.__labels.keys():
      return np_empty(np.int32)
    if not self.__valid_material_type(mat):
      self.__return_code = (ReturnCode.ERROR, 'material must be string or int')
      return np_empty(np.int32)
    if type(mat) is str:
      all_reps = set( self.__mats.keys() )
      # Check if mat is an integer passed in as a string
      if mat not in all_reps and mat.isdigit():
        mat = int(mat)
    if isinstance(mat, (int, np.integer)):
      all_reps = list( itertools.chain.from_iterable(self.__mats.values()) )
    if mat not in all_reps and mat not in self.__elems_of_mat.keys():
      return np_empty(np.int32)
    elem_idxs = np_empty(np.int32)
    if mat in self.__mats.keys():
      for mat in self.__mats[mat]:
        elem_idxs = np.concatenate( ( elem_idxs, self.__elems_of_mat[mat].get( class_name, np_empty(np.int32) ) ) )
    else:
      elem_idxs = self.__elems_of_mat[mat].get( class_name, np_empty(np.int32) )
    labels = self.__labels[ class_name ][ elem_idxs ]
    return labels

  def all_labels_of_material( self, mat: Union[str,int] ) -> Dict[str,np.ndarray]:
    """Given a specific material. Find all labels with that material and return their values.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      Dict[str,np.ndarray]: Keys are element class names. Values are numpy arrays of element labels.
    """
    if not self.__valid_material_type(mat):
      self.__return_code = (ReturnCode.ERROR, 'material must be string or int')
      return {}
    mat_nums = []
    if type(mat) is str and mat in self.__mats.keys():
      mat_nums = self.__mats.get( mat, [] )
    elif type(mat) is str and mat.isdigit():
      mat = int(mat)
    if isinstance(mat, (int, np.integer)) and mat in self.__elems_of_mat.keys():
      mat_nums = [ mat ]
    labels = {}
    for mat_num in mat_nums:
      for class_name in self.__elems_of_mat.get( mat_num , {} ).keys():
        labels[class_name] = self.class_labels_of_material( mat_num, class_name )
    return labels

  def nodes_of_elems(self, class_sname: str, elem_labels: Union[int,List[int]]) -> Tuple[np.ndarray,np.ndarray]:
    """Find nodes associated with elements by label.

    Args:
      class_sname (str): The element class name.
      elem_labels (List[int]): List of element labels.

    Returns:
      Tuple[np.ndarray,np.ndarray]: (The nodal connectivity, The element labels)
    """
    if type(elem_labels) is not list:
      if iterable(elem_labels):
        elem_labels = list(elem_labels)
      else:
        elem_labels = [ elem_labels ]
    if class_sname not in self.__class_to_sclass:
      return np.empty([1,0],dtype=np.int32), np.empty([1,0],dtype=np.int32)
    if all( label not in self.__labels[class_sname] for label in elem_labels ):
      return np.empty([1,0],dtype=np.int32), np.empty([1,0],dtype=np.int32)
    if class_sname not in self.__conns_labels:
      return np.empty([1,0],dtype=np.int32), np.empty([1,0],dtype=np.int32)

    # Only search for labels that actually exist for this processor/database
    elem_labels = [label for label in elem_labels if label in self.__labels[class_sname]]

    # get the indices of the labels we're querying in the list of local labels of the element class, so we can retrieve their connectivity
    indices = (self.__labels[class_sname][:,None] == elem_labels).argmax(axis=0)
    elem_conn = self.__conns_ids[class_sname][indices][:,:-1]
    return self.__labels['node'][elem_conn], self.__labels[class_sname][indices,None]

  def nodes_of_material(self, mat: Union[str,int] ) -> np.ndarray:
    """Find nodes associated with a material number.

    Args:
      mat (Union[str,int]): The material name or number.

    Returns:
      numpy.array: A list of all nodes associated with the material number.
    """
    if not self.__valid_material_type(mat):
      self.__return_code = (ReturnCode.ERROR, 'material must be string or int')
      return np_empty(np.int32)
    element_labels = self.all_labels_of_material( mat )
    node_labels = np_empty(np.int32)
    for class_name, class_labels in element_labels.items():
      node_labels = np.append( node_labels, self.nodes_of_elems( class_name, class_labels )[0] )
    return np.unique( node_labels )

  def __parse_query_name_and_source( self, svar_query_input, requested_result_source ):
    """Parse the svar name from a query input into a base svar, component svar list, and result source."""
    comp_start_idx = svar_query_input.find('[')
    if comp_start_idx != -1:
      svar_name = svar_query_input[:comp_start_idx]
      svar_comps = svar_query_input[comp_start_idx+1:-1].split(',')
    else:
      svar_name = svar_query_input
      svar_comps = []

    # Determine if this result exists as a primal or a derived, or both
    primal_exists = False
    derived_exists = False
    if svar_name in self.__svars.keys():
      primal_exists = True
    if not all([comp_svar_name in self.__svars.keys() for comp_svar_name in svar_comps]):
      primal_exists = False
    if svar_name in self.__derived.supported_variables():
      derived_exists = True

    # Get the result source we are goint to return
    actual_result_source = ''
    # Preferred result source is primals
    if requested_result_source == 'primal':
      if primal_exists:
        actual_result_source = 'primal'
      # If primal doesn't exist, but derived does, calculate derived.
      if not primal_exists and derived_exists:
        actual_result_source = 'derived'
    # Preferred sourec is derived
    elif requested_result_source == 'derived':
      # If derived exists, calculate derived
      if derived_exists:
        actual_result_source = 'derived'
      # If derived doesn't exist, but primal does, load primal result.
      if not derived_exists and primal_exists:
        actual_result_source = 'primal'

    return svar_query_input, svar_name, svar_comps, actual_result_source

  def __init_query_parameters( self,
                               svar_names : Union[List[str],str],
                               class_sname : str,
                               material : Optional[Union[str,int]] = None,
                               labels : Optional[Union[List[int],int]] = None,
                               states : Optional[Union[List[int],int]] = None,
                               ips : Optional[Union[List[int],int]] = None,
                               write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None ):
    """Parse the query parameters and normalize them to the types expected by the rest of the query operation."""
    any_invalid = False

    if isinstance(states, (int, np.integer)):
      states = np.array( [ states ], dtype = np.int32 )
    elif iterable( states ) and not isinstance(states, str):
      states = np.unique( np.array( states, dtype = np.int32 ) )

    min_st = 1
    max_st = len(self.__smaps)
    if states is None:
      states = np.arange( min_st, max_st+1, dtype = np.int32 )

    if not isinstance( states, np.ndarray ):
      self.__return_code = (ReturnCode.ERROR, f"'states' must be None, an integer, or a list of integers" )
      any_invalid = True
    else:
      # Support negative indexing
      if np.any( states < 0 ):
        where = np.where( states < 0 )[0]
        states[where] = (max_st + 1) + states[where]
        states = np.sort( states )
      # Check for any states that are out of bounds
      if np.any( states < min_st ) or np.any( states > max_st ):
          self.__return_code = (ReturnCode.ERROR, f"Attempting to query states that do not exist. "
                                                  f"Minimum state = {min_st}, Maximum state = {max_st}")
          any_invalid = True
      if len(states) == 0:
        self.__return_code = (ReturnCode.ERROR, f"Query failed because no states exist")
        any_invalid = True

    if isinstance(labels, (int, np.integer)):
      labels = np.array( [ labels ], dtype = np.int32 )
    if iterable( labels ) and not isinstance(labels, str):
      labels = np.unique( np.array( labels, dtype = np.int32 ) )

    # Filter labels queried by material if provided, or select all labels of given material if no other labels are given
    if material is not None:
      mat_labels = self.class_labels_of_material(material, class_sname)
      if labels is not None:
        labels = np.intersect1d( labels, mat_labels )
      else:
        labels = mat_labels

    if labels is None:
      labels = self.__labels.get( class_sname, np.empty([0],np.int32) )

    if not isinstance( labels, np.ndarray ):
      self.__return_code = (ReturnCode.ERROR, f"'labels' must be None, an integer, or a list of integers" )
      any_invalid = True

    if type( svar_names ) is not str and iterable(svar_names):
      svar_names = list(set(svar_names))
    elif type( svar_names ) is str:
      svar_names = [ svar_names ]

    if svar_names is None:
      self.__return_code = (ReturnCode.ERROR, 'State variable names must be a string or iterable of strings' )
      any_invalid = True

    if isinstance(ips, (int, np.integer)):
      ips = [ips]
    if ips is None:
      ips = []

    if not isinstance(ips, (list,np.ndarray)):
      self.__return_code = (ReturnCode.ERROR, 'comp must be an integer or list of integers' )
      any_invalid = True
    else:
      # Ensure not duplicate integration points
      ips = np.unique( np.array( ips, dtype=np.int32 ))

    if write_data is not None:
      for queried_name in svar_names:
        if not np.all( write_data[queried_name]['layout']['states'] == states ):
          self.__return_code = (ReturnCode.ERROR, f"When writing data to a database, the write_data must have the same states as the query.")
          any_invalid = True

    return any_invalid, svar_names, class_sname, material, labels, states, ips, write_data


  def query( self,
             svar_names : Union[List[str],str],
             class_sname : str,
             material : Optional[Union[str,int]] = None,
             labels : Optional[Union[List[int],int]] = None,
             states : Optional[Union[List[int],int]] = None,
             ips : Optional[Union[List[int],int]] = None,
             write_data : Optional[Dict[str,QueryDict]] = None,
             **kwargs ) -> Dict[str,QueryDict]:
    """Query the database for state variables or derived variables, returning data for the specified parameters, optionally writing data to the database.

    Args:
      svar_names (Union[List[str],str]): The names of the state variables to be queried.
      class_sname (str): The element class name being queried (e.g. brick. shell, node).
      material (Optional[Union[str,int]], default=None): Optional material name or number to select labels from.
      labels (Optional[Union[List[int],int]], default=None): Optional labels to query data for, filtered by material
        if material if material is supplied, default is all.
      states (Optional[Union[List[int],int]], default=None): Optional state numbers from which to query data, default is all.
      ips (Optional[Union[List[int],int]], default=None): Optional integration point to query for vector array state variables, default is all available.
      write_data (Optional[Dict[str,QueryDict]], default=None): Optional the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format,
        then modify the values desired, then query again with the modified result in this param
    """
    # normalize arguments to expected types / default values
    errors, svar_names, class_sname, material, labels, states, ips, write_data = self.__init_query_parameters( svar_names, class_sname, material, labels, states, ips, write_data )
    if errors:
      return {}

    # handle possible hidden keyword arguments
    output_object_labels = kwargs.get("output_object_labels", True)
    subrec = kwargs.get("subrec", None)
    # Save result source and remove from kwargs so it doesn't get passed to derived query.
    requested_result_source = kwargs.get('source', 'primal')
    if "source" in kwargs:
      del kwargs["source"]

    # Check that no unexpected keyword arguments were passed
    expected_keywords = set(["output_object_labels", "subrec", "source", "reference_state"])
    unexpected_keywords = set(kwargs.keys()) - expected_keywords
    if len(unexpected_keywords) != 0:
      self.__return_code = (ReturnCode.ERROR, f"The following unexpected keywords were provided to the query method: {unexpected_keywords}")
      return {}

    res = dict.fromkeys( svar_names )
    for ikey in res.keys():
      res[ikey] = QueryDict(
        data = np_empty( np.float32 ),    # Numpy array of results
        layout = QueryLayout(
          states = np_empty( np.int32 ),  # The state numbers in the results
          labels = np_empty( np.int32 ),  # The element labels in the results
          components  = [],              # The components of the state variable in the results.
          times = np_empty( np.float32 ) # The times at each state in the query
        ),
        source = '',                       # primals vs. derived
        class_name = class_sname,          # Element class name for the result
        title = '',                        # The state variable title
      )

    # for parallel operation it will often be the case a specific file doesn't have any labels
    labels_of_class = self.__labels.get( class_sname, np_empty(np.int32) )
    ordinals = np.where( np.isin( labels_of_class, labels ) )[0]

    query_specs = [self.__parse_query_name_and_source( queried_name, requested_result_source ) for queried_name in svar_names ]
    not_found = [ spec for spec in query_specs if spec[3] == "" ]
    derived_specs = [ spec for spec in query_specs if spec[3] == "derived" ]
    primal_specs = [ spec for spec in query_specs if spec[3] == "primal" ]

    # Error if any state variables don't exist
    if len(not_found) > 0:
      self.__return_code = (ReturnCode.ERROR, f"The state variables '{not_found}' do not exist.")
      return res

    # If no labels, then skip. We do this here to ensure non existant state variables are reported.
    if ordinals.size == 0:
      self.__return_code = (ReturnCode.ERROR, f"No labels found for the class {class_sname}")
      return res

    # Handle derived queries
    derived_query_names = [spec[0] for spec in derived_specs]
    if derived_query_names:
      try:
        derived_data = self.__derived.query(derived_query_names, class_sname, material, labels, states, ips, **kwargs)
      except Exception as e:
        self.__return_code = (ReturnCode.ERROR, f"Exception occurred during derived query: {str(e)}")
        derived_data = {}
      res.update( derived_data )

    # Handle primal queries
    for query_spec in primal_specs:
      queried_name, queried_svar_name, comp_svar_names, actual_result_source = query_spec

      res[queried_name]['layout']['states'] = states
      res[queried_name]['layout']['times'] = np.array([self.__smaps[state-1].time for state in states])
      res[queried_name]['source'] = actual_result_source
      res[queried_name]['title'] = self.__svars[queried_svar_name].title

      match_aggregate_svar = ""
      if self.__svars[queried_svar_name].agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
        match_aggregate_svar = queried_svar_name

      # if svar is an aggregate svar and we're not querying specific comps, query all the comps
      if comp_svar_names == []:
        if self.__svars[queried_svar_name].agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
          svars_to_query = self.__svars[queried_svar_name].svars
        else:
          svars_to_query = [ self.__svars[ queried_svar_name ] ]
      else:
        svars_to_query = [ self.__svars[ comp_svar_name ] for comp_svar_name in comp_svar_names ]

      # discover subrecords holding the queried svars, should precompute this if it becomes a bottleneck
      srecs_to_query : List[Subrecord] = []
      for svar in svars_to_query:
        # filter so we only have those subrecords with the appropriate class_sname
        if "es_" in queried_svar_name:
          srecs_with_svar_and_class = [ srec for srec in svar.srecs if (srec.class_name == class_sname and queried_svar_name in srec.svar_names) ]
        else:
          srecs_with_svar_and_class = [ srec for srec in svar.srecs if srec.class_name == class_sname ]
        for srec in srecs_with_svar_and_class:
          if srec not in srecs_to_query: # would prefer to just unique after this, but they're unhashable so the trivial list(set()) isn't possible
            srecs_to_query.append( srec )

      # If user only requested single subrecord, only query single subrecord
      if subrec is not None:
        srecs_to_query = [srec for srec in srecs_to_query if srec.name == subrec]

      # if we're looking for ip data determine which int_points we can get for each svar in each subrec
      matching_int_points = dict()
      int_point_labels = dict()
      for svar in svars_to_query:
        comp_svar_name = svar.name
        candidate_ip_svars = list( self.__int_points.get( comp_svar_name, [] ) )
        for candidate in candidate_ip_svars:
          candidate_srecs = self.__svars[ candidate ].srecs
          for srec in candidate_srecs:
            if srec.class_name == class_sname:
              if comp_svar_name not in matching_int_points:
                matching_int_points[ comp_svar_name ] = {}
              if len(ips) > 0:
                # Matching subset
                matching_int_points[ comp_svar_name ][ srec.name ] = [ self.__int_points[comp_svar_name][candidate].index(ip) for ip in ips ]
              else:
                # Matching All
                matching_int_points[ comp_svar_name ][ srec.name ] = [ i for i in range(self.__int_points[comp_svar_name][candidate][-1]) ]

              if comp_svar_name not in int_point_labels:
                int_point_labels[ comp_svar_name ] = {}
              int_point_labels[ comp_svar_name ] = [ self.__int_points[comp_svar_name][candidate][idx] for idx in matching_int_points[comp_svar_name][srec.name] ]

      # For each subrecord, determine which elements (labels) appear in that subrecord
      # and create a dictionary entry for each subrecord that contains the labels in that
      # subrecord and the indexes at which the data for that label appears in the subrecord
      srec_element_ordinals = {}
      srec_internal_offsets = {} # { subrecord name --> [ label_cols, index_cols... ] }
      for srec in srecs_to_query:
        srec_memory_offsets, ordinals_in_srec = srec.calculate_memory_offsets( match_aggregate_svar, svars_to_query, ordinals, matching_int_points )
        if len( ordinals_in_srec ) > 0:
          if output_object_labels:
            res[queried_name]['layout']['labels'] = np.concatenate( ( res[queried_name]['layout']['labels'], self.__labels[ class_sname ][ ordinals_in_srec ] ) )
          else:
            res[queried_name]['layout']['labels'] = ordinals_in_srec

        srec_internal_offsets[srec.name] = srec_memory_offsets
        srec_element_ordinals[srec.name] = ordinals_in_srec

      # Check for any duplcate labels for the result.
      unique_labels, counts = np.unique( res[queried_name]['layout']['labels'], return_counts=True )
      if np.any( counts > 1 ):
        duplicate_labels = unique_labels[ counts > 1 ]
        self.__return_code(ReturnCode.CRITICAL,
                           f'Invalid database: The following labels appear in multiple subrecords for the result "{queried_name}\n"'
                           f'Labels = {duplicate_labels}')
        break

      # filter out any subrecords we have no labels for
      srecs_to_query = [ srec for srec in srecs_to_query if srec_internal_offsets[srec.name].size != 0 ]
      if not srecs_to_query:
        self.__return_code = (ReturnCode.ERROR, f"No subrecords found for the result/class combination {queried_name}, {class_sname}")
        break

      # Get state variable data type
      svar_np_dtype = self.__svars[queried_svar_name].data_type.numpy_dtype()

      # filter data to be written to each subrecord
      filtered_write_data = None
      if write_data is not None and queried_name in write_data:
        filtered_write_data = {}
        for srec in srecs_to_query:
          elements_to_write = self.__labels[class_sname][srec_element_ordinals[srec.name]]
          if elements_to_write.size > 0:
            # We need to find the labels in write_data that are in this subrecord and use the same ordering
            # of the labels in write_data so that the data is written correctly.
            write_data_labels = write_data[queried_name]['layout']['labels']
            sorter = np.argsort(write_data_labels)
            rows_to_write = sorter[ np.searchsorted( write_data_labels, elements_to_write, sorter=sorter ) ]
            filtered_write_data[srec.name] = write_data[queried_name]['data'][:,rows_to_write,:]

      # initialize the results structure for this queried name
      qty_states = len(states) # dim 1 is number of states
      qty_elems = 0
      comp_qtys = []
      # Loop over subrecords to get the total number of elements and components for each subrecord
      for srec in srecs_to_query:
        srec_result_shape = srec_internal_offsets[srec.name].shape
        qty_elems += srec_result_shape[0]
        comp_qtys.append( srec_result_shape[1] )

      # Verify that number of components is consistants across all elements/materials/srecs that we are querying
      if len(list(set(comp_qtys))) != 1:
        debug_str = "Integration points by Labels:\n"
        for srec, comp_qty in zip(srecs_to_query, comp_qtys):
          srec_labels_in_query = self.__labels[srec.class_name][srec_element_ordinals[srec.name]]
          debug_str += f"\tElements {srec_labels_in_query} have {comp_qty} integration points.\n"
        self.__return_code = (ReturnCode.ERROR,
                              "Integration points not consistant across all elements/materials being queried. "
                              "Please use the argument 'ips' to provide a specific integration point or "
                              "to specify a set of integration points that exist for all the elements being queried\n"
                              f"{debug_str}")
        break

      qty_comps = comp_qtys[0]
      res[queried_name]['data'] = np.empty( [ qty_states, qty_elems, qty_comps ], dtype = svar_np_dtype )

      # Specify result components in result dictionary
      for svar in svars_to_query:
        int_points = int_point_labels.get(svar.name, [])
        if len(int_points) > 0:
          for label in int_points:
            res[queried_name]['layout']['components'].append(f"{svar.name} ipt. {label}")
        else:
          res[queried_name]['layout']['components'].append(f"{svar.name}")

      # Determine which states (of those requested) appear in each of the state files.
      # This way we can open each file only once and process all the states that appear in it
      # rather than opening a state file for each iteration
      state_file_dict = {}
      for state in states:
        state_map = self.__smaps[state - 1]
        sfn = self.__state_files[state_map.file_number]
        if sfn not in state_file_dict.keys():
          state_file_dict[sfn] = []
        state_file_dict[sfn].append( state )

      # use the calculated offsets to retrieve the underlying data from the state files
      access_mode = 'rb' if filtered_write_data is None else 'rb+'
      for state_filename, file_state_nums in state_file_dict.items():
        with open(state_filename, access_mode) as state_file:
          for state in file_state_nums:
            sidx = np.where(states == state)[0][0]
            state_offset = self.__smaps[state-1].file_offset + 8

            # Shape of the result array: Tuple with form (state_cnt, label_cnt, comp_cnt)
            # We want to use comp_cnt to initialize var data so that numpy can concatenate arrays
            result_shape = res[queried_name]['data'].shape
            var_data = np.empty( [ 0, result_shape[2] ], svar_np_dtype)
            for srec in srecs_to_query:
              srec_offsets = srec_internal_offsets[srec.name]
              state_file.seek( state_offset + srec.state_byte_offset )
              byte_read_data = state_file.read( srec.byte_size )

              if filtered_write_data is not None and srec.name in filtered_write_data:
                srec_var_data, byte_write_data = srec.extract_ordinals( byte_read_data, srec_offsets, write_data = filtered_write_data[srec.name][ sidx,:,: ] )
                state_file.seek( state_offset + srec.state_byte_offset )
                state_file.write( byte_write_data )
              else:
                srec_var_data, _ = srec.extract_ordinals( byte_read_data, srec_offsets )
              var_data = np.concatenate( ( var_data, srec_var_data ), axis=0 )

            res[ queried_name ][ "data" ][ sidx, :, : ] = var_data

    return res

  def __get_state_byte_data(self, state: int, zero_out: bool, byte_data: BinaryIO) -> None:
    state_size = np.sum([srec.byte_size for srec in self.__srecs])
    if state == 0 or zero_out:
      # No data exists, just write out zeros
      byte_data.write( bytes(state_size) )
    else:
      # Copy previous states data
      smap = self.__smaps[state-1]
      state_file_name = self.__state_files[smap.file_number]
      with open(state_file_name, 'rb') as state_file:
        state_file.seek( smap.file_offset + 8 )
        byte_data.write( state_file.read( state_size ) )

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
    n_states = len(self.__smaps)
    creating_new_state_file = False

    # Verify state_time makes sense
    if n_states > 0 and new_state_time <= self.__smaps[-1].time:
      self.__return_code = (ReturnCode.ERROR, "The time of the appended state must be greater than the current last state.")
      return n_states

    if len(self.__srecs) == 0:
      self.__return_code = (ReturnCode.ERROR, "Cannot append a state. No subrecords exist.")
      return n_states
    state_size = np.sum([srec.byte_size for srec in self.__srecs]) + 8

    # Compute the offset of the new state
    if n_states == 0:
      # We assume the subrecords already exist and we just need update the state maps/file
      new_smap_offset = 0
      new_state_file = 0
      creating_new_state_file = True
    else:
      new_smap_offset = self.__smaps[-1].file_offset + state_size
      # Assume we are writing to same state file as previous state
      new_state_file = self.__smaps[-1].file_number

    # Handle state count and file size limits
    if n_states != 0:
      if limit_states_per_file:
        states_in_current_file = len([smap for smap in self.__smaps if smap.file_number == new_state_file])
        if states_in_current_file + 1 > limit_states_per_file:
          creating_new_state_file = True
          new_state_file += 1
          new_smap_offset = 0
      if limit_bytes_per_file and (new_smap_offset + state_size) > limit_bytes_per_file:
        creating_new_state_file = True
        new_state_file += 1
        new_smap_offset = 0

    # Create new StateMap object
    new_smap = StateMap(new_state_file, new_smap_offset, new_state_time, 0)

    # Update Afile
    self.__afile.smaps.append(new_smap)
    self.__afile.state_map_count += 1
    if 'state_count' in self.__afile.dirs[DirectoryDecl.Type.APPLICATION_PARAM]:
      self.__afile.dirs[DirectoryDecl.Type.APPLICATION_PARAM]['state_count'] += 1

    afile_writer = AFileWriter()
    afile_name = f"{os.path.join( self.__dir_name, self.__base_filename )}"
    afile_writer.write(self.__afile, afile_name)

    # Update State file
    state_file_suffix = "{:02d}".format(new_smap.file_number)
    state_filename = os.path.join( self.__dir_name, f"{self.__base_filename}{state_file_suffix}" )
    if creating_new_state_file:
      self.__state_files.append(state_filename)

    # Set up bytes to be written to new state
    bytes_to_write = io.BytesIO()
    bytes_to_write.write( struct.pack('fi', new_smap.time, new_smap.state_map_id) )
    self.__get_state_byte_data( n_states, zero_out, bytes_to_write )

    # Write out new state
    access_mode = 'wb' if creating_new_state_file else 'rb+'
    with open(state_filename, access_mode) as state_file:
      state_file.seek( new_smap.file_offset )
      state_file.write( bytes_to_write.getvalue() )

    # We need to write out the nodal positions and sand flags so the database can be visualized.
    if n_states == 0 or zero_out:
      state_to_write = n_states + 1
      if n_states == 0:
        # If this is the first state being written to the database
        # then we need to write out the initial nodal positions (db.nodes())
        nodpos = self.query("nodpos", "node", states=[state_to_write])
        nodpos['nodpos']['data'][0] = self.__nodes
        self.query("nodpos", "node", states=[state_to_write], write_data=nodpos)
      else:
        # We need to query the previous states nodal positions and copy them to the current state
        nodpos = self.query("nodpos", "node", states=[state_to_write-1])
        nodpos['nodpos']['layout']['states'][0] = state_to_write
        self.query("nodpos", "node", states=[state_to_write], write_data=nodpos)

      # Need to set sand flags as well so everything can be visualized
      sand_classes = self.classes_of_state_variable("sand")
      for class_name in sand_classes:
        sand_flags = self.query("sand", class_name, states=[state_to_write])
        sand_flags['sand']['data'][0] = 1.0
        self.query("sand", class_name, states=[state_to_write], write_data=sand_flags)

    return len(self.__smaps)

  def copy_non_state_data(self, new_base_name: str) -> None:
    """Copy non state data from an existing Mili database into a new database without any states (Just the A file).

    Non state data include the geometry, states variables, subrecords and parameters.

    Args:
      new_base_name (str): The base name of the new database.
    """
    new_afile =  self.__afile.copy_non_state_data()

    # Write out the new AFile
    afile_writer = AFileWriter()
    # Copy trailing digits from basename to mimic processor numbers for uncombined databases
    trailing_digits_regex = re.compile(r'(\d+)$').search(self.__base_filename)
    trailing_digits = "" if trailing_digits_regex is None else trailing_digits_regex.group(1)
    new_base_name = new_base_name + trailing_digits
    afile_writer.write(new_afile, new_base_name)