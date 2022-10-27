"""
Copyright (c) 2016-2022, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by
 William Tobin (tobin6@llnl.hov), Kevin Durrenberger (durrenberger1@llnl.gov),
 and Ryan Hathaway (hathaway6@llnl.gov).
 CODE-OCEC-16-056.
 All rights reserved.

 This file is part of Mili. For details, see
 https://rzlc.llnl.gov/gitlab/mdg/mili/mili-python/. For read access to this repo
 please contact the authors listed above.

 Our Notice and GNU Lesser General Public License.

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

# defer evaluation of type-annotations until after the module is processed, allowing class members to refer to the class
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, IntEnum
import numpy as np
import numpy.typing as npt
from numpy.core.fromnumeric import prod
import reprlib
import warnings

from typing import *

class MiliType(IntEnum):
  M_INVALID = 0
  M_STRING = 1
  M_FLOAT = 2
  M_FLOAT4 = 3
  M_FLOAT8 = 4
  M_INT = 5
  M_INT4 = 6
  M_INT8 = 7
  def byte_size(self):
    return [ -1, 1, 4, 4, 8, 4, 4, 8 ][self.value]
  def numpy_dtype(self):
    return [ None, object, np.float32, np.float32, np.float64, np.int32, np.int32, np.int64 ][self.value]
  def struct_repr(self):
    return 'sffdiiq '[self.value-1]

class Superclass(IntEnum):
  ''' The superclass denotes what mesh class an object belongs to. '''
  M_UNIT = 0
  M_NODE = 1
  M_TRUSS = 2
  M_BEAM = 3
  M_TRI = 4
  M_QUAD = 5
  M_TET = 6
  M_PYRAMID = 7
  M_WEDGE = 8
  M_HEX = 9
  M_MAT = 10
  M_MESH = 11
  M_SURFACE = 12
  M_PARTICLE = 13
  M_TET10 = 14
  M_QTY_SUPERCLASS = 16
  M_SHARED = 100
  M_ALL = 200
  M_INVALID_LABEL = -1

  def node_count(self):
    return [ 0, 0, 2, 3, 4, 4, 4, 5, 6, 8, 0, 0, 0, 1, 10 ][self.value]

@dataclass
class StateMap:
  file_number : int = -1
  file_offset : int = -1
  time : float = -1.0
  state_map_id : int = -1

@dataclass
class Directory:
  class Type(Enum):
    ''' Every direcotry has a type that dictates what informaiton
         is in the directory. '''
    NODES = 0
    ELEM_CONNS = 1
    CLASS_IDENTS = 2
    STATE_VAR_DICT = 3
    STATE_REC_DATA = 4
    MILI_PARAM = 5
    APPLICATION_PARAM = 6
    CLASS_DEF = 7
    SURFACE_CONNS = 8
    TI_PARAM = 9
    QTY_DIR_ENTRY_TYPES = 10

  dir_type : Type = Type.QTY_DIR_ENTRY_TYPES
  modifier_idx1 : int = -1
  modifier_idx2 : int = -1
  str_cnt : int = -1
  offset : int = -1
  length : int = 0
  strings : List[str] = dataclasses.field(default_factory=list)

@dataclass
class StateVariable:
  class Aggregation(IntEnum):
    '''
    The aggregate determines how a state variable contains other state variables
    and/or stores vectors of data.
    '''
    SCALAR = 0        # has a single value
    VECTOR = 1        # contains svar components
    ARRAY = 2         # contains an array of multiple values
    VEC_ARRAY = 3     # contains an array of svar components
    NUM_AGG_TYPES = 4

  ''' Describes the layout of an aggregation of atoms of data, see StateVariable.Aggregation for more layout info '''
  name : str = ''
  title : str = ''
  agg_type : Aggregation = Aggregation.NUM_AGG_TYPES
  data_type : MiliType = MiliType.M_INVALID
  list_size : int = 0
  order : int = 0
  dims : List[int] = dataclasses.field(default_factory=list)
  svars : List[StateVariable] = dataclasses.field(default_factory=list)
  srecs : List[Subrecord] = dataclasses.field(default_factory=list)

  def __repr__( self ) -> str:
    r = reprlib.Repr()
    r.maxlevel = 1
    return r.repr(self)

  @property
  def comp_svar_names( self ):
    return [ ssvar.name for ssvar in self.svars ]

  @property
  def comp_svar_titles( self ):
    return [ ssvar.title for ssvar in self.svars ]

  @property
  def atom_qty( self ):
    ''' Returns the quantity of atoms (scalars of the contained type) based on the aggregate type. '''
    if self.agg_type == StateVariable.Aggregation.VECTOR:
      return sum( svar.atom_qty for svar in self.svars )
    elif self.agg_type == StateVariable.Aggregation.ARRAY:
      return int( prod( self.dims ) )
    elif self.agg_type == StateVariable.Aggregation.VEC_ARRAY:
      return int( prod( self.dims ) ) * sum( svar.atom_qty for svar in self.svars )
    return 1


@dataclass
class Subrecord:
  class Org(IntEnum):
    ''' The organization of data in a subrecord is either ordered by variable or object. '''
    RESULT = 0
    OBJECT = 1
    INVALID = -1

  '''   A subrecord contains a number of state varaibles organized in a certain order. '''
  ## atom variables describe the layout of the svars by their scalar components, e.g. svar_atom_lengths[0] gives the number of scalars in the first svar in the srec
  ## ordinal variables describe the association of the svars in the srec with blocks of locally-numbered (ordinal) mesh entities by ordinal
  name : str = ''
  class_name : str = ''
  superclass : Superclass = Superclass.M_QTY_SUPERCLASS
  organization : Org = Org.INVALID
  qty_svars : int = -1
  svar_names : List[str] = dataclasses.field(default_factory=list)
  svars : List[StateVariable] = dataclasses.field(default_factory=list)
  svar_atom_lengths : np.ndarray = np.empty([0],dtype = np.int64)
  svar_atom_offsets : np.ndarray = np.empty([0],dtype = np.int64)
  svar_svar_comp_layout : np.ndarray = np.empty([0],dtype = object )
  ordinal_blocks : np.ndarray = np.empty([0], dtype = np.int64)
  ordinal_block_counts : np.ndarray = np.empty([0], dtype = np.int64)
  ordinal_block_offsets : np.ndarray = np.empty([0], dtype = np.int64)
  state_byte_offset : int = -1
  atoms_per_label : int = -1
  byte_size : int = 0
  # for each svar in the srec, the offset
  svar_ordinal_offsets : np.ndarray = np.empty([0], dtype = np.int64)
  svar_byte_offsets : np.ndarray = np.empty([0], dtype = np.int64)
  srec_fmt_id: int = 0

  def __repr__( self ) -> str:
    r = reprlib.Repr()
    r.maxlevel = 1
    return r.repr(self)

  def scalar_svar_coords( self, aggregate_match, scalar_svar_name ):
    coords = []
    for idx, (svar_name, svar_comps) in enumerate( zip(self.svar_names, self.svar_svar_comp_layout) ):
      matches = [ ( idx, jdx ) for jdx, svar in enumerate(svar_comps) if svar == scalar_svar_name ]
      if ( svar_name == aggregate_match or aggregate_match == "" ) and len( matches ) > 0:
        coords.append( matches )
    return np.array( *coords )

  # right now we're assuming we have the ordinals, we could add bounds checking as a debug version, or auto-filter for parallel versions
  # we're also assuming the write_data is filtered appropriately so that only the data being written out to this procs database is included
  def extract_ordinals( self, buffer : bytes, ordinals : npt.ArrayLike, write_data : npt.ArrayLike = None ):
    # TODO(wrt): check that buffer is appropriately sized
    all_same = self.organization == Subrecord.Org.OBJECT or all( svar.data_type == self.svars[0].data_type for svar in self.svars )
    if all_same:
      svar_idx = 0
      buffer_slice = (0 , len(buffer))
    else:
      # identify which svar the labels are indexing into (we only query a single svar at a time)
      svar_idxs = np.digitize( ordinals, self.svar_ordinal_offsets )
      # assert all labels are from the same svar
      assert( np.min(svar_idxs) == np.max(svar_idxs) )
      svar_idx = np.min(svar_idxs) - 1
      # modify the ordinals to offset in the svar for this srec
      ordinals -= self.svar_ordinal_offsets[svar_idx]
      # only read the portion of the subrecord related to this svar
      buffer_slice = (self.svar_byte_offsets[svar_idx], self.svar_byte_offsets[svar_idx+1])
    var_data = np.frombuffer( buffer[ buffer_slice[0] : buffer_slice[1] ], dtype = self.svars[svar_idx].data_type.numpy_dtype() )
    if write_data is not None:
      var_data = var_data.copy( )
      var_data[ ordinals ] = write_data
      write_buffer = bytearray(buffer)
      write_buffer[ buffer_slice[0] : buffer_slice[1] ] = var_data.tobytes()
      buffer = bytes(write_buffer)
    return var_data[ ordinals ], buffer

  @property
  def total_ordinal_count( self ):
    return max( 1, np.sum( self.ordinal_block_counts ) ) #+ 1 - len(self.ordinal_block_counts) )

  def struct_repr( self ):
    '''
    This function returns the string representation of a subrecord, to be used when interpreting the bytes during reading of state.
    '''
    if self.organization == Subrecord.Org.OBJECT:
      # if subrecord is Object ordered there will only be 1 data type, so just pull the datatype of the first svar
      return str(self.atoms_per_label * self.total_ordinal_count) + self.svars[0].data_type.struct_repr(), True
    else:
      # if all the svars are the same type, merge the struct repr into one to make querying easier
      base_repr = self.svars[0].data_type.struct_repr()
      if ( all( base_repr == svar.data_type.struct_repr() for svar in self.svars ) ):
        return str(self.atoms_per_label * self.total_ordinal_count) + self.svars[0].data_type.struct_repr(), True
      else:
        warnings.warn( "Querying data from a mixed-type subrecord is currently unsupported." )
        return [str(svar.atom_qty * self.total_ordinal_count) + svar.data_type.struct_repr() for svar in self.svars ], False

@dataclass
class MeshObjectClass:
  """Dataclass storing basic information about each Mesh Element Class."""
  mesh_id : int = 0
  short_name : str = ''
  long_name : str = ''
  sclass : Superclass = Superclass.M_QTY_SUPERCLASS
  elem_qty : int = 0
  idents_exist: bool = True
