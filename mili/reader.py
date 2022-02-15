"""
Copyright (c) 2016-2021, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by 
 William Tobin (tobin6@llnl.hov) and Kevin Durrenberger (durrenberger1@llnl.gov). 
 CODE-OCEC-16-056.
 All rights reserved.

 This file is part of Mili. For details, see #TODO: <URL describing code and how to download source>.

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

__version__ = (0,2,0)
__suppress_parallel__ = False
__profiling__ = None #"memory"

# TODO : when we read raw data into numpy buffers, modify the dtype to account for endianness: https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html

import copy
import io
import itertools
import numpy as np
import numpy.typing as npt
import os
import re
import struct
import sys

from collections import defaultdict
from numpy.lib.function_base import iterable
from typing import *

from mili.datatypes import *
from mili.afileIO import *
from mili.parallel import *

if sys.version_info < (3, 8):
  raise ImportError(f"This module requires python version > 3.8!")
if tuple( int(vnum) for vnum in np.__version__.split('.') ) < (1,20,0):
  raise ImportError(f"This module requires numpy version > 1.20.0")

if __profiling__ == "memory":
  from memory_profiler import profile
else:
  def profile(func):
    return func

def np_empty(dtype):
  return np.empty([0],dtype=dtype)

class MiliDatabase:

  @profile
  def __init__( self, att_file : BinaryIO, sfile_base : os.PathLike, dir_name : os.PathLike):
    """
    A Mili database querying object, able to parse a single A-file and query state information
      from the state files described by that A-file.
    Parameters:
      att_file (BinaryIO) : a BinaryIO stream containing the data of an A-file describing the mili database layout
      sfile_base (os.PathLike) : an os.PathLike describing the base name of the mili state files, to match against based on the A-file description of the state file naming
      dir_name (os.PathLike) : an os.PathLike denoting the file sytem directory to search for all state files described by the A-file
    """
    self.__state_files = []
    self.__smaps = []

    # file data layout
    self.__svars : Mapping[ str, StateVariable ] = {}
    self.__srecs : List[ Subrecord ] = []

    # indexing / index discovery 
    self.__labels = defaultdict(lambda : defaultdict( lambda : np_empty(np.int64)) )
    self.__mats = defaultdict(list)
    self.__int_points = defaultdict(lambda : defaultdict( list ) )

    # mostly used for input validation
    self.__class_to_sclass = {}

    # allow query by material
    self.__elems_of_mat = defaultdict(lambda : defaultdict( lambda : np_empty(np.int64)) )  # map from material number to dict of class to elems

    # mesh data
    self.__conns = {}
    self.__nodes = None
    self.__mesh_dim = 0

    self.__params = defaultdict(lambda : defaultdict( list ) )

    if isinstance( att_file, str ):
      with open( os.path.join( dir_name, att_file ), 'rb' ) as f:
        self.__parse_att_file( f )
    else:
      self.__parse_att_file( att_file )
    sfile_re = re.escape(sfile_base) + f"(\d{{{self.__sfile_suf_len},}})$"
    self.__state_files = list(filter(re.compile(sfile_re).match,os.listdir(dir_name)))
    self.__state_files.sort()
    self.__state_files = [ os.path.join(dir_name,sfile) for sfile in self.__state_files ]

  def nodes(self):
    return self.__nodes

  def state_maps(self):
    return self.__smaps

  def state_variables(self):
    return self.__svars

  def labels(self):
    return self.__labels

  def materials(self):
    return self.__mats

  def material_classes(self, mat):
    return self.__elems_of_mat.get(mat,{}).keys()

  @profile
  def __parse_att_file( self, att_file : BinaryIO ):
    """ 
    Parse the provided AFile and setup all internal data structures to allow 
      querying the described database.
    Parameters:
      att_file (BinaryIO) : a BinaryIO stream containing the data of an A-file describing the mili database layout
    """
    parser = AFileParser()
    parser.register( AFileParser.Section.HEADER, self.__parse_header )
    parser.register( AFileParser.Section.STATE_MAP, self.__parse_smap )

    # read order is important as some later phases depend on 
    #   data from earlier phases
    dir_callbacks = { Directory.Type.MILI_PARAM : self.__parse_param,
                      Directory.Type.TI_PARAM : self.__parse_ti_param,
                      Directory.Type.STATE_VAR_DICT : self.__parse_svars,
                      Directory.Type.CLASS_IDENTS : self.__parse_mesh_class_ident,
                      Directory.Type.CLASS_DEF : self.__parse_mesh_class_def,
                      Directory.Type.NODES : self.__parse_nodes,
                      Directory.Type.ELEM_CONNS : self.__parse_elem_conn,
                      Directory.Type.STATE_REC_DATA : self.__parse_srec }
    for key, func in dir_callbacks.items():
      parser.register( key, func )

    parser.read( att_file )
    for dir_type in dir_callbacks.keys():
      parser.read_dirs( att_file, dir_type )

  def __parse_header( self, header_data : bytes ):
    self.__sfile_suf_len = header_data[8]

  def __parse_smap( self, smap_data : bytes ):
    self.__smaps.append( StateMap( *struct.unpack('<iqfi',smap_data) ) )

  def __parse_param( self, param_data : bytes, directory : Directory ):
    sname = directory.strings[0]
    if sname.startswith('mesh dimensions'):
      self.__params[sname] = struct.unpack(f'{len(param_data)//4}i',param_data)
      self.__mesh_dim = self.__params[sname][0]
      self.__nodes = np.empty( [0,self.__mesh_dim], dtype = np.float32 )
    # TODO: parse other params (mostly strings) correctly
    # else:
    #   self.__params[sname] = 

  def __parse_ti_param( self, ti_param_data : bytes, directory : Directory ):
    sname = directory.strings[0]
    if sname.startswith('Node Labels'):
      self.__labels['node'] = np.frombuffer( ti_param_data[8:], dtype = np.int32 )
    elif sname.startswith('Element Labels') and not 'ElemIds' in sname:
      elem_labels = np.frombuffer( ti_param_data[8:], dtype = np.int32 )
      class_name = re.search( r'Sname-(\w*)', sname ).group(1)
      if class_name not in self.__labels.keys():
        self.__labels[class_name] = np_empty(np.int32)
      self.__labels[class_name] = np.append( self.__labels[class_name], elem_labels )
    elif sname.startswith('MAT_NAME'):
      mat_name = str(struct.unpack(f"{directory.length}s", ti_param_data)[0])[2:].split('\\x00')[0]
      mat_num = int( re.match(r"MAT_NAME_(\d+)", sname).group(1) )
      self.__mats[mat_name].append( mat_num )
    elif sname.startswith('IntLabel_es_'):
      ip_name = sname[ sname.find('es_'): ]
      ips = struct.unpack(f'{len(ti_param_data)//4}i', ti_param_data)[2:-1]
      self.__int_points[ip_name] = ips

  def __parse_svars( self, svars_data : bytes, _ : Directory ):
    svar_words, svar_bytes = struct.unpack('2i', svars_data[:8])
    int_cnt = svar_words - 2
    int_bytes = int_cnt * 4
    byte_data = svars_data[8:]
    int_data = struct.unpack(f'{int_cnt}i', byte_data[:int_bytes])
    svar_strs = str(struct.unpack(f'{svar_bytes}s', byte_data[int_bytes:])[0])[2:].split('\\x00')

    int_iter = int_data.__iter__()
    str_iter = svar_strs.__iter__()

    # TODO: still don't like this, mostly due to the looping mechanism, 
    #       but since we don't know a-priori how many bytes and the layout of
    #       each top-level svar, we have to iteratively discover as we parse
    while True:
      try:
        agg_type = StateVariable.Aggregation(next(int_iter))
      except:
        break
      data_type = MiliType(next(int_iter))
      svar_name = next( str_iter )
      svar_title = next( str_iter )
      svar = StateVariable(svar_name, svar_title, agg_type, data_type)

      if agg_type == StateVariable.Aggregation.ARRAY or agg_type == StateVariable.Aggregation.VEC_ARRAY:
        svar.order = next( int_iter )
        svar.dims = list( itertools.islice( int_iter, svar.order ) )

      if agg_type == StateVariable.Aggregation.VECTOR or agg_type == StateVariable.Aggregation.VEC_ARRAY:
        svar.list_size = next( int_iter )
        svar_comp_names = list( itertools.islice( str_iter, svar.list_size ) )
        for svar_comp_name in svar_comp_names:
          comp_svar = self.__svars.get( svar_comp_name )
          if comp_svar is None:
            _ = next( str_iter )
            svar_comp_title = next( str_iter )
            comp_agg_type = StateVariable.Aggregation(next(int_iter)),
            comp_data_type = MiliType(next(int_iter))
            comp_svar = self.__svars[svar_comp_name] = StateVariable( svar_comp_name, svar_comp_title, comp_agg_type, comp_data_type )
          svar.svars.append( comp_svar )

        # TODO : clean this nonsense up, handling integration points with hard-coding specific svar names is cumbersome...
        #        really we don't need specific ip handling, just StateVariable.Aggregation.ARRAY and VEC_ARRAY handling, since
        #        in that case we want to allow indexing specific svar-indices (as in integration point data)
        if svar_name[:-1] in self.__int_points:
          stress = self.__svars.get('stress')
          stress_comps = [ svar.name for svar in stress.svars ] if stress is not None else []
          strain = self.__svars.get('strain')
          strain_comps = [ svar.name for svar in strain.svars ] if strain is not None else [] 

          if len( list(set(svar_comp_names) & set(stress_comps)) ) == 6:
            if 'stress' not in self.__int_points.keys():
              self.__int_points['stress'] = {}
            self.__int_points['stress'][svar_name] = self.__int_points[svar_name[:-1]]

          if len( list(set(svar_comp_names) & set(strain_comps)) ) == 6:
            if 'strain' not in self.__int_points.keys():
              self.__int_points['strain'] = {}
              self.__int_points['strain'][svar_name] = self.__int_points[svar_name[:-1]]

          for svar_comp_name in svar_comp_names:
            if not svar_comp_name in self.__int_points.keys():
              self.__int_points[svar_comp_name] = {}
            self.__int_points[svar_comp_name][svar_name] = self.__int_points[svar_name[:-1]]

      self.__svars[svar_name] = svar

  def __parse_mesh_class_ident( self, class_data : bytes, directory : Directory ):
    sname = directory.strings[0]
    start, stop = struct.unpack('2i',class_data[4:12])
    if sname not in self.__labels.keys():
      self.__labels[sname] = np_empty(np.int32)
    self.__labels[sname] = np.append( self.__labels[sname], np.arange( start, stop+1, dtype = np.int32 ))

  def __parse_mesh_class_def( self, _, directory ):
    self.__class_to_sclass[directory.strings[0]] = Superclass(directory.modifier_idx2)

  def __parse_nodes( self, class_data : bytes, _ : Directory ):
    f = io.BytesIO( class_data )
    blocks = np.reshape( np.frombuffer(f.read(8), dtype = np.int32 ), [-1,2] )
    blocks_atoms = np.sum( np.diff( blocks, axis=1 ).flatten( ) + 1 )
    self.__nodes = np.concatenate( ( self.__nodes, np.reshape( np.frombuffer( f.read( 4 * blocks_atoms * self.__mesh_dim ), dtype = np.float32 ), [-1, self.__mesh_dim] ) ) )

  def __parse_elem_conn( self, conn_data : bytes, directory : Directory ):
    f = io.BytesIO( conn_data )
    sname = directory.strings[0]
    sclass, block_cnt = struct.unpack( '2i', f.read(8) )
    f.read( 8 * block_cnt )
    elem_qty = directory.modifier_idx2
    conn_qty = Superclass(sclass).node_count()
    word_qty = conn_qty + 2
    conn = np.reshape( np.frombuffer( f.read( elem_qty * word_qty * 4 ), dtype = np.int32 ), [-1,word_qty] )
    self.__conns[sname] = conn[:,:conn_qty] - 1 # account for fortran indexing
    # self.__conns[sname] = np.array( [ self.__labels['node'][elem_conn] for elem_conn in node_conn ], dtype = np.int32 )
    mats = np.unique( conn[:,conn_qty] )
    for mat in mats:
      mat_2_conn_idxs = np.nonzero( conn[:,conn_qty] == mat )[0]
      # self.__elems_of_mat[ mat ][ sname ].append( mat_2_conn_idxs )
      self.__elems_of_mat[ mat ][ sname ] = np.concatenate( ( self.__elems_of_mat[ mat ][ sname ], mat_2_conn_idxs ) )

  def __parse_srec( self, srec_data : bytes, directory : Directory ):
    f = io.BytesIO( srec_data )
    srec_int_data = directory.modifier_idx1 - 4
    srec_c_data = directory.modifier_idx2
    _, _, _, srec_qty_subrecs = struct.unpack('4i', f.read( 16 ))
    idata = struct.unpack(str(srec_int_data) + 'i', f.read( srec_int_data * 4 ))
    cdata = [ name.rstrip('\x00') for name in struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].decode('utf-8').split('\x00') ]
    cdata.remove('')
    int_pos = 0
    c_pos = 0
    offset = 0

    # TODO: refactor this to be more pythonic/performant, pull as much into the subrecord init as possible:
    for kk in range(srec_qty_subrecs):
      org, qty_svars, ord_blk_cnt = Subrecord.Org(idata[int_pos]), idata[int_pos + 1], idata[int_pos + 2]

      int_pos += 3
      name, class_name = cdata[c_pos:c_pos + 2]
      c_pos += 2
      svar_names = cdata[c_pos:c_pos + qty_svars]
      c_pos += qty_svars

      svars = [ self.__svars[svar_name] for svar_name in svar_names ]
      srec = Subrecord(name, class_name, org, qty_svars, svar_names, svars = svars)
      superclass = self.__class_to_sclass[class_name]
      if superclass != Superclass.M_MESH:
        # TODO: these are NUM blocks, not label blocks, so we need to store the labels and map from label to num and operate on num internaly instead of mapping conn to labels
        srec.ordinal_blocks = np.array( idata[ int_pos : int_pos + (2 * ord_blk_cnt) ], dtype = np.int32 )
        srec.ordinal_blocks[1::2] = srec.ordinal_blocks[1::2] + 1 # add 1 to each STOP to allow inclusive bining during queries
        srec.ordinal_counts = np.concatenate( ( [0], np.diff( srec.ordinal_blocks.reshape(-1,2), axis=1 ).flatten( ) ) ) # sum stop - start + 1 over all blocks
        srec.ordinal_blocks -= 1 # account for 1-indexing used by mili/fortran
        int_pos += 2 * ord_blk_cnt
      else:
        int_pos += 2 * ord_blk_cnt

      # atom-based lengths and offsets to make querying operations easier... this can be done after the read is complete
      srec.svar_atom_lengths = np.fromiter( ( svar.atom_qty for svar in srec.svars ), dtype = np.int64 )
      srec.svar_atom_offsets = np.zeros( srec.svar_atom_lengths.size, dtype = np.int64 )
      srec.svar_atom_offsets[1:] = srec.svar_atom_lengths.cumsum()[:-1]
      srec.atoms_per_label = sum( srec.svar_atom_lengths )
      srec.svar_svar_comp_layout = np.array( [ [ svar.name ] if ( svar.order == 0 and len( svar.svars ) == 0 ) else svar.comp_svar_names * int( max( 1, np.prod(svar.dims) ) ) for svar in srec.svars ], dtype = object )

      for svar in srec.svars:
        svar.srecs.append(srec)
        for comp_svar in svar.svars:
          comp_svar.srecs.append(srec)

      srec.byte_size = srec.total_ordinal_count * sum( svar.data_type.byte_size() * svar.atom_qty for svar in srec.svars )
      srec.state_byte_offset = offset
      offset += srec.byte_size

      if srec.organization == Subrecord.Org.RESULT:
        srec.svar_ordinal_offsets = np.cumsum( np.fromiter( ( svar.atom_qty * srec.total_ordinal_count for svar in srec.svars ), dtype = np.int64 ) )
        srec.svar_byte_offsets = np.cumsum( np.fromiter( ( svar.atom_qty * srec.total_ordinal_count * svar.data_type.byte_size() for svar in srec.svars ), dtype = np.int64 ) )
        srec.svar_byte_offsets = np.append( srec.svar_byte_offsets, [srec.byte_size] )

      self.__srecs.append( srec )

    # TODO: end of last refactoring block

  def __parse_query_name( self, svar_query_input ):
    """
    Parse the svar name from a query input into a base svar and component svar list
    """
    comp_start_idx = svar_query_input.find('[')
    if comp_start_idx != -1:
      svar_name = svar_query_input[:comp_start_idx]
      svar_comps = svar_query_input[comp_start_idx+1:-1].split(',')
    else:
      svar_name = svar_query_input
      svar_comps = []
    return svar_name, svar_comps

  def class_labels_of_material( self, mat, class_name ):
    '''
    Convert a material name into labels of the specified class (if any)
    '''
    if class_name not in self.__labels.keys():
      return np_empty(np.int32)
    if type(mat) is not str and type(mat) is not int:
      raise ValueError('material must be string or int')
    if type(mat) is str:
      all_reps = set( self.__mats.keys() )
    elif type(mat) is int:
      all_reps = list( itertools.chain.from_iterable(self.__mats.values()) )
    if mat not in all_reps and mat not in self.__elems_of_mat.keys():
      raise ValueError('There is no ' + str(mat) + ' material')
    elem_idxs = np_empty(np.int32)
    if mat in self.__mats.keys():
      for mat in self.__mats[mat]:
        elem_idxs = np.concatenate( ( elem_idxs, self.__elems_of_mat[mat].get( class_name, np_empty(np.int32) ) ) )
    else:
      elem_idxs = self.__elems_of_mat[mat].get( class_name, np_empty(np.int32) )
    labels = self.__labels[ class_name ][ elem_idxs ]
    return labels

  def all_labels_of_material( self, mat ):
    ''' Given a specific material. Find all labels with that material and return their values. '''
    if mat not in self.__mats.keys() and mat not in self.__elems_of_mat.keys():
      raise ValueError('There is no material ' + str(mat))
    if type(mat) == int:
      mat_nums = [ mat ]
    else:
      mat_nums = self.__mats.get( mat, [] )
    labels = {}
    for mat_num in mat_nums:
      for class_name in self.__elems_of_mat.get( mat_num , {} ).keys():
        labels[class_name] = self.class_labels_of_material( mat_num, class_name )
    return labels

  def nodes_of_elems( self, class_sname, elem_labels ):
    ''' Find nodes associated with elements by label '''
    if type(elem_labels) is not list:
      if iterable(elem_labels):
        elem_labels = list(elem_labels)
      else:
        elem_labels = [ elem_labels ]
    if class_sname not in self.__class_to_sclass:
      return np.empty([1,0],dtype=np.int32)
    if any( label not in self.__labels[class_sname] for label in elem_labels ):
      return np.empty([1,0],dtype=np.int32)
    if class_sname not in self.__conns:
      return np.empty([1,0],dtype=np.int32)

    # get the indices of the labels we're querying in the list of local labels of the element class, so we can retrieve their connectivity
    indices = (self.__labels[class_sname][:,None] == elem_labels).argmax(axis=0)
    elem_conn = self.__conns[class_sname][indices]
    return self.__labels['node'][elem_conn]

  def nodes_of_material( self, mat ):
    ''' Find nodes associated with a material number '''
    if mat not in self.__mats and mat not in self.__elems_of_mat:
      raise ValueError( f'There is no material {mat}' )
    element_labels = self.all_labels_of_material( mat )
    node_labels = np_empty(np.int32)
    for class_name, class_labels in element_labels.items():
      node_labels = np.append( node_labels, self.nodes_of_elems( class_name, class_labels ) )
    return np.unique( node_labels )

  def __init_query_parameters( self,
                               svar_names : Union[List[str],str],
                               class_sname : str,
                               material : Optional[Union[str,int]] = None,
                               labels : Optional[Union[List[int],int]] = None,
                               states : Optional[Union[List[int],int]] = None,
                               ips : Optional[Union[List[int],int]] = None,
                               write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None ):
    """
      Parse the query parameters and normalize them to the types expected by the rest of the query operation,
        throws TypeException and/or ValueException as appropriate. Does not throw exceptions in cases where
        the result of the argument deviation from the norm would be expected to be encountered when operating in parallel.

    """
    if states is None:
      states = np.arange( 1, len(self.__smaps) + 1, dtype = np.int32 )

    if type(states) is int:
      states = np.array( [ states ], dtype = np.int32 )
    elif iterable( states ) and not type( states ) == str:
      states = np.array( states, dtype = np.int32 )

    if not isinstance( states, np.ndarray ):
      raise TypeError( f"'states' must be None, an integer, or a list of integers" )

    if iterable( labels ):
      labels = np.array( labels, dtype = np.int32 )
    elif labels is not None:
      labels = np.array( [ labels ], dtype = np.int32 )

    # Filter labels queried by material if provided, or select all labels of given material if no other labels are given
    if material is not None:
      mat_labels = self.class_labels_of_material(material, class_sname)
      if labels is not None:
        labels = np.intersect1d( labels, mat_labels )
      else:
        labels = mat_labels

    if labels is None:
      labels = self.__labels.get( class_sname, np.empty([0],np.int32) )

    if type( svar_names ) is not str and iterable(svar_names):
      svar_names = list( svar_names )
    elif type( svar_names ) is str:
      svar_names = [ svar_names ]

    if svar_names is None:
      raise TypeError( 'State variable names must be a string or iterable of strings' )

    if type(ips) is int:
      ips = [ips]
    if ips is None:
      ips = []

    if type(ips) is not list:
      raise TypeError( 'ip must be an integer or list of integers' )

    if any( sn < 0 for sn in states ) or any( sn > len(self.__smaps) for sn in states ):
      raise ValueError(f'state numbers outside of range [0,{len(self.__smaps)}] requested but not in database')

    if write_data is not None:
      for queried_name in svar_names:
        if write_data[queried_name]['layout']['states'] != states:
          raise ValueError( f"When writing data to a database, the write_data must have the same states as the query.")

    return svar_names, class_sname, material, labels, states, ips, write_data

  @profile
  def query( self,
             svar_names : Union[List[str],str],
             class_sname : str,
             material : Optional[Union[str,int]] = None,
             labels : Optional[Union[List[int],int]] = None,
             states : Optional[Union[List[int],int]] = None,
             ips : Optional[Union[List[int],int]] = None,
             write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None ):
    '''
    Query the database for svars, returning data for the specified parameters, optionally writing data to the database.
    The parameters passed to query can be identical across a parallel invocation of query, since each individual
      database query object will filter only for the subset of the query it has available.

    : param svar_names : short names for state variables being queried
    : param class_sname : mesh class name being queried
    : param material : sname or material number to select labels from, optional
    : param labels : labels to query data about, filtered by material if material if material is supplied
    : param state_numbers: state numbers from which to query data
    : param ips : integration points (really just indices into svars that have array aggregation) to specifically query
    : param write_data : the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format, 
                   then modify the values desired, then query again with the modified result in this param
    '''

    # normalize arguments to expected types / default values
    svar_names, class_sname, material, labels, states, ips, write_data = self.__init_query_parameters( svar_names, class_sname, material, labels, states, ips, write_data )

    res = dict.fromkeys( svar_names )
    for ikey in res.keys():
      res[ikey] = { 'data' : {}, 'layout' : { 'states' : states } }

    # for parallel operation it will often be the case a specific file doesn't have any labels
    labels_of_class = self.__labels.get( class_sname, np_empty(np.int32) )
    ordinals_we_have = np.where( np.isin( labels_of_class, labels ) )[0]
    if ordinals_we_have.size == 0:
      return res

    for queried_name in svar_names:
      svar_name, comp_svar_names = self.__parse_query_name( queried_name )

      if not svar_name in self.__svars.keys():
        raise ValueError( f"No state variable '{svar_name}' found in database." )
      for comp_svar_name in comp_svar_names:
        if not comp_svar_name in self.__svars.keys():
          raise ValueError( f"No state variable '{comp_svar_name}' found in database." )

      # if svar is an aggregate svar and we're not querying specific comps, query all the comps
      if comp_svar_names == []:
        if self.__svars[svar_name].agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
          svars_to_query = self.__svars[svar_name].svars
        else:
          svars_to_query = [ self.__svars[ svar_name ] ]
      else:
        svars_to_query = [ self.__svars[ comp_svar_name ] for comp_svar_name in comp_svar_names ]

      # discover subrecords holding the queried svars, should precompute this if it becomes a bottleneck
      srecs_to_query : List[Subrecord] = []
      for svar in svars_to_query:
        # filter so we only have those subrecords with the appropriate class_sname
        srecs_with_svar_and_class = [ srec for srec in svar.srecs if srec.class_name == class_sname ]
        for srec in srecs_with_svar_and_class:
          if srec not in srecs_to_query:
            srecs_to_query.append( srec )

      res[queried_name]['layout'].update( { srec.name : np_empty(np.int32) for srec in srecs_to_query } )

      filtered_write_data = copy.deepcopy( write_data )
      if write_data is not None and ordinals_we_have.size != labels.size:
      #   # if we don't have all the labels queried and we're writing data, we potentially need
      #   #  to filter for the subset of the labels we *do* have locally
        for srec in srecs_to_query:
          local_write_labels = labels_of_class[ ordinals_we_have ]
          rows_to_write = np.where( np.isin( local_write_labels,  filtered_write_data[queried_name]['layout'][srec.name] ) )[0]
          filtered_write_data[queried_name]['layout'][srec.name] = local_write_labels
          filtered_write_data[queried_name]['data'][srec.name] = filtered_write_data[queried_name]['data'][srec.name][ :, rows_to_write, : ]

        # if we're looking for ip data determine which int_points we can get for each svar in each subrec
      matching_int_points = dict()
      if len( ips ) > 0:
        for svar in svars_to_query:
          comp_svar_name = svar.name
          candidate_ip_svars = list( self.__int_points.get( comp_svar_name, [] ) )
          for candidate in candidate_ip_svars:
            candidate_srecs = self.__svars[ candidate ].srecs
            for srec in candidate_srecs:
              if srec.class_name == class_sname:
                matching_int_points[ comp_svar_name ] = [ self.__int_points[comp_svar_name][candidate].index(ip) for ip in ips ]

      # For each subrecord, determine which elements (labels) appear in that subrecord
      # and create a dictionary entry for each subrecord that contains the labels in that
      # subrecord and the indexes at which the data for that label appears in the subrecord
      srecs_ordinal_indices = {} # { subrecord name --> [ label_cols, index_cols... ] }
      for srec in srecs_to_query:

        #  determine if any of the queried svar components are in the subrecord.. and extract only the ips we're querying for
        # col 0 is supposed to be the svar, col 2 is supposed to be the comp in the svar for agg svars
        # TODO: technically we can arbitrarily nest svars, this really only accounts for a two-deep nesting, same with the srec.svar_svar_comp_layout...
        #        also we're searching by svar sname which isn't technically incorrect, but could become slow if many svars are in subrecs
        qd_svar_comps = np.empty([0,2],dtype=np.int64)
        for svar in svars_to_query:
          matches = np.argwhere( svar.name == srec.svar_svar_comp_layout )
          ips = matching_int_points.get( svar.name, None )
          if ips is not None:
            matches = matches[ips]
          qd_svar_comps = np.concatenate( ( qd_svar_comps, matches ), axis = 0 )

        # discover which labels we want to query are in the current subrecord
        # TODO : can we eliminate this if/else, will digitize with no bins return empty?
        if len(srec.ordinal_blocks) != 0:
          # at this point we use the subrecord mo_blocks to find which block each label belongs to using a bining algorithym from numpy
          #   we have [start1, stop1, start2, stop2, etc], but digitize operates as [ start1, start2, start3, etc...] where start2 = stop1
          #   can't just ::2 the blocks and use that to digitize, since we need to exclude labels falling OUTSIDE the upper limit
          # this is currently where basically ALL the time cost for calculating labels/indices for a query
          #   and there isn't much to do to improve performance at the python level unless there is a better numpy/pandas algo to apply
          #   all labels will be in bins, but only odd # bins are actual bins, even bins are between the block ranges so mask to only grab rows with odd bins
          # THE ORDINAL BLOCKS MUST BE MONOTONICALLY INCREASING FOR THIS TO WORK, since they denote local ordinals this should always be the case
          #  if not we need to order them on read
          srec_ordinal_bins = np.digitize( ordinals_we_have, srec.ordinal_blocks )
          # eliminate ordinals where the bin is not odd (not in this srec)
          srec_ordinals = ordinals_we_have[ ( srec_ordinal_bins & 0x1 ).astype( bool ) ]
        else:
          # case for global values
          srec_ordinals = np_empty(np.int64)

        srec_ordinal_indices = np.empty( [ srec_ordinals.size, qd_svar_comps.shape[0] ], dtype = np.int64 )
        srec_ordinal_indices[:,0] = srec_ordinals

        res[queried_name]['layout'][srec.name] = self.__labels[class_sname][ srec_ordinals ]

        # we work from the last column / svar to the first the first contains the ordinal info needed to compute the rest (in the RESULT organization case.. so we just do the same in both cases)
        if srec.organization == Subrecord.Org.OBJECT:
          for idx_col, comp in zip( srec_ordinal_indices.T[::-1,:], qd_svar_comps[::-1,:]):
            idx_col[:] = srec_ordinals * srec.atoms_per_label + srec.svar_atom_offsets[ comp[0] ] + comp[1]
        elif srec.organization == Subrecord.Org.RESULT:
          for idx_col, comp in zip( srec_ordinal_indices.T[::-1,:], qd_svar_comps[::-1,:]):
            idx_col[:] = srec.svar_atom_offsets[ comp[0] ] * srec.total_ordinal_count + srec.svar_atom_lengths[ comp[1] ] * srec_ordinal_indices[:,0] + comp[1]
        srecs_ordinal_indices[srec.name] = srec_ordinal_indices

      # filter out any subrecords we have no labels for
      srecs_to_query = [ srec for srec in srecs_to_query if srecs_ordinal_indices[srec.name].size != 0 ]
      if not srecs_to_query:
        break

      # initialize the results structure for this queried name
      for srec in srecs_to_query:
        # TODO: this assumes one dtype over the srec, need to account for multiple dtypes
        res_shape = [ len(states), *srecs_ordinal_indices[srec.name].shape ]
        res[queried_name]['data'][srec.name] = np.empty( res_shape, dtype = np.float32 )

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

            for srec in srecs_to_query:
              ordinal_indices = srecs_ordinal_indices[srec.name]
              state_file.seek( state_offset + srec.state_byte_offset )
              byte_read_data = state_file.read( srec.byte_size )

              if filtered_write_data is not None:
                var_data, byte_write_data = srec.extract_ordinals( byte_read_data, ordinal_indices, write_data = filtered_write_data[ queried_name ][ 'data' ][ srec.name ][ sidx,:,: ] )
                state_file.seek( state_offset + srec.state_byte_offset )
                state_file.write( byte_write_data )
              else:
                var_data, _ = srec.extract_ordinals( byte_read_data, ordinal_indices )

              res[ queried_name ][ "data" ][ srec.name ][ sidx,:,: ] = var_data

    return res

@profile
def open_database( base_filename : os.PathLike, procs = [], suppress_parallel = __suppress_parallel__, experimental = False ):
  """
  Args: 
   base_file (os.PathLike):
   procs (Optional[List[int]]) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Optional[Bool]) : optionally return a serial database reader object if possible
  """
  # ensure dir_name is the containing dir and base_filename is only the file name
  dir_name = os.path.dirname( base_filename )
  if dir_name == '':
    dir_name = os.getcwd()
  if not os.path.isdir( dir_name ):
    raise ValueError( f"Cannot locate mili file directory {dir_name}.")

  base_filename = os.path.basename( base_filename )

  afile_re = re.compile( re.escape(base_filename) + f"(\d*)A$" )
  afiles = list(filter(afile_re.match,os.listdir(dir_name)))
  afiles.sort()

  proc_from_file = lambda fn : int( proc ) if (proc := afile_re.match(fn).group(1)) != '' else None
  # determine which attribute files to remove ( if any )
  procs_we_have = [ int( proc ) for afile in afiles if ( proc := afile_re.match(afile).group(1) ) != '' ]
  procs = [ int(proc) for proc in procs ]
  procs = procs_we_have if len(procs) == 0 else procs
  procs_to_drop = list( set(procs_we_have) - set(procs) )
  afiles = [ afile for afile in afiles if proc_from_file(afile) not in procs_to_drop ]

  if len(afiles) == 0:
    raise ValueError(f"No attribute files for procs {procs} with base name '{base_filename}' discovered in {dir_name}!")

  sfiles = [ base_filename + afile_re.match(afile).group(1) for afile in afiles ]
  dir_names = [ dir_name ] * len(sfiles)

  proc_pargs = [ [afile,sfile,dir_name] for afile, sfile, dir_name in zip(afiles,sfiles,dir_names) ]
  if suppress_parallel:
    if len(proc_pargs) == 1:
      mili_database = MiliDatabase( *proc_pargs[0] )
    else:
      mili_database = LoopWrapper( MiliDatabase, proc_pargs )
  else:
    if experimental:
      mili_database = ServerWrapper( MiliDatabase, proc_pargs )
    else: 
      mili_database = PoolWrapper( MiliDatabase, proc_pargs )
  return mili_database
