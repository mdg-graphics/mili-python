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

from contextlib import ExitStack
import functools
import io
import inspect
import logging
import os
import re
import struct
from typing import *
import typing
if hasattr( typing, 'io' ):
  from typing.io import * # type: ignore

from collections import defaultdict
from mili.datatypes import *
from mili.parallel import *

def getdatamembers( cls ):
  return [ member for member in inspect.getmembers(cls) if not ( inspect.ismethod(member) or member[0].startswith('__') ) ]

def afiles_by_base( dir_name, base_filename, proc_whitelist = [] ):
  file_re = re.compile( re.escape(base_filename) + r"(\d*)A$" )
  files = list(filter(file_re.match,os.listdir(dir_name)))
  files.sort()

  def proc_from_file( fn ):
    proc = file_re.match(fn).group(1)
    return int( proc ) if proc != '' else None

  procs_we_have = list( proc_from_file(file) for file in files )
  procs_we_have = list( filter( lambda val : val != None, procs_we_have ) )
  proc_whitelist = [ int(proc) for proc in proc_whitelist ]
  proc_whitelist = procs_we_have if len(proc_whitelist) == 0 else proc_whitelist
  # drop unspecified proc A-files from the set of A-files we have
  to_drop = list( set(procs_we_have) - set(proc_whitelist) )

  # TODO : ensure we drop sfiles with different names but the same basename+proc
  files = [ file for file in files if proc_from_file(file) not in to_drop ]

  if len(files) == 0:
    raise ValueError(f"No A-files for procs '{', '.join(proc_whitelist)}' with base name '{base_filename}' discovered in {dir_name}!")
  return files

class MiliAParseError(Exception):
  pass


class AFileReader:
  """
    Incrementally traverse the sections / directories in an AFile.
    During traverse call user-registered callbacks with the binary of each section/directory.
    When directories are being processed, the directory declaration parsed into a Directory object is also passed to the callback.
    Only directly parses information and retains state required to continue traversing the file.
  """
  def __init__(self):
    self.__callbacks = defaultdict(list)
    self.__dir_decls_list = []
    self.__dir_decls = defaultdict(list)
    self.register( AFile.Section.HEADER, self.__parse_header )
    self.register( AFile.Section.FOOTER, self.__parse_footer )
    self.register( AFile.Section.DIR_DECL, self.__parse_dir )
    self.register( AFile.Section.STRINGS, self.__parse_strings )

  def __parse_header( self, header_data : bytes ):
    self.__mili_file_version = int(header_data[4])
    self.__dir_dtype = 'i' if int(header_data[7]) == 1 else 'q'
    self.__endian_str = 'big' if int(header_data[8]) == 1 else 'little'

  def __parse_footer( self, footer_data : bytes ):
    self.__string_bytes = int.from_bytes( footer_data[0:3], byteorder = self.__endian_str )

  def __parse_dir( self, dir_data : bytes ):
    ddecl = DirectoryDecl( *struct.unpack(f'<6{self.__dir_dtype}',dir_data ) )
    ddecl.dir_type = DirectoryDecl.Type(ddecl.dir_type)
    self.__dir_decls_list.append( ddecl )
    self.__dir_decls[ddecl.dir_type].append( ddecl )

  def __parse_strings( self, strings_data : bytes ):
    strings = str(struct.unpack( f'>{self.__string_bytes}s', strings_data)[0])[2:].split('\\x00')
    str_offset = 0
    # assign string to dirs (needs to be ordered)
    for dir in self.__dir_decls_list:
      dir.strings = strings[ str_offset : str_offset + dir.str_cnt ]
      str_offset += dir.str_cnt

  def __callback( self, key : Union[AFile.Section,DirectoryDecl.Type], *pargs, **kwargs ) -> None:
    for cback in self.__callbacks.get(key, []):
      cback(*pargs,**kwargs)

  def __have_callback( self, key : Union[AFile.Section,DirectoryDecl.Type] ) -> bool:
    return key in self.__callbacks.keys()

  def __read_successive( self, f : BinaryIO, key : Union[AFile.Section,DirectoryDecl.Type], num_items : int, item_bytes : int ) -> None:
    ''' Read a buffer of num_items * item_bytes and iterate over it in item_bytes slices, calling the 'key' callback. '''
    data = f.read( num_items * item_bytes )
    byte_data = io.BytesIO( data )
    item_data = byte_data.read( item_bytes )
    while item_data:
      self.__callback( key, item_data )
      item_data = byte_data.read( item_bytes )

  def register( self, key : Union[AFile.Section,DirectoryDecl.Type], cback : Callable[[bytearray],None]) -> None:
    self.__callbacks[key].append(cback)

  def read( self, f : BinaryIO, t : BinaryIO ) -> None:
    header = f.read(16)
    self.__callback( AFile.Section.HEADER, header )
    dirs_bytes = 24 if header[5] == 1 else 48

    if self.__mili_file_version > 2:
      if t is None:
        raise MiliAParseError(f"Expected T-file for mili file format v{self.__mili_file_version} not found!")

    f.seek(-16, os.SEEK_END)
    footer = f.read(16)

    self.__callback( AFile.Section.FOOTER, footer )
    strings_bytes, _, num_dirs, num_smaps = struct.unpack( '4i', footer )

    if self.__mili_file_version > 2:
      t.seek( 0, os.SEEK_END )
      num_smaps = (t.tell() - 1) // 20 # tmap is an array of smaps plus a trailing '~'
      t.seek( 0, os.SEEK_SET )
      struct.pack('i',int.from_bytes(footer[11:15], byteorder=self.__endian_str))

    smap_offset = -16 # footer
    if self.__mili_file_version > 2:
      self.__read_successive( t, AFile.Section.STATE_MAP, num_smaps, 20 )
      check = t.read(1)
      if check != b'~':
        logging.info(f"TFile ended with unexpected value {check}. Possible corruption.")
    else:
      smap_offset -= ( num_smaps * 20 ) # num_smaps * smap_bytes
      f.seek( smap_offset, os.SEEK_END )
      self.__read_successive( f, AFile.Section.STATE_MAP, num_smaps, 20 )

    dirs_offset = smap_offset - ( num_dirs * dirs_bytes )
    f.seek( dirs_offset, os.SEEK_END )
    self.__read_successive( f, AFile.Section.DIR_DECL, num_dirs, dirs_bytes )

    strings_offset = dirs_offset - strings_bytes
    f.seek( strings_offset, os.SEEK_END )
    strings_data = f.read( strings_bytes )
    self.__callback( AFile.Section.STRINGS, strings_data )

  def read_dirs( self, f : BinaryIO, dir_type : DirectoryDecl.Type ):
    if self.__have_callback( dir_type ):
      for dd in self.__dir_decls.get(dir_type,[]):
        f.seek( dd.offset )
        self.__callback( dd.dir_type, f.read( dd.length ), dd )

class AFileParser:
  """
    Uses the AFileReader to register parsing callbacks for all sections and directories in an A-file.
    This parses ALL information in the AFile, passed via callback for the AFileReader, and populates a new AFile object.
  """
  def __init__(self, allow_exceptions = False):
    self.__except = allow_exceptions

  def parse( self, afile : AFile,  base : os.PathLike, dir_whitelist = None ):
    if dir_whitelist is None:
      # the whitelist order matters, we need to parse params prior to the mesh nodes since the dimensionality is set by the params (2d v 3d)
      dir_whitelist = [ DirectoryDecl.Type.MILI_PARAM,
                        DirectoryDecl.Type.APPLICATION_PARAM,
                        DirectoryDecl.Type.TI_PARAM,
                        DirectoryDecl.Type.STATE_VAR_DICT,
                        DirectoryDecl.Type.CLASS_IDENTS,
                        DirectoryDecl.Type.CLASS_DEF,
                        DirectoryDecl.Type.NODES,
                        DirectoryDecl.Type.ELEM_CONNS,
                        DirectoryDecl.Type.SREC_DATA ]

    dir_callbacks = { AFile.Section.HEADER : functools.partial(self.__parse_header, afile),
                      AFile.Section.FOOTER : functools.partial(self.__parse_footer, afile),
                      AFile.Section.STATE_MAP : functools.partial(self.__parse_smap, afile),
                      AFile.Section.DIR_DECL : functools.partial(self.__parse_dir_decl, afile),
                      AFile.Section.STRINGS : functools.partial(self.__parse_strings, afile),
                      DirectoryDecl.Type.MILI_PARAM : functools.partial(self.__parse_mili_param, afile),
                      DirectoryDecl.Type.APPLICATION_PARAM : functools.partial(self.__parse_app_param, afile),
                      DirectoryDecl.Type.TI_PARAM : functools.partial(self.__parse_ti_param, afile),
                      DirectoryDecl.Type.STATE_VAR_DICT : functools.partial(self.__parse_svars, afile),
                      DirectoryDecl.Type.CLASS_IDENTS : functools.partial(self.__parse_class_ident, afile),
                      DirectoryDecl.Type.CLASS_DEF : functools.partial(self.__parse_class_def, afile),
                      DirectoryDecl.Type.NODES : functools.partial(self.__parse_nodes, afile),
                      DirectoryDecl.Type.ELEM_CONNS : functools.partial(self.__parse_elem_conn, afile),
                      DirectoryDecl.Type.SREC_DATA : functools.partial(self.__parse_srec, afile) }

    reader = AFileReader( )
    for key, func in dir_callbacks.items( ):
      reader.register( key, func )

    self.__all_valid = True
    self.afilename = base + 'A'
    self.tfilename = base + 'T'
    with ExitStack() as stack:
      af = stack.enter_context( open(self.afilename,'rb') )
      tf = None
      if os.path.isfile( self.tfilename ):
        tf = stack.enter_context( open( self.tfilename, 'rb' ) )
      reader.read( af, tf )
      for dir_type in dir_whitelist:
        reader.read_dirs( af, dir_type )
    rval = self.__all_valid
    self.__all_valid = True
    return rval

  def verify( self, name, value, valid = None ):
    if valid is None:
      valid = lambda x : True
    is_valid = valid( value )
    if not is_valid:
      self.__all_valid = False
    log_str = f"{self.afilename}\n{name}\n  validity = {is_valid}\n  value = {value}\n  validator = '{inspect.getsource(valid).strip()}'\n"
    logging.debug(log_str)
    if not is_valid and self.__except:
      raise MiliAParseError( f"invalid value encounterd:\n  {log_str}" )
    return value

  def __parse_header( self, afile : AFile, header_data : bytes ):
    if len( header_data ) != 16 :
      raise MiliAParseError( "Header data is not 16 bytes. This should never be seen unless something is wrong with I/O itself.")
    formats = [ dmem[1] for dmem in getdatamembers( AFile.Format ) ]
    afile.file_format = self.verify('header/file_format', str(header_data[0:4].decode('utf-8')), lambda x : x in formats )
    version_validator = lambda x : AFile.Version.MIN <= x <= AFile.Version.MAX
    afile.file_version = self.verify('header/file_version', int(header_data[4]), version_validator )
    directory_version_validator = lambda x : AFile.DirectoryVersion.MIN <= x <= AFile.DirectoryVersion.MAX
    afile.directory_version = self.verify('header/directory_verion', int(header_data[5]), directory_version_validator )
    self.__dir_dtype = 'i' if afile.directory_version == 1 else 'q'
    endians = [ dmem[1] for dmem in getdatamembers( AFile.Endian ) ]
    afile.endian_flag = self.verify('header/endian_flag', int(header_data[6]), lambda x : x in endians)
    self.__endian_str = 'big' if afile.endian_flag == 1 else 'little'
    precisions = [ dmem[1] for dmem in getdatamembers( AFile.Precision ) ]
    afile.precision_flag = self.verify('header/precision_flag', int(header_data[7]), lambda x : x in precisions )
    afile.sfile_suffix_length = self.verify('header/sfile_suffix_length', int(header_data[8]) )
    afile.partition_scheme = self.verify('header/partition_scheme', int(header_data[9]) )

  def __parse_footer( self, afile : AFile, footer_data : bytes ):
    if len(footer_data) != 16 :
      raise MiliAParseError( "Footer data is not 16 bytes. This should never be seen unless something is wrong with I/O itself.")
    afile.string_bytes = self.verify( 'footer/string_bytes', int.from_bytes( footer_data[0:3], byteorder = self.__endian_str ) )
    afile.commit_count = self.verify( 'footer/commit_count', int.from_bytes( footer_data[4:7], byteorder = self.__endian_str ) )
    afile.directory_count = self.verify( 'footer/directory_count', int.from_bytes( footer_data[8:11], byteorder = self.__endian_str ) )
    afile.srec_count = self.verify( 'footer/srec_count', int.from_bytes( footer_data[12:15], byteorder = self.__endian_str ) )

  def __parse_smap( self, afile : AFile, smap_data : bytes ):
    smap = StateMap( *struct.unpack('<iqfi',smap_data) )
    idx = len(afile.smaps)
    self.verify( f"smap[{idx}].sfile_number", smap.file_number )
    self.verify( f"smap[{idx}].sfile_offset", smap.file_offset )
    self.verify( f"smap[{idx}].time", smap.time )
    self.verify( f"smap[{idx}].srec_id", smap.state_map_id )
    afile.smaps.append( smap )

  def __parse_dir_decl( self, afile : AFile, dir_data : bytes ):
    directory = DirectoryDecl( *struct.unpack(f'<6{self.__dir_dtype}',dir_data ) )
    directory.dir_type = DirectoryDecl.Type( directory.dir_type )
    idx = len( afile.dir_decls_list )
    dir_types = [ dmem[1] for dmem in getdatamembers(DirectoryDecl.Type) ]
    self.verify( f"dir_decls_list[{idx}].dir_type", directory.dir_type, lambda x : x in dir_types )
    self.verify( f"dir_decls_list[{idx}].modifier_idx1", directory.modifier_idx1)
    self.verify( f"dir_decls_list[{idx}].modifier_idx2", directory.modifier_idx2)
    self.verify( f"dir_decls_list[{idx}].string_count", directory.str_cnt)
    self.verify( f"dir_decls_list[{idx}].offset", directory.offset)
    self.verify( f"dir_decls_list[{idx}].length", directory.length)
    afile.dir_decls_list.append( directory )
    afile.dir_decls[ directory.dir_type ].append( directory )

  def __parse_strings( self, afile : AFile, strings_data : bytes ):
    strings = str(struct.unpack( f'>{afile.string_bytes}s', strings_data)[0])[2:].split('\\x00')
    str_offset = 0
    for idx, s in enumerate(strings):
      self.verify( f"strings[{idx}]", s )
    # assign string to dirs (needs to be ordered)
    for dir in afile.dir_decls_list:
      dir.strings = strings[ str_offset : str_offset + dir.str_cnt ]
      str_offset += dir.str_cnt
    afile.strings = strings

  def __parse_mili_param( self, afile : AFile, param_data : bytes, dir_decl : DirectoryDecl ):
    sname = dir_decl.strings[0]
    if sname in [ "mesh dimensions", "states per file", "mesh type" ]:
      # single int params
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'MILI_PARAM/{sname}', int.from_bytes(param_data, byteorder = self.__endian_str ) )
    else:
      # default to strings
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'MILI_PARAM/{sname}', struct.unpack(f"{dir_decl.length}s", param_data)[0].decode('utf-8').split('\x00') )

  def __parse_app_param( self, afile : AFile, param_data : bytes, dir_decl : DirectoryDecl ):
    sname = dir_decl.strings[0]
    if sname in [ "state_count", "OPEN_FOR_WRITE", "max size per file" ] :
      # single int params
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'APPLICATION_PARAM/{sname}', int.from_bytes(param_data, byteorder = self.__endian_str ) )
    elif sname == "title":
      # strings
      param = struct.unpack(f"{dir_decl.length}s", param_data)[0].decode('utf-8')
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'APPLICATION_PARAM/{sname}', param )
    else:
      # default to just byte data
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'APPLICATION_PARAM/{sname}', param_data )

  def __parse_ti_param( self, afile : AFile, param_data : bytes, dir_decl : DirectoryDecl ):
    sname = dir_decl.strings[0]
    if "Labels" in sname or sname.startswith('GLOBAL_IDS') or sname.startswith('IntLabel_es_'):
      # int arrays
      param = np.frombuffer( param_data[8:], dtype = np.int32 )
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'TI_PARAM/{sname}', param )
    elif sname.startswith("SetRGB"):
      # float arrays
      param = np.frombuffer( param_data[8:], dtype = np.float32 )
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'TI_PARAM/{sname}', param )
    elif sname.startswith('GLOBAL_COUNT') or sname.startswith('LOCAL_COUNT') or sname == "nproc":
      # single int params
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'TI_PARAM/{sname}', int.from_bytes(param_data, byteorder = self.__endian_str ) )
    else:
      # default to string
      param = struct.unpack(f"{dir_decl.length}s", param_data)[0].decode('utf-8').split('\x00')[0]
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'TI_PARAM/{sname}', param )

  def __parse_svars( self, afile : AFile, svar_data : bytes, dir_decl : DirectoryDecl ):
    header_bytes = 8
    svar_words, svar_bytes = struct.unpack('2i', svar_data[:header_bytes])
    int_cnt = svar_words - 2
    int_bytes = int_cnt * 4

    int_data = np.frombuffer( svar_data[ header_bytes : header_bytes + int_bytes ], np.int32 )
    strings = struct.unpack(f'{svar_bytes}s', svar_data[ header_bytes + int_bytes : ])[0].decode('utf-8').split('\x00')
    strings = list(filter(('').__ne__,strings))

    iidx = 0
    sidx = 0
    while iidx < len(int_data):
      iidx, sidx = self.__parse_svar( afile, iidx, int_data, sidx, strings, dir_decl )

  def __parse_svar( self, afile : AFile, iidx : int, int_data, sidx : int, strings, dir_decl ):
    agg_type = int_data[ iidx ]
    sname = strings[ sidx ]
    kwargs = { 'name' : sname,
               'title' : strings[ sidx + 1 ],
               'data_type' : MiliType(int_data[ iidx + 1 ]),
               'agg_type' : agg_type }
    iidx += 2
    sidx += 2
    if agg_type in [ StateVariable.Aggregation.ARRAY, StateVariable.Aggregation.VEC_ARRAY ]:
      kwargs['order'] = order = int_data[ iidx ]
      kwargs['dims'] = int_data[ iidx + 1 : iidx + 1 + order ]
      iidx += 1 + order
    if agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
      kwargs['list_size'] = list_size = int_data[ iidx ]
      kwargs['comp_names'] = comp_snames = strings[ sidx : sidx + list_size ]
      iidx += 1
      sidx += list_size
      for comp_sname in comp_snames:
        if comp_sname not in afile.dirs[dir_decl.dir_type].keys():
          iidx, sidx = self.__parse_svar( afile, iidx, int_data, sidx, strings, dir_decl )
    # if sname == "stress":
    #   breakpoint()
    afile.dirs[dir_decl.dir_type][sname] = self.verify( f'STATE_VAR_DICT/{sname}', StateVariable( **kwargs ) )
    return iidx, sidx

  def __parse_class_ident( self, afile : AFile, class_data : bytes, dir_decl : DirectoryDecl ):
    sname = dir_decl.strings[0]
    if sname not in afile.dirs[dir_decl.dir_type].keys():
      afile.dirs[dir_decl.dir_type][sname] = np.empty([0],np.int32)
    append_to = afile.dirs[dir_decl.dir_type][sname]
    append = np.frombuffer( class_data[4:12], dtype = np.int32 )
    afile.dirs[dir_decl.dir_type][sname] = self.verify( f'CLASS_IDENTS/{sname}', np.append( append_to, append ) )

  def __parse_class_def( self, afile : AFile, _ : bytes, dir_decl : DirectoryDecl ):
    short_name = dir_decl.strings[0]
    long_name = dir_decl.strings[1]
    mesh_id = dir_decl.modifier_idx1
    sclass = Superclass(dir_decl.modifier_idx2)
    afile.dirs[dir_decl.dir_type][short_name] =  self.verify( f'CLASS_DEF/{short_name}', MeshObjectClass(mesh_id, short_name, long_name, sclass) )

  def __parse_nodes( self, afile : AFile, nodes_data : bytes, dir_decl : DirectoryDecl ):
    sname = dir_decl.strings[0]
    f = io.BytesIO( nodes_data )
    blocks = np.reshape( np.frombuffer( f.read(8), dtype = np.int32 ), [-1,2] )
    blocks_atoms = np.sum( np.diff( blocks, axis=1 ).flatten( ) + 1 )
    if sname not in afile.dirs[dir_decl.dir_type]:
      afile.dirs[dir_decl.dir_type][sname] = np.empty( [0], np.float32 )
    # append_to = afile.dirs[dir_decl.dir_type][sname]
    mesh_dim = afile.dirs[DirectoryDecl.Type.MILI_PARAM]["mesh dimensions"]
    append = np.reshape( np.frombuffer( f.read( 4 * blocks_atoms * mesh_dim ), dtype = np.float32 ), [-1, mesh_dim] )
    afile.dirs[dir_decl.dir_type][sname] = self.verify( f'NODES/{sname}', append )

  def __parse_elem_conn( self, afile : AFile, conn_data : bytes, dir_decl : DirectoryDecl ):
    f = io.BytesIO( conn_data )
    sname = dir_decl.strings[0]
    sclass, block_cnt = struct.unpack( '2i', f.read(8) )
    f.read( 8 * block_cnt )
    elem_qty = dir_decl.modifier_idx2
    conn_qty = Superclass(sclass).node_count()
    word_qty = conn_qty + 2
    conn = np.reshape( np.frombuffer( f.read( elem_qty * word_qty * 4 ), dtype = np.int32 ), [-1,word_qty] )
    if not sname in afile.dirs[dir_decl.dir_type].keys():
      afile.dirs[dir_decl.dir_type][sname] = np.empty( (0,word_qty), dtype = np.int32 )
    afile.dirs[dir_decl.dir_type][sname] = np.concatenate( ( afile.dirs[dir_decl.dir_type][sname], self.verify( f'ELEM_CONN/{sname}', conn ) ) )

  def __parse_srec( self, afile : AFile, srec_data : bytes, dir_decl : DirectoryDecl ):
    srec_count = int.from_bytes( srec_data[12:16], self.__endian_str )
    srec_header_bytes = 16
    srec_idata_bytes = ( dir_decl.modifier_idx1 - 4 ) * 4
    srec_strdata_bytes = dir_decl.modifier_idx2
    int_data = np.frombuffer( srec_data[ srec_header_bytes : srec_header_bytes + srec_idata_bytes ], np.int32 )
    strings = [ name.rstrip('\x00') for name in struct.unpack(str(srec_strdata_bytes) + 's', srec_data[ srec_header_bytes + srec_idata_bytes : ])[0].decode('utf-8').split('\x00') ]
    strings.remove('')
    sidx = iidx = 0
    for _ in range( srec_count ):
      sname = strings[ sidx + 0 ]
      class_name = strings[ sidx + 1 ]
      qty_svars = int_data[ iidx + 1 ]
      block_count = int_data[ iidx + 2 ]
      kwargs = { 'name' : sname,
                 'class_name' : class_name,
                 'organization' : int_data[ iidx + 0 ],
                 'qty_svars' : qty_svars,
                 'svar_names' : strings[ sidx + 2 : sidx + 2 + qty_svars ] }
      sidx += 2 + qty_svars
      iidx += 3
      # TODO: slow ( how to solve without adding state to the parser? )
      superclass = -1
      for class_def in afile.dir_decls[ DirectoryDecl.Type.CLASS_DEF ]:
        if class_name == class_def.strings[ 0 ]:
          superclass = class_def.modifier_idx2

      ordinal_blocks = np.array( [0,1], dtype = np.int64 )

      if superclass != Superclass.M_MESH:
        ordinal_blocks = np.array( int_data[ iidx : iidx + ( 2 * block_count ) ], dtype = np.int32 )
        ## Need to either account for these on write, or move them into just the reader/querier state (which would increase memory usage)
        ordinal_blocks[1::2] = ordinal_blocks[1::2] + 1 # add one to each STOP to allow inclusive bining during queries..
        ordinal_blocks = ordinal_blocks - 1 # account for 1-indexing used by mili(?)

      ordinal_block_counts = np.concatenate( ( [0], np.diff( ordinal_blocks.reshape(-1,2), axis=1 ).flatten( ) ) ) # sum stop - start + 1 over all blocks
      ordinal_block_offsets = np.cumsum( ordinal_block_counts )

      # Check that ordinal blocks are monotonically increasing
      monotonically_increasing = np.all( ordinal_blocks[1:] >= ordinal_blocks[:-1] )
      if not monotonically_increasing:
        # Sort ordinal_blocks and ordinal_block offsets together
        ordinal_blocks = np.reshape( ordinal_blocks, (-1,2) )
        blocks, offsets = zip( *(sorted( zip(ordinal_blocks, ordinal_block_offsets[:-1]), key=lambda x: x[0][0]) ) )
        ordinal_blocks = np.array( blocks ).flatten()
        ordinal_block_offsets = np.array( offsets )
        # Recalculate ordinal_block_counts
        ordinal_block_counts = np.concatenate( ( [0], np.diff( ordinal_blocks.reshape(-1,2), axis=1 ).flatten( ) ) ) # sum stop - start + 1 over all blocks

      kwargs['ordinal_blocks'] = ordinal_blocks
      kwargs['ordinal_block_counts'] = ordinal_block_counts
      kwargs['ordinal_block_offsets'] = ordinal_block_offsets
      iidx += 2 * block_count
      afile.dirs[dir_decl.dir_type][sname] = self.verify( f'STATE_REC_DATA/{sname}', Subrecord( **kwargs ) )

class AFileParallelHelper:
  def __init__( self, base : os.PathLike, allow_exceptions : bool = True ):
    self.__afile = AFile()
    self.__parser = AFileParser( allow_exceptions )
    self.__rval = self.__parser.parse( self.__afile, base )
  def afile( self ):
    return self.__afile
  def rval( self ):
    return self.__rval

def parse_database( base : os.PathLike, procs = [], suppress_parallel = False, experimental = False ) -> List[AFile]:
  """
   Open and parse a database. This only opens the database metadata files, and can be useful for verifying
   parallel databases are in valid and consistent states. The object returned by this function will have the
   same interface as an AFile object, though will return a list of results from the specified proc files
   instead of a single result.
  Args:
   base (os.PathLike): the base filename of the mili database (e.g. for 'pltA', just 'plt', for parallel
                            databases like 'dblplt00A', also exclude the rank-digits, giving 'dblplt')
   procs (Optional[List[int]]) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Optional[Bool]) : optionally return a serial database layout object if possible.
                                        Note: if the database is parallel, suppress_parallel==True will return a reader that will
                                        query each processes database files serially.
   experimental (Optional[Bool]) : optional developer-only argument to try experimental parallel features
  """
  # ensure dir_name is the containing dir and base is only the file name
  dir_name = os.path.dirname( base )
  if dir_name == '':
    dir_name = os.getcwd()
  if not os.path.isdir( dir_name ):
    raise ValueError( f"Cannot locate mili file directory {dir_name}.")
  base = os.path.basename( base )
  afiles = afiles_by_base( dir_name, base, procs )
  proc_bases = [ afile[:-1] for afile in afiles ] # drop the A to get each processes base filename for A,T, and S files
  proc_pargs = [ [os.path.join(dir_name,base_file)] for base_file in proc_bases ]
  parse_wrapper = get_wrapper( suppress_parallel, experimental )( AFileParallelHelper, proc_pargs )
  afiles = parse_wrapper.afile()
  rvals = parse_wrapper.rval()
  return afiles, rvals
