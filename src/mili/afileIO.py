"""
SPDX-License-Identifier: (MIT)
"""

from contextlib import ExitStack
import functools
import sys
import io
import inspect
import logging
import os
import re
import struct
import math
from typing import Literal, BinaryIO, Callable, Set, Mapping

import numpy as np
from numpy.typing import NDArray

from collections import defaultdict
from mili.datatypes import *
from mili.parallel import *

def getdatamembers(cls: Any) -> List[Any]:
  return [ member for member in inspect.getmembers(cls) if not ( inspect.ismethod(member) or member[0].startswith('__') ) ]

class MiliFileNotFoundError(Exception):
  pass

class MiliAParseError(Exception):
  pass


def afiles_by_base(dir_name: str, base_filename: str, proc_whitelist: List[int] = []) -> List[str]:
  file_re = re.compile( re.escape(base_filename) + r"(\d*)A$" )
  files = list(filter(file_re.match,os.listdir(dir_name)))
  files.sort()

  def proc_from_file(fn: str) -> Optional[int]:
    proc_match = file_re.match(fn)
    proc = proc_match.group(1) if proc_match is not None else ''
    return int( proc ) if proc != '' else None

  procs_we_have = list( proc_from_file(file) for file in files )
  procs_we_have = [proc for proc in procs_we_have if proc is not None]
  proc_whitelist = [ int(proc) for proc in proc_whitelist ]
  proc_whitelist = procs_we_have if len(proc_whitelist) == 0 else proc_whitelist  #  type: ignore  # mypy thinks procs_we_have is type List[int|None]
  # drop unspecified proc A-files from the set of A-files we have
  to_drop = list( set(procs_we_have) - set(proc_whitelist) )

  # TODO : ensure we drop sfiles with different names but the same basename+proc
  files = [ file for file in files if proc_from_file(file) not in to_drop ]

  if len(files) == 0:
    raise MiliFileNotFoundError(f"No A-files for procs '{', '.join([str(proc) for proc in proc_whitelist])}' "
                                f"with base name '{base_filename}' discovered in {dir_name}!")
  return files


class AFileReader:
  """
    Incrementally traverse the sections / directories in an AFile.
    During traverse call user-registered callbacks with the binary of each section/directory.
    When directories are being processed, the directory declaration parsed into a Directory object is also passed to the callback.
    Only directly parses information and retains state required to continue traversing the file.
  """
  def __init__(self) -> None:
    self.__callbacks: Dict[Union[AFile.Section,DirectoryDecl.Type],List[Callable[...,Any]]] = defaultdict(list)
    self.__dir_decls_list: List[DirectoryDecl] = []
    self.__dir_decls: Dict[DirectoryDecl.Type,List[DirectoryDecl]] = defaultdict(list)
    self.register( AFile.Section.HEADER, self.__parse_header )
    self.register( AFile.Section.FOOTER, self.__parse_footer )
    self.register( AFile.Section.DIR_DECL, self.__parse_dir )
    self.register( AFile.Section.STRINGS, self.__parse_strings )

  def __parse_header(self, header_data: bytes) -> None:
    self.__mili_file_version = int(header_data[4])
    self.__dir_dtype = 'i' if int(header_data[5]) <= 2 else 'q'
    self.__endian_str: Literal['big','little'] = 'big' if int(header_data[6]) == AFile.Endian.BIG else 'little'

  def __parse_footer(self, footer_data: bytes) -> None:
    self.__string_bytes = int.from_bytes( footer_data[0:3], byteorder = self.__endian_str )

  def __parse_dir(self, dir_data: bytes) -> None:
    ddecl = DirectoryDecl( *struct.unpack(f'<6{self.__dir_dtype}',dir_data ) )
    ddecl.dir_type = DirectoryDecl.Type(ddecl.dir_type)
    self.__dir_decls_list.append( ddecl )
    self.__dir_decls[ddecl.dir_type].append( ddecl )

  def __parse_strings(self, strings_data: bytes) -> None:
    strings = str(struct.unpack( f'>{self.__string_bytes}s', strings_data)[0])[2:].split('\\x00')
    str_offset = 0
    # assign string to dirs (needs to be ordered)
    for dir in self.__dir_decls_list:
      dir.strings = strings[ str_offset : str_offset + dir.str_cnt ]
      str_offset += dir.str_cnt

  def __callback(self, key : Union[AFile.Section,DirectoryDecl.Type], *pargs: Any, **kwargs: Any) -> None:
    for cback in self.__callbacks.get(key, []):
      cback(*pargs,**kwargs)

  def __have_callback(self, key :Union[AFile.Section,DirectoryDecl.Type]) -> bool:
    return key in self.__callbacks.keys()

  def __read_successive(self, f: BinaryIO, key: Union[AFile.Section,DirectoryDecl.Type], num_items: int, item_bytes: int ) -> None:
    ''' Read a buffer of num_items * item_bytes and iterate over it in item_bytes slices, calling the 'key' callback. '''
    data = f.read( num_items * item_bytes )
    byte_data = io.BytesIO( data )
    item_data = byte_data.read( item_bytes )
    while item_data:
      self.__callback( key, item_data )
      item_data = byte_data.read( item_bytes )

  def register(self, key: Union[AFile.Section,DirectoryDecl.Type], cback: Callable[[bytearray],None]) -> None:
    self.__callbacks[key].append(cback)

  def read(self, f: BinaryIO, t: Optional[BinaryIO]) -> None:
    header = f.read(16)
    self.__callback( AFile.Section.HEADER, header )
    dirs_bytes = 24 if header[5] <= 2 else 48

    f.seek(-16, os.SEEK_END)
    footer = f.read(16)

    self.__callback( AFile.Section.FOOTER, footer )
    strings_bytes, _, num_dirs, num_smaps = struct.unpack( '4i', footer )

    if self.__mili_file_version > 2 and t is not None:
      t.seek( 0, os.SEEK_END )
      num_smaps = (t.tell() - 1) // 20 # tmap is an array of smaps plus a trailing '~'
      t.seek( 0, os.SEEK_SET )
      struct.pack('i',int.from_bytes(footer[11:15], byteorder=self.__endian_str))

    smap_offset = -16 # footer
    if self.__mili_file_version > 2 and t is not None:
      self.__read_successive( t, AFile.Section.STATE_MAP, num_smaps, 20 )
      check = t.read(1)
      if check != b'~':
        logging.info(f"TFile ended with unexpected value {check!r}. Possible corruption.")
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

  def read_dirs(self, f: BinaryIO, dir_type: DirectoryDecl.Type ) -> None:
    if self.__have_callback( dir_type ):
      for dd in self.__dir_decls.get(dir_type,[]):
        f.seek( dd.offset )
        self.__callback( dd.dir_type, f.read( dd.length ), dd )

class AFileParser:
  """
    Uses the AFileReader to register parsing callbacks for all sections and directories in an A-file.
    This parses ALL information in the AFile, passed via callback for the AFileReader, and populates a new AFile object.
  """
  def __init__(self, allow_exceptions: bool = False, log_validator: bool = True) -> None:
    self.__except = allow_exceptions
    self.__log_validator = log_validator

  def parse( self, afile: AFile,  base: Union[str,os.PathLike], dir_whitelist: Optional[List[DirectoryDecl.Type]] = None ) -> bool:
    if dir_whitelist is None:
      # the whitelist order matters, we need to parse params prior to the mesh nodes since the dimensionality is set by the params (2d v 3d)
      dir_whitelist = [ DirectoryDecl.Type.MILI_PARAM,
                        DirectoryDecl.Type.APPLICATION_PARAM,
                        DirectoryDecl.Type.TI_PARAM,
                        DirectoryDecl.Type.CLASS_DEF,
                        DirectoryDecl.Type.CLASS_IDENTS,
                        DirectoryDecl.Type.NODES,
                        DirectoryDecl.Type.ELEM_CONNS,
                        DirectoryDecl.Type.STATE_VAR_DICT,
                        DirectoryDecl.Type.SREC_DATA ]

    dir_callbacks: Dict[Union[AFile.Section,DirectoryDecl.Type],Any]
    dir_callbacks  = { AFile.Section.HEADER : functools.partial(self.__parse_header, afile),
                      AFile.Section.FOOTER : functools.partial(self.__parse_footer, afile),
                      AFile.Section.STATE_MAP : functools.partial(self.__parse_smap, afile),
                      AFile.Section.DIR_DECL : functools.partial(self.__parse_dir_decl, afile),
                      AFile.Section.STRINGS : functools.partial(self.__parse_strings, afile),
                      DirectoryDecl.Type.MILI_PARAM : functools.partial(self.__parse_param, afile),
                      DirectoryDecl.Type.APPLICATION_PARAM : functools.partial(self.__parse_param, afile),
                      DirectoryDecl.Type.TI_PARAM : functools.partial(self.__parse_param, afile),
                      DirectoryDecl.Type.CLASS_IDENTS : functools.partial(self.__parse_class_ident, afile),
                      DirectoryDecl.Type.CLASS_DEF : functools.partial(self.__parse_class_def, afile),
                      DirectoryDecl.Type.NODES : functools.partial(self.__parse_nodes, afile),
                      DirectoryDecl.Type.ELEM_CONNS : functools.partial(self.__parse_elem_conn, afile),
                      DirectoryDecl.Type.STATE_VAR_DICT : functools.partial(self.__parse_svars, afile),
                      DirectoryDecl.Type.SREC_DATA : functools.partial(self.__parse_srec, afile) }

    reader = AFileReader( )
    for key, func in dir_callbacks.items( ):
      reader.register( key, func )

    self.__all_valid = True
    self.afilename = str(base) + 'A'
    self.tfilename = str(base) + 'T'
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

  def verify( self, name: str, value: Any, valid: Optional[Callable[...,bool]] = None ) -> Any:
    if valid is None:
      valid = lambda x : True
    is_valid = valid( value )
    if not is_valid:
      self.__all_valid = False
    if not self.__log_validator:
      log_str = f"{self.afilename}\n{name}\n  validity = {is_valid}\n  value = {value}\n"
    else:
      log_str = f"{self.afilename}\n{name}\n  validity = {is_valid}\n  value = {value}\n  validator = '{inspect.getsource(valid).strip()}'\n"
    logging.debug(log_str)
    if not is_valid and self.__except:
      raise MiliAParseError( f"invalid value encounterd:\n  {log_str}" )
    return value

  def __parse_header(self, afile: AFile, header_data: bytes ) -> None:
    if len( header_data ) != 16 :
      raise MiliAParseError( "Header data is not 16 bytes. This should never be seen unless something is wrong with I/O itself.")
    formats = [ dmem[1] for dmem in getdatamembers( AFile.Format ) ]
    afile.file_format = self.verify('header/file_format', str(header_data[0:4].decode('ascii')), lambda x : x in formats )
    version_validator = lambda x : AFile.Version.MIN <= x <= AFile.Version.MAX
    afile.file_version = self.verify('header/file_version', int(header_data[4]), version_validator )
    directory_version_validator = lambda x : AFile.DirectoryVersion.MIN <= x <= AFile.DirectoryVersion.MAX
    afile.directory_version = self.verify('header/directory_verion', int(header_data[5]), directory_version_validator )
    self.__dir_dtype = 'i' if afile.directory_version <= 2 else 'q'
    endians = [ dmem[1] for dmem in getdatamembers( AFile.Endian ) ]
    afile.endian_flag = self.verify('header/endian_flag', int(header_data[6]), lambda x : x in endians)
    self.__endian_str: Literal['big','little'] = 'big' if afile.endian_flag == AFile.Endian.BIG else 'little'
    precisions = [ dmem[1] for dmem in getdatamembers( AFile.Precision ) ]
    afile.precision_flag = self.verify('header/precision_flag', int(header_data[7]), lambda x : x in precisions )
    afile.sfile_suffix_length = self.verify('header/sfile_suffix_length', int(header_data[8]) )
    afile.partition_scheme = self.verify('header/partition_scheme', int(header_data[9]) )

  def __parse_footer(self, afile: AFile, footer_data: bytes) -> None:
    if len(footer_data) != 16 :
      raise MiliAParseError( "Footer data is not 16 bytes. This should never be seen unless something is wrong with I/O itself.")
    afile.string_bytes = self.verify( 'footer/string_bytes', int.from_bytes( footer_data[0:3], byteorder = self.__endian_str ) )
    afile.commit_count = self.verify( 'footer/commit_count', int.from_bytes( footer_data[4:7], byteorder = self.__endian_str ) )
    afile.directory_count = self.verify( 'footer/directory_count', int.from_bytes( footer_data[8:11], byteorder = self.__endian_str ) )
    afile.state_map_count = self.verify( 'footer/state_map_count', int.from_bytes( footer_data[12:15], byteorder = self.__endian_str ) )

  def __parse_smap(self, afile: AFile, smap_data: bytes) -> None:
    smap = StateMap( *struct.unpack('<iqfi',smap_data) )
    idx = len(afile.smaps)
    self.verify( f"smap[{idx}].sfile_number", smap.file_number )
    self.verify( f"smap[{idx}].sfile_offset", smap.file_offset )
    self.verify( f"smap[{idx}].time", smap.time )
    self.verify( f"smap[{idx}].srec_id", smap.state_map_id )
    afile.smaps.append( smap )

  def __parse_dir_decl(self, afile: AFile, dir_data: bytes) -> None:
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

  def __parse_strings(self, afile: AFile, strings_data: bytes ) -> None:
    strings = str(struct.unpack( f'>{afile.string_bytes}s', strings_data)[0])[2:].split('\\x00')
    str_offset = 0
    for idx, s in enumerate(strings):
      self.verify( f"strings[{idx}]", s )
    # assign string to dirs (needs to be ordered)
    for dir in afile.dir_decls_list:
      dir.strings = strings[ str_offset : str_offset + dir.str_cnt ]
      str_offset += dir.str_cnt
    afile.strings = strings

  def __parse_param(self, afile: AFile, param_data: bytes, dir_decl: DirectoryDecl) -> None:
    sname = dir_decl.strings[0]
    log_str = f'{dir_decl.dir_type.name}/{sname}'
    # modifier_idx1 is the type of the parameter
    # modified_idx2 is SCALAR (0) or ARRAY (2)
    param_dtype = MiliType(dir_decl.modifier_idx1)
    param_type = ParameterType(dir_decl.modifier_idx2)
    if param_type == ParameterType.SCALAR:
      if param_dtype in (MiliType.M_INT, MiliType.M_INT4, MiliType.M_INT8):
        afile.dirs[dir_decl.dir_type][sname] = self.verify( log_str, int.from_bytes(param_data, byteorder = self.__endian_str ) )
      elif param_dtype in (MiliType.M_FLOAT, MiliType.M_FLOAT4, MiliType.M_FLOAT8):
        afile.dirs[dir_decl.dir_type][sname] = self.verify( log_str, struct.unpack( param_dtype.struct_repr(), param_data ) )
      elif param_dtype == MiliType.M_STRING:
        param = struct.unpack(f"{dir_decl.length}s", param_data)[0].decode('ascii').split('\x00')[0]
        afile.dirs[dir_decl.dir_type][sname] = self.verify( log_str, param )
    elif param_type == ParameterType.ARRAY:
      # The first 2 integers are order and rank
      ndims = int.from_bytes(param_data[:4], byteorder=self.__endian_str)
      dim_bytes = 4 + 4 * ndims
      shape = np.frombuffer( param_data[4:dim_bytes], np.int32 )
      param = np.frombuffer( param_data[8:], dtype = param_dtype.numpy_dtype() )
      param = np.reshape( param, shape )
      afile.dirs[dir_decl.dir_type][sname] = self.verify( log_str, param )

  def __parse_svars(self, afile: AFile, svar_data: bytes, dir_decl: DirectoryDecl) -> None:
    header_bytes = 8
    svar_words, svar_bytes = struct.unpack('2i', svar_data[:header_bytes])
    int_cnt = svar_words - 2
    int_bytes = int_cnt * 4

    int_data = np.frombuffer( svar_data[ header_bytes : header_bytes + int_bytes ], np.int32 )
    strings = struct.unpack(f'{svar_bytes}s', svar_data[ header_bytes + int_bytes : ])[0].decode('ascii').split('\x00')
    strings = list(filter(('').__ne__,strings))

    iidx = 0
    sidx = 0
    while iidx < len(int_data):
      iidx, sidx = self.__parse_svar( afile, iidx, int_data, sidx, strings, dir_decl )

  def __parse_svar(self, afile: AFile, iidx: int, int_data: NDArray[np.int32], sidx: int, strings: List[str], dir_decl: DirectoryDecl) -> Tuple[int,int]:
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
    afile.dirs[dir_decl.dir_type][sname] = self.verify( f'{dir_decl.dir_type.name}/{sname}', StateVariable( **kwargs ) )
    return iidx, sidx

  def __parse_class_ident(self, afile: AFile, class_data: bytes, dir_decl: DirectoryDecl) -> None:
    sname = dir_decl.strings[0]
    superclass = int.from_bytes( class_data[:4], byteorder=self.__endian_str )
    if sname not in afile.dirs[dir_decl.dir_type].keys():
      afile.dirs[dir_decl.dir_type][sname] = {
        "superclass": superclass,
        "idents": np.empty([0],np.int32)
      }
    append_to = afile.dirs[dir_decl.dir_type][sname]['idents']
    append = np.frombuffer( class_data[4:12], dtype = np.int32 )
    afile.dirs[dir_decl.dir_type][sname]['idents'] = self.verify( f'{dir_decl.dir_type.name}/{sname}', np.append( append_to, append ) )

  def __parse_class_def(self, afile: AFile, _: bytes, dir_decl: DirectoryDecl) -> None:
    short_name = dir_decl.strings[0]
    long_name = dir_decl.strings[1]
    mesh_id = dir_decl.modifier_idx1
    sclass = Superclass(dir_decl.modifier_idx2)
    afile.dirs[dir_decl.dir_type][short_name] =  self.verify( f'{dir_decl.dir_type.name}/{short_name}', MeshObjectClass(mesh_id, short_name, long_name, sclass) )

  def __parse_nodes(self, afile: AFile, nodes_data: bytes, dir_decl: DirectoryDecl) -> None:
    sname = dir_decl.strings[0]
    f = io.BytesIO( nodes_data )
    blocks = np.reshape( np.frombuffer( f.read(8), dtype = np.int32 ), [-1,2] )
    blocks_atoms = np.sum( np.diff( blocks, axis=1 ).flatten( ) + 1 )
    mesh_dim = afile.dirs[DirectoryDecl.Type.MILI_PARAM]["mesh dimensions"]
    append = np.reshape( np.frombuffer( f.read( 4 * blocks_atoms * mesh_dim ), dtype = np.float32 ), [-1, mesh_dim] )
    if sname not in afile.dirs[dir_decl.dir_type]:
      afile.dirs[dir_decl.dir_type][sname] = {
        'nodes': self.verify( f'{dir_decl.dir_type.name}/{sname}', append ),
        'blocks': blocks
      }
    else:
      afile.dirs[dir_decl.dir_type][sname]['nodes'] = np.append( afile.dirs[dir_decl.dir_type][sname]['nodes'], self.verify( f'{dir_decl.dir_type.name}/{sname}', append ), axis=0 )
      afile.dirs[dir_decl.dir_type][sname]['blocks'] = np.append( afile.dirs[dir_decl.dir_type][sname]['blocks'], blocks, axis=0 )

  def __parse_elem_conn(self, afile: AFile, conn_data: bytes, dir_decl: DirectoryDecl) -> None:
    f = io.BytesIO( conn_data )
    sname = dir_decl.strings[0]
    sclass, block_cnt = struct.unpack( '2i', f.read(8) )
    blocks = np.frombuffer( f.read( 8 * block_cnt ), dtype=np.int32 )
    elem_qty = dir_decl.modifier_idx2
    conn_qty = Superclass(sclass).node_count()
    word_qty = conn_qty + 2
    conn = np.reshape( np.frombuffer( f.read( elem_qty * word_qty * 4 ), dtype = np.int32 ), [-1,word_qty] )
    if not sname in afile.dirs[dir_decl.dir_type].keys():
      afile.dirs[dir_decl.dir_type][sname] = {
        'conns': np.empty( (0,word_qty), dtype = np.int32 ),
        'superclass': sclass,
        'block_cnt': 0,
        'blocks': np.empty([0],dtype=np.int32),
      }
    afile.dirs[dir_decl.dir_type][sname]['conns'] = np.concatenate( ( afile.dirs[dir_decl.dir_type][sname]['conns'], self.verify( f'{dir_decl.dir_type.name}/{sname}', conn ) ) )
    afile.dirs[dir_decl.dir_type][sname]['blocks'] = np.concatenate( (afile.dirs[dir_decl.dir_type][sname]['blocks'], blocks) )
    afile.dirs[dir_decl.dir_type][sname]['block_cnt'] += block_cnt

  def __parse_srec(self, afile: AFile, srec_data: bytes, dir_decl: DirectoryDecl) -> None:
    srec_count = int.from_bytes( srec_data[12:16], self.__endian_str )
    # We store the subrecord data header information to be use when writing an A file
    afile.dirs[dir_decl.dir_type]['header'] = struct.unpack('4i', srec_data[0:16])
    afile.dirs[dir_decl.dir_type]['subrecords'] = {}
    srec_header_bytes = 16
    srec_idata_bytes = ( dir_decl.modifier_idx1 - 4 ) * 4
    srec_strdata_bytes = dir_decl.modifier_idx2
    int_data = np.frombuffer( srec_data[ srec_header_bytes : srec_header_bytes + srec_idata_bytes ], np.int32 )
    strings = [ name.rstrip('\x00') for name in struct.unpack(str(srec_strdata_bytes) + 's', srec_data[ srec_header_bytes + srec_idata_bytes : ])[0].decode('ascii').split('\x00') ]
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
        blocks, offsets = zip( *(sorted( zip(ordinal_blocks, ordinal_block_offsets[:-1]), key=lambda x: x[0][0]) ) )  # type: ignore
        ordinal_blocks = np.array( blocks ).flatten()
        ordinal_block_offsets = np.array( offsets )
        # Recalculate ordinal_block_counts
        ordinal_block_counts = np.concatenate( ( [0], np.diff( ordinal_blocks.reshape(-1,2), axis=1 ).flatten( ) ) ) # sum stop - start + 1 over all blocks

      kwargs['superclass'] = Superclass(superclass)
      kwargs['ordinal_blocks'] = ordinal_blocks
      kwargs['ordinal_block_counts'] = ordinal_block_counts
      kwargs['ordinal_block_offsets'] = ordinal_block_offsets
      iidx += 2 * block_count
      afile.dirs[dir_decl.dir_type]['subrecords'][sname] = self.verify( f'{dir_decl.dir_type.name}/{sname}', Subrecord( **kwargs ) )


def _ceil_to_nearest(val: int, mult: int) -> int:
  return mult * math.ceil( val / mult )

class AFileParallelHelper:
  def __init__(self, base: Union[str,os.PathLike], allow_exceptions: bool = True) -> None:
    self.__afile = AFile()
    self.__parser = AFileParser(allow_exceptions)
    self.__rval = self.__parser.parse(self.__afile, base)

  def afile(self) -> AFile:
    return self.__afile

  def rval(self) -> int:
    return self.__rval

_log_dir = lambda sname, dir_decl, data : logging.debug(f'ti\n  sname = {sname}\n  dd = {dir_decl}\n  data = {data}')

def endian_info(endian_flag: int) -> Tuple[Literal['big','little'],Literal['>','<']]:
  if endian_flag == AFile.Endian.BIG:
    return 'big', '>'
  else:  # endian_flag == AFile.Endian.LITTLE:
    return 'little', '<'

class AFileWriter:
  def __init__(self) -> None:
    self.__byteorder = sys.byteorder
    self.__bo = '>' if self.__byteorder == 'big' else '<'
    self.__callbacks = {
      DirectoryDecl.Type.MILI_PARAM : self.__write_param,
      DirectoryDecl.Type.APPLICATION_PARAM : self.__write_param,
      DirectoryDecl.Type.TI_PARAM : self.__write_param,
      DirectoryDecl.Type.CLASS_DEF : self.__write_class_def,
      DirectoryDecl.Type.CLASS_IDENTS : self.__write_class_ident,
      DirectoryDecl.Type.NODES : self.__write_nodes,
      DirectoryDecl.Type.ELEM_CONNS : self.__write_elem_conn,
      # DirectoryDecl.Type.STATE_VAR_DICT : self.__write_svars,
      # DirectoryDecl.Type.STATE_REC_DATA : self.__write_srecs
    }

  def write(self, afile: AFile, base: Union[str,os.PathLike], allow_tfile: bool = False) -> int:
    self.__byteorder, self.__bo = endian_info(afile.endian_flag)

    # TODO: check for file existence and reject if they already exist
    afilename = str(base) + 'A'
    tfilename = str(base) + 'T'

    adata = io.BytesIO( )
    tdata = io.BytesIO( )

    self.__write_header(afile, adata)
    strings = []
    dir_order = [ DirectoryDecl.Type.MILI_PARAM,
                  DirectoryDecl.Type.APPLICATION_PARAM,
                  DirectoryDecl.Type.TI_PARAM,
                  DirectoryDecl.Type.CLASS_DEF,
                  DirectoryDecl.Type.CLASS_IDENTS,
                  DirectoryDecl.Type.NODES,
                  DirectoryDecl.Type.ELEM_CONNS ]
    for dir_type in dir_order:
      _, dirs_strings = self.__write_directories(afile, adata, dir_type) # modifies the offset and length of the decls based on the write, str_count stays the same
      strings.extend( dirs_strings )

    # if we can have more than 1 this needs to change
    svars: List[StateVariable] = afile.dirs[ DirectoryDecl.Type.STATE_VAR_DICT ].values()  # type: ignore
    svar_dd : DirectoryDecl = afile.dir_decls[ DirectoryDecl.Type.STATE_VAR_DICT ][ 0 ]
    svar_dd.offset = adata.tell( )
    svar_dd.length = self.__write_svars( svars, adata )

    # again if we can have more than 1 this needs to changes
    srecs = afile.dirs[ DirectoryDecl.Type.SREC_DATA ]
    srec_dd : DirectoryDecl = afile.dir_decls[ DirectoryDecl.Type.SREC_DATA ][ 0 ]
    srec_dd.offset = adata.tell( )
    srec_dd.length = self.__write_srecs(srecs, adata, srec_dd)

    afile.strings = strings
    afile.string_bytes = self.__write_strings(afile, adata) # write the strings in their new order (based on the order we wrote the dirs)

    # we write the decls in offset order.. which since we change the offsets based on the above order, is the same order, so the reordered strings are valid
    afile.directory_count = self.__write_dir_decls(afile, adata)

    afile.state_map_count = self.__write_smaps(afile, tdata)
    if afile.file_version < 3 or not allow_tfile: # write smaps directly to AFile before v3 file format
      adata.write( tdata.getvalue() )
    footer_binary = io.BytesIO()
    self.__write_footer(afile, footer_binary)
    adata.write(footer_binary.getvalue())

    # actually write to file(s)
    with open(afilename,'wb') as af:
      af.write(adata.getvalue())
    if afile.file_version > 2 and allow_tfile:
      with open(tfilename, 'wb') as tf:
        tf.write(tdata.getvalue())
        tf.write(b'~')
    return 0

  def __write_header(self, afile: AFile, out: BinaryIO ) -> None:
    out.write( bytes( afile.file_format, 'ascii' ) )
    out.write( afile.file_version.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( afile.directory_version.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( afile.endian_flag.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( afile.precision_flag.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( afile.sfile_suffix_length.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( afile.partition_scheme.to_bytes( 1, byteorder = self.__byteorder ) )
    out.write( bytes(6) )

  def __write_directories(self, afile: AFile, out: BinaryIO, dir_type: DirectoryDecl.Type ) -> Tuple[int,List[str]]:
    init = out.tell( )
    strings: List[str] = []
    for sname, data in afile.dirs[ dir_type ].items():
      dd: DirectoryDecl
      dd = next( filter( lambda x: x.strings[0] == sname, afile.dir_decls[dir_type]) )
      dd.offset = out.tell( ) # this assumes the out argument is the final buffer / file
      dd.length = self.__callbacks[ dir_type ]( data, out, dd )
      strings.extend( dd.strings )
    return out.tell( ) - init, strings

  def __write_nodes(self, data: Any, out: BinaryIO, _: DirectoryDecl) -> int:
    init = out.tell( )
    out.write( data['blocks'].tobytes( ) ) # node block
    out.write( data['nodes'].tobytes( ) ) # coord data for the block
    return out.tell( ) - init

  def __write_elem_conn(self, data: Any, out: BinaryIO, _: DirectoryDecl) -> int:
    init = out.tell( )
    out.write( int(data['superclass']).to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( int(data['block_cnt']).to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( data['blocks'].tobytes() ) # we never bother converting the blocks from bytes internally
    out.write( data['conns'].tobytes() )  # elem_conns
    return out.tell( ) - init

  def __write_class_ident(self, data: Any, out: BinaryIO, _: DirectoryDecl ) -> int:
    init = out.tell( )
    superclass = data['superclass']
    out.write( superclass.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( data['idents'].tobytes() )
    return out.tell() - init

  def __write_class_def(self, _1: Any, _2: BinaryIO, _3: DirectoryDecl) -> int:
    # we actually just don't do anything here, the dir_decl is the class identifier
    return 0

  def __write_param(self, data: Any, out: BinaryIO, dir_decl: DirectoryDecl) -> int:
    init = out.tell( )
    # modifier_idx1 is the type of the parameter
    # modified_idx2 is SCALAR (0) or ARRAY (2)
    param_dtype = MiliType(dir_decl.modifier_idx1)
    param_type = ParameterType(dir_decl.modifier_idx2)
    if param_type == ParameterType.SCALAR:
      if param_dtype in (MiliType.M_INT, MiliType.M_INT4, MiliType.M_INT8):
        out.write( data.to_bytes( param_dtype.byte_size(), byteorder = self.__byteorder ) )
      elif param_dtype in (MiliType.M_FLOAT, MiliType.M_FLOAT4, MiliType.M_FLOAT8):
        out.write( struct.pack( param_dtype.struct_repr(), data ) )
      elif param_dtype == MiliType.M_STRING:
        out.write( bytes( data, 'ascii' ) )
        out.write( bytes(1) )
        str_bytes = out.tell( ) - init
        align_bytes = _ceil_to_nearest( str_bytes, 8 ) - str_bytes # param strings get rounded to the nearest 8 instead of 4...
        out.write( bytes( align_bytes ) )
    elif param_type == ParameterType.ARRAY:
      order = data.ndim
      dims = np.array(data.shape, dtype=np.int32)
      out.write( order.to_bytes( 4, byteorder = self.__byteorder ) )
      out.write( dims.tobytes() )
      out.write( data.tobytes() )
    bytes_written = out.tell( ) - init
    return bytes_written

  def __write_svars(self, svars: List[StateVariable], out: BinaryIO) -> int:
    init = out.tell( )
    int_data: List[int] = []
    str_data: List[str] = []
    processed: Set[str] = set()
    for svar in svars:
      if svar.name not in processed:
        self.__collect_svar_data( svar, int_data, str_data, svars, processed )

    # write an intermediate buffer of the int and str data to get the sizes
    svar_out = io.BytesIO()
    svar_out.write( struct.pack(f'{len(int_data)}i', *int_data ) )
    int_bytes = svar_out.tell( )
    for ss in str_data:
      svar_out.write( bytes( ss, 'ascii' ) + bytes(1) )
    int_cnt = ( int_bytes // 4 ) + 2
    str_bytes = svar_out.tell() - int_bytes
    align_bytes = _ceil_to_nearest( str_bytes, 4 ) - str_bytes # round up to the nearest 4-byte word (to mimic the core mili library)
    svar_out.write( bytes( align_bytes ) )
    str_bytes = str_bytes + align_bytes

    # do the actual write
    out.write( int_cnt.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( str_bytes.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( svar_out.getvalue() )

    return out.tell() - init

  def __collect_svar_data(self, svar: StateVariable, int_data: List[int], str_data: List[str], svars: List[StateVariable], processed: Set[str] ) -> None:
    processed.add( svar.name )
    int_data.append( svar.agg_type )
    int_data.append( svar.data_type.value )
    str_data.append( svar.name )
    str_data.append( svar.title )
    if svar.agg_type in [ StateVariable.Aggregation.ARRAY, StateVariable.Aggregation.VEC_ARRAY ]:
      int_data.append( svar.order )
      int_data.extend( svar.dims )
    if svar.agg_type in [ StateVariable.Aggregation.VECTOR, StateVariable.Aggregation.VEC_ARRAY ]:
      int_data.append( svar.list_size )
      str_data.extend( svar.comp_names )
      for comp_name in svar.comp_names:
        if comp_name not in processed:
          comp_svar = next( filter( lambda x : x.name == comp_name, svars ) )
          self.__collect_svar_data( comp_svar, int_data, str_data, svars, processed )

  def __write_srecs(self, srecs: Mapping[str,Mapping[str,Subrecord]], out: BinaryIO, srec_dd: DirectoryDecl) -> int:
    init = out.tell()
    int_data: List[int] = []
    str_data: List[str] = []
    for _, srec in srecs['subrecords'].items():
      int_data.append( srec.organization )
      int_data.append( srec.qty_svars )
      if srec.superclass != Superclass.M_MESH:
        int_data.append( np.size( srec.ordinal_blocks, 0 ) // 2 )
        ordinal_blocks = srec.ordinal_blocks + 1
        ordinal_blocks[1::2] = ordinal_blocks[1::2] - 1 # undo parse-in change to speed up query algo
        int_data.extend( ordinal_blocks.flatten().tolist() )
      else:
        # block size is zero for M_MESH
        int_data.append( 0 )
      str_data.append( srec.name )
      str_data.append( srec.class_name )
      str_data.extend( srec.svar_names )

    srec_out = io.BytesIO()
    srec_out.write( struct.pack( f'{len(int_data)}i', *int_data ) )
    int_bytes = srec_out.tell( )
    for ss in str_data:
      srec_out.write( bytes( ss, 'ascii' ) + bytes(1) )
    str_bytes = srec_out.tell() - int_bytes
    align_bytes = _ceil_to_nearest( str_bytes, 4 ) - str_bytes  # round up to the nearest 4-byte word (to mimic the core mili library)
    srec_out.write( bytes( align_bytes ) )
    str_bytes = str_bytes + align_bytes

    srec_dd.modifier_idx1 = ( int_bytes // 4 ) + 4
    srec_dd.modifier_idx2 = str_bytes

    srec_count: int = len( srecs )
    srec_id: int
    mesh_id: int
    size: int
    srec_id, mesh_id, size, srec_count = srecs['header']  # type: ignore   # Key header has special value(s) in the srecs dictionary.
    out.write( srec_id.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( mesh_id.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( size.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( srec_count.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( srec_out.getvalue() )

    return out.tell() - init

  def __write_dir_decls(self, afile: AFile, out: BinaryIO) -> int:
    size = 4 if afile.directory_version == 1 else 8
    dir_decl_order = [ DirectoryDecl.Type.MILI_PARAM,
                       DirectoryDecl.Type.APPLICATION_PARAM,
                       DirectoryDecl.Type.TI_PARAM,
                       DirectoryDecl.Type.CLASS_DEF,
                       DirectoryDecl.Type.CLASS_IDENTS,
                       DirectoryDecl.Type.NODES,
                       DirectoryDecl.Type.ELEM_CONNS,
                       DirectoryDecl.Type.STATE_VAR_DICT,
                       DirectoryDecl.Type.SREC_DATA ]
    dir_count = 0
    for dir_type in dir_decl_order:
      for dd in afile.dir_decls[dir_type]:
        out.write( int(dd.dir_type.value).to_bytes( size, byteorder = self.__byteorder ) )
        out.write( dd.modifier_idx1.to_bytes( size, byteorder = self.__byteorder ) )
        out.write( dd.modifier_idx2.to_bytes( size, byteorder = self.__byteorder ) )
        out.write( dd.str_cnt.to_bytes( size, byteorder = self.__byteorder ) )
        out.write( dd.offset.to_bytes( size, byteorder = self.__byteorder ) )
        out.write( dd.length.to_bytes( size, byteorder = self.__byteorder ) )
        dir_count += 1
    return dir_count

  def __write_strings(self, afile: AFile, out: BinaryIO) -> int:
    init = out.tell()
    for strdata in afile.strings:
      out.write( bytes( strdata, 'ascii' ) + bytes(1) )
    str_bytes = out.tell( ) - init
    align_bytes: int = _ceil_to_nearest( str_bytes, 4 ) - str_bytes  # round up to the nearest 4-byte word (to mimic the core mili library)
    out.write( bytes( align_bytes ) )
    str_bytes = str_bytes + align_bytes
    return str_bytes

  def __write_smaps(self, afile: AFile, out: BinaryIO ) -> int:
    smap_count = 0
    for smap in afile.smaps:
      out.write( struct.pack( f'{self.__bo}iqfi', smap.file_number, smap.file_offset, smap.time, smap.state_map_id ) )
      smap_count += 1
    return smap_count

  def __write_footer(self, afile: AFile, out: BinaryIO) -> None:
    out.write( afile.string_bytes.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( afile.commit_count.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( afile.directory_count.to_bytes( 4, byteorder = self.__byteorder ) )
    out.write( afile.state_map_count.to_bytes( 4, byteorder = self.__byteorder ) )

class AFileWriteParallel:
  '''
    Just a small utility class to make the default writer usage in write_database cleaner with the parallel wrappers.
  '''
  def __init__(self, base: Union[str,os.PathLike], afile: AFile) -> None:
    self.__writer = AFileWriter()
    self.__rval = self.__writer.write( afile, base )

  def rval(self) -> int:
    return self.__rval


def parse_database(base: Union[str,os.PathLike], procs: List[int] = [],
                   suppress_parallel: bool = False, experimental: bool = False ) -> Tuple[List[AFile],List[int]]:
  """
   Open and parse a database. This only opens the database metadata files, and can be useful for verifying
   parallel databases are in valid and consistent states. The object returned by this function will have the
   same interface as an AFile object, though will return a list of results from the specified proc files
   instead of a single result.
  Args:
   base (os.PathLike): the base filename of the mili database (e.g. for 'pltA', just 'plt', for parallel
                            databases like 'dblplt00A', also exclude the rank-digits, giving 'dblplt')
   procs (List[int], default = []) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Bool, default = False) : optionally return a serial database layout object if possible.
                                        Note: if the database is parallel, suppress_parallel==True will return a reader that will
                                        query each processes database files serially.
   experimental (Bool, default = False) : optional developer-only argument to try experimental parallel features
  """
  # ensure dir_name is the containing dir and base is only the file name
  dir_name = os.path.dirname( base )
  if dir_name == '':
    dir_name = os.getcwd()
  if not os.path.isdir( dir_name ):
    raise MiliFileNotFoundError( f"Cannot locate mili file directory {dir_name}.")
  base = os.path.basename( base )
  afile_names = afiles_by_base( dir_name, base, procs )
  proc_bases = [ afile[:-1] for afile in afile_names ] # drop the A to get each processes base filename for A,T, and S files
  proc_pargs = [ [os.path.join(dir_name,base_file)] for base_file in proc_bases ]
  parse_wrapper = get_wrapper( suppress_parallel, experimental )( AFileParallelHelper, proc_pargs )
  afiles: List[AFile] = parse_wrapper.afile()  # type: ignore  # mypy error becase parse_wrapper is a parallel wrapper and has no attribute afile.
  rvals = parse_wrapper.rval()  # type: ignore  # mypy error becase parse_wrapper is a parallel wrapper and has no attribute rval.
  return afiles, rvals

def write_database(afiles: Union[AFile,List[AFile]],
                   afilenames: Union[Union[str,os.PathLike],List[Union[str,os.PathLike]]],
                   suppress_parallel: bool = False,
                   experimental: bool = False ) -> List[int]:
  """
   Write a mili database. This only writes the database metadata files, and can be useful for repairing invalid/broken metadata files.

  Args:
    afiles (Union[AFile,List[Afile]]):
    afilenames (Union[os.PathLike,List[os.PathLike]]):
    suppress_parallel (Optional[Bool]) : optionally execute write operations serially
    experimental (Optional[Bool]) : optional developer-only argument to try experimental parallel features
  """
  if not isinstance( afiles, List ):
    afiles = [afiles]
  if not isinstance( afilenames, List ):
    afilenames = [afilenames]
  proc_pargs = [list(zipped) for zipped in zip(afilenames,afiles)]
  write_wrapper = get_wrapper( True, experimental )( AFileWriteParallel, proc_pargs )
  rvals: List[int] = write_wrapper.rval()  # type: ignore  # mypy error becase write_wrapper is a parallel wrapper and has no attribute rval.
  return rvals