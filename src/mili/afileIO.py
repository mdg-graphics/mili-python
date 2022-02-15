"""
Copyright (c) 2016-2022, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by 
 William Tobin (tobin6@llnl.hov) and Kevin Durrenberger (durrenberger1@llnl.gov). 
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

from collections import defaultdict
from enum import Enum
import io
import os
import struct 
from typing import *
import typing
if hasattr( typing, 'io' ):
  from typing.io import * # type: ignore

from mili.datatypes import *

# Callback keys: AFileParser.Section values and Directory.Type values
class AFileParser:
  class Section(Enum):
    HEADER = 0
    FOOTER = 1
    STATE_MAP = 2
    DIR_DECL = 3
    STRINGS = 4

  def __init__(self):
    self.__callbacks = defaultdict(list)
    self.__dirs = []
    self.register( AFileParser.Section.HEADER, self.__parse_header )
    self.register( AFileParser.Section.FOOTER, self.__parse_footer )
    self.register( AFileParser.Section.DIR_DECL, self.__parse_dir )
    self.register( AFileParser.Section.STRINGS, self.__parse_strings)

  def __parse_header( self, header_data : bytes ):
    self.__dir_dtype = 'i' if header_data[5] == 1 else 'q'

  def __parse_footer( self, footer_data : bytes ):
    self.__strings_bytes = struct.unpack( 'i', footer_data[0:4] )[0]

  def __parse_dir( self, dir_data : bytes ):
    directory = Directory( *struct.unpack(f'<6{self.__dir_dtype}',dir_data ) )
    directory.dir_type = Directory.Type( directory.dir_type )
    self.__dirs.append( directory )

  def __parse_strings( self, strings_data ):
    strings = str(struct.unpack( f'>{self.__strings_bytes}s', strings_data)[0])[2:].split('\\x00')
    str_offset = 0
    # assign string to dirs (needs to be ordered)
    for dir in self.__dirs:
      dir.strings = strings[ str_offset : str_offset + dir.str_cnt ]
      str_offset += dir.str_cnt
    # restructure dirs to be mapped by Directory.Type
    dirs = self.__dirs
    self.__dirs = defaultdict(list)
    for dir in dirs:
      self.__dirs[dir.dir_type].append(dir)

  def __callback( self, key : Union[Section,Directory.Type], *pargs ) -> None:
    [ foo(*pargs) for foo in self.__callbacks.get(key, []) ]

  def __have_callback( self, key : Union[Section,Directory.Type] ) -> bool:
    return key in self.__callbacks.keys()

  def __read_successive( self, f : BinaryIO, key : Union[Section,Directory.Type], num_items : int, item_bytes : int ) -> None:
    ''' Read a buffer of num_items * item_bytes and iterate over it in item_bytes slices, calling the 
         key callback. '''
    data = f.read( num_items * item_bytes )
    byte_data = io.BytesIO( data )
    while item_data := byte_data.read( item_bytes ):
      self.__callback( key, item_data )

  def register( self, key : Union[Section,Directory.Type], cback : Callable[[bytearray],None]) -> None:
    self.__callbacks[key].append(cback)

  def read( self, f : BinaryIO ) -> None:
    header = f.read(16)
    self.__callback(AFileParser.Section.HEADER,header)
    dirs_bytes = 24 if header[5] == 1 else 48
    f.seek(-16, os.SEEK_END)
    footer = f.read(16)
    self.__callback(AFileParser.Section.FOOTER,footer)
    strings_bytes, _, num_dirs, num_smaps = struct.unpack('4i', footer)

    smap_offset = - ( 16 + num_smaps * 20 ) # footer + num_smaps * smap_bytes
    f.seek( smap_offset, os.SEEK_END )
    self.__read_successive( f, AFileParser.Section.STATE_MAP, num_smaps, 20 )

    dirs_offset = smap_offset - ( num_dirs * dirs_bytes )
    f.seek( dirs_offset, os.SEEK_END )
    self.__read_successive( f, AFileParser.Section.DIR_DECL, num_dirs, dirs_bytes )

    strings_offset = dirs_offset - strings_bytes
    f.seek( strings_offset, os.SEEK_END )
    strings_data = f.read( strings_bytes )
    self.__callback( AFileParser.Section.STRINGS, strings_data )

  def read_dirs( self, f : BinaryIO, dir_type : Directory.Type ):
    # read any subscribed directories in the above order
    # for dir_type in dirs_order:
      for dd in self.__dirs.get(dir_type,[]):
        if self.__have_callback( dd.dir_type ):
          f.seek( dd.offset )
          self.__callback( dd.dir_type, f.read( dd.length ), dd )
