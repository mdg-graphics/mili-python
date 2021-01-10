from multiprocessing import Process, Manager
from enum import Enum
from collections import OrderedDict
import os
import struct
import sys
import re
import psutil


    ######################## classes / indexes #########################
    # Might not use but can still reference

class HeaderArray(Enum):
    # MILI_TAUR not kept in array, but remember to add during write (4s)
    HEADER_VERSION = 0
    DIR_VERSION = 1
    ENDIAN_FLAG = 2
    PRECISION_FLAG = 3
    STATE_FILE_SUFFIX_LEN = 4
    PARTITION_FLAG = 5

class IndexingArray(Enum):
    NULL_TERMED_NAMES_BYTES = 0
    NUM_COMMITS = 1
    NUM_DIRS = 2
    NUM_STATE_MAPS = 3

class StateMapsArray(Enum):
    FILE_NUM = 0
    FILE_OFFSET = 1
    TIME = 2
    STATE_MAP_ID = 3

class ParamsNode(Enum):
    FIRST = 0
    LAST = 1
    NODE_LABELS = 2 # START INDEX

class ParamsElem(Enum):
    FIRST = 0
    TOTAL = 1
    INTS = 2 # START INDEX, ARE MULT

class ParamsEs(Enum):
    FIRST = 0
    TOTAL = 1
    I_POINTS = 2
    NUM_I_POINTS = 3

class SvarInts(Enum):
    AGG_TYPE = 0
    DATA_TYPE = 1
    ORDER = 2
    LIST_SIZE = 4 # BUT NEED TO ADD ORDER

class SvarS(Enum):
    SV_NAME = 0
    TITLE = 1
    SV_NAMES = 2
    # INNER SV NAME, TITLE

class MeshElemConns(Enum):
    SUP_CLASS = 0
    QTY_BLOCKS = 1

class MeshECElemBlocks(Enum):
    START = 0
    STOP = 1

class MeshECEbuf(Enum):
    NODES = 0 # MULT
    MAT = 1 # 1 + NUM NODES
    PART = 2 # 2 + NUM NODES

class SrecHeader(Enum):
    SREC_QTY_SUBRECS = 3

class SubrecIdata(Enum):
    ORG = 0
    QTY_SVARS = 1
    QTY_ID_BLOCKS = 2
    START = 3
    STOP = 4

class SubrecCdata(Enum):
    NAME = 0
    CLASS_NAME = 1
    SVARS = 2

class DirArray(Enum):
    TYPE_IDX = 0
    MOD_IDX1 = 1
    MOD_IDX2 = 2
    STR_QTY_IDX = 3
    OFFSET_IDX = 4
    LENGTH_IDX = 5

class DirStringsArray(Enum):
    STRING_OFFSET_IDX = 0
    STRINGS = 1 # strings start



    ######################### read functions ###########################


def ReadNames(p, f, file_tag, offset, this_indexing_array, this_proc_dirs, dir_num_strings_dict, global_names):
    # 1. Keep track of which strings associated w which directories
    # 2. Keep list of all names

    offset -= this_indexing_array[IndexingArray.NULL_TERMED_NAMES_BYTES.value]
    f.seek(offset, os.SEEK_END)
    fmt = str(this_indexing_array[IndexingArray.NULL_TERMED_NAMES_BYTES.value]) + 's'
    byte_array = struct.unpack(file_tag + fmt, f.read(this_indexing_array[0]))[0]

    if (sys.version_info > (3, 0)):
        strings = str(byte_array)[2:].split('\\x00')  # only works for Python3
    else:
        strings = byte_array.split(b'\x00')

    num_dirs = this_indexing_array[IndexingArray.NUM_DIRS.value]

    # Need to make sure list of keys is in order
    dir_types = [k for k in this_proc_dirs]
    for dir_type in dir_types:
        for item in this_proc_dirs[dir_type].items():
            item = item[0]
            dir_num, dir_offset = item

            string_offset = dir_num_strings_dict[dir_type][dir_num][DirStringsArray.STRING_OFFSET_IDX.value]
            string_count = this_proc_dirs[dir_type][item][DirArray.STR_QTY_IDX.value]
            
            directory_strings = [ strings[sidx] for sidx in range( string_offset, string_offset + string_count) ]
            dir_num_strings_dict[dir_type][dir_num][DirStringsArray.STRINGS.value] = directory_strings

            # Add to global_names if not in there already
            if global_names is None:
                global_names = directory_strings
            else:
                for string in directory_strings:
                    if string not in global_names:
                        global_names.append(string)


    
def ReadDirectories(p, f, file_size, offset, file_tag, this_header_array, this_indexing_array, dir_strings_dict, dir_type_dict):
    # This function only reads and keeps track of local dictionaries - so uses offsets in current file
    # In other read functions, another directory dictionary is updated that uses short names bc
    # # is used between all processors

    # Should the directory strings be global - need short_name to id them?

    # OrderedDicts - find better indexing system - dictionary appears as (key, value) ?
    # local_dir_dict = dir_type_dict[this_dir_type][(local_dir_number, this_dir_offset)] = dir_entry
    # local_dir_strings_dict = dir_strings_dict[this_dir_type][local_dir_number] = [num_strings, [strings]]

    number_of_strings = 0
    directory_length = 4 * 6
    state_map_length = 20
    int_long = 'i'

    dir_version = this_header_array[HeaderArray.DIR_VERSION.value]

    if dir_version > 2:
        directory_length = 8 * 6
        int_long = 'q'

    num_state_maps = this_indexing_array[IndexingArray.NUM_STATE_MAPS.value]
    num_dirs = this_indexing_array[IndexingArray.NUM_DIRS.value]

    offset -= state_map_length * num_state_maps + directory_length * num_dirs  # make this not a hard coded 6 eventually
    f.seek(offset, os.SEEK_END)
    for i in range(1, 1 + num_dirs): # for num dirs
        byte_array = f.read(directory_length)

        # number of strings isn't in the byte array - create another var or just sum dir index ?
        this_byte_array = struct.unpack(file_tag + '6' + int_long, byte_array)
        # Dictionary to map offset to number of strings for that dir
        this_dir_offset = this_byte_array[DirArray.OFFSET_IDX.value]

        this_dir_type = this_byte_array[DirArray.TYPE_IDX.value]
        if this_dir_type not in dir_type_dict:
            dir_type_dict[this_dir_type] = OrderedDict()
            dir_strings_dict[this_dir_type] = OrderedDict()
            
        dir_type_dict[this_dir_type][(i, this_dir_offset)] = this_byte_array
        dir_strings_dict[this_dir_type][i] = {}
        dir_strings_dict[this_dir_type][i][DirStringsArray.STRING_OFFSET_IDX.value] = number_of_strings

        this_dir_string_qty = this_byte_array[DirArray.STR_QTY_IDX.value]
        number_of_strings += this_dir_string_qty
        
    return offset

    
def ReadStateMaps(p, f, file_tag, this_proc_indexing_array):
    # Why does this return offset
    offset = -16
    state_map_length = 20
    f.seek(offset, os.SEEK_END)
    number_of_state_maps = this_proc_indexing_array[IndexingArray.NUM_STATE_MAPS.value]
    state_maps = {}

    for num in range(1, 1 + number_of_state_maps):
        f.seek(-1 * state_map_length, 1)
        byte_array = f.read(state_map_length)
        state_map = struct.unpack(file_tag + 'iqfi', byte_array)
        state_map = list(state_map)
        file_number, file_offset, time, state_map_id = state_map

        state_maps[file_offset] = state_map
        f.seek(-1 * state_map_length, 1)

    return state_maps, offset

def ReadHeader(p, f, local_file_tag, header_dict, indexing_dict): # add file_tag to header_array
    # Precision flag should be same across procs ??
    header = f.read(16)
    mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')

    header_array = struct.unpack('6b', header[4:10])
    header_dict[p] = (list(header_array))
    header_version, dir_version, endian_flag, precision_flag, state_file_suffix_length, partition_flag = header_array

    if endian_flag == 1:
        local_file_tag = '>'
    else:
        local_file_tag = '<'

    # Read Indexing #
    offset = -16
    f.seek(offset, os.SEEK_END)
    indexing_dict[p] = struct.unpack('4i', f.read(16))

    return local_file_tag, mili_taur



def StartRead(p, file_name, header_dict, indexing_dict, global_state_maps, global_names):
    file_size = os.path.getsize(file_name)
    local_file_tag = None
    mili_taur = None
    offset = None

    # Is it better to create new lists or to continue to access the list in the dictionary
    local_header = []
    local_indexing = []

    # index match directory to strings
    local_directory_dict = OrderedDict()
    local_dir_strings_dict = OrderedDict()

    with open(file_name, 'rb') as f:
        local_file_tag, mili_taur = ReadHeader(p, f, local_file_tag, header_dict, indexing_dict)
        local_header = header_dict[p]
        local_indexing = list(indexing_dict[p])

        global_state_maps[p], offset = ReadStateMaps(p, f, local_file_tag, local_indexing)

        offset = ReadDirectories(p, f, file_size, offset, local_file_tag, local_header, local_indexing, local_dir_strings_dict, local_directory_dict)

        if local_indexing[IndexingArray.NULL_TERMED_NAMES_BYTES.value] > 0:
            ReadNames(p, f, local_file_tag, offset, local_indexing, local_directory_dict, local_dir_strings_dict, global_names)
            print(global_names)
            #ReadParams()
            #ReadStateVariables()
            #ReadSubrecords()

    #print("Local Dir Dict", local_directory_dict)
    #print("Local Dir Strings Dict", local_dir_strings_dict)
        


def main():
    
    #file_name = '/g/g12/pham22/mili-python/d3samp6.pltA'
    file_names = ['/g/g12/pham22/mili-python/parallel/d3samp6.plt000A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt001A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt002A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt003A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt004A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt005A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt006A', \
                  '/g/g12/pham22/mili-python/parallel/d3samp6.plt007A']

    manager = Manager()
    header_dict = manager.dict()
    indexing_dict = manager.dict()
    global_state_maps = manager.dict()
    global_names = manager.list()

    for i,file_name in enumerate(file_names):

        p = Process(target=StartRead, args=(i, file_name, header_dict, \
                                            indexing_dict, global_state_maps, \
                                            global_names))

        p.start()
        p.join()

    print("Header Dict", header_dict)
    print("Indexing Dict", indexing_dict)
    #print("Global State Maps Dict", global_state_maps)

if __name__ == '__main__':
    main()
