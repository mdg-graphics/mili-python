from multiprocessing import Process, Manager
from enum import Enum
import os
import struct
import sys
import re
import psutil


    ######################## classes / indexes #########################
    # Might not use but can still reference

class HeaderArray(Enum):
    MILI_TAUR = 0
    HEADER_VERSION = 1
    DIR_VERSION = 2
    ENDIAN_FLAG = 3
    PRECISION_FLAG = 4
    STATE_FILE_SUFFIX_LEN = 5
    PARTITION_FLAG = 6

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



    
def ReadStateMaps(i, f, file_tag, this_proc_indexing_array):
    offset = -16
    state_map_length = 20
    f.seek(offset, os.SEEK_END)
    number_of_state_maps = this_proc_indexing_array[i][IndexingArray.NUM_STATE_MAPS.value]
    state_maps = {}

    for num in range(1, 1 + number_of_state_maps):
        f.seek(-1 * state_map_length, 1)
        byte_array = f.read(state_map_length)
        state_map = struct.unpack(file_tag + 'iqfi', byte_array)
        state_map = list(state_map)
        file_number, file_offset, time, state_map_id = state_map

        state_maps[file_offset] = state_map
        f.seek(-1 * state_map_length, 1)
    offset = f.tell()
    return state_maps, offset

def ReadHeader(i, f, local_file_tag, header_dict, indexing_dict): # add file_tag to header_array
    # Precision flag should be same across procs ??
    header = f.read(16)
    mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')

    header_array = struct.unpack('6b', header[4:10])
    header_dict[i] = (list(header_array))
    header_version, dir_version, endian_flag, precision_flag, state_file_suffix_length, partition_flag = header_array

    if endian_flag == 1:
        local_file_tag = '>'
    else:
        local_file_tag = '<'

    # Read Indexing #
    offset = -16
    f.seek(offset, os.SEEK_END)
    indexing_dict[i] = struct.unpack('4i', f.read(16))

    return local_file_tag



def StartRead(i, file_name, header_dict, indexing_dict, global_state_maps):
    local_file_tag = None
    offset = None

    with open(file_name, 'rb') as f:
        local_file_tag = ReadHeader(i, f, local_file_tag, header_dict, indexing_dict)

        global_state_maps[i], offset = ReadStateMaps(i, f, local_file_tag, indexing_dict)

        


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


    for i,file_name in enumerate(file_names):

        p = Process(target=StartRead, args=(i, file_name, header_dict, \
                                            indexing_dict, global_state_maps))

        p.start()
        p.join()

    print(header_dict)
    print(indexing_dict)
    #print(global_state_maps)

if __name__ == '__main__':
    main()
