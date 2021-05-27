 #!/usr/bin/env python3
from multiprocessing import Process, Manager
from enum import Enum
import collections
from collections import OrderedDict
import os
import struct
import sys
import re
import psutil
from mili_combiner_optparser import parseParameters, gather_A_files
from datetime import datetime as dt
import platform
from mili_reader_lib import DirectoryType, Superclass, ConnWords, DataType, ExtSize, AggregateType, DataOrganization


    ######################## classes / indexes #########################

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
    DIMS = 3
    LIST_SIZE = 4 # BUT NEED TO ADD ORDER

class SvarS(Enum):
    SV_NAME = 0
    TITLE = 1
    SV_NAMES = 2
    # INNER SV NAME, TITLE

class InnerSvar(Enum):
    INTS = 0
    S = 1
    OWNER = 2 # The SV that has this inner sv info as part of its entries
    PROCS = 3

# Change the Mesh Enums
class MeshElemConns(Enum):
    SUP_CLASS = 0
    QTY_BLOCKS = 1
    ELEM_BLOCKS = 2
    EBUF = 3

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
    START_STOP = 3

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

# '''
# Every direcotry has a type that dictates what informaiton
# is in the directory.
# '''
# class DirectoryType(Enum):
#     NODES = 0
#     ELEM_CONNS = 1
#     CLASS_IDENTS = 2
#     STATE_VAR_DICT = 3
#     STATE_REC_DATA = 4
#     MILI_PARAM = 5
#     APPLICATION_PARAM = 6
#     CLASS_DEF = 7
#     SURFACE_CONNS = 8
#     TI_PARAM = 9
#     QTY_DIR_ENTRY_TYPES = 10


# '''
# The superclass tells what class an object belongs to.
# '''
# class Superclass(Enum):
#     M_UNIT = 0
#     M_NODE = 1
#     M_TRUSS = 2
#     M_BEAM = 3
#     M_TRI = 4
#     M_QUAD = 5
#     M_TET = 6
#     M_PYRAMID = 7
#     M_WEDGE = 8
#     M_HEX = 9
#     M_MAT = 10
#     M_MESH = 11
#     M_SURFACE = 12
#     M_PARTICLE = 13
#     M_TET10 = 14
#     M_QTY_SUPERCLASS = 16
#     M_SHARED = 100
#     M_ALL = 200
#     M_INVALID_LABEL = -1

# '''
# Connections per superclass
# '''
# class ConnWords(Enum):
#     M_UNIT = 0
#     M_NODE = 0
#     M_TRUSS = 4
#     M_BEAM = 5
#     M_TRI = 5
#     M_QUAD = 6
#     M_TET = 6
#     M_PYRAMID = 7
#     M_WEDGE = 8
#     M_HEX = 10
#     M_MAT = 0
#     M_MESH = 0
#     M_SURFACE = 0
#     M_PARTICLE = 3
#     M_TET10 = 12

# '''
# The numbering system used by Mili files to state type.
# '''
# class DataType(Enum):
#     M_STRING = 1
#     M_FLOAT = 2
#     M_FLOAT4 = 3
#     M_FLOAT8 = 4
#     M_INT = 5
#     M_INT4 = 6
#     M_INT8 = 7

# '''
# The size of each data type in bytes.
# '''
# class ExtSize(Enum):
#     M_STRING = 1
#     M_FLOAT = 4
#     M_FLOAT4 = 4
#     M_FLOAT8 = 8
#     M_INT = 4
#     M_INT4 = 4
#     M_INT8 = 8

# '''
# The aggregate type describes what a state variable can be.
# '''
# class AggregateType(Enum):
#     SCALAR = 0
#     VECTOR = 1
#     ARRAY = 2
#     VEC_ARRAY = 3

# '''
# The organization of data in a subrecord is either ordered by variable
# or object.
# '''
# class DataOrganization(Enum):
#     RESULT = 0
#     OBJECT = 1


    
    ######################### commit funtions ##########################
    

def __InitWrite(afile_name, mili_taur, header_array):
    header_offset = 0
    indexing_offset = -16

    with open(afile_name, 'wb') as a_file:
        a_file.seek(header_offset)
        #Encode string, add 6 header ints, add 6 more ints to create full length 16 byte header
        mili_taur = bytes(str(mili_taur), 'ascii')
        a_file.write(mili_taur)

        header_array = list(header_array)
        header_array.extend([0,0,0,0,0,0])
        #print("PRINT INIT", header_array)
        header_array = struct.pack('12b', *header_array) # don't recpy to same variable
        a_file.write(header_array)
        #print("PRINT init", a_file.tell())


def __CommitIndexing(a_file, offset, indexing_array, null_termed_names_bytes, num_dirs):
    # Write indexing array at end

    indexing_array[IndexingArray.NULL_TERMED_NAMES_BYTES.value] = null_termed_names_bytes
    indexing_array[IndexingArray.NUM_DIRS.value] = num_dirs
    #print("PRINT index", indexing_array)
    a_file.seek(offset)
    indexing_array = struct.pack('4i', *indexing_array)
    a_file.write(indexing_array)
    
    return offset + 16


def __CommitStateMaps(a_file, offset, file_tag, state_maps, number_state_maps, global_dirs):
    num_items_in_sm = 4
    
    # Check number state maps against length of array we have
    if number_state_maps != (len(state_maps)/4):
        # 4 numbers per state map, fix hardcode later
        print("State Map Error")

    state_map_length = 20
    a_file.seek(offset)
    
    # for state_map_num in range(number_state_maps):
    #     state_map = state_maps[state_map_num*4:state_map_num*4 + 4]
    #     state_map = struct.pack('=iqfi', *state_map)        
    #     a_file.write(state_map)

    print("STATE MAPS", state_maps)

    format_str = '=' + 'iqfi'*number_state_maps
    packed_state_maps = struct.pack(format_str, *state_maps)
    a_file.write(packed_state_maps)
       
    return offset + number_state_maps*state_map_length
        
    
def __CommitDirectories(a_file, offset, header_array, global_dirs, dir_order, testfile, no_off_testfile):    

    directory_length = 4 * 6
    state_map_length = 20
    int_long = 'i'

    dir_version = header_array[HeaderArray.DIR_VERSION.value]

    if dir_version > 2:
        directory_length = 8 * 6
        int_long = 'q'

    # num_state_maps = this_indexing_array[IndexingArray.NUM_STATE_MAPS.value]
    # num_dirs = this_indexing_array[IndexingArray.NUM_DIRS.value]

    # offset -= state_map_length * num_state_maps + directory_length * num_dirs  # make this not a hard coded 6 eventually
    # f.seek(offset, os.SEEK_END)
    # for i in range(1, 1 + num_dirs): # for num dirs
    #     byte_array = f.read(directory_length) ###
    
    # If appending, in order, need to use seek and offsets?
    all_dirs = []
    num_dirs = 0
    for dir_type in dir_order:
        for dir_name in dir_order[dir_type]:
            all_dirs = all_dirs + global_dirs[dir_type][dir_name]
            num_dirs += 1

    print("dir offset", a_file.tell(), offset)
    a_file.seek(offset)
    #print("PRINT dir offset2 + dir", a_file.tell(), offset, all_dirs)
    print(len(all_dirs), "All Dirs", all_dirs)
    all_dirs = struct.pack(str(6*num_dirs) + int_long, *all_dirs)
    a_file.write(all_dirs)
    print("dir offset3", a_file.tell(), offset, "num dirs, offset + dir_len*num_dirs:", num_dirs, offset + (directory_length*num_dirs))
    return offset + (directory_length*num_dirs), num_dirs


def __RoundDecimals(unrounded_floats):
    # Mesh (floats), State Maps
    for i,fl in enumerate(unrounded_floats):
        unrounded_floats[i] = float("%1.6e"%fl)

    return unrounded_floats
    
    
def __CommitMesh(a_file, offset, header_array, global_dirs, global_dir_strings, mesh_dim, floats, elem_conns, node_local_id_to_global_id, global_class_idents, testfile, no_off_testfile):
    ### CLASS DEFS ### - Global directory is same as local, only need to commit directories
    print("MESH START", a_file.tell(), offset)
    #print("PRINT CLAS IDENTS", global_class_idents)
    ### CLASS IDENTS ###
    for short_name in global_class_idents:
        # Change offset in directories
        temp_dir = global_dirs[DirectoryType.CLASS_IDENTS.value] #FIX: #pass class ident etc to commitparams
        temp_dir[short_name][DirArray.OFFSET_IDX.value] = offset
        global_dirs[DirectoryType.CLASS_IDENTS.value] = temp_dir

        testfile.seek(offset)
        testfile.write(str(global_class_idents[short_name]))
        no_off_testfile.write("\nMESH, name: " + short_name + ", " + str(global_class_idents[short_name]))
        
        class_ident = struct.pack('3i', *(global_class_idents[short_name]))
        a_file.seek(offset)
        a_file.write(class_ident)
        offset = offset + global_dirs[DirectoryType.CLASS_IDENTS.value][short_name][DirArray.LENGTH_IDX.value]
        
    ### NODES ###
    node_start = 1
    node_stop = len(floats)

    # Create nodes dictionary entry
    node_type_idx = DirectoryType.NODES.value
    node_mesh_id = 0
    node_num_nodes = int(node_stop/mesh_dim)
    node_str_qty_idx = len(global_dir_strings[DirectoryType.NODES.value]['node'][DirStringsArray.STRINGS.value])
    node_offset = offset

    # Get precision before calculating bytes - header array from every proc is the same (start + stop + prec*floats)
    #prec = header_array[HeaderArray.PRECISION_FLAG.value] # find how to interpret prec flag??
    prec = 4 #FIX
    node_len = 8 + prec*node_stop

    # Add directory to global dirs
    global_dirs[node_type_idx] = {}
    temp_dirs = global_dirs[node_type_idx]
    temp_dirs['node'] = [node_type_idx, node_mesh_id, node_num_nodes, node_str_qty_idx, node_offset, node_len]
    global_dirs[node_type_idx] = temp_dirs

    # Clean up coords
    print("pre cleaned floats", floats)
    floats = __RoundDecimals(floats)
    print("cleaned floats", floats)

    # add start, stop to beginning of floats array
    floats.insert(0, node_stop)
    floats.insert(0, node_start)
    print("PRINT NODES", floats)

    testfile.seek(offset)
    testfile.write(str(floats))
    no_off_testfile.write("\nMESH, nodes: " + str(floats))
    
    # Write nodes to a_file
    floats = struct.pack('2i' + str(node_stop) + 'f', *floats) # * to unpack not a pointer, floats are double prec in python?
    a_file.seek(offset)
    a_file.write(floats)
    
    offset = offset + node_len
    
    ### ELEM_CONNS ###
    # Go through different kinds of elems
    __CheckElemConns(elem_conns)

    for elem_type in elem_conns.keys():
        # Create elem_conns entries (each short_name has own entry)
        ec_type_idx = DirectoryType.ELEM_CONNS.value
        ec_mesh_id = 0
        ec_num_elems = elem_conns[elem_type][MeshElemConns.ELEM_BLOCKS.value][MeshECElemBlocks.STOP.value]
        ec_str_qty_idx = len(global_dir_strings[DirectoryType.ELEM_CONNS.value][elem_type][DirStringsArray.STRINGS.value])
        ec_offset = offset

        ec_entry_start = elem_conns[elem_type][:2] + [num for num in elem_conns[elem_type][2]]
        print("EC ENTRY START", ec_entry_start)

        prec = 4
        sup_class_name = Superclass(elem_conns[elem_type][MeshElemConns.SUP_CLASS.value]).name
        sup_class_conns = ConnWords[sup_class_name].value
        print(elem_type, "PRINT ELEM CONNS ", ec_entry_start, elem_conns[elem_type][MeshElemConns.EBUF.value])

        testfile.seek(offset)
        testfile.write(str(ec_entry_start) + str(elem_conns[elem_type][MeshElemConns.EBUF.value]))
        no_off_testfile.write("\nMESH elem conns, " + elem_type + ": " + str(ec_entry_start) + str(elem_conns[elem_type][MeshElemConns.EBUF.value]))
        
        # Write length of entire elem conns entry for this elem type
        ec_num_ints = 2 + 2*elem_conns[elem_type][MeshElemConns.QTY_BLOCKS.value] + ec_num_elems*sup_class_conns
        ec_entry = struct.pack(str(ec_num_ints) + 'i', *(ec_entry_start + elem_conns[elem_type][MeshElemConns.EBUF.value]))

        a_file.seek(offset)
        a_file.write(ec_entry)

        ec_len = prec*ec_num_ints

        # Add directory to global dirs
        if ec_type_idx not in global_dirs:
            global_dirs[ec_type_idx] = {}
        temp_dirs = global_dirs[ec_type_idx]
        temp_dirs[elem_type] = [ec_type_idx, ec_mesh_id, ec_num_elems, ec_str_qty_idx, ec_offset, ec_len]
        global_dirs[ec_type_idx] = temp_dirs
        print(offset + ec_len, elem_type, "temp dirs", temp_dirs[elem_type])
        
        offset = offset + ec_len
    print("MESH STOP", a_file.tell(), offset)
    return offset


def __CheckElemConns(elem_conns):
    for elem_type in elem_conns.keys():
        ec_num_elems = elem_conns[elem_type][MeshElemConns.ELEM_BLOCKS.value][MeshECElemBlocks.STOP.value]
        stride = int(len(elem_conns[elem_type][3]) / ec_num_elems)
        print("STRIDE", stride, len(elem_conns[elem_type][3]), ec_num_elems, elem_conns[elem_type])

        count = 0
        ec_map = {}
        while count < ec_num_elems:
            conns_list = elem_conns[elem_type][3][count*stride:count*stride + stride]
            temp_tup_key = tuple(conns_list[:len(conns_list)-2])
            temp_tup_val = tuple(conns_list[len(conns_list)-2:])
            print("CONNS LIST", conns_list, temp_tup_key, temp_tup_val)
            if tuple(temp_tup_key) in ec_map:
                print("bad", temp_tup_key, temp_tup_val)
            else:
                ec_map[temp_tup_key] = []
           
            ec_map[temp_tup_key].append(temp_tup_val)
            count += 1

        print("EC MAP", ec_map)
          
    

# def __MakeMultOfFour(num):
#     if num % 4 != 0:
#         num += ((num % 4))
#     return num

# def __MakeMultOfEight(num):
#     if num % 8 != 0:
#         num += ((num % 8))
#     return num

def __MakeMultOfFour(num):
    if num % 4 != 0:
        num += (4 - (num % 4))
    return num

def __MakeMultOfEight(num):
    if num % 8 != 0:
        num += (8 - (num % 8))
    return num

def __RoundNumBytes(num_bytes_calculated, all_bytes):
    # Params, Subrecs
    added = 0
    diff = num_bytes_calculated-len(all_bytes)

    while added < diff:
        all_bytes.append(0)
        added += 1

def __RoundNumPadding(num_bytes_calculated, all_bytes):
    added = 0
    diff = num_bytes_calculated-len(all_bytes)

    while added < diff:
        all_bytes = all_bytes + bytes('\x00', 'ascii') #'\x00'
        added += 1
    print("All Bytes", all_bytes)

    return all_bytes


def __CommitSvars(a_file, offset, global_dirs, global_svar_s, global_svar_ints, testfile, no_off_testfile):
    # Need to check if svar_bytes is larger than single file size ?
    # svar_bytes = svar_header[1]
    # if svar_bytes > FILE_MAX

    combined_header = []
    all_svar_ints = []
    all_svar_s = []
    #all_svar_s_bytes = ''
    all_svar_s_bytes = bytearray('', 'ascii')
    svar_words, svar_bytes = 0, 0 # svar_words - 2 = num_ints
    for svar_name in global_svar_ints:
        size_ints = len(global_svar_ints[svar_name])
        #size_s = len(global_svar_s[svar_name])
        
        svar_words += size_ints
        #svar_bytes += size_s

        for i, sv_piece in enumerate(global_svar_s[svar_name]):
            size_s = len(sv_piece) + 1 # get length before converting
            no_off_testfile.write("\nSVAR s_bytes: " + svar_name + ", " + str(sv_piece)) 
            sv_piece = bytes(sv_piece, 'ascii')
            #all_svar_s_bytes = all_svar_s_bytes + sv_piece + '\x00'
            all_svar_s_bytes = all_svar_s_bytes + sv_piece
            all_svar_s_bytes.append(0)
            svar_bytes += size_s

        all_svar_ints.extend(global_svar_ints[svar_name])
        #all_svar_s.extend(global_svar_s[svar_name])

    # Round up to multiple of 4
    #if svar_bytes % 4 != 0:
    #    svar_bytes += (4 - (svar_bytes % 4))
    svar_bytes = __MakeMultOfFour(svar_bytes)
        
    combined_header = [svar_words, svar_bytes]

    a_file.seek(offset)

    #testfile.seek(offset)
    #testfile.write(str(combined_header))
    #testfile.write(str(all_svar_ints))
    #testfile.write(str(all_svar_s_bytes))
    no_off_testfile.write("\nSVAR combined header: " + str(combined_header))
    no_off_testfile.write("\nSVAR ints: " + str(all_svar_ints))
   
    
    # Write header
    #print("PRINT SVAR HEADER", combined_header, offset)
    svar_header = struct.pack('2i', *combined_header)
    a_file.write(svar_header)

    # Write ints
    #print("PRINT SVAR I", all_svar_ints)
    svar_ints = struct.pack(str(svar_words) + 'i', *all_svar_ints)
    a_file.write(svar_ints)

    # Write s
    __RoundNumBytes(svar_bytes, all_svar_s_bytes)
    print("PRINT SVAR S", svar_words, svar_bytes, len(all_svar_s_bytes), len(combined_header)*4 + combined_header[0]*4 + combined_header[1], a_file.tell())
    #all_svar_s_bytes = bytes(all_svar_s_bytes, 'ascii')
    a_file.write(all_svar_s_bytes)
    print("SVAR end", a_file.tell())

    entry = [0] * 6
    entry[DirArray.OFFSET_IDX.value] = offset
    entry[DirArray.LENGTH_IDX.value] = len(combined_header)*4 + combined_header[0]*4 + combined_header[1]

    # Add directory - will this ever need to be appended instead?
    global_dirs[DirectoryType.STATE_VAR_DICT.value] = {}
    temp_dict = global_dirs[DirectoryType.STATE_VAR_DICT.value]
    temp_dict['svars'] = entry
    global_dirs[DirectoryType.STATE_VAR_DICT.value] = temp_dict

    offset = offset + entry[DirArray.LENGTH_IDX.value]
    return offset


def __CommitNames(a_file, offset, global_dir_strings, global_dirs, testfile, no_off_testfile):
    # null_termed_names_bytes

    dir_order = OrderedDict() # dictionaries don't keep order, so keep track of order writing strings so they align

    all_dir_strings = ''
    #all_dir_strings = bytearray('', 'ascii')
    all_string_sum = 0

    for dir_type in global_dirs:
        if dir_type not in dir_order:
            dir_order[dir_type] = []
            
        if dir_type in global_dir_strings:
            if len(global_dir_strings[dir_type]) != len(global_dirs[dir_type]):
                print("Commit Names Error: Mismatch in Directories and Directory Strings")

            for dir_name in global_dir_strings[dir_type]:
                dir_order[dir_type].append(dir_name)

                # String Entries have 2 possible formats - FIX (list of str vs dict of list of str)
                strings = global_dir_strings[dir_type][dir_name]
                if isinstance(strings, dict):
                    strings = strings[1]

                for st in strings:
                    print("String Sum ", st, offset+all_string_sum, offset+all_string_sum+len(st)+1) 
                    all_string_sum += len(st)+1
                    no_off_testfile.write("\nNAMES dir_name: " + dir_name + ", name/dir_string: " + str(st))
                    all_dir_strings = all_dir_strings + st + '\x00'
                    #all_dir_strings = all_dir_strings + bytes(st, 'ascii')
                    #all_dir_strings.append(0)
                
        else:
            # Add to dir_order anyway so can just loop through in CommitDirectories without checking
            for dir_name in global_dirs[dir_type]:
                dir_order[dir_type].append(dir_name)
                
    all_dir_strings = bytes(all_dir_strings, 'ascii')
    
    # Round all_string_sum up to a multiple of four if not
    #if all_string_sum % 4 != 0:
    #    all_string_sum += (4 - (all_string_sum % 4))
    all_string_sum = __MakeMultOfFour(all_string_sum)

    a_file.seek(offset)
    print("\nPRINT NAMES 1", len(all_dir_strings), offset, all_string_sum)
    all_dir_strings = __RoundNumPadding(all_string_sum, all_dir_strings)
    print("\nPRINT NAMES 2", len(all_dir_strings), offset, all_string_sum)

    #all_dir_strings.encode('ascii')
    
    #a_file.write(all_dir_strings)
    print("all_string_sum", offset, str(all_string_sum), all_dir_strings)
    packed_dir_strs = struct.pack(str(all_string_sum) + 's', all_dir_strings)
    a_file.write(packed_dir_strs)
    
    # Update and Return offset
    print("Names NULL TERMED NAMES BYTES", all_string_sum, offset, offset+all_string_sum)
    offset = offset + all_string_sum
    return offset, all_string_sum, dir_order
    


def __ExtractParam(param, type_rep, max_len):
    if isinstance(param, list) is False: return param
    
    save_param = None
    while isinstance(param, type_rep) is False:
        save_param = param
        param = param[0]

    if type_rep == str:
        save_param = [i for i in save_param if i.strip("'")]
    if len(save_param) > 1:
        return save_param
    
    return param


def __AddNewDate():
    # Create date combined param
    now = dt.now()

    date_string = now.strftime("%a %b %d %H:%M:%S %Y")
    return date_string

def __AddMiliParams(global_params, global_dirs, global_dir_strings):
    origin_keys = [('pcombiner_alpha','pcombiner_alpha'), ('origin lib version','lib version'),
                   ('origin arch name','arch name'), ('origin host name','host name'),
                   ('origin date','date'), ('origin username','username')]
    
    param_map = { 'pcombiner_alpha': 'pcombiner_alpha',
                      'lib version': global_params['lib version'],
                      'host name': platform.node(),
                      'arch name': platform.system() + ' + ' + platform.release() + platform.node(),
                      'date': __AddNewDate(),
                      'username': os.getlogin() }
    
    type_id = DirectoryType.MILI_PARAM.value

    for origin_param,param in origin_keys:
        if origin_param != 'pcombiner_alpha':
            global_params[origin_param] = global_params[param]
        global_params[param] = param_map[param]

        temp_dirs = global_dirs[type_id]
        print("adding new params", global_params[param], len(global_params[param]), __ExtractParam(global_params[param], str, 1), len(__ExtractParam(global_params[param], str, 1)))
        temp_dirs[param][DirArray.LENGTH_IDX.value] = __MakeMultOfEight(len(__ExtractParam(global_params[param], str, 1)))
        print(param, "temp dir", len(global_params[param]), temp_dirs[param][DirArray.LENGTH_IDX.value])
        global_dirs[type_id] = temp_dirs


def __Debug_CheckOverlap(arr):
    for i,a in enumerate(arr):
        for j,aa in enumerate(arr):
            if ( a == aa and i != j ):
                print("OVERLAP ERROR", a, " indexes: ", i, j)

def __Debug_CheckUnpacked(fmt_str, unpacked, len_orig_byte_array):
    # unpacked should be passed in as a list not tuple
    # Check len and contents
    orig_len = len(unpacked)
    print("CheckUnpacked", fmt_str, unpacked)
    # Pack and unpack and check - assumes unpacked is a list
    packed = struct.pack(fmt_str, *unpacked)

    # Check if the length of this packed byte array is same as the byte array that
    # # was originally read from an A file
    len_new_byte_array = len(packed)

    new_unpacked = list(struct.unpack(fmt_str, packed))

    new_len = len(new_unpacked)

    if orig_len != new_len or unpacked != new_unpacked or len_orig_byte_array != len_new_byte_array:
        print("Error - __Debug_CheckUnpacked() Unpacked Lengths:", orig_len, new_len, "Packed Lengths:", len_orig_byte_array, len_new_byte_array, " Unpacked Arrays:", unpacked, new_unpacked)
    else:
        print("__Debug_CheckUnpacked() Unpacked Lengths Equal = True: ", orig_len, new_len)
        print("__Debug_CheckUnpacked() Packed Lengths Equal = True: ", len_orig_byte_array, len_new_byte_array)
        print("__Debug_CheckUnpacked() True: ", unpacked, new_unpacked)

def __Debug_CheckPacked(fmt_str, packed):
    # Check len in bytes
    orig_len = len(packed)

    # Unpack
    unpacked = struct.unpack(fmt_str, *packed)

    # Repack and check
    new_packed = struct.pack(fmt_str, *unpacked)

    new_len = len(new_packed)

    if orig_len != new_len or packed != new_packed:
        print("Error - __Debug_CheckPacked()", packed)
    
    
def __CommitParams(a_file, offset, global_params, global_dirs, global_dir_strings, global_node_labels, global_elem_labels, global_elem_offsets,     header_array, dim, floats, elem_conns, node_local_id_to_global_id, global_class_idents, testfile, no_off_testfile):
    type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}
    print("PARAMS START", a_file.tell(), offset)
    param_ids = [ DirectoryType.MILI_PARAM.value, DirectoryType.APPLICATION_PARAM.value, DirectoryType.TI_PARAM.value ]
    elem_types = global_elem_labels.keys()
    
    mesh_committed = True #False
    total_param_len = 0
    byte_len = 0
    for type_id in param_ids:
        for param_name in global_dir_strings[type_id]:
            param = global_params[param_name]
            print("PARAM PRINT", param_name, param)
            # Mesh commit interleaved with param commit - didn't change anything
            param_name_list = ['GLOBAL_COUNT', 'GLOBAL_IDS_LOCAL_MAP', 'GLOBAL_IDS', 'LOCAL_COUNT', 'Node Labels', 'Element Labels']
            if any(substring in param_name for substring in param_name_list) and not mesh_committed:
                print("pre MESH offset", offset)
                #offset = __CommitMesh(a_file, offset, header_array, global_dirs, global_dir_strings, global_params['mesh dimensions'], floats, elem_conns, node_local_id_to_global_id, global_class_idents)
                print("after MESH offset", offset)
                mesh_committed = True

            # get dir entry - use length
            dir_entry = global_dirs[type_id][param_name]

            param_type = dir_entry[DirArray.MOD_IDX1.value]

            type_rep = type_to_str[DataType(param_type).name]
            type_value = ExtSize[DataType(param_type).name].value

            # Find length we need to write out this param
            full_len = dir_entry[DirArray.LENGTH_IDX.value]
            num = int(full_len / type_value)

            # if param_name == 'date':
            #     # Date --> Origin Date
            #     origin_date = global_params[param_name]
            #     global_params['origin date'] = origin_date
            #     global_dirs[type_id]['origin date'] = dir_entry
                
            #     global_params[param_name] = __AddNewDate()
            #     full_len = len(global_params[param_name])
            #     num = int(full_len / type_value)

            #     # Update dir_entry
            #     dir_entry[DirArray.LENGTH_IDX.value] = len(global_params[param_name]) * type_value

            if 'GLOBAL_COUNT' in param_name:
                # # GLOBAL_COUNT-<elem>
                # Can combine GLOBAL_COUNT and LOCAL_COUNT if only use '_COUNT' for search index
                elem_name = param_name[param_name.index('GLOBAL_COUNT-') + len('GLOBAL_COUNT-'):]
                superclass = Superclass(__FindSuperclass(global_dirs, global_dir_strings, elem_name)).name

                if (superclass, elem_name) not in elem_types:
                    print("CommitParams Error: GLOBAL_COUNT:", superclass, elem_name)

                global_params[param_name] = [len(global_elem_labels[(superclass, elem_name)])]

                
            elif 'GLOBAL_IDS_LOCAL_MAP' in param_name:
                elem_start_idx = param_name.index('Sname-') + len('Sname-')
                elem_end_idx = param_name.rfind('/')                
                elem_name = param_name[elem_start_idx : elem_end_idx]

                superclass = Superclass(__FindSuperclassCommit(global_dirs, global_dir_strings, elem_name)).name

                if (superclass, elem_name) not in elem_types:
                    print("CommitParams Error: GLOBAL_IDS_LOCAL_MAP:", superclass, elem_name)

                global_params[param_name] = [1, len(global_elem_labels[(superclass, elem_name)])] + global_elem_labels[(superclass, elem_name)]

                # Update the length of this parameter in dir_entry
                dir_entry[DirArray.LENGTH_IDX.value] = len(global_params[param_name]) * type_value

                
            elif 'GLOBAL_IDS' in param_name:
                # Same as 'Elem Ids'
                elem_start_idx = param_name.index('Sname-') + len('Sname-')
                elem_end_idx = param_name.rfind('/')                
                elem_name = param_name[elem_start_idx : elem_end_idx]

                num_of_elem = global_elem_offsets[elem_name][-1]

                global_params[param_name] = [1, num_of_elem] + [i+1 for i in range(num_of_elem)]

                # Update the length of this parameter in dir_entry
                dir_entry[DirArray.LENGTH_IDX.value] = len(global_params[param_name]) * type_value

                
            elif 'LOCAL_COUNT' in param_name:
                # # LOCAL_COUNT-<elem>
                elem_name = param_name[param_name.index('LOCAL_COUNT-') + len('LOCAL_COUNT-'):]
                superclass = Superclass(__FindSuperclass(global_dirs, global_dir_strings, elem_name)).name

                if (superclass, elem_name) not in elem_types:
                    print("CommitParams Error: LOCAL_COUNT:", superclass, elem_name)
                    
                global_params[param_name] = [len(global_elem_labels[(superclass, elem_name)])]

                
            elif 'Node Labels' in param_name:
                # first, last, node_labels
                __Debug_CheckOverlap(global_params[param_name])
                global_params[param_name] = [1, len(global_node_labels)] + global_node_labels
                dir_entry[DirArray.LENGTH_IDX.value] = len(global_params[param_name]) * type_value
                print("NODE LABELS", global_params[param_name])

                
            elif 'Element Labels' in param_name:
                # first, last, ints
                sup_class_idx = param_name.index('Scls-') + len('Scls-')
                sup_class_end_idx = param_name.index('/', sup_class_idx, len(param_name))
                class_idx = param_name.index('Sname-') + len('Sname-')
                class_end_idx = param_name.index('/', class_idx, len(param_name))

                sup_class = param_name[sup_class_idx : sup_class_end_idx]
                clas = param_name[class_idx : class_end_idx]

                elem_labels = global_elem_labels[(sup_class, clas)]
                
                if 'ElemIds' in param_name:
                    elem_labels = [1, len(elem_labels)] + [i+1 for i in range(len(elem_labels))]
                else:
                    elem_labels = [1, len(elem_labels)] + elem_labels

                global_params[param_name] = elem_labels
                dir_entry[DirArray.LENGTH_IDX.value] = len(elem_labels) * type_value

                print(param_name, "ELEM LABELS", global_params[param_name])


            param = global_params[param_name] # cut num of variables in function
            print("PRINT PARAM (", param_name, ") ", global_params[param_name])
            
            if type_rep == 's':
                param = __ExtractParam(param, str, 1)
                global_params[param_name] = param
                
                testfile.seek(offset)
                testfile.write(param)
                no_off_testfile.write("\nPARAM s, param_name: " + param_name + ", param: " + param)
                
                print(byte_len, "s param len pre", param, len(str(param)), dir_entry[DirArray.LENGTH_IDX.value])
                #param = bytes(str(param), 'ascii')
                param = bytearray(param, 'ascii')
                param.append(0)
                
                dir_entry[DirArray.LENGTH_IDX.value] = __MakeMultOfEight(len(param))
                
                __RoundNumBytes(dir_entry[DirArray.LENGTH_IDX.value], param)

                byte_len+=dir_entry[DirArray.LENGTH_IDX.value]#len(param)
                print("length of bytes in str parm", len(param))
                
            elif type_rep == 'i':
                param = __ExtractParam(param, int, 1)
                global_params[param_name] = param

                testfile.seek(offset)
                testfile.write(str(param))
                print(param_name, "PARAMS I", str(param))
                no_off_testfile.write("\nPARAM i, param_name: " + param_name + ", param: " + str(param))
                
                if isinstance(param, int):
                    param = struct.pack(type_rep, param)

                    print("PARAM unpacked i, param_name: ", param_name, ", ", struct.unpack(type_rep, param))
                    
                else:
                    temp_len = len(param) #
                    param = struct.pack(str(len(param)) + type_rep, *param)

                    print("PARAM unpacked i, param_name: ", param_name, ", ", struct.unpack(str(temp_len)+type_rep, param))
                    
            elif type_rep == 'f':
                #param = __ExtractParam(param, float, 1) # use int instead of float bc first two indexes are ints
                param = __ExtractParam(param, int, 1)
                global_params[param_name] = param

                testfile.seek(offset)
                testfile.write(str(param))
                no_off_testfile.write("\nPARAM f, param_name: " + param_name + ": " + str(param))                
                
                param = struct.pack(str(len(param)) + type_rep, *param)                
            else:
                param = struct.pack(str(len(param)) + type_rep, *param)
                no_off_testfile.write("\nPARAM else, param_name: " + param_name + ": " + str(param))                 

            a_file.seek(offset)
            a_file.write(param)

            # Change the directory offset which is offset + total_param_len
            dir_entry[DirArray.OFFSET_IDX.value] = offset #+ total_param_len
            temp_dir = global_dirs[type_id]
            temp_dir[param_name] = dir_entry
            global_dirs[type_id] = temp_dir
            
            # After writing to file, add length to total_param_len
            total_param_len = total_param_len + dir_entry[DirArray.LENGTH_IDX.value]
            offset = offset + dir_entry[DirArray.LENGTH_IDX.value]
    print("PARAMS STOP", a_file.tell(), offset, byte_len,total_param_len)
    return offset
    

def __CommitSubrecords(a_file, offset, global_dirs, global_dir_strings, global_svar_s, global_svar_ints, global_svar_inners, global_subrec_idata, global_subrec_cdata, testfile, no_off_testfile):

    all_idata = []
    #all_cdata = ''
    all_cdata = bytearray('', 'ascii')
    all_cdata_size = 0
    total_num_chars = 0

    for subrec in global_subrec_idata:
        if subrec not in global_subrec_cdata:
            # There should be same number of idata and cdata entries
            print("Commit Subrecords Error: Cdata Missing")

        idata = global_subrec_idata[subrec]
            
        # Calc number of elems in this subrecord
        subrec_num_elems = 0
        for blk_num in range(idata[SubrecIdata.QTY_ID_BLOCKS.value]):
            #print("idata", idata[SubrecIdata.START_STOP.value], idata)
            block = idata[SubrecIdata.START_STOP.value][blk_num]
            subrec_num_elems += block[1] - block[0] + 1

        # Subrec Idata
        if not idata[SubrecIdata.START_STOP.value]:
            idata = idata[:SubrecIdata.START_STOP.value]
        else:
            temp = idata[:SubrecIdata.START_STOP.value]
            for tup in __ExtractParam(idata[SubrecIdata.START_STOP.value], tuple, 2):
                if isinstance(tup, int):
                    temp.append(tup)
                elif isinstance(tup, tuple):
                    temp.extend(tup)
            idata = temp

        # Subrec Cdata
        subrec_svars = global_subrec_cdata[subrec][SubrecCdata.SVARS.value]
        lump_atoms = []
        lump_sizes = []
        lump_offsets = []
        count = 0
        sz = 0
        
        subrec_size = 0

        svar_org = idata[SubrecIdata.ORG.value]
        if svar_org == DataOrganization.OBJECT.value:
            for svar in subrec_svars:
                atom_size = ExtSize[DataType(global_svar_ints[svar][SvarInts.DATA_TYPE.value]).name].value
                atoms = __atom_qty(svar, global_svar_ints, global_svar_s, global_svar_inners)

                total_atoms = atoms
                count += total_atoms
                sz += total_atoms * atom_size
            lump_atoms.append(count)
            lump_sizes.append(sz)

            superclass = __FindSuperclass(global_dirs, global_dir_strings, subrec)
            if superclass != Superclass.M_SURFACE.value:
                subrec_size = sz * subrec_num_elems
            else:
                subrec_size = sz

        elif svar_org == DataOrganization.RESULT.value:
            for svar in subrec_svars:
                atoms = __atom_qty(svar, global_svar_ints, global_svar_s, global_svar_inners)
                total_atoms = atoms

                lump_atoms.append(subrec_num_elems * total_atoms)
                lump_sizes.append(subrec_num_elems * total_atoms * ExtSize[DataType(global_svar_ints[svar][SvarInts.DATA_TYPE.value]).name].value)

            lump_offsets.append(0)
            subrec_qty_svars = idata[SubrecIdata.QTY_SVARS.value]
            for j in range(1, subrec_qty_svars):
                lump_offsets.append(lump_offsets[j - 1] + lump_sizes[j - 1])

            j = subrec_qty_svars - 1
            subrec_size = lump_offsets[j] + lump_sizes[j]

        # Do we need to keep track of where in container this subrec is? subrec_offset = all_cdata_size
        all_cdata_size += subrec_size
        #print(subrec, subrec_size, "UPDATE CDATA", all_cdata_size)

            
        cdata = global_subrec_cdata[subrec][:SubrecCdata.CLASS_NAME.value+1] + global_subrec_cdata[subrec][SubrecCdata.SVARS.value]

        #temp_cdata = ''
        temp_cdata = bytearray('', 'ascii')
        for c in cdata:
            # Calculate MOD_IDX2
            total_num_chars += len(c)+1
            no_off_testfile.write("\nSUBREC " + str(subrec) + "cdata: " + str(c))
            #temp_cdata = temp_cdata + c + '\x00'
            temp_cdata = temp_cdata + bytes(c, 'ascii')
            temp_cdata.append(0)

        all_idata.extend(idata)
        all_cdata = all_cdata + temp_cdata

    # Idata has 4 ints in front of it (read in ReadSubrecords as _,_,all_cdata_size,num_subrecs) ? 
    pre_idata = [0] * 4
    pre_idata[2] = all_cdata_size
    pre_idata[3] = len(global_subrec_idata)
    all_idata = pre_idata + all_idata

    # Directory Creation:
    entry = [0] * 6

    # Round MOD_IDX2 up to multiple of 4
    #if total_num_chars % 4 != 0:
    #total_num_chars += (4 - (total_num_chars % 4))
    total_num_chars = __MakeMultOfFour(total_num_chars)

    entry[DirArray.TYPE_IDX.value] = DirectoryType.STATE_REC_DATA.value
    entry[DirArray.MOD_IDX1.value] = len(all_idata)
    entry[DirArray.MOD_IDX2.value] = total_num_chars
    #entry[DirArray.STR_QTY_IDX.value] = 0
    entry[DirArray.OFFSET_IDX.value] = offset
    entry[DirArray.LENGTH_IDX.value] = entry[DirArray.MOD_IDX1.value]*4 + entry[DirArray.MOD_IDX2.value]

    temp_dir = {}
    temp_dir['subrecs'] = entry
    global_dirs[DirectoryType.STATE_REC_DATA.value] = temp_dir
    #print("PRINT SUBREC", all_idata, all_cdata, offset, entry[DirArray.LENGTH_IDX.value])
    # Pack/Write
    a_file.seek(offset)
    no_off_testfile.write("\nSUBREC all_idata: " + str(all_idata))
    all_idata = struct.pack(str(len(all_idata)) + 'i', *all_idata)
    a_file.write(all_idata)

    #all_cdata = bytes(all_cdata, 'ascii')
    print("PRINT SUBREC 1", total_num_chars, len(all_cdata))
    __RoundNumBytes(total_num_chars, all_cdata)
    print("PRINT SUBREC 2", total_num_chars, len(all_cdata))
    a_file.write(all_cdata)
    
    # Update and Return offset
    offset = offset + entry[DirArray.LENGTH_IDX.value]
    return offset

    
    ######################### read functions ###########################
    

def __FindSuperclass(this_proc_dirs, dir_strings_dict, short_name):
    # Only Works for Local Directory Search
    dir_type = DirectoryType.CLASS_DEF.value
    
    for dir_num in dir_strings_dict[dir_type]:
        if dir_strings_dict[dir_type][dir_num][1][0] == short_name:            
            entry = this_proc_dirs[dir_type][dir_num]
            superclass = entry[DirArray.MOD_IDX2.value]
            return superclass
    return -1

    
def __FindSuperclassCommit(this_proc_dirs, dir_strings_dict, short_name):
    superclass = -1
    # Only Works for Local Directory Search
    dir_type = DirectoryType.CLASS_DEF.value

    if short_name not in this_proc_dirs[dir_type]:
        print("FindSuperclassCommit Error: ", short_name)
        
    entry = this_proc_dirs[dir_type][short_name]
    superclass = entry[DirArray.MOD_IDX2.value]
    
    return superclass


def __CalcNodeBlocks(global_node_labels, global_subrec_idata):
    # Calculate node blocks after compressed
    # node blocks should only have unique nodes?
    if ('node', 'node') in global_subrec_idata:
        temp_sub = global_subrec_idata[('node', 'node')]
    else:
        temp_sub = {}
    temp_sub[SubrecIdata.START_STOP.value] = [(1,len(global_node_labels))]
    global_subrec_idata[('node', 'node')] = temp_sub
    

#def __ComputeNodeOwner():
    # The "owning" processor is the lowest numbered processor that contains that node
    # What info is missing that we would need to check / compensate for....

    
def __ReadSubrecords(p, f, this_proc_dirs, dir_strings_dict, node_label_consec_global_ids, global_elem_conns_dict, this_proc_labels, global_labels, global_srec_headers, global_subrec_idata, global_subrec_cdata):
    # Node and Element labels are put into separate structures (node_label_consec_global_ids, global_labels)
    # 'glob' has qty_id_blks = 0
    # 'mat' and 'glob' only need to be read by Processor 0

    # labels not in subrecords

    type_id = DirectoryType.STATE_REC_DATA.value
    global_id_node_labels = {value:key for key,value in node_label_consec_global_ids.items()}
    
    for dir_num in this_proc_dirs[type_id]:
        entry = this_proc_dirs[type_id][dir_num]
        srec_offset = entry[DirArray.OFFSET_IDX.value]

        srec_int_data = entry[DirArray.MOD_IDX1.value] - 4
        srec_c_data = entry[DirArray.MOD_IDX2.value]

        f.seek(srec_offset)

        this_srec_array = struct.unpack('4i', f.read(16))                          
        _,_,_, srec_qty_subrecs = this_srec_array

        idata = struct.unpack(str(srec_int_data) + 'i', f.read(srec_int_data * 4))

        if (sys.version_info > (3, 0)):
            cdata = str(struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0])[2:].split('\\x00')  # only works for Python3
        else:
            cdata = struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].split(b'\x00')

        int_pos = 0
        c_pos = 0
        
        this_subrec_cdata = []
        this_subrec_idata = []
        sub_mo_qtys = {} # What to do with mo qtys for each sub

        for subrec_num in range(srec_qty_subrecs):
            org, qty_svars, qty_id_blks = idata[int_pos], idata[int_pos + 1], idata[int_pos + 2]
            this_subrec_idata = [org, qty_svars, qty_id_blks, []]

            int_pos += 3
            name, class_name = cdata[c_pos:c_pos + 2]
            this_subrec_cdata = [name, class_name]
            c_pos += 2
            svars = cdata[c_pos:c_pos + qty_svars]
            this_subrec_cdata.append(svars)
            c_pos += qty_svars

            global_srec_headers[(name,class_name)] = this_srec_array
            global_subrec_cdata[(name,class_name)] = this_subrec_cdata

            # Get this subrecords local ids
            this_subrec_local_ids = []
            for block_num in range(qty_id_blks):
                start, stop = idata[int_pos], idata[int_pos + 1]
                int_pos += 2
                for id_in_block in range(stop-start+1):
                    id_num = id_in_block + start
                    this_subrec_local_ids.append(id_num)

                if start == stop:
                    this_subrec_local_ids.append(start)

            if qty_id_blks == 0:
                # M_MESH, glob has zero id_blks
                this_subrec_global_cdata = this_subrec_cdata
                this_subrec_global_idata = this_subrec_idata
            elif qty_id_blks > 0:
                # Get this subrecord's global id
                this_subrec_global_ids = []
                this_subrec_node_labels = []
                if class_name == 'node':
                    for this_subrec_node_label in this_proc_labels[('M_NODE', 'node')]:
                        if this_proc_labels[('M_NODE', 'node')][this_subrec_node_label] in this_subrec_local_ids:
                            # Get global node id for this label and add if was part of one of this subrecs start-stop blocks

                            global_node_id = node_label_consec_global_ids[(this_subrec_node_label, p)]
                            this_subrec_global_ids.append(global_node_id)
                            this_subrec_node_labels.append(this_subrec_node_label)

                elif class_name == 'mat':
                    #print(p, "mat id blocks", this_subrec_local_ids)
                    this_subrec_global_ids = this_subrec_local_ids # added
                #    if p == 0:
                #        this_subrec_global_ids = this_subrec_local_ids
                else:
                    superclass = __FindSuperclass(this_proc_dirs, dir_strings_dict, class_name)
                    sup_class_name = Superclass(superclass).name

                    for this_subrec_elem_label in this_proc_labels[(sup_class_name, class_name)]:
                        if this_proc_labels[(sup_class_name, class_name)][this_subrec_elem_label] in this_subrec_local_ids:
                            # Get global id for this label
                            global_elem_id = global_labels[(sup_class_name, class_name)].index(this_subrec_elem_label)+1
                            this_subrec_global_ids.append(global_elem_id)


                if (name,class_name) in global_subrec_idata:
                    # If already in global_subrec_idata, need to combine new labels with old labels                
                    qty_global_id_blocks = global_subrec_idata[(name,class_name)][SubrecIdata.QTY_ID_BLOCKS.value]

                    # Global cdata array - to get svars get from index 2 to end
                    this_subrec_global_cdata = global_subrec_cdata[(name,class_name)]

                    this_subrec_global_svars = this_subrec_global_cdata[SubrecCdata.SVARS.value]

                    # Add svars if missing
                    for svar in svars:
                        if svar not in this_subrec_global_svars:
                            this_subrec_global_svars.append(svar)

                    this_subrec_global_cdata[SubrecCdata.SVARS.value] = this_subrec_global_svars

                    # Global idata array
                    this_subrec_global_idata = global_subrec_idata[(name,class_name)]

                    # Get current global ids
                    for block_num in range(qty_global_id_blocks):
                        if class_name == 'node':
                            start_blk, stop_blk = global_subrec_idata[('node', 'node')][SubrecIdata.START_STOP.value][0 + block_num]

                            # Get the labels associated with these global ids
                            for id_in_block in range(stop_blk-start_blk+1):
                                id_in_block = start_blk + id_in_block
                                node_label, node_label_proc = global_id_node_labels[id_in_block]
                                if node_label not in this_subrec_node_labels:
                                    this_subrec_global_ids.append(id_in_block)
                                    this_subrec_node_labels.append(node_label)
                                
                        #elif class_name == 'mat':
                        #    if p == 0:
                        #        print("MAT:", name, class_name)
                        else:
                            block_tuple = None
                            for i, b_t in enumerate(global_subrec_idata[(name,class_name)][SubrecIdata.START_STOP.value]):
                                if i == block_num:
                                    block_tuple = b_t
                            start_blk, stop_blk = block_tuple

                            # Add older global ids
                            # For elem labels there will be no repeats, so can add all labels
                            for id_in_block in range(stop_blk-start_blk+1):
                                id_num = start_blk+id_in_block
                                this_subrec_global_ids.append(id_num)

                elif (name,class_name) not in global_subrec_idata:
                    this_subrec_global_cdata = this_subrec_cdata
                    this_subrec_global_idata = this_subrec_idata
                    
                if class_name != 'mat' and class_name != 'node':
                    # Go through and recalc new blocks made with global ids
                    this_subrec_global_ids.sort()
                    id_blocks = []
                    start_blk = this_subrec_global_ids[0]
                    stop_blk = this_subrec_global_ids[0]

                    for i, global_node_id in enumerate(this_subrec_global_ids[1:], 1):
                        if global_node_id == (this_subrec_global_ids[i-1]+1):
                            stop_blk = global_node_id
                            if i == (len(this_subrec_global_ids)-1):
                                id_blocks.append((start_blk,stop_blk))
                        else:
                            id_blocks.append((start_blk,stop_blk))
                            if i != (len(this_subrec_global_ids)-1):
                                start_blk = this_subrec_global_ids[i]
                                stop_blk = this_subrec_global_ids[i]
                                
                    this_subrec_global_idata[SubrecIdata.QTY_ID_BLOCKS.value] = len(id_blocks)
                    this_subrec_global_idata[SubrecIdata.START_STOP.value] = id_blocks
                #elif class_name == 'node':
                    #this_subrec_global_idata[SubrecIdata.START_STOP.value] = 
                else:
                    this_subrec_global_cdata = this_subrec_cdata
                    this_subrec_global_idata = this_subrec_idata

                    # Assumes there is only one block in MAT...
                    #print(name, class_name, "mat block", this_subrec_global_ids, this_subrec_global_idata, this_subrec_global_cdata)
                    if qty_id_blks > 0:
                        mat_block = (this_subrec_global_ids[0], this_subrec_global_ids[-1])
                        this_subrec_global_idata[SubrecIdata.START_STOP.value] = [mat_block]

            global_subrec_idata[(name,class_name)] = this_subrec_global_idata
            global_subrec_cdata[(name,class_name)] = this_subrec_global_cdata

        

def __CompressNodes(p, num_node_labels, add_node, have_label, global_floats_consec, node_local_id_to_global_id, node_label_consec_global_ids):
    # Go through mask array and node info and compress data into final array
    floats = []
    node_mapping = {}
    nodes_per_proc = {}

    # this array aligns with the floats (items at indexes map between arrays), the range of numbers for node
    # # labels is larger than number of unique nodes because some of the lower numbers mapped to repeat labels
    node_labels = []

    # Switch keys and values in node_local_id_to_global_id so we can access entries by id
    node_global_id_to_local_id = {value:key for key,value in node_local_id_to_global_id.items()}

    # How to do this better?
    global_id_node_labels = {value:key for key,value in node_label_consec_global_ids.items()}

    for node_flag in range(len(add_node)):
        gid = None
        if add_node[node_flag] == 1:
            # need to add 1 below bc global_floats_consec starts at 1
            floats.extend(global_floats_consec[node_flag+1])
            
            label = global_id_node_labels[node_flag+1][0]
            node_labels.append(label)
        else:
            # If is a repeat, replace this global id with older global id
            label = global_id_node_labels[node_flag+1][0]

        # need to add 1 below bc node_global_id_to_local_id starts at 1
        proc = node_global_id_to_local_id[node_flag+1][1]
            
        if proc not in node_mapping:
            node_mapping[proc] = []
        node_mapping[proc].append(label)

        # increment number of nodes for that proc
        if proc not in nodes_per_proc:
            nodes_per_proc[proc] = 0
        nodes_per_proc[proc] = nodes_per_proc[proc] + 1

    return floats, node_mapping, nodes_per_proc, node_labels


def __ExtendArray(len_needed, curr_array, item_to_add):
    # Add len if we now have a label that is longer than curr len(have_label)
    curr_len = len(curr_array)
    length_to_add = len_needed - curr-len

    for i in range(length_to_add):
        curr_array.append(item_to_add)
                            
                
def __AlreadyHaveNode(p, sorted_labels, this_proc_node_labels, have_label, add_node):
    # global_labels = OrderedDict(label: global_id,...)
    # # index is global id 
    # this_proc_labels = local labels
    
    # have_label flag array shows if that label has already been added
    # add_node flag array will have len > (num of unique labels) bc will have repeats

    # Add all node labels to list of global labels
    # order needs to be the same as array for global ids
    for node_label in this_proc_node_labels:
        sorted_labels[(node_label,p)] = len(sorted_labels) + 1

    # Sort labels in ascending order
    ascending_labels = [key[0] for key in sorted_labels.keys()]
    ascending_labels.sort()

    # Max label
    max_label = ascending_labels[-1]

    if max_label > len(have_label):
        # Make sure have_label is long enough, curr length should = old max label + 1
        curr_len_have_label = len(have_label)
        length_to_add = max_label - curr_len_have_label

        for i in range(length_to_add):
            have_label.append(-1)

    # len of add_node should be sum of all nodes on all procs, not unique
    for i in range(len(this_proc_node_labels)):
        add_node.append(0)

    for label in this_proc_node_labels:
        if have_label[label-1] == -1:
            # Have not come across this node yet
            have_label[label-1] = p

            # Currently, there is only one of label in the array, so can get index in global_label_array
            global_node_id = sorted_labels[(label,p)]
            add_node[global_node_id-1] = 1
    
    

# M_MAT and M_MESH are same across all procs?
def __ReadMesh(p, f, this_proc_dim, this_indexing_array, this_proc_dirs, dir_strings_dict, global_class_idents, global_dir_dict, global_dir_string_dict, node_label_consec_global_ids, this_proc_connectivity, this_proc_materials, this_proc_labels, global_floats_consec, have_label, add_node, node_local_id_to_global_id, uniq_nodes):
    # Turn global dir creation into function
    
    can_proc = [ DirectoryType.CLASS_DEF.value , 
                 DirectoryType.CLASS_IDENTS.value, 
                 DirectoryType.NODES.value, 
                 DirectoryType.ELEM_CONNS.value ]

    for type_id in can_proc:
        for dir_num in this_proc_dirs[type_id]: # check dir_type_dict
            entry = this_proc_dirs[type_id][dir_num]
            mesh_offset = entry[DirArray.OFFSET_IDX.value]
            
            if type_id == DirectoryType.CLASS_DEF.value:
                dir_strings_whole = dir_strings_dict[type_id][dir_num]
                dir_strings = dir_strings_whole[1]
                short_name = dir_strings[0]

                # Need to create a global dictionary definiton
                if type_id not in global_dir_dict:
                    global_dir_dict[type_id] = {}
                if short_name not in global_dir_dict[type_id]:
                    temp_dirs = global_dir_dict[type_id]
                    temp_dirs[short_name] = entry
                    global_dir_dict[type_id] = temp_dirs

                # Directory string dictionary entry
                if type_id not in global_dir_string_dict:
                    global_dir_string_dict[type_id] = {}
                if short_name not in global_dir_string_dict[type_id]:
                    global_dir_string_dict[type_id][short_name] = {}
                    temp_string_dict = global_dir_string_dict[type_id]
                    # The number of strings here is meaningless...
                    temp_string_dict[short_name] = dir_strings_whole
                    global_dir_string_dict[type_id] = temp_string_dict
                else:
                    for string in dir_strings:
                        curr_global_strings_whole = global_dir_string_dict[type_id][short_name]
                        global_num_strings = curr_global_strings_whole[0]
                        curr_global_strings = curr_global_strings_whole[1]
                        if string not in curr_global_strings:
                            curr_global_strings.append(string)
                            global_num_strings += 1
                    global_dir_string_dict[type_id][short_name] = [global_num_strings, curr_global_strings]
                    #global_dir_string_dict[type_id][short_name] = curr_global_strings
                    #FIX: #global_dir_string_dict has global_num_strings in types 1,7
                    
            if type_id == DirectoryType.CLASS_IDENTS.value:
                f.seek(mesh_offset)
                
                dir_strings_whole = dir_strings_dict[type_id][dir_num]
                dir_strings = dir_strings_whole[1]
                short_name = dir_strings[0]

                # Need to create a global dictionary definiton
                if type_id not in global_dir_dict:
                    global_dir_dict[type_id] = {}
                if short_name not in global_dir_dict[type_id]:
                    temp_dirs = global_dir_dict[type_id]
                    temp_dirs[short_name] = this_proc_dirs[type_id][dir_num] # NOTE: offset and len given here will not be correct for global
                    global_dir_dict[type_id] = temp_dirs

                if type_id not in global_dir_string_dict:
                    global_dir_string_dict[type_id] = {}
                if short_name not in global_dir_string_dict[type_id]:
                    global_dir_string_dict[type_id][short_name] = {}
                    temp_string_dict = global_dir_string_dict[type_id]
                    # The number of strings here is meaningless...
                    temp_string_dict[short_name] = dir_strings_whole
                    global_dir_string_dict[type_id] = temp_string_dict
                else:
                    for string in dir_strings:
                        curr_global_strings_whole = global_dir_string_dict[type_id][short_name]
                        global_num_strings = curr_global_strings_whole[0]
                        curr_global_strings = curr_global_strings_whole[1]
                        if string not in curr_global_strings:
                            curr_global_strings.append(string)
                            global_num_strings += 1
                    global_dir_string_dict[type_id][short_name] = [global_num_strings, curr_global_strings]
                    #global_dir_string_dict[type_id][short_name] = curr_global_strings #FIX

                # Add superclass, start, stop
                sup_class_start_stop = struct.unpack('3i', f.read(12))
                superclass, start, stop = sup_class_start_stop

                # If processor 0, save these bc same across all procs?
                #if p == 0: #FIX? problem after changing readmesh?
                if short_name not in global_class_idents:
                    global_class_idents[short_name] = sup_class_start_stop
                else:
                    print(p, "error w global class idents", short_name, sup_class_start_stop, global_class_idents)

                # Make local dictionary
                superclass = Superclass(superclass).name
                if (superclass, short_name) not in this_proc_labels:
                    this_proc_labels[(superclass, short_name)] = {}

                id = len(this_proc_labels[(superclass, short_name)])
                for label in range(start, stop + 1):
                    id += 1
                    this_proc_labels[(superclass, short_name)][label] = id


            if type_id == DirectoryType.NODES.value:
                f.seek(mesh_offset)

                dir_strings_whole = dir_strings_dict[type_id][dir_num]
                dir_strings = dir_strings_whole[1]
                short_name = dir_strings[0]

                # Need to create a global dictionary definiton
                if type_id not in global_dir_dict:
                    global_dir_dict[type_id] = {}
                if short_name not in global_dir_dict[type_id]:
                    global_dir_dict[type_id][short_name] = {}

                if type_id not in global_dir_string_dict:
                    global_dir_string_dict[type_id] = {}
                if short_name not in global_dir_string_dict[type_id]:
                    global_dir_string_dict[type_id][short_name] = {}
                    temp_string_dict = global_dir_string_dict[type_id]
                    # The number of strings here is meaningless...
                    temp_string_dict[short_name] = dir_strings_whole
                    global_dir_string_dict[type_id] = temp_string_dict
                else:
                    for string in dir_strings:
                        curr_global_strings_whole = global_dir_string_dict[type_id][short_name]
                        global_num_strings = curr_global_strings_whole[0]
                        curr_global_strings = curr_global_strings_whole[1]
                        if string not in curr_global_strings:
                            curr_global_strings.append(string)
                            global_num_strings += 1
                    global_dir_string_dict[type_id][short_name] = [global_num_strings, curr_global_strings]
                    #global_dir_string_dict[type_id][short_name] = curr_global_strings #FIX

                # Add start, stop
                start, stop = struct.unpack('2i', f.read(8))                    

                num_coordinates = this_proc_dim * (stop - start + 1)
                byte_array = f.read(4 * num_coordinates)
                floats = struct.unpack('<' + str(num_coordinates) + 'f', byte_array)
                # tuple_floats = struct.unpack('<' + str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                # floats = []
                # for i,fl in enumerate(tuple_floats):
                #     floats.append(float("{0:.7f}".format(fl)))
                __Debug_CheckUnpacked('<' + str(num_coordinates) + 'f', __RoundDecimals(list(floats)), len(byte_array))
                #__Debug_CheckUnpacked('<' + str(num_coordinates) + 'f', list(floats), len(byte_array))

                class_name = short_name
                print("READ MESH", short_name, floats)
                node_label_local_id_dict = {}

                # For label in labels - if not in new dict label:global_id then add to dict and add associated floats to NODES array
                if ('M_NODE','node') not in this_proc_labels:
                    print("ReadMesh Node Error")
                    return 0
                else:
                    node_label_local_id_dict = this_proc_labels[('M_NODE','node')]

                if len(node_label_consec_global_ids) < 1:
                    labels = list(node_label_local_id_dict) #we are relying on this to be in order

                    # Set up have_label - need to find max label, add_node
                    max_label = labels
                    max_label.sort()
                    max_label = max_label[-1]

                    for i in range(len(labels)):
                        add_node.append(0)
                    for i in range(max_label):
                        have_label.append(-1)

                    floats_pos = 0
                    for i, label in enumerate(labels):
                        # consec dictionaries can have multiple global ids for a label (from diff procs)
                        node_label_consec_global_ids[(label, p)] = node_label_local_id_dict[label] 
                        node_local_id_to_global_id[(node_label_local_id_dict[label], p)] = node_label_local_id_dict[label]
                        
                        global_floats_consec[node_label_local_id_dict[label]] = list(floats[floats_pos:floats_pos + this_proc_dim])
                        floats_pos = floats_pos + this_proc_dim

                        # add_node, have_label
                        add_node[i] = 1
                        have_label[label-1] = p
                        uniq_nodes[label] = len(uniq_nodes) + 1
                        

                elif len(node_label_consec_global_ids) > 0:
                    # Remap to global ids
                    sorted_labels = dict(node_label_consec_global_ids)

                    floats_pos = 0 # keep track of local pos
                    id = len(sorted_labels) + 1
                    
                    __AlreadyHaveNode(p, sorted_labels, node_label_local_id_dict, have_label, add_node)

                    for label in node_label_local_id_dict:
                        if have_label[label-1] == -1 or have_label[label-1] == p:
                            uniq_nodes[label] = len(uniq_nodes) + 1
                            
                        # To keep in consecutive order - should we just have repeats of labels?:
                        sorted_labels[(label, p)] = id
                        global_floats_consec[id] = list(floats[floats_pos:floats_pos + this_proc_dim])

                        # Map (local id, proc): global_id
                        node_local_id_to_global_id[(node_label_local_id_dict[label], p)] = id
                        id += 1

                        # Increment floats_pos of local
                        floats_pos = floats_pos + this_proc_dim

                    # Sorted_ids_to_labels[id] = label
                    sorted_ids_to_labels = collections.OrderedDict(sorted(sorted_labels.items()))

                    for piece in sorted_ids_to_labels:
                        # # sorted_ids_to_labels[(label, proc)] = global_id
                        # # piece = (label, p)
                        node_label_consec_global_ids[piece] = sorted_ids_to_labels[piece]


            if type_id == DirectoryType.ELEM_CONNS.value: #ignore
                # cannot combine this until have node mapping - skip these and have separate mesh routine for elem_conns
                # or can put all of these together in a lpcal dictionary to read/collect and then combine all local w global dictionary

                f.seek(mesh_offset)

                dir_strings_whole = dir_strings_dict[type_id][dir_num]
                dir_strings = dir_strings_whole[1]
                short_name = dir_strings[0]

                # Need to create a global dictionary definiton
                if type_id not in global_dir_dict:
                    global_dir_dict[type_id] = {}
                if short_name not in global_dir_dict[type_id]:
                    global_dir_dict[type_id][short_name] = {}

                if type_id not in global_dir_string_dict:
                    global_dir_string_dict[type_id] = {}
                if short_name not in global_dir_string_dict[type_id]:
                    global_dir_string_dict[type_id][short_name] = {}
                    temp_string_dict = global_dir_string_dict[type_id]
                    # The number of strings here is meaningless...
                    temp_string_dict[short_name] = dir_strings_whole
                    global_dir_string_dict[type_id] = temp_string_dict
                else:
                    for string in dir_strings:
                        curr_global_strings_whole = global_dir_string_dict[type_id][short_name]
                        global_num_strings = curr_global_strings_whole[0]
                        curr_global_strings = curr_global_strings_whole[1]
                        if string not in curr_global_strings:
                            curr_global_strings.append(string)
                            global_num_strings += 1
                    global_dir_string_dict[type_id][short_name] = [global_num_strings, curr_global_strings]
                    #global_dir_string_dict[type_id][short_name] =  curr_global_strings # FIX

                # this dict holds the local connectivities that we will combine later when we have node mapping
                this_proc_connectivity[short_name] = {} # could it already have an entry with this short name?

                superclass, qty_blocks = struct.unpack('2i', f.read(8))
                elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))

                elem_qty = entry[DirArray.MOD_IDX2.value]
                word_qty = ConnWords[Superclass(superclass).name].value
                conn_qty = word_qty - 2
                mat_offset = word_qty - 1
                ebuf = struct.unpack(str(elem_qty * word_qty) + 'i', f.read(elem_qty * word_qty * 4))
                index = 0

                for j in range(qty_blocks):
                    off = elem_blocks[j * 2] - 1
                    elem_qty = elem_blocks[j * 2 + 1] - elem_blocks[j * 2] + 1

                    mo_id = 1
                    for k in range(index, len(ebuf), word_qty):

                        this_proc_connectivity[short_name][mo_id] = []
                        mat = ebuf[k + conn_qty]
                        for m in range(0, conn_qty):
                            node = ebuf[k + m]
                            this_proc_connectivity[short_name][mo_id].append(node)
                        part = ebuf[k + mat_offset]
                        # materials needs to be kept specific to a mesh
                        if mat not in this_proc_materials:
                            this_proc_materials[mat] = {}
                        if short_name not in this_proc_materials[mat]: this_proc_materials[mat][short_name] = []
                        this_proc_materials[mat][short_name].append(mo_id)
                        mo_id += 1
                    index = word_qty * elem_qty


# Remap local to global    
def __RemapConns(p, f, this_proc_dirs, dir_strings_dict, this_proc_labels, node_label_consec_global_ids, global_elem_conns_dict, global_labels, have_label, elem_processor_offsets, uniq_nodes):

    #this_proc_labels = getLabels() # return this_proc_labels dictionary with (sup_class, class): {label: local_id} entries
    classes = list(this_proc_labels)
    uniform_types = ['M_NODE', 'M_MESH', 'M_MAT'] # For these types can just copy once, so P0 can be the only one to read

    local_node_id_to_label = {value:key for key,value in this_proc_labels[('M_NODE','node')].items()}

    for dir_num in this_proc_dirs[DirectoryType.ELEM_CONNS.value]:
        entry = this_proc_dirs[DirectoryType.ELEM_CONNS.value][dir_num]

        elem_conns_offset_from_dir = entry[DirArray.OFFSET_IDX.value]

        f.seek(elem_conns_offset_from_dir)

        dir_strings = dir_strings_dict[DirectoryType.ELEM_CONNS.value][dir_num][1]
        short_name = dir_strings[0] # ! Fix so get short_name from associated strings array

        sup_class, qty_blocks = struct.unpack('2i', f.read(8))
        sup_class_name = Superclass(sup_class).name
        local_elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))

        # ! Need to calc these things from arrays:
        elem_qty = entry[DirArray.MOD_IDX2.value]
        word_qty = ConnWords[Superclass(sup_class).name].value
        conn_qty = word_qty - 2
        mat_offset = word_qty - 1

        ebuf = struct.unpack(str(elem_qty * word_qty) + 'i', f.read(elem_qty * word_qty * 4))

        if sup_class not in uniform_types: # Else same across all procs or nodes
            new_ebuf = [] # easier to have separate array to add to new_elem_conns_array afterwards

            local_elem_id_to_label = {value:key for key,value in this_proc_labels[(sup_class_name, short_name)].items()}

            # Go through elem_blocks to get chunk of ebuf we need
            # Go through ELEM_CONNS and use label:local_id from getLabels() again
            index = 0
            this_proc_label_index = 1
            for block_num in range(qty_blocks):
                start = local_elem_blocks[block_num * 2]
                stop = local_elem_blocks[block_num * 2 + 1]

                for k in range(index, len(ebuf), word_qty): # all elems of this type have same len
                    mat = ebuf[k + conn_qty]
                    part = ebuf[k + mat_offset]

                    if (sup_class_name,short_name) not in global_labels:
                        global_labels[(sup_class_name,short_name)] = []

                    # Keep track of new global mo_id
                    mo_id = len(global_labels[(sup_class_name,short_name)]) + 1
                    
                    new_sub_entry = global_labels[(sup_class_name,short_name)]

                    new_sub_entry.append(local_elem_id_to_label[this_proc_label_index])
                    global_labels[(sup_class_name,short_name)] = new_sub_entry

                    # Calc processor offsets for element labels
                    if short_name not in elem_processor_offsets:
                        # Add 1 so last index can hold total number of that type of elem
                        elem_processor_offsets[short_name] = [0] * (elem_processor_offsets["num_procs"] + 1)

                    temp = elem_processor_offsets[short_name]

                    for ind in range(p+1,len(temp)-1):
                        temp[ind] = mo_id

                    if k == (len(ebuf)-word_qty):
                        # check if should update max
                        if p != elem_processor_offsets["num_procs"]:
                            temp[-1] = mo_id

                    elem_processor_offsets[short_name] = temp

                    # Replace local node ids with global node ids
                    for m in range(0, conn_qty):
                        # Get the local node 
                        local_node_id = ebuf[k + m]

                        # Get label from local_node_id
                        node_label = local_node_id_to_label[local_node_id]
                        
                        # Get global_node_id from node_label
                        # dicts below - label:global vs local:global
                        #global_node_id = node_label_consec_global_ids[(node_label, p)]
                        #global_node_id = node_local_id_to_global_id[(local_node_id, p)]
                        global_node_id = uniq_nodes[node_label]

                        # Update new_ebuf (nodes,mat,part)
                        new_ebuf.append(global_node_id)
                        
                    # After all conns add mat and part
                    new_ebuf.extend([mat, part])

                    # Increment indices after m loop within k loop
                    this_proc_label_index += 1

                # Update index after k loop
                index = word_qty * elem_qty

                # Append/add new_ebuf to ebuf_dict[short_name]
                if short_name in global_elem_conns_dict:
                    new_conns_entry = global_elem_conns_dict[short_name]
                    new_conns_entry[MeshElemConns.EBUF.value] = new_conns_entry[MeshElemConns.EBUF.value] + new_ebuf
                    global_elem_conns_dict[short_name] = new_conns_entry # create ebuf enum index

                    # Change stop
                    new_stop = int(len(global_elem_conns_dict[short_name][MeshElemConns.EBUF.value])/word_qty)

                    # Does this affect number of blocks
                    if qty_blocks == 1:
                        new_block_entry = global_elem_conns_dict[short_name]
                        #new_block_entry[MeshElemConns.ELEM_BLOCKS.value] = (global_elem_conns_dict[short_name][MeshElemConns.ELEM_BLOCKS.value][MeshECElemBlocks.STOP.value], new_stop)
                        new_block_entry[MeshElemConns.ELEM_BLOCKS.value] = (1, new_stop)
                        global_elem_conns_dict[short_name] = new_block_entry
                    else:
                        print("ReadMesh Error: More than one elem block and not doing anything about it")

                elif short_name not in global_elem_conns_dict:
                    # And create short_name in elem_conns_dict
                    global_elem_conns_dict[short_name] = [sup_class, qty_blocks, local_elem_blocks, new_ebuf] # ?

                    
'''
Returns the quantity of atoms based on the aggregate type. (from mili_reader_lib.py)
'''
def __atom_qty(svar_name, global_svar_ints, global_svar_s, global_svar_inners):
    qty = 1
    base = 1
    svar_ints = global_svar_ints[svar_name]
    svar_agg_type = svar_ints[SvarInts.AGG_TYPE.value]
    svar_s = global_svar_s[svar_name]
    
    if svar_agg_type == AggregateType.VECTOR.value:
        qty = svar_ints[SvarInts.LIST_SIZE.value]
        
    elif svar_agg_type == AggregateType.ARRAY.value:
        # Number of dims = Order
        for i in range(svar_s[SvarS.ORDER.value]):
            qty *= svar_s[SvarS.DIMS.value + i]
            
    elif svar_agg_type == AggregateType.VEC_ARRAY.value:
        # Number of dims = Order
        qty = 0
        for i in range(svar_ints[SvarInts.ORDER.value]):
            base *= svar_ints[SvarInts.DIMS.value + i]

        svar_list_size = svar_ints[SvarInts.LIST_SIZE.value]

        for i in range(svar_list_size):
            # Main/input state var = svar_*, inner state vars = sv_*
            sv_name = svar_s[SvarS.SV_NAMES.value + i]
            
            if sv_name in global_svar_inners:
                sv_ints = global_svar_inners[sv_name][InnerSvar.INTS.value]
            elif sv_name in global_svar_ints:
                sv_ints = global_svar_ints[sv_name]
            else:
                print("Atom Qty Error: Inner Svar Missing", sv_name)
            sv_agg_type = sv_ints[SvarInts.AGG_TYPE.value]
            
            if sv_agg_type == AggregateType.SCALAR.value:
                qty += base
            else:
                qty += base * __atom_qty(sv_name, global_svar_ints, global_svar_s, global_svar_inners)
    return qty                    


def __ReadSvars(p, f, this_proc_dirs, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners):
    for dir_num in this_proc_dirs[DirectoryType.STATE_VAR_DICT.value]:
        entry = this_proc_dirs[DirectoryType.STATE_VAR_DICT.value][dir_num]
        svar_offset = entry[DirArray.OFFSET_IDX.value]
        f.seek(svar_offset)
        
        this_proc_svar_header =  struct.unpack('2i', f.read(8))
        svar_words, svar_bytes = this_proc_svar_header
        num_ints = (svar_words - 2)
        ints = struct.unpack(str(num_ints) + 'i', f.read(num_ints * 4))

        if (sys.version_info > (3, 0)):
            s = str(struct.unpack(str(svar_bytes) + 's', f.read(svar_bytes))[0])[2:].split('\\x00')  # only works for Python3
        else:
            s = struct.unpack(str(svar_bytes) + 's', f.read(svar_bytes))[0].split(b'\x00')
        int_pos = 0
        c_pos = 0
        all_ints = 0

        while int_pos < len(ints):
            exists = False
            newly_exists = False 
            this_svar_s = []
            this_svar_ints = []
            ints_len = 0
            s_len = 0

            sv_name, title = s[c_pos], s[c_pos + 1]
            this_svar_s = [sv_name, title]

            agg_type, data_type = ints[int_pos], ints[int_pos + 1]
            this_svar_ints = [agg_type, data_type]

            if sv_name in global_svar_s:
                exists = True           
            
            if agg_type == AggregateType.SCALAR.value:
                # If scalar, add to global array before incrementing in case last svar is a scalar
                if sv_name not in global_svar_ints:
                    global_svar_ints[sv_name] = this_svar_ints
                    ints_len += 2
                if sv_name not in global_svar_s:
                    global_svar_s[sv_name] = this_svar_s
                    s_len += 2

            int_pos += 2
            c_pos += 2
            all_ints += 2

            if agg_type == AggregateType.ARRAY.value or agg_type == AggregateType.VEC_ARRAY.value:
                order, dims = ints[int_pos], []
                this_svar_ints.append(order)
                int_pos += 1
                all_ints += 1
                for k in range(order):
                    dims.append(ints[int_pos])
                    int_pos += 1
                    all_ints += 1
                this_svar_ints.extend(dims)

                if not exists:
                    global_svar_s[sv_name] = this_svar_s
                    global_svar_ints[sv_name] = this_svar_ints

                    # Add to sizes - will add these sizes to total
                    this_svar_s = "".join(this_svar_s)
                    ints_len += len(this_svar_ints)
                    s_len += sys.getsizeof(this_svar_s)
                    
                    newly_exists = True

            if agg_type == AggregateType.VECTOR.value or agg_type == AggregateType.VEC_ARRAY.value:
                svar_list_size = ints[int_pos]

                if sv_name not in global_svar_ints:
                    this_svar_ints.append(svar_list_size)
                    global_svar_ints[sv_name] = this_svar_ints
                elif sv_name in global_svar_ints and newly_exists:
                    # exists = this entry was created in this loop, not prev
                    temp_sv = global_svar_ints[sv_name]
                    temp_sv.append(svar_list_size)
                    global_svar_ints[sv_name] = temp_sv
                    
                ints_len += 1
                int_pos += 1
                all_ints += 1

                sv_names = []
                for j in range(svar_list_size):
                    # Is it possible that some lists of sv_names would be incomplete?
                    sv_names.append(s[c_pos])
                    c_pos += 1
                if sv_name not in global_svar_s:
                    this_svar_s.extend(sv_names)
                    global_svar_s[sv_name] = this_svar_s
                elif sv_name in global_svar_s and newly_exists:
                    new_svar = global_svar_s[sv_name] + sv_names
                    global_svar_s[sv_name] = new_svar

                s_len += sys.getsizeof("".join(sv_names))

                for sv_name_inner in sv_names:
                    # inner svs will have repeats (multiple svars will have same inner svs
                    # but only the first inner_sv (if multiple) will have additional name,title,agg,data
                    # when an inner sv is already present, add to dict where dict[inner][p] = [sv_names]
                    
                    if sv_name_inner not in global_svar_inners and sv_name_inner not in global_svar_s:
                        # Add to global svars under sv_name entry and iterate
                        # We also need to keep track of this info to calc sizes for other svars w this sv_nner
                        inner_name, inner_title = s[c_pos], s[c_pos + 1]
                        inner_s = [inner_name, inner_title]
                        inner_agg, inner_data = ints[int_pos], ints[int_pos + 1]
                        inner_ints = [inner_agg, inner_data]
                        
                        if sv_name not in global_svar_s:
                            # Add all
                            this_svar_s = this_svar_s + inner_s
                            global_svar_s[sv_name] = this_svar_s

                            this_svar_ints = this_svar_ints + inner_ints
                            global_svar_ints[sv_name] = this_svar_ints                           
                        else:
                            # Append
                            new_svar_s = global_svar_s[sv_name] + inner_s
                            global_svar_s[sv_name] = new_svar_s

                            new_svar_ints = global_svar_ints[sv_name] + inner_ints
                            global_svar_ints[sv_name] = new_svar_ints

                        # Inner SV dictionary entries - [[ints],[s],[owner],[processors]]
                        global_svar_inners[sv_name_inner] = [inner_ints,inner_s,[sv_name],[p]]
                        
                        s_len += 2
                        ints_len += 2

                        int_pos += 2
                        all_ints += 2
                        c_pos += 2
                    elif sv_name_inner in global_svar_inners and sv_name_inner not in global_svar_s:
                        if p not in global_svar_inners[sv_name_inner][InnerSvar.PROCS.value]:
                            # Add this processor to procs so we know we don't have to incremement if we see it later
                            temp_procs = global_svar_inners[sv_name_inner]
                            temp_procs[InnerSvar.PROCS.value].append(p)
                            global_svar_inners[sv_name_inner] = temp_procs
                            
                            int_pos += 2
                            all_ints += 2
                            c_pos += 2
                            
                    if sv_name_inner in global_svar_s:
                        if sv_name_inner in global_svar_inners:
                            print("ReadSvars Inner Svar Error:", sv_name, sv_name_inner)

            global_svar_header[0] += ints_len
            global_svar_header[1] += s_len

            
def __CreateNewMiliParams(global_params, global_dirs, global_dir_strings):
    new_params = ['pcombiner_alpha', 'origin lib version', 'origin arch name', 'origin host name', 'origin date', 'origin username']
    type_id = DirectoryType.MILI_PARAM.value

    for param in new_params:
        if param == 'pcombiner_alpha': # what happens if there is a param name that is not recognized in database?
            new_dir = [type_id, 1, 0, 1, 0, len(param)] # MOD_IDX1 --> 1=str, 5=int4
            new_str = [param]

            global_params[param] = [param]
        else:
            split_str = 'origin '
            old_param = param[param.index(split_str) + len(split_str):]
            new_dir = global_dirs[type_id][old_param]
            new_str = [split_str + global_dir_strings[type_id][old_param][0]]
        
            global_params[param] = []

        # Add new dir entry for each new param
        temp_dir = global_dirs[type_id]
        temp_dir[param] = new_dir
        global_dirs[type_id] = temp_dir

        # Add new dir string for each new param
        temp_dir_str = global_dir_strings[type_id]
        temp_dir_str[param] = new_str
        global_dir_strings[type_id] = temp_dir_str
            

def __ReadParams(p, f, file_tag, this_proc_dirs, dir_num_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_int_points, this_proc_param_dirs, global_dirs, global_dir_string_dict):

    # global_params[name_type][name] =
    # # 's' = params
    # # 'mesh_dimensions = array of dims and use one at the end?
    # # 'node_labels' = dictionary of (superclass, name): {label, local_id}
    # # 'Element Label'
    # # # 'ElemIds'
    # # 'MAT_NAME' = [matname].append(ints)
    # # 'es_' = i_points = first, _, i_points, num_i_points - how to store - get rid of es_ and only use name as key ? 

    this_proc_params = {}
    this_proc_label_keys = []

    type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}

    file_number = 0

    can_proc = [ DirectoryType.MILI_PARAM.value, DirectoryType.APPLICATION_PARAM.value, DirectoryType.TI_PARAM.value ]
    for type_id in can_proc:
        for dir_num in this_proc_dirs[type_id]:
            dir_strings = dir_num_strings_dict[type_id][dir_num][1] #FIX
            name = dir_strings[0]

            # Add Directory Strings to the global array - these will not change...?
            if type_id not in global_dir_string_dict:
                global_dir_string_dict[type_id] = {}
            if name not in global_dir_string_dict[type_id]:
                temp_str = global_dir_string_dict[type_id]
                temp_str[name] = dir_strings
                global_dir_string_dict[type_id] = temp_str
            
            entry = this_proc_dirs[type_id][dir_num]
            
            # Add placeholder directory
            if type_id not in global_dirs:
                global_dirs[type_id] = {}
            if name not in global_dirs[type_id]:
                temp_dir = global_dirs[type_id]
                temp_dir[name] = entry
                global_dirs[type_id] = temp_dir
            
            param_offset = entry[DirArray.OFFSET_IDX.value]
            f.seek(param_offset)

            dir_length = entry[DirArray.LENGTH_IDX.value]
            
            byte_array = f.read(dir_length)
            param_type = entry[DirArray.MOD_IDX1.value]

            type_rep = type_to_str[DataType(param_type).name]
            type_value = ExtSize[DataType(param_type).name].value

            if type_rep == 's':
                if (sys.version_info > (3, 0)):
                    this_proc_params[name] = [str(struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)[0])[2:].split('\\x00')]

                else:
                    this_proc_params[name] = [struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)[0].split(b'\x00')[0]]
            else:
                if type_rep == 'f':
                    this_proc_params[name] = [struct.unpack(file_tag + '2i' + str(int(dir_length / type_value) - 2) + type_rep, byte_array)]
                    #__Debug_CheckUnpacked(file_tag + '2i' + str(int(dir_length / type_value) - 2) + type_rep, this_proc_params[name][0])
                else:
                    this_proc_params[name] = [struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)]


            if name not in global_params:
                # Same across procs except for things like "LOCAL_COUNT"
                global_params[name] = this_proc_params[name]
                
                # Add in new mili params if right place in order
                if name == 'username':
                    __CreateNewMiliParams(global_params, global_dirs, global_dir_string_dict)

            if name == 'mesh dimensions':
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                this_proc_dim = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)[0]

            if 'Node Labels' in name:
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                ints = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)
                first, last, node_labels = ints[0], ints[1], ints[2:]

                print(p, "READING NODE LABELS", first, last, node_labels)

                if ('M_NODE', 'node') not in this_proc_labels:
                    this_proc_labels[('M_NODE', 'node')] = {}
                for j in range (first - 1, last):
                    this_proc_labels[('M_NODE', 'node')][node_labels[j]] = j + 1 # change GetLabels()

            if 'Element Label' in name:
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                ints = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)
                first, _, ints = ints[0], ints[1], ints[2:]

                print(p, name, "READ ELEM LABELS", first, _, ints)

                sup_class_idx = name.index('Scls-') + len('Scls-')
                sup_class_end_idx = name.index('/', sup_class_idx, len(name))
                class_idx = name.index('Sname-') + len('Sname-')
                class_end_idx = name.index('/', class_idx, len(name))

                sup_class = name[sup_class_idx : sup_class_end_idx]
                clas = name[class_idx : class_end_idx]

                if (sup_class, clas) not in this_proc_labels:
                    this_proc_labels[(sup_class, clas)] = {}

                for j in range(len(ints)):
                    if 'ElemIds' in name:
                        this_proc_labels[(sup_class, clas)][this_proc_label_keys[j]] = ints[j]
                    else:
                        this_proc_label_keys.append(ints[j])

                if 'ElemIds' in name:
                    this_proc_label_keys = []

                if (sup_class, clas) not in this_proc_param_dirs:
                    # when remap we need the local dir
                    this_proc_param_dirs[(sup_class, clas)] = [dir_num]
                else:
                    new_arr = this_proc_param_dirs[(sup_class, clas)] + [dir_num]
                    this_proc_param_dirs[(sup_class, clas)] = new_arr

            if 'MAT_NAME' in name: # need to combine? all procs same?
                # init 1-1000?
                f.seek(param_offset)
                byte_array = f.read(dir_length)

                if (sys.version_info > (3, 0)):
                    matname = str(struct.unpack(str(dir_length) + 's', byte_array)[0])[2:].split('\\x00')[0]  # only works for Python3
                else:
                    matname = struct.unpack(str(dir_length) + 's', byte_array)[0].split(b'\x00')[0]

                num = name[-1:]
                if matname in this_proc_matname:
                    this_proc_matname[matname].append(int(num))
                else:
                    this_proc_matname[matname] = [int(num)]

            if 'es_' in name: # check if need to combine or all procs same
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                i_points = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)
                first, total, i_points, num_i_ponts = i_points[0], i_points[1], i_points[2:len(i_points) - 1], i_points[len(i_points) - 1]
                index = name.find('es_')
                # Note: first, total included in thsi array
                this_proc_int_points[name[index:]] = [first, total, i_points, num_i_ponts] # what is diff between total and num_i_points

                # determine stress or strain

            print(type_rep, "READ PARAMS", name, this_proc_params[name])

    return this_proc_dim
    

def __ReadNames(p, f, file_tag, offset, this_indexing_array, this_proc_dirs, dir_num_strings_dict, global_names):
    
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
        for dir_num in this_proc_dirs[dir_type]:
            dir_offset = this_proc_dirs[dir_type][dir_num][DirArray.OFFSET_IDX.value]
            string_offset = dir_num_strings_dict[dir_type][dir_num][DirStringsArray.STRING_OFFSET_IDX.value]
            string_count = this_proc_dirs[dir_type][dir_num][DirArray.STR_QTY_IDX.value]
            
            directory_strings = [ strings[sidx] for sidx in range( string_offset, string_offset + string_count) ]

            dir_num_strings_dict[dir_type][dir_num][DirStringsArray.STRINGS.value] = directory_strings

            

            # Add to global_names if not in there already
            #if global_names is None:
            #    global_names = directory_strings
            #else:
            #    for string in directory_strings:
            #        if string not in global_names:
            #            global_names.append(string)


    
def __ReadDirectories(p, f, file_size, offset, file_tag, this_header_array, indexing_dict, dir_strings_dict, dir_type_dict):

    this_indexing_array = indexing_dict[p]

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


        this_byte_array = struct.unpack(file_tag + '6' + int_long, byte_array)

        # Get dir num using indexing_dict
        dir_num = indexing_dict['totals'][IndexingArray.NUM_DIRS.value]

        this_dir_type = this_byte_array[DirArray.TYPE_IDX.value]
        if this_dir_type not in dir_type_dict:
            dir_type_dict[this_dir_type] = OrderedDict()
            dir_strings_dict[this_dir_type] = OrderedDict()
        dir_type_dict[this_dir_type][dir_num] = list(this_byte_array)
        dir_strings_dict[this_dir_type][dir_num] = {}
        dir_strings_dict[this_dir_type][dir_num][DirStringsArray.STRING_OFFSET_IDX.value] = number_of_strings

        this_dir_string_qty = this_byte_array[DirArray.STR_QTY_IDX.value]
        number_of_strings += this_dir_string_qty

        # Increment NUM_DIRS kept in 'totals' index of indexing dict
        new_num_total_dirs = indexing_dict['totals']
        new_num_total_dirs[IndexingArray.NUM_DIRS.value] = dir_num + 1
        indexing_dict['totals'] = new_num_total_dirs
        
    return offset

    
def __ReadStateMaps(p, f, file_tag, this_proc_indexing_array, state_maps):
    # Why does this return offset, use file tag or '='?
    offset = -16
    state_map_length = 20

    # number_of_state_maps = this_proc_indexing_array[IndexingArray.NUM_STATE_MAPS.value]
    # f.seek(-(number_of_state_maps*state_map_length) + offset, os.SEEK_END)

    # for num in range(1, 1 + number_of_state_maps):
    #     byte_array = f.read(state_map_length)
    #     state_map = struct.unpack('=iqfi', byte_array)
    #     state_map = list(state_map)
        
    #     # 32 -> python float double ? format float in list
    #     state_map[2] = float(f'{state_map[2]:.7f}')
    #     for i, num in enumerate(state_map):   
    #         print(i, "TYPES", type(num), num)#, f'{num:.4f}')
    #     file_number, file_offset, time, state_map_id = state_map
        
    #     state_maps.extend(state_map)


    # Change so that reads them all at once
    number_of_state_maps = this_proc_indexing_array[IndexingArray.NUM_STATE_MAPS.value]
    f.seek(-(number_of_state_maps*state_map_length) + offset, os.SEEK_END)
    byte_array = f.read(number_of_state_maps*state_map_length)

    #format_str = 'iqfi' * number_of_state_maps    
    #state_maps[:number_of_state_maps*4] = list(struct.unpack('=' + format_str, byte_array))

    format_str = 'iqfi'
    tup_list = list(struct.iter_unpack('=' + format_str, byte_array))
    state_maps[:number_of_state_maps*4] = [i for tup in tup_list for i in tup]
    print("state maps1 ", state_maps)

    for sm_count in range(number_of_state_maps):
        state_maps[sm_count*4 + 2] = __RoundDecimals([state_maps[sm_count*4 + 2]])[0]

    print("state maps2", state_maps)

    return state_maps, offset


def __ReadHeader(p, f, local_file_tag, indexing_dict): # add file_tag to header_array
    # Precision flag should be same across procs ??
    header = f.read(16)
    mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')

    header_array = struct.unpack('6b', header[4:10])
    header_version, dir_version, endian_flag, precision_flag, state_file_suffix_length, partition_flag = header_array

    if endian_flag == 1:
        local_file_tag = '>'
    else:
        local_file_tag = '<'

    # Read Indexing #
    offset = -16
    f.seek(offset, os.SEEK_END)
    indexing_dict[p] = struct.unpack('4i', f.read(16))

    if 'totals' not in indexing_dict:
        # Use to keep track of total number of directories across all processors to keep track of dir_num assignment
        indexing_dict['totals'] = [0, 0, 0, 0]

    return local_file_tag, mili_taur, header_array


def __TransformFileName(f, file_stat):
    if file_stat == "open":
        # Get name of open file
        f = str(f.name)

    # Trim beginning directory info
    if '/' in f:
        f = f[f.rfind('/')+1:]

    # Trim extension
    if '.' in f:
        f = f[:f.rfind('.')]

    # Add "afile_" - temp naming convention
    f = "afile_" + f + 'A'

    # Add ".pltA"
    #f = f + ".pltA"
    return f


def __StartRead(p, file_name, header_array, indexing_dict, dims, global_state_maps, global_class_idents, global_names, global_params, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners, global_dir_dict, global_dir_string_dict, global_elem_conns_dict, global_labels, have_label, add_node, global_subrec_idata, global_subrec_cdata, global_srec_headers, node_label_consec_global_ids, node_local_id_to_global_id, elem_processor_offsets, global_floats_consec, uniq_nodes):
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

    # Local label dictionaries
    this_proc_labels = {}
    this_proc_matname = {}
    this_proc_i_points = {}

    this_proc_dim = None
    this_proc_param_dirs = {}

    # Local mesh dictionaries
    local_connectivity = {}
    local_materials = {}

    afile_name = __TransformFileName(file_name, "closed")

    with open(file_name, 'rb') as f:
        # Fix so passing dictionaries isnt necessary
        local_file_tag, mili_taur, local_header = __ReadHeader(p, f, local_file_tag, indexing_dict)
        local_indexing = list(indexing_dict[p])

        if p == 0:
            # If processor 0, write header array and save local header as global header - need precision when committing?
            for num in local_header:
                header_array.append(num)

            __InitWrite(afile_name, mili_taur, local_header)

            offset = __ReadStateMaps(p, f, local_file_tag, local_indexing, global_state_maps)

        offset = -16 # can assume P0 will always set offset, so can leave as uninitialized var above...
        offset = __ReadDirectories(p, f, file_size, offset, local_file_tag, local_header, indexing_dict, local_dir_strings_dict, local_directory_dict)

        if local_indexing[IndexingArray.NULL_TERMED_NAMES_BYTES.value] > 0:
            __ReadNames(p, f, local_file_tag, offset, local_indexing, local_directory_dict, local_dir_strings_dict, global_names)

            this_proc_dim = __ReadParams(p, f, local_file_tag, local_directory_dict, local_dir_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_i_points, this_proc_param_dirs, global_dir_dict, global_dir_string_dict)
            dims[p] = this_proc_dim
            print("global params", global_params)

            __ReadSvars(p, f, local_directory_dict, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners)
            
            __ReadMesh(p, f, this_proc_dim, local_indexing, local_directory_dict, local_dir_strings_dict, global_class_idents, global_dir_dict, global_dir_string_dict, node_label_consec_global_ids, local_connectivity, local_materials, this_proc_labels, global_floats_consec, have_label, add_node, node_local_id_to_global_id, uniq_nodes)

            __RemapConns(p, f, local_directory_dict, local_dir_strings_dict, this_proc_labels, node_label_consec_global_ids, global_elem_conns_dict, global_labels, have_label, elem_processor_offsets, uniq_nodes)
            
            __ReadSubrecords(p, f, local_directory_dict, local_dir_strings_dict, node_label_consec_global_ids, global_elem_conns_dict, this_proc_labels, global_labels, global_srec_headers, global_subrec_idata, global_subrec_cdata)


def main():
    
    params = parseParameters(sys.argv[1:]) 
    if params is None:
        sys.exit(1)

    file_names = gather_A_files(params)
    if len(file_names) <= 0:
        print(f"Error while gathering A files. No A files were found in the directory '{params['input_dir']}' with the base name '{params['input_file']}'")
        sys.exit(1)


    manager = Manager()
    header_array = manager.list()
    indexing_dict = manager.dict()
    dims = manager.dict()
    global_state_maps = manager.list()
    global_class_idents = manager.dict()
    global_names = manager.list() # not necessary
    global_params = manager.dict()

    global_svar_header = manager.list()
    init_svar_header = [0,0]
    global_svar_header.extend(init_svar_header)

    global_svar_s = manager.dict()
    global_svar_ints = manager.dict()
    global_svar_inners = manager.dict()

    global_dir_dict = manager.dict()
    global_dir_string_dict = manager.dict()
    
    node_label_consec_global_ids = manager.dict() # (label, proc): global_id --> labels can have more than one global id
    node_local_id_to_global_id = manager.dict() # (local id, proc): global_id

    global_floats_consec = manager.dict() # consec
    
    global_elem_conns_dict = manager.dict() # all elem_conns entries
    global_labels = manager.dict() # element labels separated by class
    
    elem_processor_offsets = manager.dict()
    elem_processor_offsets["num_procs"] = len(file_names)

    # List to check if we have node label in global space already
    have_label = manager.list()
    # List to act as mask for node adding to global data
    add_node = manager.list()

    global_subrec_idata = manager.dict()
    global_subrec_cdata = manager.dict()
    global_srec_headers = manager.dict()

    uniq_nodes = manager.dict()


    processes = []
    for i,file_name in enumerate(file_names):
        
        p = Process(target=__StartRead, args=(i, file_name, header_array, \
                                            indexing_dict, dims, global_state_maps, \
                                            global_class_idents, global_names, global_params, \
                                            global_svar_header, global_svar_s, \
                                            global_svar_ints, global_svar_inners, \
                                            global_dir_dict, global_dir_string_dict, \
                                            global_elem_conns_dict, global_labels, \
                                            have_label, add_node, \
                                            global_subrec_idata, global_subrec_cdata, \
                                            global_srec_headers, \
                                            node_label_consec_global_ids, \
                                            node_local_id_to_global_id, \
                                            elem_processor_offsets, global_floats_consec, \
                                            uniq_nodes))

        p.start()
        p.join()
        processes.append(p)

    floats, node_mapping, nodes_per_proc, node_labels = __CompressNodes(p, len(have_label), add_node, have_label, global_floats_consec, node_local_id_to_global_id, node_label_consec_global_ids)

    __CalcNodeBlocks(node_labels, global_subrec_idata)

    for p in processes:
        p.join()
  
#################### Commits ######################

    afile_name = __TransformFileName(file_names[0], "closed")
    curr_wrt_index = 16
    mili_taur = 'mili'
    with open(afile_name, 'ab+') as a_file, open ('ascii_testfile_no_offset', 'a+') as no_off_testfile, open ('ascii_testfile', 'a+') as testfile:        

        __AddMiliParams(global_params, global_dir_dict, global_dir_string_dict)
        print("COMMITS", global_class_idents)
        curr_wrt_index = __CommitParams(a_file, curr_wrt_index, global_params, global_dir_dict, global_dir_string_dict, node_labels, global_labels, elem_processor_offsets,    header_array, dims[0], floats, global_elem_conns_dict, node_local_id_to_global_id, global_class_idents, testfile, no_off_testfile)
        print("after params", a_file.tell(), curr_wrt_index)

        # Called in CommitParams
        curr_wrt_index = __CommitMesh(a_file, curr_wrt_index, header_array, global_dir_dict, global_dir_string_dict, dims[0], floats, global_elem_conns_dict, node_local_id_to_global_id, global_class_idents, testfile, no_off_testfile)

        print("PRE commitSvars", curr_wrt_index, a_file.tell())
        curr_wrt_index = __CommitSvars(a_file, curr_wrt_index, global_dir_dict, global_svar_s, global_svar_ints, testfile, no_off_testfile)
        print("POST commitSvars", curr_wrt_index, a_file.tell())

        curr_wrt_index = __CommitSubrecords(a_file, curr_wrt_index, global_dir_dict, global_dir_string_dict, global_svar_s, global_svar_ints, global_svar_inners, global_subrec_idata, global_subrec_cdata, testfile, no_off_testfile)

        print("CommitNames1 all_string_sum", curr_wrt_index)
        curr_wrt_index, null_termed_names_bytes, dir_order =  __CommitNames(a_file, curr_wrt_index, global_dir_string_dict, global_dir_dict, testfile, no_off_testfile)
        print("CommitNames2 all_string_sum", curr_wrt_index)
        print("Dir Order", dir_order)
        print("Dirs", global_dir_dict)
              
        curr_wrt_index, num_dirs  = __CommitDirectories(a_file, curr_wrt_index, header_array, global_dir_dict, dir_order, testfile, no_off_testfile)
        print("dir", a_file.tell(), curr_wrt_index)
        
        file_tag = ''
        endian_flag = header_array[HeaderArray.ENDIAN_FLAG.value]
        if endian_flag == 1:
            file_tag = '>'
        else:
            file_tag = '<'

        curr_wrt_index = __CommitStateMaps(a_file, curr_wrt_index, file_tag, global_state_maps, indexing_dict[0][IndexingArray.NUM_STATE_MAPS.value], global_dir_dict)
        print("sm", a_file.tell(), curr_wrt_index)

        
        curr_wrt_index = __CommitIndexing(a_file, curr_wrt_index, list(indexing_dict[0]), null_termed_names_bytes, num_dirs)
        print("index", a_file.tell(),curr_wrt_index)
        
        
        print("\n")
        
    if __debug__:
        print("\nGlobal Params", global_params)
        
        print("\nGlobal Node Labels", len(node_label_consec_global_ids), node_label_consec_global_ids)
        print("\nGlobal Node Local ID:Global ID, per proc", node_mapping)
        print("\nGlobal Elem Conns", global_elem_conns_dict)

        print("\nGlobal Strings", global_dir_string_dict)
        print("\nGlobal Directories", global_dir_dict)

        print("\nGlobal Labels", global_labels)
        print("\nGlobal Element Label Offsets", elem_processor_offsets)

        print("\nGlobal svar_s", global_svar_s)

        print("\nGlobal svar_ints", global_svar_ints)        

        print("\nGlobal State Maps Dict", global_state_maps)

        print("\nGlobal Subrecord Idata", global_subrec_idata)
        print("\nGlobal Subrecord Cdata", global_subrec_cdata)

        print("\nnodes per proc", nodes_per_proc)

        print("\nnode local id to global id", node_local_id_to_global_id)
        print("\nhave label, add_node", have_label, add_node)
        
        print("\nTo print less: 'python3 -O para_mrl.py'")
    else:
        print("To Print Debug Strings: 'python3 para_mrl.py'")

    print("uniq nodes", uniq_nodes)

    print(len(floats)/3, "floats", floats)

    print("node mapping", node_mapping)

    print("node labels", node_labels)

        
if __name__ == '__main__':
    main()

### NOTES ###
# Questions:
# # check file size using 'max size per file' param

# # Mili Library version does not get updated...

# # stop hardcoding precision

# # file_tag in CommitStateMaps is '<' from ReadStateMaps

# # trying to format floats --> params get messed up ? mat/glob

# # mesh nodes read as -0.0 
