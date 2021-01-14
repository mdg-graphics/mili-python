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

'''
Every direcotry has a type that dictates what informaiton
is in the directory.
'''
class DirectoryType(Enum):
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


'''
The superclass tells what class an object belongs to.
'''
class Superclass(Enum):
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

'''
Connections per superclass
'''
class ConnWords(Enum):
    M_UNIT = 0
    M_NODE = 0
    M_TRUSS = 4
    M_BEAM = 5
    M_TRI = 5
    M_QUAD = 6
    M_TET = 6
    M_PYRAMID = 7
    M_WEDGE = 8
    M_HEX = 10
    M_MAT = 0
    M_MESH = 0
    M_SURFACE = 0
    M_PARTICLE = 3
    M_TET10 = 12

'''
The numbering system used by Mili files to state type.
'''
class DataType(Enum):
    M_STRING = 1
    M_FLOAT = 2
    M_FLOAT4 = 3
    M_FLOAT8 = 4
    M_INT = 5
    M_INT4 = 6
    M_INT8 = 7

'''
The size of each data type in bytes.
'''
class ExtSize(Enum):
    M_STRING = 1
    M_FLOAT = 4
    M_FLOAT4 = 4
    M_FLOAT8 = 8
    M_INT = 4
    M_INT4 = 4
    M_INT8 = 8

'''
The aggregate type describes what a state variable can be.
'''
class AggregateType(Enum):
    SCALAR = 0
    VECTOR = 1
    ARRAY = 2
    VEC_ARRAY = 3

'''
The organization of data in a subrecord is either ordered by variable
or object.
'''
class DataOrganization(Enum):
    RESULT = 0
    OBJECT = 1


    ######################### read functions ###########################


#def CheckSvarExists(p, sv_name_inner, global_svar_s, global_svar_inners):
    

def ReadSvars(p, f, this_proc_dirs, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners):
    # svar.c: "delete previous successful entries" ?
    for item in this_proc_dirs[DirectoryType.STATE_VAR_DICT.value]:
        svar_offset = item[1]
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
                #print("try this", sv_name, agg_type, int_pos, len(ints))
                svar_list_size = ints[int_pos]

                if sv_name not in global_svar_ints:
                    this_svar_ints.append(svar_list_size)
                    global_svar_ints[sv_name] = this_svar_ints
                elif sv_name in global_svar_ints and newly_exists:
                    # exists = this entry was created in this loop, not prev
                    global_svar_ints[sv_name].append(svar_list_size)
                    
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
                    
                    #print(int_pos, "all stress inners", svar_list_size, sv_name_inner, "sv_name", sv_name, "agg", agg_type)
                    if sv_name_inner not in global_svar_inners and sv_name_inner not in global_svar_s:
                        # Add to global svars and iterate
                        inner_name, inner_title = s[c_pos], s[c_pos + 1]
                        inner_s = [inner_name, inner_title]
                        inner_agg, inner_data = ints[int_pos], ints[int_pos + 1]
                        inner_ints = [inner_agg, inner_data]
                        
                        if sv_name not in global_svar_s:
                            # Add all
                            this_svar_s = this_svar_s + inner_s
                            this_svar_ints = this_svar_ints + inner_ints
                            
                            global_svar_s[sv_name] = this_svar_s
                            global_svar_ints[sv_name] = this_svar_ints
                        else:
                            # Append
                            new_svar_s = global_svar_s[sv_name] + inner_s
                            global_svar_s[sv_name] = new_svar_s

                            new_svar_ints = global_svar_ints[sv_name] + inner_ints
                            global_svar_ints[sv_name] = new_svar_ints

                        global_svar_inners[sv_name_inner] = [p]
                        
                        s_len += 2
                        ints_len += 2

                        int_pos += 2
                        all_ints += 2
                        c_pos += 2
                    elif sv_name_inner in global_svar_inners and sv_name_inner not in global_svar_s:
                        if p not in global_svar_inners[sv_name_inner]:
                            new_str = global_svar_inners[sv_name_inner] + [p]
                            global_svar_inners[sv_name_inner] = new_str
                            int_pos += 2
                            all_ints += 2
                            c_pos += 2
                            
                    if sv_name_inner in global_svar_s:
                        if sv_name_inner in global_svar_inners:
                            print("what is here in inners", sv_name, sv_name_inner)

            global_svar_header[0] += ints_len
            global_svar_header[1] += s_len


def ReadParams(p, f, file_tag, this_proc_dirs, dir_num_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_int_points):

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
        for item in this_proc_dirs[type_id]:
            dir_num = item[0]
            dir_strings = dir_num_strings_dict[type_id][dir_num][1]
            name = dir_strings[0]
            param_offset = item[1]

            f.seek(param_offset)

            entry = this_proc_dirs[type_id][item]
            dir_length = entry[DirArray.LENGTH_IDX.value]
            
            byte_array = f.read(dir_length)
            param_type = entry[DirArray.MOD_IDX1.value]

            type_rep = type_to_str[DataType(param_type).name]
            type_value = ExtSize[DataType(param_type).name].value

            if type_rep == 's': # replace hard coded strings w variable equivalents
                if (sys.version_info > (3, 0)):
                    this_proc_params[name] = [str(struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)[0])[2:].split('\\x00')]
                else:
                    this_proc_params[name] = [struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)[0].split(b'\x00')[0]]
            else:
                this_proc_params[name] = [struct.unpack(file_tag + str(int(dir_length / type_value)) + type_rep, byte_array)]

            if name not in global_params:
                global_params[name] = this_proc_params[name]                
            for param in this_proc_params[name]:
                if param not in global_params[name]:
                    global_params[name].append(param)

            if name == 'mesh dimensions':
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                this_proc_dim = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)[0]
                # Do I need to keep track of dims from all?

            if 'Node Labels' in name:
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                ints = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)
                first, last, node_labels = ints[0], ints[1], ints[2:]

                if ('M_NODE', 'node') not in this_proc_labels:
                    this_proc_labels[('M_NODE', 'node')] = {}
                for j in range (first - 1, last):
                    this_proc_labels[('M_NODE', 'node')][node_labels[j]] = j + 1 # change GetLabels()

            if 'Element Label' in name:
                f.seek(param_offset)
                byte_array = f.read(dir_length)
                ints = struct.unpack(file_tag + str(int(dir_length / 4)) + 'i', byte_array)
                first, _, ints = ints[0], ints[1], ints[2:]

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

    return this_proc_dim
    

def ReadNames(p, f, file_tag, offset, this_indexing_array, this_proc_dirs, dir_num_strings_dict, global_names):
    # Should all 8 procs have same names bc two of them are shorter?
    
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
    # Should get rid of this_dir_offset in local_dir_dict index so is not a tuple? Not sure if it made things
    # # easier like i imagined it would
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
        dir_type_dict[this_dir_type][(i, this_dir_offset)] = list(this_byte_array)
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



def StartRead(p, file_name, header_dict, indexing_dict, global_state_maps, global_names, global_params, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners):
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

    with open(file_name, 'rb') as f:
        local_file_tag, mili_taur = ReadHeader(p, f, local_file_tag, header_dict, indexing_dict)
        local_header = header_dict[p]
        local_indexing = list(indexing_dict[p])

        global_state_maps[p], offset = ReadStateMaps(p, f, local_file_tag, local_indexing)

        offset = ReadDirectories(p, f, file_size, offset, local_file_tag, local_header, local_indexing, local_dir_strings_dict, local_directory_dict)

        if local_indexing[IndexingArray.NULL_TERMED_NAMES_BYTES.value] > 0:
            ReadNames(p, f, local_file_tag, offset, local_indexing, local_directory_dict, local_dir_strings_dict, global_names)

            this_proc_dim = ReadParams(p, f, local_file_tag, local_directory_dict, local_dir_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_i_points)

            
            ReadSvars(p, f, local_directory_dict, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners)
            #ReadMesh()
            #ReadSubrecords()

    if __debug__:
        print("\nLocal Dir Dict", local_directory_dict)
        print("\nLocal Dir Strings Dict", local_dir_strings_dict)
        print("\nGlobal Names", global_names)
        print("\nParams", global_params,"\n\nLabels", this_proc_labels, "\n\nMatname", this_proc_matname, "\n\nI_pts", this_proc_i_points)
        

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
    global_params = manager.dict()

    global_svar_header = manager.list()
    init_svar_header = [0,0]
    global_svar_header.extend(init_svar_header)

    global_svar_s = manager.dict()
    global_svar_ints = manager.dict()
    global_svar_inners = manager.dict()

    processes = []
    for i,file_name in enumerate(file_names):
        
        p = Process(target=StartRead, args=(i, file_name, header_dict, \
                                            indexing_dict, global_state_maps, \
                                            global_names, global_params, \
                                            global_svar_header, global_svar_s, \
                                            global_svar_ints, global_svar_inners))

        p.start()
        p.join()
        processes.append(p)

    if __debug__:
        print("\nHeader Dict", header_dict)
        print("\nIndexing Dict", indexing_dict)
        print("\nGlobal State Maps Dict", global_state_maps)

        print("\nGlobal State Variable Ints", global_svar_ints)
    print("\nGlobal State Variable S", global_svar_s)
    print("\nGlobal Inner State Variables", global_svar_inners)

if __name__ == '__main__':
    main()
