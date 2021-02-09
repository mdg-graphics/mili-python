from multiprocessing import Process, Manager
from enum import Enum
import collections
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


    
    ######################### commit funtions ##########################

def CommitMesh(a_file, offset, header_array, global_dirs, global_dir_strings, dim, floats, elem_conns, node_local_id_to_global_id):

    ### NODES ###
    node_start = 1
    node_stop = len(floats)

    # Create nodes dictionary entry
    node_type_idx = DirectoryType.NODES.value
    node_mesh_id = 1
    node_num_nodes = int(node_stop/dim)
    node_str_qty_idx = len(global_dir_strings[DirectoryType.NODES.value]['node'][DirStringsArray.STRINGS.value])
    node_offset = offset

    # Get precision before calculating bytes - header array from every proc is the same (start + stop + prec*floats)
    #prec = header_array[HeaderArray.PRECISION_FLAG.value] # find how to interpret prec flag??
    prec = 4
    node_len = 8 + prec*node_stop

    # Add directory to global dirs
    global_dirs[node_type_idx] = {}
    temp_dirs = global_dirs[node_type_idx]
    temp_dirs['node'] = [node_type_idx, node_mesh_id, node_num_nodes, node_str_qty_idx, node_offset, node_len]
    global_dirs[node_type_idx] = temp_dirs

    # add start, stop to beginning of floats array
    floats.insert(0, node_stop)
    floats.insert(0, node_start)
    
    # Write nodes to a_file
    floats = struct.pack('2i' + str(node_stop) + 'f', *floats) # * to unpack not a pointer, floats are double prec in python?
    a_file.write(floats)
    
    offset = offset + node_len # Q: Do we need to multiply by size of item to get offset
    
    ### ELEM_CONNS ###
    
    # Go through different kinds of elems
    for elem_type in elem_conns.keys():
        # Create elem_conns entries (each short_name has own entry)
        ec_type_idx = DirectoryType.ELEM_CONNS.value
        ec_mesh_id = 1
        ec_num_elems = elem_conns[elem_type][MeshElemConns.ELEM_BLOCKS.value][MeshECElemBlocks.STOP.value]
        ec_str_qty_idx = len(global_dir_strings[DirectoryType.ELEM_CONNS.value][elem_type][DirStringsArray.STRINGS.value])
        ec_offset = offset

        ec_entry_start = elem_conns[elem_type][:2] + [num for num in elem_conns[elem_type][2]]

        prec = 4
        sup_class_name = Superclass(elem_conns[elem_type][MeshElemConns.SUP_CLASS.value]).name
        sup_class_conns = ConnWords[sup_class_name].value

        # Write length of entire elem conns entry for this elem type
        ec_num_ints = 4 + ec_num_elems*sup_class_conns
        ec_entry = struct.pack(str(ec_num_ints) + 'i', *(ec_entry_start + elem_conns[elem_type][MeshElemConns.EBUF.value]))
        a_file.write(ec_entry)

        ec_len = prec*ec_num_ints

        # Add directory to global dirs
        if ec_type_idx not in global_dirs:
            global_dirs[ec_type_idx] = {}
        temp_dirs = global_dirs[ec_type_idx]
        temp_dirs[elem_type] = [ec_type_idx, ec_mesh_id, ec_num_elems, ec_str_qty_idx, ec_offset, ec_len]
        global_dirs[ec_type_idx] = temp_dirs
        
        offset = offset + ec_len
    

# NOTE: Put combined info into a Mili from mili_python_lib.py - so commit_svars should use Mili attributes

# This is not working yay
def CommitSvars(a_file, curr_index, global_svar_s, global_svar_ints):
    # Assume we are calling this bc we already have info - not checking for more than header
    # Need to have size of data
    # 1 - svar_hdr ( qty int words idx, qty bytes idx), svar_words, svar_bytes
    # 2 - add directory entry?
    # 3 - write out svar_i_ios then svar_c_ios

    # Need to check if svar_bytes is larger than single file size ?
    # svar_bytes = svar_header[1]
    # if svar_bytes > FILE_MAX

    # encode('ascii')
    # turn list of strings into long string?

    combined_header = []
    all_svar_ints = []
    all_svar_s = []
    all_svar_s_bytes = []
    #formatting_string = None
    svar_words, svar_bytes = 0, 0 # svar_words - 2 = num_ints
    for svar_name in global_svar_ints:
        # Calc sizes here, get rid of in ReadSvars()
        # Should we create new arrays or keep accessing the arrays repeatedly

        size_ints = len(global_svar_ints[svar_name])
        size_s = len(global_svar_s[svar_name])
        
        svar_words += size_ints
        svar_bytes += size_s

        for i, sv_piece in enumerate(global_svar_s[svar_name]):
            # Turn each string into bytes
                
            if sys.version_info < (3, 0):
                sv_piece = bytes(sv_piece)
            else:
                sv_piece = bytes(sv_piece, 'utf8')

            all_svar_s_bytes = all_svar_s_bytes + [sv_piece]

        all_svar_ints.extend(global_svar_ints[svar_name])
        all_svar_s.extend(global_svar_s[svar_name])

    combined_header = [svar_words, svar_bytes]

    a_file.seek(curr_index)

    svar_header = struct.pack('2i', *combined_header)
    
    svar_ints = struct.pack(str(svar_words) + 'i', *all_svar_ints) # * to unpack not a pointer
    a_file.write(svar_ints)

    #print("bytes", all_svar_s_bytes.encode('utf8'))
    print("list", list(all_svar_s_bytes))
    for piece in all_svar_s_bytes:
        a_file.write(piece)


    # update directory = [type_idx  mod_idx1  mod_idx2  string_qty  offset  length]
    #AddDirectory(SVAR, _, _, str_qty_idx, curr_index, svar_byte_len)
    # update curr_index - maybe return curr_index ?


    
    ######################### read functions ###########################
    
    
def FindSuperclass(this_proc_dirs, dir_strings_dict, short_name):
    dir_type = DirectoryType.CLASS_DEF.value
    for dir_num in dir_strings_dict[dir_type]:
        if dir_strings_dict[dir_type][dir_num][1][0] == short_name:
            entry = this_proc_dirs[dir_type][dir_num]
            superclass = entry[DirArray.MOD_IDX2.value]
            return superclass
    return 0
    
    
def ReadSubrecords(p, f, this_proc_dirs, dir_strings_dict, node_label_global_id_dict, node_label_consec_global_ids, global_elem_conns_dict, this_proc_labels, global_labels, global_srec_headers, global_subrec_idata, global_subrec_cdata):

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
            this_subrec_cdata.extend(svars)
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


            if qty_id_blks > 0:
                # Get this subrecord's global id
                this_subrec_global_ids = []
                this_subrec_node_labels = []
                if class_name == 'node':
                    for this_subrec_node_label in this_proc_labels[('M_NODE', 'node')]:
                        if this_proc_labels[('M_NODE', 'node')][this_subrec_node_label] in this_subrec_local_ids:
                            # Get global node id for this label and add if was part of one of this subrecs start-stop blocks
                            # All these ids should match
                            #global_node_id = node_label_global_id_dict[this_subrec_node_label]

                            global_node_id = node_label_consec_global_ids[(this_subrec_node_label, p)]
                            this_subrec_global_ids.append(global_node_id)
                            this_subrec_node_labels.append(this_subrec_node_label)
                elif class_name == 'mat':
                    if p == 0:
                        this_subrec_global_ids = this_subrec_local_ids
                else:
                    superclass = FindSuperclass(this_proc_dirs, dir_strings_dict, class_name)
                    sup_class_name = Superclass(superclass).name

                    for this_subrec_elem_label in this_proc_labels[(sup_class_name, class_name)]:
                        if this_proc_labels[(sup_class_name, class_name)][this_subrec_elem_label] in this_subrec_local_ids:
                            # Get global id for this label
                            # These groups of ids should be the same
                            global_elem_id = global_labels[(sup_class_name, class_name)][this_subrec_elem_label]
                            this_subrec_global_ids.append(global_elem_id)


                if (name,class_name) in global_subrec_idata:
                    # If already in global_subrec_idata, need to combine new labels with old labels                
                    qty_global_id_blocks = global_subrec_idata[(name,class_name)][SubrecIdata.QTY_ID_BLOCKS.value]

                    # Global cdata array - to get svars get from index 2 to end
                    this_subrec_global_cdata = global_subrec_cdata[(name,class_name)]
                    this_subrec_global_svars = this_subrec_global_cdata[SubrecCdata.SVARS.value:]

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
                            # if label already added to list of this subrecords labels, don't add older - THIS MIGHT NOT BE RIGHT
                            for id_in_block in range(stop_blk-start_blk+1):
                                id_in_block = start_blk + id_in_block
                                node_label, node_label_proc = global_id_node_labels[id_in_block]
                                if node_label not in this_subrec_node_labels:
                                    this_subrec_global_ids.append(id_in_block)
                                    this_subrec_node_labels.append(node_label)
                        elif class_name == 'mat':
                            if p == 0:
                                print("MAT")
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
                    
                if class_name != 'mat':
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
                else:
                    this_subrec_global_cdata = this_subrec_cdata
                    this_subrec_global_idata = this_subrec_idata

                    # Assumes there is only one block in MAT...
                    mat_block = (this_subrec_global_ids[0], this_subrec_global_ids[-1])
                    this_subrec_global_idata[SubrecIdata.START_STOP.value] = [mat_block]

                global_subrec_idata[(name,class_name)] = this_subrec_global_idata
                global_subrec_cdata[(name,class_name)] = this_subrec_global_cdata


def CompressNodes(p, add_node, global_floats_consec, node_local_id_to_global_id):
    # Go through mask array and node info and compress data into final array
    floats = []
    node_mapping = {}
    nodes_per_proc = {}

    # Switch keys and values in node_local_id_to_global_id so we can access entries by id
    node_global_id_to_local_id = {value:key for key,value in node_local_id_to_global_id.items()}
    
    for node_flag in range(len(add_node)):
        if add_node[node_flag] == 1:
            # need to add 1 below bc global_floats_consec starts at 1
            floats.extend(global_floats_consec[node_flag+1])
            
            # need to add 1 below bc node_global_id_to_local_id starts at 1
            node_mapping[node_global_id_to_local_id[node_flag+1]] = node_flag+1

            # increment number of nodes for that proc
            proc = node_global_id_to_local_id[node_flag+1][1]
            if proc not in nodes_per_proc:
                nodes_per_proc[proc] = 0

            nodes_per_proc[proc] = nodes_per_proc[proc] + 1

    #print(len(floats)/3, "COMPRESSED FLOATS", floats)
    #print(len(node_mapping), "COMPRESSED NODE IDS", node_mapping)
    #print("NODES PER PROC", nodes_per_proc)

    return floats, node_mapping, nodes_per_proc
            
                            
                
def AlreadyHaveNode(p, sorted_labels, this_proc_node_labels, have_label, add_node):
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
            have_label.append(0)

    # len of add_node should be sum of all nodes on all procs, not unique
    for i in range(len(this_proc_node_labels)):
        add_node.append(0)

    for label in this_proc_node_labels:
        if have_label[label-1] != 1:
            # Have not come across this node yet
            have_label[label-1] = 1

            # Currently, there is only one of label in the array, so can get index in global_label_array
            global_node_id = sorted_labels[(label,p)]
            add_node[global_node_id-1] = 1
    
    

# M_MAT and M_MESH are same across all procs?
def ReadMesh(p, f, this_proc_dim, this_indexing_array, this_proc_dirs, dir_strings_dict, global_dir_dict, global_dir_string_dict, node_label_global_id_dict, node_label_consec_global_ids, this_proc_connectivity, this_proc_materials, this_proc_labels, global_floats, global_floats_consec, have_label, add_node, node_local_id_to_global_id):
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
                    global_dir_dict[type_id][short_name] = {}

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
                    
            if type_id == DirectoryType.CLASS_IDENTS.value:
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

                # Add superclass, start, stop
                sup_class_start_stop = struct.unpack('3i', f.read(12))
                superclass, start, stop = sup_class_start_stop

                #if short_name not in global_mesh_dictionary[type_id]:
                #    global_mesh_dictionary[type_id][short_name] = sup_class_start_stop
                # Maybe we don't need to keep a dictionary for this, can use directories and dir strings to create later?

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

                # Add start, stop
                start, stop = struct.unpack('2i', f.read(8))                    

                num_coordinates = this_proc_dim * (stop - start + 1)
                floats = struct.unpack(str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                class_name = short_name

                node_label_local_id_dict = {}

                # For label in labels - if not in new dict label:global_id then add to dict and add associated floats to NODES array
                if ('M_NODE','node') not in this_proc_labels:
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
                        have_label.append(0)

                    floats_pos = 0
                    for i, label in enumerate(labels):
                        # consec dictionaries can have multiple global ids for a label (from diff procs)
                        #node_label_global_id_dict[label] = node_label_local_id_dict[label]
                        node_label_consec_global_ids[(label, p)] = node_label_local_id_dict[label]
                        node_local_id_to_global_id[(node_label_local_id_dict[label], p)] = node_label_local_id_dict[label]
                        
                        #global_floats[node_label_local_id_dict[label]] = list(floats[floats_pos:floats_pos + this_proc_dim])
                        global_floats_consec[node_label_local_id_dict[label]] = list(floats[floats_pos:floats_pos + this_proc_dim])
                        floats_pos = floats_pos + this_proc_dim

                        # add_node, have_label
                        add_node[i] = 1
                        have_label[label-1] = 1
                        

                elif len(node_label_consec_global_ids) > 0:
                    # Remap to global ids
                    sorted_labels = dict(node_label_consec_global_ids)

                    floats_pos = 0 # keep track of local pos
                    id = len(sorted_labels) + 1
                    
                    AlreadyHaveNode(p, sorted_labels, node_label_local_id_dict, have_label, add_node)
                    
                    for label in node_label_local_id_dict:
                        #if label not in node_label_global_id_dict:
                            # Sorted_labels in order of ID 1-n
                            #sorted_labels[label] = id
                            #global_floats[id] = list(floats[floats_pos:floats_pos + this_proc_dim])
                            #id += 1

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
                        # sorted_ids_to_labels[(label, proc)] = global_id
                        # piece = (label, p)
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
def RemapConns(p, f, this_proc_dirs, dir_strings_dict, this_proc_param_dirs, this_proc_labels, node_label_global_id_dict, node_label_consec_global_ids, elem_conns_label_global_id_dict, global_elem_conns_dict, global_labels, node_local_id_to_global_id):

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

        if short_name not in elem_conns_label_global_id_dict:
            elem_conns_label_global_id_dict[short_name] = {} # CHECK if short_name already exists
        #elem_conns_label_global_id_length = 0
        sup_class, qty_blocks = struct.unpack('2i', f.read(8))
        sup_class_name = Superclass(sup_class).name
        local_elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))

        # ! Need to calc these things from arrays:
        elem_qty = entry[DirArray.MOD_IDX2.value]
        word_qty = ConnWords[Superclass(sup_class).name].value
        conn_qty = word_qty - 2
        mat_offset = word_qty - 1

        ebuf = struct.unpack(str(elem_qty * word_qty) + 'i', f.read(elem_qty * word_qty * 4))

        if sup_class not in uniform_types: # Else same across all procs
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

                    # Keep track of new global mo_id
                    mo_id = len(elem_conns_label_global_id_dict[short_name]) + 1

                    if (sup_class_name,short_name) not in global_labels:
                        global_labels[(sup_class_name,short_name)] = {}
                    new_sub_entry = global_labels[(sup_class_name,short_name)]
                    new_sub_entry[local_elem_id_to_label[this_proc_label_index]] = mo_id
                    global_labels[(sup_class_name,short_name)] = new_sub_entry

                    for m in range(0, conn_qty):
                        # Get the local node 
                        local_node_id = ebuf[k + m]

                        # Get label from local_node_id
                        node_label = local_node_id_to_label[local_node_id]
                        # Get global_node_id from node_label
                        #global_node_id = node_label_global_id_dict[node_label]

                        # label:global vs local:global
                        #global_node_id = node_label_consec_global_ids[(node_label, p)]
                        global_node_id = node_local_id_to_global_id[(local_node_id, p)]
                        
                        ##### Update Global Label Dictionary - DONT NEED THIS, just to check
                        new_elem_conn_entry = elem_conns_label_global_id_dict[short_name]
                        if mo_id in new_elem_conn_entry:
                            new_global_id_list = new_elem_conn_entry[mo_id] + [global_node_id]
                            new_elem_conn_entry[mo_id] = new_global_id_list
                        else:
                            new_elem_conn_entry[mo_id] = [global_node_id]
                        elem_conns_label_global_id_dict[short_name] = new_elem_conn_entry  # mo_id = elem mo_id
                        #####

                        # Update new_ebuf (nodes,mat,part)
                        new_ebuf.append(global_node_id)
                        
                    # After all conns add mat and part
                    new_ebuf.extend([mat, part])

                    # Increment indices after m loop within k loop
                    mo_id += 1
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



def ReadSvars(p, f, this_proc_dirs, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners):
    # svar.c: "delete previous successful entries" ?
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


def ReadParams(p, f, file_tag, this_proc_dirs, dir_num_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_int_points, this_proc_param_dirs):

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
            dir_strings = dir_num_strings_dict[type_id][dir_num][1]
            name = dir_strings[0]
            entry = this_proc_dirs[type_id][dir_num]
            param_offset = entry[DirArray.OFFSET_IDX.value]

            f.seek(param_offset)

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

    return this_proc_dim
    

def ReadNames(p, f, file_tag, offset, this_indexing_array, this_proc_dirs, dir_num_strings_dict, global_names):
    
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


    
def ReadDirectories(p, f, file_size, offset, file_tag, this_header_array, indexing_dict, dir_strings_dict, dir_type_dict):

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

    if 'totals' not in indexing_dict:
        # Use to keep track of total number of directories across all processors to keep track of dir_num assignment
        indexing_dict['totals'] = [0, 0, 0, 0]

    return local_file_tag, mili_taur



def StartRead(p, file_name, header_dict, indexing_dict, dims, global_state_maps, global_names, global_params, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners, global_dir_dict, global_dir_string_dict, node_label_global_id_dict, global_floats, elem_conns_label_global_id_dict, global_elem_conns_dict, global_labels, have_label, add_node, global_subrec_idata, global_subrec_cdata, global_srec_headers, node_label_consec_global_ids, node_local_id_to_global_id, global_floats_consec):
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

    with open(file_name, 'rb') as f:
        # Fix so passing dictionaries isnt necessary
        local_file_tag, mili_taur = ReadHeader(p, f, local_file_tag, header_dict, indexing_dict)
        local_header = header_dict[p]
        local_indexing = list(indexing_dict[p])

        global_state_maps[p], offset = ReadStateMaps(p, f, local_file_tag, local_indexing)

        offset = ReadDirectories(p, f, file_size, offset, local_file_tag, local_header, indexing_dict, local_dir_strings_dict, local_directory_dict)

        if local_indexing[IndexingArray.NULL_TERMED_NAMES_BYTES.value] > 0:
            ReadNames(p, f, local_file_tag, offset, local_indexing, local_directory_dict, local_dir_strings_dict, global_names)

            this_proc_dim = ReadParams(p, f, local_file_tag, local_directory_dict, local_dir_strings_dict, global_params, this_proc_dim, this_proc_labels, this_proc_matname, this_proc_i_points, this_proc_param_dirs)
            dims[p] = this_proc_dim

            ReadSvars(p, f, local_directory_dict, global_svar_header, global_svar_s, global_svar_ints, global_svar_inners)
            
            ReadMesh(p, f, this_proc_dim, local_indexing, local_directory_dict, local_dir_strings_dict, global_dir_dict, global_dir_string_dict, node_label_global_id_dict, node_label_consec_global_ids, local_connectivity, local_materials, this_proc_labels, global_floats, global_floats_consec, have_label, add_node, node_local_id_to_global_id)

            RemapConns(p, f, local_directory_dict, local_dir_strings_dict, this_proc_param_dirs, this_proc_labels, node_label_global_id_dict, node_label_consec_global_ids, elem_conns_label_global_id_dict, global_elem_conns_dict, global_labels, node_local_id_to_global_id)
            #CompressNodes(p, add_node, global_floats_consec, node_local_id_to_global_id)
            
            #ReadSubrecords(p, f, local_directory_dict, local_dir_strings_dict, node_label_global_id_dict, node_label_consec_global_ids, global_elem_conns_dict, this_proc_labels, global_labels, global_srec_headers, global_subrec_idata, global_subrec_cdata)

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
    dims = manager.dict()
    global_state_maps = manager.dict()
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
    node_label_global_id_dict = manager.dict()
    node_label_consec_global_ids = manager.dict() # (label, proc): global_id --> labels can have more than one global id
    node_local_id_to_global_id = manager.dict() # (local id, proc): global_id
    global_floats = manager.dict()
    global_floats_consec = manager.dict() # consec
    
    elem_conns_label_global_id_dict = manager.dict() # not necessary - Use to check
    global_elem_conns_dict = manager.dict()
    global_labels = manager.dict()

    # List to check if we have label in global space already
    have_label = manager.list()
    # List to act as mask for node adding to global data
    add_node = manager.list()

    global_subrec_idata = manager.dict()
    global_subrec_cdata = manager.dict()
    global_srec_headers = manager.dict()


    #barrier = manager.Barrier(len(file_names))
    processes = []
    for i,file_name in enumerate(file_names):
        
        p = Process(target=StartRead, args=(i, file_name, header_dict, \
                                            indexing_dict, dims, global_state_maps, \
                                            global_names, global_params, \
                                            global_svar_header, global_svar_s, \
                                            global_svar_ints, global_svar_inners, \
                                            global_dir_dict, global_dir_string_dict, \
                                            node_label_global_id_dict, global_floats, \
                                            elem_conns_label_global_id_dict, \
                                            global_elem_conns_dict, global_labels, \
                                            have_label, add_node, \
                                            global_subrec_idata, global_subrec_cdata, \
                                            global_srec_headers, \
                                            node_label_consec_global_ids, \
                                            node_local_id_to_global_id, global_floats_consec))

        p.start()
        p.join()
        processes.append(p)

    floats, node_mapping, nodes_per_proc = CompressNodes(p, add_node, global_floats_consec, node_local_id_to_global_id)

    #for p in processes:
    #    p.join()

    # Commits
    file_name = 'afile_d3samp6'
    curr_wrt_index = 0
    with open(file_name, 'wb') as a_file:
        #CommitSvars(a_file, curr_wrt_index, global_svar_s, global_svar_ints)

        CommitMesh(a_file, curr_wrt_index, header_dict[0], global_dir_dict, global_dir_string_dict, dims[0], floats, global_elem_conns_dict, node_local_id_to_global_id) #, elem_local_id_to_global_id)
        

    if __debug__:
        print("\nHeader Dict", header_dict)
        print("\nIndexing Dict", indexing_dict)
        print("\nGlobal State Maps Dict", global_state_maps)

        print("\nGlobal State Variable Ints", global_svar_ints)
        print("\nGlobal State Variable S", global_svar_s)
        print("\nGlobal Inner State Variables", global_svar_inners)
        print("\nGlobal Node Labels", node_label_consec_global_ids)
        print("\nGlobal Labels", global_labels)
        print("\nGlobal Floats", global_floats_consec, len(global_floats_consec))

        print("\nTo print less: 'python3 -0 para_mrl.py'")
    else:
        print("To Print Debug Strings: 'python3 para_mrl.py'")

    #print("\nGlobal \"Have Label\" Array", len(have_label), have_label)
    #print("\nGlobal Mask Array", len(add_node), add_node)
    print("\nGlobal Node Labels", node_label_consec_global_ids)
    print("\nGlobal Local ID:Global ID", node_local_id_to_global_id)
    print("\nGlobal Elem Conns", global_elem_conns_dict)

    print("\nGlobal Strings", global_dir_string_dict)
    print("\nGlobal Directories", global_dir_dict)
        
    #print("\nGlobal Subrecord Idata", global_subrec_idata)
    #print("\nGlobal Subrecord Cdata", global_subrec_cdata)

        
if __name__ == '__main__':
    main()
