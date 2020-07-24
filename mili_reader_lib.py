"""
Copyright (c) 2016, Lawrence Livermore National Security, LLC. 
 Produced at the Lawrence Livermore National Laboratory. Written 
 by Kevin Durrenberger: durrenberger1@llnl.gov. CODE-OCEC-16-056. 
 All rights reserved.

 This file is part of Mili. For details, see <URL describing code 
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
'''
This is the 'meta' information for a statemap, but not the statemap itself.
'''

from collections import defaultdict
import os
import struct
import sys
import time

from enum import Enum
import psutil

import multiprocessing as mp


class StateMap:
    def __init__(self, file_number, file_offset, time, state_map_id):
        self.file_number = file_number
        self.file_offset = file_offset
        self.time = time
        self.state_map_id = state_map_id

'''
Directories dictate where information can be found in the .pltA file.
'''
class Directory:
    def __init__(self, type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx):
        self.type_idx = type_idx
        self.modifier_idx1 = modifier_idx1
        self.modifier_idx2 = modifier_idx2
        self.string_qty_idx = string_qty_idx
        self.offset_idx = offset_idx
        self.length_idx = length_idx

'''
A state variable is the information being stored in subrecords throughout time
at different states.
'''
class StateVariable:

    '''
    Initiliazer for a state variable.
    '''
    def __init__(self, name, title, agg_type, data_type):
        self.name = name
        self.title = title
        self.agg_type = agg_type
        self.data_type = data_type
        self.list_size = 0
        self.order = 0
        self.dims = []
        self.svars = []  # list of string names included in this if it is vector

    '''
    Returns the quantity of atoms based on the aggregate type.
    '''
    def atom_qty(self, state_variables):
        qty = 1
        base = 1
        if self.agg_type == AggregateType.VECTOR.value:
            qty = self.list_size
        elif self.agg_type == AggregateType.ARRAY.value:
            for i in range(self.order):
                qty *= self.dims[i]
        elif self.agg_type == AggregateType.VEC_ARRAY.value:
            qty = 0
            for i in range(self.order):
                base *= self.dims[i]
            for i in range(len(self.svars)):
                sv = state_variables[self.svars[i]][0]
                if sv.agg_type == AggregateType.SCALAR.value:
                    qty += base
                else:
                    qty += base * sv.atom_qty()
        return qty

'''
A subrecord contains a number of state varaibles organized in a certain order.
'''
class Subrecord:
    def __init__(self, name, class_name, organization, qty_svars, svar_names):
        self.name = name
        self.class_name = class_name
        self.organization = organization
        self.qty_svars = qty_svars
        self.svar_names = svar_names
        self.qty_blocks = None
        self.mo_blocks = []
        self.mo_qty = None
        self.offset = 0
        self.size = 0  # set this mo_qty * size of variables

'''
Each Mili object has a subrecord container holding all the subrecords.
'''
class SubrecordContainer:
    def __init__(self):
        self.subrecs = []
        self.size = 0

'''
The mesh information needed is stored in this data type.
'''
class MeshObjectClassData:

    '''
    Initializer for necccessary mesh information.
    '''
    def __init__(self, short_name, long_name, superclass):
        self.short_name = short_name
        self.long_name = long_name
        self.superclass = superclass
        self.blocklist = BlockList(0, 0, [])

    '''
    Adding a block to the meshobjectclassdata class
    '''
    def add_block(self, start, stop):
        self.blocklist.blocks.append((start, stop))
        self.blocklist.block_qty += 1
        self.blocklist.obj_qty += start - stop + 1

'''
Each mesh object contains a blocklist data structure.
The blocklist has information about which objects are
part of the mesh object.
'''
class BlockList:
    def __init__(self, obj_qty, block_qty, blocks):
        self.obj_qty = obj_qty
        self.block_qty = block_qty
        self.blocks = blocks  # array of tuples with start, stop

'''
An attribute has a name and a value
e.g. name = 'mo_id' value = '5'
Always print is used in case the value is 0 for displaying in repr
'''
class Attribute:
    def __init__(self, name, value, always_print=False):
        self.name = name
        self.value = value
        self.always_print = always_print

    def __repr__(self):
        ret = ''
        if self.value or self.always_print:
            return self.name + ': ' + str(self.value) + '\n'
        return ret

'''
An item is an object that represents a component of an answer
Any number of its attributes may be None if they aren't filled
'''
class Item:
    def __init__(self, name=None, material=None, mo_id=None, label=None, class_name=None, modify=None, value=None):
        self.name = name
        self.material = material
        self.mo_id = mo_id
        self.label = label
        self.class_name = class_name
        self.modify = modify
        self.value = value
        self.always_print = False  # always print the value

    def set(self, value):
        self.value = value
        self.always_print = True

    def __str__(self):
        attributes = []
        attvalues = [self.name, self.material, self.mo_id, self.label, self.class_name, self.modify, self.value]
        attnames = ['name', 'material', 'mo_id', 'label', 'class_name', 'modify', 'value']

        for i in range(len(attvalues)):
            name, value = attnames[i], attvalues[i]
            if name == 'value' and self.always_print: attributes.append(Attribute(name, value, True))
            else: attributes.append(Attribute(name, value))

        ret = ''
        for attribute in attributes:
            ret += str(attribute)
        return ret + '\n'

'''
Each state in a query has one StateAnswer
Each contains a list of items and a state number
'''
class StateAnswer:
    def __init__(self):
        self.items = []
        self.state_number = None

    def __str__(self):
        ret = '\nstate number: ' + str(self.state_number) + '\n'
        for item in self.items:
            ret += str(item)
        return ret

'''
The encompassing object that is returned by a query.
This object contains either:
1. A list of items
2. A list of state answers
If the answer is state based (comes from a query with state_numbers),
the answer contains a list of state answers.
'''
class Answer:

    def __init__(self):
        self.state_answers = []
        self.items = []

    def __str__(self):
        ret = ''

        if not len(self.state_answers):
            for item in self.items:
                ret += str(item)
        else:
            for state in self.state_answers:
                ret += str(state)
        return ret

    def set(self, names, materials, mo_ids, labels, class_name, modify):
        num = 0
        if names: num = max(num, len(names))
        if materials: num = max(num, len(materials))
        if mo_ids: num = max(num, len(mo_ids))
        if labels: num = max(num, len(labels))

        for i in range(num):
            name = material = label = mo_id = None
            if names: name = names[i]
            if materials: material = materials[i]
            if labels: label = labels[i]
            if mo_ids: mo_id = mo_ids[i]
            if type(class_name) is list: name = class_name[i]
            else: name = class_name
            item = Item(name, material, mo_id, label, name, modify)
            self.items.append(item)

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

'''
A Mili object contains data structures to store objects such as directories
and subrecords. It also has a number of querying functions that a user can call
to access this data. Takes in the path of the file to read
'''
class Mili:
    def __init__(self, read_file=None, parallel_read=False):
        self.__milis = []  # list of parallel mili files
        self.__parent_conns = []  # list of parent connection, index number
        self.__mili_num = None  # Number of mili file (processor number)
        self.__labeltomili = defaultdict(lambda: defaultdict(list))  # map from (superclass, label) to dict of label:file
        self.__state_maps = []
        self.__directories = []
        self.__names = []
        self.__params = {}  # maps param name: [value]
        self.__state_variables = {}  # map to state variable, list of subrecords it is in
        self.__mesh_object_class_datas = {}  # shortname to object
        self.__labels = {}  # maps from label : elem_id
        self.__label_keys = []
        self.__int_points = {}
        self.__nodes = []
        self.__materials = defaultdict(dict)  # map from material number to dict of class to elems
        self.__matname = {}  # from material name to number
        for i in range(1000):
            self.__matname[str(i)] = [i]  # this is the default in case they don't have an entry for material names
        self.__connectivity = {}  # shortname : {mo_id : node}
        self.__dim = None  # dimensions from mesh dimensions
        self.__srec_container = None
        self.__header_version = None
        self.__directory_version = None
        self.__endian_flag = None
        self.__precision_flag = None
        self.__state_file_suffix_length = None
        self.__partition_flag = None
        self.__tag = None
        self.__null_termed_names_bytes = None
        self.__number_of_commits = None
        self.__number_of_directories = None
        self.__number_of_state_maps = None
        self.__filename = None
        self.__state_map_filename = None
        self.__error_file = None
        self.__parallel_mode = parallel_read

        if read_file: self.read(read_file, parallel_read=self.__parallel_mode)
    
    '''
    Close down all connections
    '''
    def __del__(self):
        for conn in self.__parent_conns:
            conn, i = conn
            conn.send("End")
            conn.close()
    
    '''
    Getter for params
    '''
    def getParams(self):
        if self.__parallel_mode: return self.__getHelper("get params")
        if len(self.__milis) > 1: return [m.getParams() for m in self.__milis]
        return self.__params

    '''
    Getter for state maps
    '''
    def getStateMaps(self):
        if self.__parallel_mode: return self.__getHelper("get state maps")
        if len(self.__milis) > 1: return [m.getStateMpas() for m in self.__milis]
        return self.__state_maps

    '''
    Getter for directories
    '''
    def getDirectories(self):
        if self.__parallel_mode: return self.__getHelper("get directories")
        if len(self.__milis) > 1: return [m.getDirectories() for m in self.__milis]
        return self.__directories

    '''
    Getter for state variables
    '''
    def getStateVariables(self):
        if self.__parallel_mode: return self.__getHelper("get state variables")
        if len(self.__milis) > 1: return [m.getStateVariables() for m in self.__milis]
        return self.__state_variables

    '''
    Getter for labels
    '''
    def getLabels(self):
        if self.__parallel_mode: return self.__getHelper("get labels")
        if len(self.__milis) > 1: return [m.getLabels() for m in self.__milis]
        return self.__labels

    '''
    Getter for materials
    '''
    def getMaterials(self):
        if self.__parallel_mode: return self.__getHelper("get materials")
        if len(self.__milis) > 1: return [m.getMaterials() for m in self.__milis]
        return self.__materials
    
    '''
    Get error file
    '''
    def getErrorFile(self):
        return self.__error_file
        
    
    def __getHelper(self, message):
        mili_conns = []
        ret = []
        for mili_index in range(len(self.__parent_conns)):
            mili_conn, i = self.__parent_conns[mili_index]
            mili_conn.send([message])
            mili_conns.append(mili_conn)
        
        while len(ret) < len(mili_conns):
            for mili_conn in mili_conns:
                if mili_conn.poll():
                    query_response = mili_conn.recv()
                    ret.append(query_response)
        return ret
    '''
    Reads the data for statemap locations from the Mili file.
    This doesn't read the .pltA file, which contains data for the actual
    state maps
    '''
    def __readStateMaps(self, f):
        offset = -16
        state_map_length = 20
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + self.__number_of_state_maps):
            f.seek(-1 * state_map_length, 1)
            byte_array = f.read(state_map_length)
            file_number, file_offset, time, state_map_id = struct.unpack(self.__tag + 'iqfi', byte_array)
            self.__state_maps = [StateMap(file_number, file_offset, time, state_map_id)] + self.__state_maps
            f.seek(-1 * state_map_length, 1)
        return offset

    '''
    Reads the state variables and parameters from the Mili file.
    '''
    def __readStateVariablesAndParams(self, f, offset):
        type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}

        offset -= self.__null_termed_names_bytes
        f.seek(offset, os.SEEK_END)
        fmt = str(self.__null_termed_names_bytes) + 's'
        byte_array = struct.unpack(self.__tag + fmt, f.read(self.__null_termed_names_bytes))[0]
        
        if (sys.version_info > (3, 0)):
            strings = str(byte_array)[2:].split('\\x00')  # only works for Python3
        else:
            strings = byte_array.split(b'\x00')
        nnames = 0
        file_number = 0

        for i in range(len(self.__directories)):
            directory = self.__directories[i]
            for j in range(directory.string_qty_idx):
                name = strings[nnames]
                self.__names.append(name)
                nnames += 1

            if directory.type_idx == DirectoryType.MILI_PARAM.value or directory.type_idx == DirectoryType.APPLICATION_PARAM.value or \
                self.__directory_version >= 2 and directory.type_idx == DirectoryType.TI_PARAM.value:
                self.__params[name] = file_number, directory
                f.seek(directory.offset_idx)
                byte_array = f.read(directory.length_idx)
                type = directory.modifier_idx1
                num_entries = directory.modifier_idx2

                type_rep = type_to_str[DataType(type).name]
                type_value = ExtSize[DataType(type).name].value
        
                if type_to_str[DataType(type).name] == 's':
                    if (sys.version_info > (3, 0)):
                        self.__params[name] = [str(struct.unpack(self.__tag + str(int(directory.length_idx / type_value)) + type_rep, byte_array)[0])[2:].split('\\x00'), directory]
                    else:
                        self.__params[name] = [struct.unpack(self.__tag + str(int(directory.length_idx / type_value)) + type_rep, byte_array)[0].split(b'\x00')[0], directory]                
                else:
                    self.__params[name] = [struct.unpack(self.__tag + str(int(directory.length_idx / type_value)) + type_rep, byte_array), directory]
                        
                if name == 'mesh dimensions':
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    self.__dim = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)[0]
                
                if 'Node Labels' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    ints = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)
                    first, last, node_labels = ints[0], ints[1], ints[2:]

                    self.__labels[('M_NODE', 'node')] = {}
                    for j in range (first - 1, last):
                        self.__labels[('M_NODE', 'node')][node_labels[j]] = j + 1
                        self.__labeltomili[('M_NODE', 'node')][node_labels[j]].append(self.__mili_num)


                if 'Element Label' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    ints = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)
                    first, total, ints = ints[0], ints[1], ints[2:]

                    sup_class_idx = name.index('Scls-') + len('Scls-')
                    sup_class_end_idx = name.index('/', sup_class_idx, len(name))
                    class_idx = name.index('Sname-') + len('Sname-')
                    class_end_idx = name.index('/', class_idx, len(name))

                    sup_class = name[sup_class_idx : sup_class_end_idx]
                    clas = name[class_idx : class_end_idx]
                    self.__labels[(sup_class, clas)] = {}

                    for j in range(len(ints)):
                        if 'ElemIds' in name:
                            self.__labels[(sup_class, clas)][self.__label_keys[j]] = ints[j]
                            self.__labeltomili[(sup_class, clas)][self.__label_keys[j]].append(self.__mili_num)
                        else:
                            self.__label_keys.append(ints[j])

                    if 'ElemIds' in name:
                        self.__label_keys = []

                if 'MAT_NAME' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    
                    if (sys.version_info > (3, 0)):
                        matname = str(struct.unpack(str(directory.length_idx) + 's', byte_array)[0])[2:].split('\\x00')[0]  # only works for Python3
                    else:
                        matname = struct.unpack(str(directory.length_idx) + 's', byte_array)[0].split(b'\x00')[0]
                    
                    num = name[-1:]
                    if matname in self.__matname:
                        self.__matname[matname].append(int(num))
                    else:
                        self.__matname[matname] = [int(num)]

                if 'es_' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    i_points = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)
                    first, total, i_points, num_i_ponts = i_points[0], i_points[1], i_points[2:len(i_points) - 1], i_points[len(i_points) - 1]
                    index = name.find('es_')
                    self.__int_points[name[index:]] = [i_points, num_i_ponts]
                    # determine stress or strain


            if directory.type_idx == DirectoryType.STATE_VAR_DICT.value:
                f.seek(directory.offset_idx)
                svar_words, svar_bytes = struct.unpack('2i', f.read(8))
                num_ints = (svar_words - 2)
                ints = struct.unpack(str(num_ints) + 'i', f.read(num_ints * 4))  # what is this
                if (sys.version_info > (3, 0)):
                    s = str(struct.unpack(str(svar_bytes) + 's', f.read(svar_bytes))[0])[2:].split('\\x00')  # only works for Python3
                else:
                    s = struct.unpack(str(svar_bytes) + 's', f.read(svar_bytes))[0].split(b'\x00')
                
                int_pos = 0
                c_pos = 0
                while int_pos < len(ints):
                    sv_name, title = s[c_pos], s[c_pos + 1]

                    agg_type, data_type = ints[int_pos], ints[int_pos + 1]
                    state_variable = StateVariable(sv_name, title, agg_type, data_type)

                    int_pos += 2
                    c_pos += 2

                    if agg_type == AggregateType.ARRAY.value:
                        order, dims = ints[int_pos], []
                        int_pos += 1
                        for k in range(order):
                            dims.append(ints[int_pos])
                            int_pos += 1
                        state_variable.order = order
                        state_variable.dims = dims

                    if agg_type == AggregateType.VECTOR.value or agg_type == AggregateType.VEC_ARRAY.value:
                        if agg_type == AggregateType.VEC_ARRAY.value:
                            order, dims = ints[int_pos], []
                            int_pos += 1
                            for k in range(order):
                                dims.append(ints[int_pos])
                                int_pos += 1
                            state_variable.order = order
                            state_variable.dims = dims

                        state_variable.list_size = ints[int_pos]
                        int_pos += 1
                        sv_names = []
                        for j in range(state_variable.list_size):
                            sv_names.append(s[c_pos])
                            c_pos += 1
                        for sv_name_inner in sv_names:
                            if sv_name_inner in self.__state_variables:
                                sv = self.__state_variables[sv_name_inner]
                            else:
                                sv_name_inner, title = s[c_pos], s[c_pos + 1]
                                agg_type, data_type = ints[int_pos], ints[int_pos + 1]
                                int_pos += 2
                                c_pos += 2
                                sv = StateVariable(sv_name_inner, title, agg_type, data_type)
                                self.__state_variables[sv_name_inner] = [sv, []]
                        state_variable.svars = sv_names

                    self.__state_variables[sv_name] = [state_variable, []]

                    if (sv_name == 'es_1a' or sv_name == 'es_3a' or sv_name == 'es_3c') and sv_name[:-1] in self.__int_points:
                        stresscount = straincount = 0
                        stress = strain = []

                        if 'stress' in self.__state_variables: stress = self.__state_variables['stress'][0].svars
                        if 'strain' in self.__state_variables: strain = self.__state_variables['strain'][0].svars

                        for p in sv_names:
                            if p in stress: stresscount += 1
                            if p in strain: straincount += 1

                        if stresscount == 6:
                            if not 'stress' in self.__int_points: self.__int_points['stress'] = {}
                            self.__int_points['stress'][sv_name] = self.__int_points[sv_name[:-1]]

                        if straincount == 6:
                            if not 'strain' in self.__int_points: self.__int_points['strain'] = {}
                            self.__int_points['strain'][sv_name] = self.__int_points[sv_name[:-1]]

    '''
    Reads in directory information from the Mili file
    '''
    def __readDirectories(self, f, offset):
        number_of_strings = 0
        directory_length = 4 * 6
        state_map_length = 20
        int_long = 'i'
        if self.__directory_version > 2:
            directory_length = 8 * 6
            int_long = 'q'
        offset -= state_map_length * self.__number_of_state_maps + directory_length * self.__number_of_directories  # make this not a hard coded 6 eventually
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + self.__number_of_directories):
            byte_array = f.read(directory_length)
            type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx = \
                struct.unpack(self.__tag + '6' + int_long, byte_array)
            self.__directories.append(Directory(type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx))
            number_of_strings += string_qty_idx
        return offset

    '''
    Reads in information such as nodes and element connections from the Mili fiel
    '''
    def __readMesh(self, f):
        name_cnt = 0
        for i in range(len(self.__directories)):
            directory = self.__directories[i]


            if directory.type_idx == DirectoryType.CLASS_DEF.value:
                superclass = directory.modifier_idx2
                short_name = self.__names[name_cnt]
                long_name = self.__names[name_cnt + 1]
                mocd = MeshObjectClassData(short_name, long_name, superclass)
                self.__mesh_object_class_datas[short_name] = mocd

            if directory.type_idx == DirectoryType.CLASS_IDENTS.value:
                f.seek(directory.offset_idx)
                short_name = self.__names[name_cnt]
                superclass, start, stop = struct.unpack('3i', f.read(12))
                self.__mesh_object_class_datas[short_name].add_block(start, stop)
                superclass = Superclass(superclass).name
                if (superclass, short_name) not in self.__labels:
                    self.__labels[(superclass, short_name)] = {}
                for label in range(start, stop + 1):
                    self.__labels[(superclass, short_name)][label] = label
                    self.__labeltomili[(superclass, short_name)][label].append(self.__mili_num)

            if directory.type_idx == DirectoryType.NODES.value:
                f.seek(directory.offset_idx)
                short_name = self.__names[name_cnt]
                start, stop = struct.unpack('2i', f.read(8))
                num_coordinates = self.__dim * (stop - start + 1)
                floats = struct.unpack(str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                class_name = self.__names[name_cnt]
                sup_class = self.__mesh_object_class_datas[class_name].superclass
                sup_class = Superclass(sup_class).name

                for n in range(0, len(floats), self.__dim):
                    self.__nodes.append([floats[n:n + self.__dim]])

                self.__mesh_object_class_datas[short_name].add_block(start, stop)

            if directory.type_idx == DirectoryType.ELEM_CONNS.value:
                f.seek(directory.offset_idx)
                short_name = self.__names[name_cnt]
                self.__connectivity[short_name] = {}
                superclass, qty_blocks = struct.unpack('2i', f.read(8))
                elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))
                for j in range(0, len(elem_blocks), 2):
                    self.__mesh_object_class_datas[short_name].add_block(elem_blocks[j], elem_blocks[j + 1])

                elem_qty = directory.modifier_idx2
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
                        self.__connectivity[short_name][mo_id] = []
                        mat = ebuf[k + conn_qty]
                        for m in range(0, conn_qty):
                            node = ebuf[k + m]
                            self.__connectivity[short_name][mo_id].append(node)
                        part = ebuf[k + mat_offset]
                        if short_name not in self.__materials[mat]: self.__materials[mat][short_name] = []
                        self.__materials[mat][short_name].append(mo_id)
                        mo_id += 1
                    index = word_qty * elem_qty
            name_cnt += directory.string_qty_idx

    '''
    Reads in all the subrecords for a Mili file
    '''
    def __readSubrecords(self, f):
        for i in range(len(self.__directories)):
            directory = self.__directories[i]
            if directory.type_idx == DirectoryType.STATE_REC_DATA.value:
                srec_int_data = directory.modifier_idx1 - 4
                srec_c_data = directory.modifier_idx2
                f.seek(directory.offset_idx)
                srec_id, srec_parent_mesh_id, srec_size_bytes, srec_qty_subrecs = struct.unpack('4i', f.read(16))
                idata = struct.unpack(str(srec_int_data) + 'i', f.read(srec_int_data * 4))
                if (sys.version_info > (3, 0)):
                    cdata = str(struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0])[2:].split('\\x00')  # only works for Python3
                else:
                    cdata = struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].split(b'\x00')
                
        
                int_pos = 0
                c_pos = 0
                self.__srec_container = SubrecordContainer()

                for k in range(srec_qty_subrecs):
                    org, qty_svars, qty_id_blks = idata[int_pos], idata[int_pos + 1], idata[int_pos + 2]
            
            
                    # for j in range(qty_id_blks):

                    int_pos += 3
                    name, class_name = cdata[c_pos:c_pos + 2]
                    c_pos += 2
                    svars = cdata[c_pos:c_pos + qty_svars]
                    c_pos += qty_svars

                    superclass = self.__mesh_object_class_datas[class_name].superclass

                    sub = Subrecord(name, class_name, org, qty_svars, svars)

                    if superclass != Superclass.M_MESH.value:
                        sub.mo_qty = 0
                        sub.qty_blocks = qty_id_blks
                        for j in range(qty_id_blks):
                            start, stop = idata[int_pos], idata[int_pos + 1]
                            int_pos += 2
                            sub.mo_blocks.append([start, stop])
                            sub.mo_qty += stop - start + 1
                    else:
                        for j in range(qty_id_blks):
                            start, stop = idata[int_pos], idata[int_pos + 1]
                            int_pos += 2
                        sub.mo_qty = 1


                    lump_atoms = []
                    lump_sizes = []
                    lump_offsets = []
                    count = 0
                    sz = 0
                    
                    # Handle Aggregate Types
                    for sv in svars:
                        sv_name = sv
                        sv = self.__state_variables[sv][0]
                        for sv_sv in sv.svars:
                            self.__state_variables[sv_sv][1].append(k)

                    if org == DataOrganization.OBJECT.value:
                        for sv in svars:
                            state_var = self.__state_variables[sv][0]
                            self.__state_variables[sv][1].append(k)

                            atom_size = ExtSize[DataType(state_var.data_type).name].value
                            atoms = state_var.atom_qty(self.__state_variables)  # define this

                            # ## stuff about surface here

                            total_atoms = atoms
                            count += total_atoms
                            sz += total_atoms * atom_size
                        lump_atoms.append(count)
                        lump_sizes.append(sz)

                        if superclass != Superclass.M_SURFACE.value:
                            sub.size = sz * sub.mo_qty
                        else:
                            sub.size = sz

                    elif org == DataOrganization.RESULT.value:
                        for sv in svars:
                            state_var = self.__state_variables[sv][0]
                            self.__state_variables[sv][1].append(k)

                            atoms = state_var.atom_qty(self.__state_variables)
                            # ## stuff about surface here
                            total_atoms = atoms

                            lump_atoms.append(sub.mo_qty * total_atoms)
                            lump_sizes.append(sub.mo_qty * total_atoms * ExtSize[DataType(state_var.data_type).name].value)

                        lump_offsets.append(0)
                        for j in range(1, sub.qty_svars):
                            lump_offsets.append(lump_offsets[j - 1] + lump_sizes[j - 1])

                        j = sub.qty_svars - 1
                        sub.size = lump_offsets[j] + lump_sizes[j]


                    sub.offset = self.__srec_container.size
                    self.__srec_container.size += sub.size
                    self.__srec_container.subrecs.append(sub)

    '''
    This function returns the string representation of a subrecord, to be used
    when interpreting the bytes during reading of state.
    '''
    def __set_string(self, subrecord):
        ret = ''
        type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}

        for sv_name in subrecord.svar_names:
            sv, sub = self.__state_variables[sv_name]
            datatype = DataType(sv.data_type).name
            ret += type_to_str[datatype] * sv.atom_qty(self.__state_variables)


        if subrecord.organization == DataOrganization.OBJECT.value:
            ret_final = ret * subrecord.mo_qty
        else:
            ret_final = str()
            for c in ret:
                ret_final += c * subrecord.mo_qty

        return ret_final

    '''
    This function calls the other reader functions one at a time, spanning the entire
    Mili file.
    '''
    def read(self, file_name, labeltomili=None, mili_num=None, parallel_read=False):
        if self.__filename:
            self.__error('This mili object is already instantiated. You must create another.')
            return
        
        self.__parallel_mode = parallel_read
        orig = file_name
        # Handle case of multiple state files
        end_dir = file_name.rfind(os.sep)
        # dir_name = os.getcwd()
        dir_name = os.getcwd()
        if end_dir != -1:
            dir_name = file_name[:end_dir]
            file_name = file_name[end_dir + 1:]

        parallel = []
        for f in os.listdir(dir_name):
            if file_name in f and f[-1] == 'A':
                parallel.append(f[:-1])

        i = 0
        if len(parallel) > 1:
            self.__filename = file_name
            self.__split_reads(orig, parallel_read)
            return
        else:
            if parallel_read:
                self.__error('Reading in serial mode, since there are less than 2 mili files')
                parallel_read = False
                self.__parallel_mode = False
            state_files = []
            for f in os.listdir(dir_name):
                if file_name in f and f[-1] != 'A':
                    num = f[len(file_name):]
                    state_files.append(dir_name + os.sep + file_name + num)
            state_files.sort()

            file_name = dir_name + os.sep + file_name

            self.__filename = file_name + 'A'
            self.__state_map_filename = state_files

    # Open file with 'b' to specify binary mode
        with open(self.__filename, 'rb') as f:       
            ### Handle parallel information ###
            if type(mili_num) is int: 
                if labeltomili: self.__labeltomili = labeltomili
                self.__mili_num = mili_num
            
            ### Read Header ###
            header = f.read(16)
            mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')
            self.__header_version, self.__directory_version, self.__endian_flag, self.__precision_flag, self.__state_file_suffix_length, self.__partition_flag = \
               struct.unpack('6b', header[4:10])

            if self.__endian_flag == 1:
                self.__tag = '>'
            else:
                self.__tag = '<'

            ### Read Indexing ###
            offset = -16
            f.seek(offset, os.SEEK_END)
            self.__null_termed_names_bytes, self.__number_of_commits, self.__number_of_directories, self.__number_of_state_maps = \
                struct.unpack('4i', f.read(16))

            ### Read State Maps ####
            offset = self.__readStateMaps(f)

            ### Read Directories ###
            offset = self.__readDirectories(f, offset)


            if self.__null_termed_names_bytes > 0:
                 # ## DIRECTORY AND SV DATA #
                self.__readStateVariablesAndParams(f, offset)


                ### MESH DATA ###
                self.__readMesh(f)


                ### SUBRECORD DATA ###
                self.__readSubrecords(f)
        
        if self.__mili_num is not None:
            return [self.__labeltomili, self.__mesh_object_class_datas]

    '''
    Take name and turn it into vector and component names
    '''
    def __parse_name(self, name):
        vector, component = None, name
        if '[' in name:
            pos = name.find('[')
            vector = name[:pos]
            component = name[pos + 1:len(name) - 1]
        return [vector, component]

    '''
    Create an answer from res dictionary
    This is what is returned by query
    '''
    def __create_answer(self, res, names, materials, labels, class_name, state_numbers, modify, raw_data):
        if raw_data:
            return res
        answer = Answer()

        for state_number in state_numbers:
            state = StateAnswer()
            answer.state_answers.append(state)
            state.state_number = state_number

            for name in names:
                for label in labels:
                    item = Item(name, None, None, label, class_name, modify)
                    if name in res[state_number]:
                        item.set(res[state_number][name][label])
                        state.items.append(item)

        return answer

    '''
    Return true if this class name combo is a vector array
    '''
    def __is_vec_array(self, name, class_name):
        if not name: return False
        if 'stress' in name and 'stress' in self.__int_points:
            elem_sets = self.__int_points['stress']
            vars = self.__state_variables['stress'][0].svars
        elif 'strain' in name and 'strain' in self.__int_points:
            elem_sets = self.__int_points['strain']
            vars = self.__state_variables['strain'][0].svars
        else:
            return False

        set_names = elem_sets.keys()
        for set_name in set_names:
            temp_sv, temp_subrecords = self.__state_variables[set_name]
            temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
            if temp_subrecord.class_name == class_name:
                return elem_sets[set_name][0]

        return False

    '''
    Given a single state variable name, a number of states, and a number of label(s) return the requested value(s)
    '''
    def __variable_at_state(self, subrecord, labels, name, vars, sup_class, clas, sub, res, state, modify=False, int_points=False):
        mo_search_arr = []
        indices = {}
        values = {}  # from label to value

        # Deal with names like vector[component]
        vector, variables = self.__parse_name(name)
        variables = [variables]

        # These are the mo_ids we are interested in for this subrecord
        for label in labels:
            if label not in self.__labels[(sup_class, clas)]:
                # return self.__error('label ' + str(label) + ' was not found in ' + clas)
                nothing = 0
            else:
                mo_search_arr.append([label, self.__labels[(sup_class, clas)][label]])

        mo_index = 0
        mo_idx_found = []

        # Search subrecord blocks to find what index the mo_ids are at
        for range in subrecord.mo_blocks:
            start, end = range
            for mo_search in mo_search_arr:
                label, mo_search = mo_search
                if mo_search >= start and mo_search <= end:
                    mo_idx_found.append([label, mo_index + mo_search - start])
            mo_index += end - start

        indices[sub] = defaultdict(list)

        # Deal with aggregate types and create list of state variable names
                
        if name in self.__state_variables and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR':
            variables = self.__state_variables[name][0].svars
        
        sv_names = []
        sv_group_start = {}
        sv_group_start[0] = 0
        sv_group_len = {}
        group_idx = 0
        for sv in subrecord.svar_names:
            sv_var = self.__state_variables[sv][0]
            sv_group_len[group_idx] = max(1, len(sv_var.svars))
            if group_idx: sv_group_start[group_idx] = sv_group_start[group_idx - 1] + sv_group_len[group_idx]
            if len(sv_var.svars) > 0:
                for sv_name in sv_var.svars:
                    sv_names.append(sv_name)
            else:
                sv_names.append(sv)
            group_idx += 1
            
        var_indexes = []
        for child in variables:
            if child not in sv_names:
                return self.__error(child + ' not a valid variable name')
            # var_indexes.append(sv_names.index(child))
            for sv_group in subrecord.svar_names:
                if sv_group == child: 
                    var_indexes.append([subrecord.svar_names.index(sv_group), 0])
                sv = self.__state_variables[sv_group][0].svars
                if child in sv:
                    var_indexes.append([subrecord.svar_names.index(sv_group), sv.index(child)])
                    
        # Add correct values given organizational structure and correct indexing
        if int_points:
            int_points, num_int_points = int_points[:-1], int_points[-1:][0]

        for mo_index in mo_idx_found:
            if int_points: indexes = {}
            else: indexes = []
            label, mo_index = mo_index
            for var_index in var_indexes:
                var_index, var_in_group = var_index
                if int_points: var_index = var_in_group
                if int_points and var_index not in indexes: indexes[var_index] = {}
                if int_points:
                    offset = mo_index * len(sv_names) * num_int_points
                    for int_point in int_points:
                        index = offset + var_index * num_int_points + int_point - 1
                        indexes[var_index][int_point] = index
                else:
                    if subrecord.organization == DataOrganization.OBJECT.value:
                        indexes.append(mo_index * len(sv_names) + sv_group_start[var_index] + var_in_group)
                    else:
                        indexes.append(sv_group_start[var_index] * subrecord.mo_qty + sv_group_len[var_index] * mo_index + var_in_group)

            # 3 different aggregate types here - contructing the res
            if int_points:
                indices[sub][label] = {}
                if label not in res[state][name]: res[state][name][label] = {}
                v_index = 0
                for index in indexes.keys():
                    indices[sub][label][sv_names[index]] = {}
                    sv_name = sv_names[index]
                    res[state][name][label][sv_name] = {}
                    for int_point in int_points:
                        res[state][name][label][sv_name][int_point] = vars[indexes[index][int_point]]
                        indices[sub][label][sv_names[index]][int_point] = indexes[index][int_point]
                    v_index += 1
            elif name in self.__state_variables and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR':
                res[state][name][label] = []
                for index in indexes:
                    res[state][name][label].append(vars[index])
                    indices[sub][label].append(index)
            else:
                res[state][name][label] = vars[indexes[0]]
                indices[sub][label].append(indexes[0])

        if modify:
            return [res, indices]

        return res

    '''
    Turn a class name and element list into the appropriate label array
    '''
    def __getLabelsFromClassElems(self, class_name, elems):
        ret = []
        if class_name not in self.__mesh_object_class_datas:
            return ret
        sup_class = self.__mesh_object_class_datas[class_name].superclass
        sup_class = Superclass(sup_class).name
        labels = self.__labels[(sup_class, class_name)]
        for elem in elems:
            id, class_name_elem = elem
            for k in labels.keys():
                label, mo_id = k, labels[k]
                if mo_id == id and class_name_elem == class_name:
                    ret.append(label)

        return ret

    '''
    Add two default dictionaries together
    '''
    def __addDicts(self, a, b):
        if not a:
            return b
        for k in a:
            b[k] = b[k].union(a[k])
        return b

    '''
    Set the output file for error messages (screen output)
    '''
    def setErrorFile(self, file_name=None):
        if not file_name:
            self.__error_file = self.__filename + '_error'
        else:
            self.__error_file = file_name
        with open(self.__error_file, 'w') as f:
            f.write('')
        if len(self.__milis) > 1:
            if self.__parallel_mode:
                for i in range(len(self.__parent_conns)):
                    conn, k = self.__parent_conns[i]
                    conn.send(["Error file", self.__error_file])
            else:
                for mili in self.__milis:
                    mili.setErrorFile(file_name)

    '''
    Write error to error file if it exists. Otherwise write it to screen.
    '''
    def __error(self, msg):
        if self.__error_file:
            with open(self.__error_file, 'a+') as f:
                f.write(msg + '\n')
            return
        print(msg)
        return

    '''
    Convert a material name into the labels
    '''
    def __material_to_labels(self, material, class_name, labels):
        if type(material) is not str and type(material) is not int:
            return self.__error('material must be string or int')
        if material not in self.__matname and material not in self.__materials:
            return self.__error('There is no ' + str(material) + ' material')
        elems = []
        if material in self.__matname:
            for mat in self.__matname[material]:
                elems += self.__materials[mat][class_name]
        else:
            if class_name not in self.__materials[material]:
                self.__error("There are no elements of material " + str(material) + " and class " + str(class_name))
            elems = self.__materials[material][class_name]
        elems = [[i, class_name] for i in elems]
        labels_mat = self.__getLabelsFromClassElems(class_name, elems)
        if labels:
           for label in labels:
               if label not in labels_mat:
                   labels.remove(label)
        else:
           labels = labels_mat
        return labels
        
    
    '''
    Create processes and read the files
    '''
    def __split_reads(self, file_name, parallel_read):
        # Handle case of multiple state files
        end_dir = file_name.rfind(os.sep)
        # dir_name = os.getcwd()
        dir_name = os.getcwd()
        if end_dir != -1:
            dir_name = file_name[:end_dir]
            file_name = file_name[end_dir + 1:]
        parallel = []
        for f in os.listdir(dir_name):
            if file_name in f and f[-1] == 'A':
                parallel.append(f[:-1])
        
        cpus = psutil.cpu_count(logical=False)
        self.__parent_conns = []
        i = 0
        if len(parallel) > 1:
            self.__filename = file_name
            for p in parallel:
                mili = Mili()
                self.__milis.append(mili)
                if parallel_read:
                    parent_conn, child_conn = mp.Pipe()
                    pr = mp.Process(target=self.__child_read, args=(mili, child_conn, dir_name + os.sep + p, i,))
                    self.__parent_conns.append([parent_conn, i])
                    pr.start()
                else:
                    labeltomili, mesh_objects = mili.read(dir_name + os.sep + p, mili.__labeltomili, i)
                    for mesh_obj in mesh_objects:
                        if mesh_obj not in self.__mesh_object_class_datas:
                            self.__mesh_object_class_datas[mesh_obj] = mesh_objects[mesh_obj]

                    for class_name in mili.__labels:
                        labmap = mili.__labels[class_name]
                        for label_key in labmap:
                            self.__labeltomili[class_name][label_key].append(i)
                i += 1
                
        if parallel_read:
            # Receive reads from children
            activeChildren = len(self.__parent_conns)
            while activeChildren > 0:
                for conn in self.__parent_conns:
                    conn, i = conn
                    if (conn.poll()):
                        child_number, mesh_objects, labels = conn.recv()
                        for mesh_obj in mesh_objects:
                            if mesh_obj not in self.__mesh_object_class_datas:
                                self.__mesh_object_class_datas[mesh_obj] = mesh_objects[mesh_obj]

                        for class_name in labels:
                            labmap = labels[class_name]
                            for label_key in labmap:
                                self.__labeltomili[class_name][label_key].append(i)
                
                        activeChildren -= 1
                        
        
    '''
    Child read and wait process
    '''
    def __child_read(self, mili, conn, file_name, i):
        # Read the file
        mili.__labeltomili, mesh_objects = mili.read(file_name, mili.__labeltomili, i)
        
        # Send back mesh information and labels
        conn.send([i, mesh_objects, mili.__labels])
        
        answ = defaultdict(dict)
        # ## Wait for querys
        while True:    
            if conn.poll():
                query = conn.recv()
                if query == "End":
                    conn.close()
                    return
                if query[0] == "Error file":
                    mili.__error_file = query[1]
                elif query[0] == "labels of material":
                    material = query[1]
                    conn.send(mili.labels_of_material(material))
                elif query[0] == "nodes of material":
                    material = query[1]
                    conn.send(mili.nodes_of_material(material))
                elif query[0] == "nodes of elem":
                    label, class_name = query[1]
                    conn.send(mili.nodes_of_elem(label, class_name))
                elif query[0] == "modify":
                    state_variable, class_name, value, labels, state_numbers, int_points = query[1:]
                    mili.modify_state_variable(state_variable, class_name, value, labels, state_numbers, int_points)
                elif query[0] == "get params":
                    params = mili.getParams()
                    conn.send(params)
                elif query[0] == "get state maps":
                    state_maps = mili.getStateMaps()
                    conn.send(state_maps)
                elif query[0] == "get directories":
                    directories = mili.getDirectories()
                    conn.send(directories)
                elif query[0] == "get state variables":
                    state_vars = mili.getStateVariables()
                    conn.send(state_vars)
                elif query[0] == "get labels":
                    labels = mili.getLabels()
                    conn.send(labels)
                elif query[0] == "get materials":
                    materials = mili.getMaterials()
                    conn.send(materials)
                else:
                    sv, class_name, material, label, state_numbers, modify, int_points, raw, answ = query
                    
                    answer = mili.query(sv, class_name, material, label, state_numbers, modify, int_points, raw, answ)
                
                    if not answer:
                        conn.send("Fail")
                    else:
                        state_number_zero = state_numbers if not isinstance(state_numbers, list) else state_numbers[0]
                        if answ and sv in answ[state_number_zero]:
                            send_answer = self.__create_answer(answ, sv, material, label, class_name, state_numbers, modify, raw)
                            conn.send(send_answer)
                        else:
                            conn.send("Fail")
        
    '''
    Add the to_add to res. This combines to results from two queries into one and returns it
    '''
    def __addres(self, to_add, res):
        for state in to_add:
            if state not in res:
                res[state] = defaultdict(defaultdict)
            for name in to_add[state]:
                for label in to_add[state][name]:
                    res[state][name][label] = to_add[state][name][label]
        return res
    

    '''
    Single query function that given inputs can call the other functions

    Arguments:
    1. names: short names for all the shortnames you are interested in
    2. class: class name interested in
    3. material: string name of material you are interested in
            Empty: Ignore material
    4. labels: labels that you are interested in
            Empty: Find all possible labels
    5. state_numbers: States you are interested in
            Empty: all states
    6. modify: whether or not this is part of a modification call
    7. int_points: if this is a vector array, which integration points
    8. raw_data: whether the user wants raw data or Answer object
    9. res: the result passed around when doing parallel querying, not for the user


    The following is the structure of the result that is passed to create answer
    res[state][name][label] = value
    '''
    def query(self, names, class_name, material=None, labels=None, state_numbers=None, modify=False, int_points=False, raw_data=True, res=defaultdict(dict)):
        # Parse Arguments
        if state_numbers is None:
            state_numbers = [i - 1 for i in range(1, self.__number_of_state_maps + 1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        if type(labels) is int:
            labels = [labels]

        if material:
            if len(self.__milis):
                labels = self.labels_of_material(material)
                if class_name not in labels:
                    self.__error('There are no elements from class ' + str(class_name) + ' of material ' + str(material))
                labels = list(labels[class_name])
            else:
                labels = self.__material_to_labels(material, class_name, labels)

            if not labels or not len(labels):
                return self.__error('There are no elements from class ' + str(class_name) + ' of material ' + str(material))
        if labels:
            if type(labels) is not list or type(labels[0]) is not int:
                return self.__error('labels must of a list of ints or an int')
        
        if not labels and class_name not in self.__mesh_object_class_datas:
            return self.__create_answer({}, names, material, labels, class_name, state_numbers, modify, raw_data)
        if not labels and not self.__parallel_mode and not len(self.__milis):
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name
            labels = self.__labels[(sup_class, class_name)].keys()
        if not labels and not self.__parallel_mode and len(self.__milis):
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name
            labels = self.__labeltomili[(sup_class, class_name)].keys()

        if type(names) is str:
            names = [names]
        if type(names) is not list or type(names[0]) is not str:
            return self.__error('state variables names must be a string or list of strings')
        if modify and type(modify) is not bool:
            return self.__error('modify must be boolean')
        if raw_data and type(raw_data) is not bool:
            return self.__error('raw data must be boolean')
        if type(int_points) is int:
            int_points = [int_points]
        if int_points and type(int_points) is not list:
            return self.__error('int point must be an integer or list of integers')
        if type(state_numbers) is not list or type(state_numbers[0]) is not int:
            return self.__error('state numbers must be an integer or list of integers')

        # Deal with parallel Mili file case
        if len(self.__milis):
            answ = defaultdict(dict)
            for sv in names:
                vector, variables = self.__parse_name(sv)
                if not vector: sv_key = variables
                else: sv_key = vector

                # turn sv and class into elem
                if class_name not in self.__mesh_object_class_datas: return self.__error('invalid class name')
                sup_class = self.__mesh_object_class_datas[class_name].superclass
                sup_class = Superclass(sup_class).name
                
                failcount = 0
                milis = set()
                mili_to_labels = defaultdict(list)
                if labels != None:
                    for label in labels:
                        mili = self.__labeltomili[(sup_class, class_name)][label]
                        for m in mili:
                            mili_to_labels[m].append(label)
                            milis.add(m)
                    milis = list(milis)
                if not labels: milis = set([i for i in range(len(self.__milis))])
                
                if self.__parallel_mode:
                    mili_conns = []
                    for mili_index in milis:
                        mili_conn, i = self.__parent_conns[mili_index]
                        mili_conn.send([sv, class_name, material, mili_to_labels[mili_index], state_numbers, modify, int_points, True, answ])
                        mili_conns.append(mili_conn)
                
                    while failcount < len(milis):
                        for mili_conn in mili_conns:
                            if mili_conn.poll():
                                query_response = mili_conn.recv()
                                if query_response == "Fail": failcount += 1
                                else: 
                                    answ = self.__addres(query_response, answ)
                                    failcount += 1
                else:
                    for mili_index in milis:
                        resp = self.__milis[mili_index].query(sv, class_name, material, mili_to_labels[mili_index], state_numbers, modify, int_points, True, answ)
                        if resp: 
                            answ = resp 
                        
            return self.__create_answer(answ, names, material, labels, class_name, state_numbers, modify, raw_data)

        # Run Correct Function
        for state in state_numbers:
            if state not in res: res[state] = defaultdict(dict)
            if state < 0 or state >= len(self.__state_maps):
                return self.__error('There is no state ' + str(state))
            state_map = self.__state_maps[state]

            with open(self.__state_map_filename[state_map.file_number], 'rb') as f:
                f.seek(state_map.file_offset)
                byte_array = f.read(8)
                time, state_map_id = struct.unpack(self.__tag + 'fi', byte_array)

                for name in names:
                    # Handle case of vector[component]
                    vector, variables = self.__parse_name(name)
                    if not vector: vector = variables

                    if self.__is_vec_array(vector, class_name):
                        if 'stress' in name: elem_sets = self.__int_points['stress']
                        elif 'strain' in name: elem_sets = self.__int_points['strain']

                        for set_name in elem_sets.keys():
                            temp_sv, temp_subrecords = self.__state_variables[set_name]
                            temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
                            if temp_subrecord.class_name == class_name:
                                sv, subrecords = temp_sv, temp_subrecords
                                set_name_chosen = set_name
                        if not int_points:
                            int_points = list(self.__is_vec_array(vector, class_name))
                        else:
                            possible_int_points = self.__is_vec_array(vector, class_name)
                            for i in range(len(int_points)):
                                ip = int_points[i]
                                if ip not in possible_int_points:
                                    idx_ip, ip = min(enumerate(possible_int_points), key=lambda x: abs(x[1] - ip))
                                    self.__error(str(ip) + ' is not an integration point, but the closest is ' + str(ip))
                                int_points[i] = ip
                        int_points.append(len(self.__is_vec_array(vector, class_name)))

                    elif vector: 
                        if vector not in self.__state_variables:
                            return self.__error('There is no variable ' + vector)
                        sv, subrecords = self.__state_variables[vector]
                    else:
                        if name not in self.__state_variables:
                            return self.__error('There is no variable ' + name)
                        sv, subrecords = self.__state_variables[name]
                     
                    for sub in subrecords:
                        
                        f.seek(state_map.file_offset)
                        
                        subrecord = self.__srec_container.subrecs[sub]
                        f.seek(subrecord.offset, 1)
                        byte_array = f.read(subrecord.size)
                        s = self.__set_string(subrecord)
                        
                        vars = struct.unpack(self.__tag + s, byte_array)

                        if class_name == subrecord.class_name:
                            sup_class = self.__mesh_object_class_datas[subrecord.class_name].superclass
                            sup_class = Superclass(sup_class).name

                            if modify:
                                return self.__variable_at_state(subrecord, labels, name, vars, sup_class, subrecord.class_name, sub, res, state, modify, int_points)
                            else:
                                res = self.__variable_at_state(subrecord, labels, name, vars, sup_class, subrecord.class_name, sub, res, state, modify, int_points)

        return self.__create_answer(res, names, material, labels, class_name, state_numbers, modify, raw_data)

    '''
    Given a specific material. Find all mo_ids with that material and return
    their values.
    '''
    def __elements_of_material(self, material):
        mo_ids = defaultdict(list)

        if material not in self.__matname and material not in self.__materials:
            return self.__error('There is no material ' + str(material))

            # if they specify the material name
        if material in self.__matname:
            matnumbers = self.__matname[material]
            for matnumber in matnumbers:
                mo_id_class = self.__materials[matnumber]
                for class_name in mo_id_class:
                    mo_ids[class_name] += mo_id_class[class_name]
        else:
            mo_id_class = self.__materials[material]
            for class_name in mo_id_class:
                mo_ids[class_name] += mo_id_class[class_name]

        return mo_ids

        
    '''
    Accumulate information from children
    '''
    def __get_children_info(self, accumulater, function, message, data_send, serial_function=None):
        if self.__parallel_mode:
            failcount = 0
            mili_conns = []
            for mili_index in range(len(self.__parent_conns)):
                mili_conn, i = self.__parent_conns[mili_index]
                mili_conn.send([message, data_send])
                mili_conns.append(mili_conn)
                        
            while failcount < len(mili_conns):
                for mili_conn in mili_conns:
                    if mili_conn.poll():
                        query_response = mili_conn.recv()
                        if query_response == None: 
                            failcount += 1
                        else:
                            if function == self.__addDicts:
                                accumulater = function(query_response, accumulater)
                            if function == set.union:
                                accumulater = function(set(query_response), accumulater)
                            if function == None:
                                accumulater = query_response
                            failcount += 1
                        
                if failcount >= len(mili_conns):
                    break
        else:
            for m in self.__milis:
                if message == "nodes of material":
                    serial = serial_function(m, data_send)
                    if serial: accumulater = function(set(serial), accumulater)
                elif message == "nodes of elem":
                    label, class_name = data_send
                    val = serial_function(m, label, class_name)
                    if val: accumulater = val
                else: 
                    accumulater = function(serial_function(m, data_send), accumulater)
            
        
        return accumulater

    '''
    Find the labels associated with a material
    '''
    def labels_of_material(self, material, raw_data=True):
        labels = defaultdict(list)
        labels_list = []
        class_names = []

        if len(self.__milis) > 1:
            labels = self.__get_children_info(defaultdict(set), self.__addDicts, "labels of material", material, Mili.labels_of_material)
                
            for class_name in labels:
                l = list(labels[class_name])
                labels_list += l
                class_names += [class_name] * len(l)

        else:
            if material not in self.__matname and material not in self.__materials:
                return self.__error('There is no material ' + str(material))

            elements = self.__elements_of_material(material)
            for class_name in elements:
                l = self.__getLabelsFromClassElems(class_name, [[elements[class_name][i], class_name] for i in range(len(elements[class_name]))])
                labels[class_name] += l
                labels_list += l
                class_names += [class_name] * len(l)

        if raw_data:
            return labels

        answer = Answer()
        answer.set(None, None, labels_list, None, class_names, None)

        return answer

    '''
    Find nodes associated with a material number
    '''
    def nodes_of_material(self, material, raw_data=True):
        nodes = set()

        if len(self.__milis) > 1:
            nodes = self.__get_children_info(set(), set.union, "nodes of material", material, Mili.nodes_of_material)
        else:
            labels = self.labels_of_material(material)
            if not labels:
                return self.__error('There are no labels with material ' + str(material))
            for class_name in labels:
                labs = labels[class_name]
                for lab in labs:
                    nodes.add(lab)

        if raw_data:
            return list(nodes)

        answer = Answer()
        answer.set(None, None, None, list(nodes), 'node', None)

        return answer

    '''
    Find nodes associated with an element number
    '''
    def nodes_of_elem(self, label, class_name, raw_data=True):
        labels = None
        if len(self.__milis) > 1:
            labels = self.__get_children_info(None, None, "nodes of elem", [label, class_name], Mili.nodes_of_elem)
        
        else:
            if class_name not in self.__mesh_object_class_datas:
                return self.__error('Class name ' + class_name + ' not found')
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name

            if label not in self.__labels[(sup_class, class_name)]:
                return self.__error('label ' + str(label) + ' not found')
            mo_id = self.__labels[(sup_class, class_name)][label]
            labels = self.__connectivity[class_name][mo_id]

        if raw_data:
            return labels

        answer = Answer()
        answer.set(None, None, None, labels, class_name, None)

        return answer

    '''
    Modifies a state variable at a specific label and time state

    Arguments:
    1. state_variable: short names for all the shortnames you are interested in
    2. class_name: class name interested in
    3. value: the value to enter:
        dictionary that has the following strucute:
            value[state][state_variable][label][val]
                where val is as follows:
                    scalar or vector component:
                        val is an integer/float/etc.
                    vector:
                        val is an array
                    vector array:
                        val is a dictionary as follows:
                            val = {sv_name : {int_point : float/integet/etc.}}
    4. labels: labels that you are interested in
            Empty: Find all possible labels
    5. state_numbers: States you are interested in
            Empty: all states
    7. int_points: if this is a vector array, which integration points
    '''
    def modify_state_variable(self, state_variable, class_name, value, labels, state_numbers, int_points=False):
        if len(self.__milis) > 1:
            if self.__parallel_mode:
                for mili_index in range(len(self.__parent_conns)):
                    mili_conn, i = self.__parent_conns[mili_index]
                    mili_conn.send(["modify", state_variable, class_name, value, labels, state_numbers, int_points])
            else:
                for mili_index in range(len(self.__milis)):
                    self.__milis[mili_index].modify_state_variable(state_variable, class_name, value, labels, state_numbers, int_points)
            return
  

        query = self.query(state_variable, class_name, None, labels, state_numbers, True, int_points)
        if type(query) is not list:
            return None
        
        res, indices = query
        type_to_str = {'s' : 'M_STRING', 'f' : 'M_FLOAT', 'd' : 'M_FLOAT8', 'i' : 'M_INT', 'q' : 'M_INT8'}

        if not state_numbers:
            state_numbers = [i - 1 for i in range(1, self.__number_of_state_maps + 1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        if type(labels) is int:
            labels = [labels]
        if int_points:
            int_points = int_points[:len(int_points) - 1]
        if type(state_variable) is not str:
            return self.__error('state variable must be a string')

        for state in state_numbers:
            if state < 0 or state >= len(self.__state_maps):
                return self.__error('There is no state ' + str(state))

            state_map = self.__state_maps[state]

            with open(self.__state_map_filename[state_map.file_number], 'rb+') as f:
                f.seek(state_map.file_offset)
                byte_array = f.read(8)
                time, state_map_id = struct.unpack(self.__tag + 'fi', byte_array)

                name = state_variable

                vector, variables = self.__parse_name(name)
                if not vector: vector = variables
                if self.__is_vec_array(vector, class_name):
                    if 'stress' in name: elem_sets = self.__int_points['stress']
                    elif 'strain' in name: elem_sets = self.__int_points['strain']

                    for set_name in elem_sets.keys():
                        temp_sv, temp_subrecords = self.__state_variables[set_name]
                        temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
                        if temp_subrecord.class_name == class_name:
                            sv, subrecords = temp_sv, temp_subrecords
                            set_name_chosen = set_name
                    if not int_points:
                        int_points = list(self.__is_vec_array(vector, class_name))
                    # int_points.append(len(self.__is_vec_array(vector, class_name)))
                elif vector:
                    if vector not in self.__state_variables:
                        return self.__error('There is no variable ' + str(vector))
                    sv, subrecords = self.__state_variables[vector]
                else:
                    if name not in self.__state_variables:
                        return self.__error('There is no variable ' + str(name))
                    sv, subrecords = self.__state_variables[name]

                for sub in subrecords:
                    subrecord = self.__srec_container.subrecs[sub]
                    f.seek(subrecord.offset, 1)
                    s = self.__set_string(subrecord)

                    # Iterate through s, the string representation of the subrecord, adding
                    # offset along the way
                    for label in labels:
                        if int_points:
                            for sv_name in indices[sub][label]:
                                int_point_to_index = indices[sub][label][sv_name]
                                for ip in int_points:
                                    offset = 0
                                    for i in range(indices[sub][label][sv_name][ip]):
                                        char = s[i]
                                        offset += ExtSize[type_to_str[char]].value
                                    f.seek(offset, 1)

                                    vector, component = self.__parse_name(name)
                                    if vector: n = vector
                                    else: n = name

                                    byte_array = struct.pack(self.__tag + s[int_point_to_index[ip]], value[state][n][label][sv_name][ip])
                                    f.write(byte_array)
                                    f.seek(-offset - len(byte_array), 1)
                        else:
                            for idx in range(len(indices[sub][label])):
                                offset = 0
                                for i in range(indices[sub][label][idx]):
                                    char = s[i]
                                    offset += ExtSize[type_to_str[char]].value

                                # Seek to variable location and write in the byte array
                                f.seek(offset, 1)
                                if type(value[state][name][label]) is not list:
                                    value[state][name][label] = [value[state][name][label]]
                                byte_array = struct.pack(self.__tag + s[indices[sub][label][idx]], value[state][name][label][idx])
                                f.write(byte_array)
                                f.seek(-offset - len(byte_array), 1)

                if 'post_modified' in self.__params:
                    file_number, directory = self.__params['post_modified']
                    f.seek(directory.offset_idx)
                    one = struct.pack(self.__tag + 'i', 1)
                    self.__params['post_modified'][0] = 1
                    f.write(one)

'''
This function is an example of how a user could use the Mili reader
'''
def main():
    # You can run code here as well if you copy the library!
    
    # f = '/g/g20/legler5/Xmilics/XMILICS-toss_3_x86_64-RZTRONA/bin_debug/HexModel1.plt_c'
    # f = '/g/g20/legler5/Mili/MILI-toss_3_x86_64_ib-RZGENIE/d3samp6.plt'
    # f = "taurus/taurus.plt"
    # f = '/usr/workspace/wsrzc/legler5/BigMili/dblplt'
    # f = 'parallel/d3samp6.plt'
    f = 'd3samp6.plt'
    mili = Mili()
    # mili.read(f, parallel_read=True)
    mili.read(f, parallel_read=False)
    
    d = mili.getParams()
    
    print(mili.query('sz', 'brick', None, None, 10))
    

    # mili.setErrorFile()
        
if __name__ == '__main__':
        main()
