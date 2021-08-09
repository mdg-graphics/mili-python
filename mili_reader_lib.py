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
from cProfile import label
from _collections import OrderedDict
'''
This is the 'meta' information for a statemap, but not the statemap itself.
'''

from collections import defaultdict
import os
import struct
import sys
import time
import re
import atexit
import pdb
from mili_combiner_optparser import parseParameters, gather_A_files

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
    def __init__(self, type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx, string_offset_idx):
        self.type_idx = type_idx
        self.modifier_idx1 = modifier_idx1
        self.modifier_idx2 = modifier_idx2
        self.string_qty_idx = string_qty_idx
        self.offset_idx = offset_idx
        self.length_idx = length_idx
        self.string_offset_idx = string_offset_idx
        self.strings = []
    
    def addStrings( self, strings ):
        assert( len(strings) == self.string_qty_idx )
        self.strings = strings

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

class Search:
    def __init__(self):
        self.__table = {}
    
    def find(self, mili_class_name, variable_name, item=None):
        
        all = False
        
        if (mili_class_name == None or len(mili_class_name) == 0 ) or (variable_name == None and len(variable_name)>0) :
            print("Class name or Variable name not found")
            return None
        
        if not variable_name in self.__table.keys() or not mili_class_name in self.__table[variable_name].keys():
            return None
        
        if item == None:
            all = True
        
        subrecords = self.__table[variable_name][mili_class_name]
        
        for subrecord in subrecords:
            for block in subrecord.mo_blocks:
                if(block[0] <= item and item <=block[1]):
                    return subrecord
        
        return None    
        
    def __str__(self):
        
        for key in self.__table.keys():
            print(key)
            for class_key in self.__table[key].keys():
                print("\t",class_key)
                for subrecord in self.__table[key][class_key]:
                    print("\t\t",subrecord.name)
                    print("\t\t",subrecord.svar_names)
                        #print("\t\t\tMo_ids: ",subrecord.mo_blocks[i][0],"-",subrecord.mo_blocks[i][1])
        return "\n"
    
    def add(self, mili_class_name, variable_name, subrecord):  
        if (mili_class_name == None or len(mili_class_name) == 0) or (variable_name == None and len(variable_name)>0) :
            return False
        
        if variable_name not in self.__table.keys():
            self.__table[variable_name] = {}
        if mili_class_name not in self.__table[variable_name].keys():
            self.__table[variable_name][mili_class_name] = []
        self.__table[variable_name][mili_class_name].append(subrecord)

class MiliCombiner:
    def __init__(self, base_name, parameters):
        
        print(parameters)
        self.file_name = str()
        self.__header = None
        self.__outputName = parameters["output_file"]
        self.__complete_path = parameters["input_dir"]+os.sep+parameters["input_file"]
        mili = Mili()
        if(mili == None):
            return None
        
        mili.read(self.__complete_path)
        
        #Read header and write the information.  It is the same for all processors
        self.__header = mili.getHeader()
        mili_taur = struct.unpack('4s', self.__header[:4])[0].decode('ascii')
        assert(mili_taur == 'mili' or mili_taur == 'taur')
        
        self.__header_version, self.__directory_version, self.__endian_flag, self.__precision_flag, self.__state_file_suffix_length, self.__partition_flag = \
            struct.unpack('6b', self.__header[4:10])

        if self.__endian_flag == 1:
            self.__tag = '>'
        else:
            self.__tag = '<'

        #Get all the processor's timesteps      
        self.timestep_maps = mili.getStateMaps()
        if mili.getProcessorCount() != len(self.timestep_maps):
            print("Number of processors and state maps do not match")
            return None
        
        self.max_state = int()
        
        #Do a quick run over the time steps and only do the lowest number of steps available
        for state in range(0,len(self.timestep_maps)):
            if state == 0:
                self.max_state = len(self.timestep_maps[state])
            else:
                if self.max_state > len(self.timestep_maps[state]):
                    self.max_state = len(self.timestep_maps[state])
        
        params = mili.getParams()
        
        for param in params:
            print(type(param))
            for key in param.keys():
                print(key, param[key])
            
        
        
        self.incoming_directories = mili.getDirectories()
        
        self.master_index = [0]*4
        labels = mili.getLabels()
        
        
        self.__label_mapping, self.__global_ident_mapping = self.__createGlobalMapping(labels)
        print(self.__label_mapping)
        print(list(self.__label_mapping['shell']['map'].keys()))
        print(list(self.__label_mapping['shell']['map'].values()))
        """for proc in range(0,len(labels)):
            print('processor:', proc)
            print(labels[proc])
            print(self.__global_ident_mapping[proc])
            print()
        """
        with open(self.__outputName+"A", 'wb') as f:
            f.write(self.__header)
            f.close()
        
        del(mili)
        
    def __createGlobalMapping(self,labels):
        global_ident_map = OrderedDict()
        labels_to_global= OrderedDict()
        for proc  in range(0,len(labels)):
            if proc not in global_ident_map.keys():
                global_ident_map[proc] = {}
            for key in labels[proc].keys():
                if key not in labels_to_global.keys():
                    labels_to_global[key]={'current_size':0,'map':{}}
                if key not in global_ident_map[proc].keys():
                    global_ident_map[proc][key] = {}
                for label in labels[proc][key].keys():
                    if label not in labels_to_global[key]['map'].keys():
                        labels_to_global[key]['current_size'] = labels_to_global[key]['current_size']+1
                        global_ident_map[proc][key][labels[proc][key][label]] = labels_to_global[key]['current_size']
                        labels_to_global[key]['map'][label] = labels_to_global[key]['current_size']
                    else:
                        global_ident_map[proc][key][labels[proc][key][label]] = labels_to_global[key]['map'][label]
                        
        return labels_to_global, global_ident_map    
                        
class Mili:
    def __init__(self, read_file=None, parallel_read=False, processors=None, a_files=None):
        """Mili Class.

        A Mili object contains data structures to store objects such as directories
        and subrecords. It also has a number of querying functions that a user can call
        to access this data. Takes in the path of the file to read.

        Args:
            read_file (str): The base name of the plot file data. Defaults to None.
            parallel_read (bool): Flag to turn on/off reading in parallel. Defaults to False.
            processors (int): The number of processors to use when reading in parallel. Defaults
                to None. This means that all available processors are used.
            a_files (List[int]): List of A file processor suffixes to read in. This defaults to None,
                but can be used to limit the A files that are read in, if desired.
        """
        self.__milis = []  # list of parallel mili files
        self.__parent_conns = []  # list of parent connection, index number
        self.__mili_num = None  # Number of mili file (processor number)
        self.__labeltomili = defaultdict(lambda: defaultdict(list))  # map from (superclass, label) to dict of label:file
        self.__state_maps = []
        self.__directories = {}
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
        self.__mili_search = Search()  #added to help speed up the query routine
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
        self.__header = None
        # Number of A files. Defaults to 1. If there are multiple A files, then this will be updates by __split_reads
        self.A_file_count = 1 

        if processors is not None and  processors <= 1:
            processors = 1
            self.__parallel_mode = False

        if read_file: self.read(read_file, parallel_read=self.__parallel_mode, processors=processors, a_files=a_files)

        # This ensures that all connections are closed, when program end/ ctrl+c or ctrl+d are used
        # in the interactive console. Previously the closing of these connections was handled by
        # The __del__, which is not guaranteed to be called and could cause the reader to hang
        # when parallel_mode was set to True
        atexit.register(self.closeAllConnections)

    
    '''
    Close down all connections
    '''
    def closeAllConnections(self):
        for conn in self.__parent_conns:
            conn, i = conn
            conn.send("End")
            conn.close()
        self.__parent_conns = []
    
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
        if len(self.__milis) > 1: return [m.getStateMaps() for m in self.__milis]
        return self.__state_maps
    
    def getProcessorCount(self):
        return self.A_file_count
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
    Getter for Nodes
    '''
    def getNodes(self):
        if self.__parallel_mode: return self.__getHelper("get nodes")
        if len(self.__milis) > 1: return [m.getNodes() for m in self.__milis]
        return self.__nodes
    '''
    Getter for the header info.  Only need from the zero processor
    '''
    def getHeader(self):
        if self.__parallel_mode: return self.__getHelper("get_header")
        if len(self.__milis) >1: return self.__milis[0].getHeader()
        return self.__header
    '''
    Get error file
    '''
    def getErrorFile(self):
        return self.__error_file
        
    
    def __getHelper(self, message):
        mili_conns = []
        ret = {}
        for mili_index in range(len(self.__parent_conns)):
            mili_conn, i = self.__parent_conns[mili_index]
            mili_conn.send([message])
            mili_conns.append([mili_conn,i])
        
        while len(list(ret.keys())) < len(mili_conns):
            for mili_conn in mili_conns:
                mili_conn, i = mili_conn
                if mili_conn.poll():
                    query_response = mili_conn.recv()
                    ret[i] = query_response
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
    Reads all the string data for the mili file and associates the strings
    with the appropriate directory structures.
    '''
    def __readNames(self,f,offset):
        offset -= self.__null_termed_names_bytes
        f.seek(offset, os.SEEK_END)
        fmt = str(self.__null_termed_names_bytes) + 's'
        byte_array = struct.unpack(self.__tag + fmt, f.read(self.__null_termed_names_bytes))[0]
        
        if (sys.version_info > (3, 0)):
            strings = str(byte_array)[2:].split('\\x00')  # only works for Python3
        else:
            strings = byte_array.split(b'\x00')

        for _, directories in self.__directories.items():
            for directory in directories:
                string_offset = directory.string_offset_idx
                string_count = directory.string_qty_idx
                directory_strings = [ strings[sidx] for sidx in range( string_offset, string_offset + string_count) ]
                self.__names.extend( directory_strings  )
                directory.addStrings( directory_strings )

    '''
    Reads the state variables and parameters from the Mili file.
    '''
    def __readParams(self, f, offset):
        type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}

        file_number = 0

        can_proc = [ DirectoryType.MILI_PARAM.value, DirectoryType.APPLICATION_PARAM.value, DirectoryType.TI_PARAM.value ]
        for type_id in can_proc:
            for directory in self.__directories.get(type_id, []):

                name = directory.strings[0]

                self.__params[name] = file_number, directory
                f.seek(directory.offset_idx)
                byte_array = f.read(directory.length_idx)
                param_type = directory.modifier_idx1
                # num_entries = directory.modifier_idx2

                type_rep = type_to_str[DataType(param_type).name]
                type_value = ExtSize[DataType(param_type).name].value
        
                if type_to_str[DataType(param_type).name] == 's':
                    if (sys.version_info > (3, 0)):
                        self.__params[name] = [str(struct.unpack(self.__tag + str(int(directory.length_idx / type_value)) + type_rep, byte_array)[0])[2:].split('\\x00')[0], directory]
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
        
                    if 'node' not in self.__labels:
                        self.__labels['node'] = {}

                    for j in range (first - 1, last):
                        self.__labels['node'][node_labels[j]] = j + 1
                        self.__labeltomili['node'][node_labels[j]].append(self.__mili_num)
                    

                if 'Element Label' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    ints = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)
                    first, _, ints = ints[0], ints[1], ints[2:]

                    sup_class_idx = name.index('Scls-') + len('Scls-')
                    sup_class_end_idx = name.index('/', sup_class_idx, len(name))
                    class_idx = name.index('Sname-') + len('Sname-')
                    class_end_idx = name.index('/', class_idx, len(name))

                    sup_class = name[sup_class_idx : sup_class_end_idx]
                    clas = name[class_idx : class_end_idx]

                    if (sup_class, clas) not in self.__labels:
                        self.__labels[clas] = {}

                    for j in range(len(ints)):
                        if 'ElemIds' in name:
                            self.__labels[clas][self.__label_keys[j]] = ints[j]
                            self.__labeltomili[clas][self.__label_keys[j]].append(self.__mili_num)
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

                if 'IntLabel_es_' in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    i_points = struct.unpack(self.__tag + str(int(directory.length_idx / 4)) + 'i', byte_array)
                    first, _, i_points, num_i_ponts = i_points[0], i_points[1], i_points[2:len(i_points) - 1], i_points[len(i_points) - 1]
                    index = name.find('es_')
                    self.__int_points[name[index:]] = [i_points, num_i_ponts]
                    # determine stress or strain

    def __readStateVariables(self,f,offset_idx):

        for directory in self.__directories.get(DirectoryType.STATE_VAR_DICT.value, []):
            
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

                if sv_name[:-1] in self.__int_points:
                #if (sv_name == 'es_1a' or sv_name == 'es_3a' or sv_name == 'es_3c') and sv_name[:-1] in self.__int_points:
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

                    for sv in sv_names:
                        if not sv in self.__int_points:
                            self.__int_points[sv] = {}
                        self.__int_points[sv][sv_name] = self.__int_points[sv_name[:-1]]

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
            if not type_idx in self.__directories.keys( ):
                self.__directories[type_idx] = []
            self.__directories[type_idx].append(Directory(type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx, number_of_strings))
            number_of_strings += string_qty_idx
        return offset

    '''
    Reads in information such as nodes and element connections from the Mili fiel
    '''
    def __readMesh(self, f):
        can_proc = [ DirectoryType.CLASS_DEF.value , 
                     DirectoryType.CLASS_IDENTS.value, 
                     DirectoryType.NODES.value, 
                     DirectoryType.ELEM_CONNS.value ]

        for type_id in can_proc:
            for directory in self.__directories.get(type_id, []):

                if directory.type_idx == DirectoryType.CLASS_DEF.value:
                    superclass = directory.modifier_idx2
                    short_name = directory.strings[0]
                    long_name = directory.strings[1]
                    mocd = MeshObjectClassData(short_name, long_name, superclass)
                    self.__mesh_object_class_datas[short_name] = mocd

                if directory.type_idx == DirectoryType.CLASS_IDENTS.value:
                    f.seek(directory.offset_idx)
                    short_name = directory.strings[0]
                    superclass, start, stop = struct.unpack('3i', f.read(12))
                    self.__mesh_object_class_datas[short_name].add_block(start, stop)
                    superclass = Superclass(superclass).name
                    if (superclass, short_name) not in self.__labels:
                        self.__labels[short_name] = {}
                    for label in range(start, stop + 1):
                        self.__labels[short_name][label] = label
                        self.__labeltomili[short_name][label].append(self.__mili_num)

                if directory.type_idx == DirectoryType.NODES.value:
                    f.seek(directory.offset_idx)
                    short_name = directory.strings[0]
                    start, stop = struct.unpack('2i', f.read(8))
                    num_coordinates = self.__dim * (stop - start + 1)
                    floats = struct.unpack(str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                    class_name = short_name
                    sup_class = self.__mesh_object_class_datas[class_name].superclass
                    sup_class = Superclass(sup_class).name

                    for n in range(0, len(floats), self.__dim):
                        self.__nodes.append([floats[n:n + self.__dim]])

                    self.__mesh_object_class_datas[short_name].add_block(start, stop)

                if directory.type_idx == DirectoryType.ELEM_CONNS.value:
                    f.seek(directory.offset_idx)
                    short_name = directory.strings[0]
                    if short_name not in self.__connectivity.keys():
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

                        mo_id = len(self.__connectivity[short_name].keys()) + 1
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

    '''
    Reads in all the subrecords for a Mili file
    '''
    def __readSubrecords(self, f):
        for directory in self.__directories.get(DirectoryType.STATE_REC_DATA.value, []):
            srec_int_data = directory.modifier_idx1 - 4
            srec_c_data = directory.modifier_idx2
            f.seek(directory.offset_idx)
            _, _, _, srec_qty_subrecs = struct.unpack('4i', f.read(16))
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
                    self.__mili_search.add(class_name,sv,sub)
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

        if subrecord.organization == DataOrganization.OBJECT.value:
            # If subrecord is Object ordered there will only be 1 data type
            datatype = None
            atom_qty_per_label = 0
            for sv_name in subrecord.svar_names:
                sv, sub = self.__state_variables[sv_name]
                datatype = DataType(sv.data_type).name
                atom_qty_per_label += sv.atom_qty(self.__state_variables)
    
            subrec_mo_qty = atom_qty_per_label * subrecord.mo_qty
            datatype_str = type_to_str[datatype]
            ret_final = str(subrec_mo_qty) + datatype_str
            ret_final = f"{subrec_mo_qty}{datatype_str}"
        else:
            ret_final = ""
            for sv_name in subrecord.svar_names:
                sv, sub = self.__state_variables[sv_name]
                datatype = DataType(sv.data_type).name
                datatype_str = type_to_str[datatype]
                atom_qty_per_label = sv.atom_qty(self.__state_variables)
                ret_final += f"{(subrecord.mo_qty * atom_qty_per_label)}{datatype_str}"

        return ret_final

    def read(self, file_name, labeltomili=None, mili_num=None, parallel_read=False, processors=None, a_files=None):
        """Reads in Mili plot file.

        This function calls the other reader functions one at a time, spanning the entire Mili file.

        Args:
            file_name (str): The base name of the Mili database.
            labeltomili ():
            mili_num (int):
            processors (int): The number of processors to use if reading in parallel. None means use
                as many as are available.
            a_files (List[int]): A list of the A files to read. This can be used to read a limited
                number of the plot files for an uncombined database. This defaults to None and 
                can only be used for uncombined databases.
        """
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

        attfile_re = re.compile(re.escape(file_name) + "[0-9]*A$")
        parallel = list(filter(attfile_re.match,os.listdir(dir_name)))

        # remove the 'A' at the end of the filename for recursive parallel read
        parallel = [ rootfile[:-1] for rootfile in parallel ]
        
        if len( parallel ) > 1:
            self.__filename = file_name
            self.__split_reads(orig, parallel_read, processors, a_files)
            return
        else:
            if parallel_read:
                self.__error('Reading in serial mode, since there are less than 2 mili files')
                parallel_read = False
                self.__parallel_mode = False

            sfile_re = re.compile(re.escape(file_name) + "[0-9]*[^A]$")

            state_files = list(filter(sfile_re.match,os.listdir(dir_name)))
            state_files = [ dir_name + os.sep + fname for fname in state_files ]
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
            self.__header = f.read(16)
            mili_taur = struct.unpack('4s', self.__header[:4])[0].decode('ascii')
            assert(mili_taur == 'mili' or mili_taur == 'taur')
            self.__header_version, self.__directory_version, self.__endian_flag, self.__precision_flag, self.__state_file_suffix_length, self.__partition_flag = \
               struct.unpack('6b', self.__header[4:10])

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
                self.__readNames(f, offset)
                self.__readParams(f, offset)
                self.__readStateVariables(f, offset)

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
        if raw_data or not res:
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
        elif name in self.__int_points:
            elem_sets = self.__int_points[name]
        else:
            return False

        set_names = elem_sets.keys()
        for set_name in set_names:
            temp_sv, temp_subrecords = self.__state_variables[set_name]
            if temp_subrecords != []:
                temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
                if temp_subrecord.class_name == class_name:
                    return elem_sets[set_name][0]

        return False

    '''
    Given a single state variable name, a number of states, and a number of label(s) return the requested value(s)
    '''
    def __variable_at_state(self, state, subrecord, subrec_label_indexes, name, variables, sub, res, modify=False, int_points=False):
        indices = {}
        temp_res = { state : { name : {} } }
        
        indices[sub] = defaultdict(list)

        sv_names = []
        sv_group_start = {}
        sv_group_start[0] = 0
        sv_group_len = {}
        group_idx = 0
        
        if int_points:
            for sv in subrecord.svar_names:
                sv_var = self.__state_variables[sv][0]
                sv_group_len[group_idx] = max(1, len(sv_var.svars))
                if group_idx: sv_group_start[group_idx] = sv_group_start[group_idx - 1] + sv_group_len[group_idx-1]
                if len(sv_var.svars) > 0:
                    sv_names += [ sv_name for sv_name in sv_var.svars ]
                else:
                    sv_names.append(sv)
                group_idx += 1
            
        is_vector_type = name in self.__state_variables \
                         and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR'
        is_object_ordered = subrecord.organization == DataOrganization.OBJECT.value
        
        for label, indexes in subrec_label_indexes.items():
            if int_points:
                
                indices[sub][label] = {}
                
                if label not in temp_res[state][name]: temp_res[state][name][label] = {}
                v_index = 0
                
                for index in indexes.keys():
                    
                    indices[sub][label][sv_names[index]] = {}
                    sv_name = sv_names[index]
                    temp_res[state][name][label][sv_name] = {}
                    for int_point in int_points:
                        
                        temp_res[state][name][label][sv_name][int_point] = variables[indexes[index][int_point]]
                        if modify:
                            indices[sub][label][sv_names[index]][int_point] = indexes[index][int_point]
                    v_index += 1
            elif is_vector_type:
                temp_res[state][name][label] = [ variables[index] for index in indexes ]
                if modify:
                    indices[sub][label] += indexes
            else:
                temp_res[state][name][label] = variables[indexes[0]]
                if modify:
                    indices[sub][label].append(indexes[0])

        # the nested form of the dicts needs to be identical iirc
        self.__recurMergeDicts(res,temp_res)

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
        labels = self.__labels[class_name]

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
    Merge two dictionaries with nested dictionaries together
    '''
    def __recurMergeDicts(self, dct, merge_dct):
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):
                self.__recurMergeDicts(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]

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
        
    
    def __split_reads(self, file_name, parallel_read, processors=None, a_files=None):
        """Process and read uncombined plot files.

        If parallel_read is set to True the multiple processes are created and the plot files are
        split between them and read. If parallel_read is set to False, then the plot files are
        processed and read serially.

        Args:
            file_name (str): The base file name for the plot database.
            parallel_read (bool): Flag to turn on/off reading in parallel.
            processors (int): The number of processors to be used when reading in parallel. Defaults
                to None, meaning it uses all available processors.
            a_files (List[int]): List of A file processor suffixes to be read in. This defaults to 
                None, meaning all A files are processed and read. 
        """
        # Handle case of multiple state files
        is_first = True
        end_dir = file_name.rfind(os.sep)
        # dir_name = os.getcwd()
        dir_name = os.getcwd()
        if end_dir != -1:
            dir_name = file_name[:end_dir]
            file_name = file_name[end_dir + 1:]
        
        attfile_re = re.compile(re.escape(file_name) + "[0-9]*A$")
        parallel = list(filter(attfile_re.match,os.listdir(dir_name)))

        # Only read specified files if requested.
        if a_files is not None:
            new_parallel = []
            for a_file_suffix in a_files:
                a_file_suffix_re = f"{re.escape(file_name)}[0]*{str(a_file_suffix)}A$"
                for pfile in parallel:
                    if re.match(a_file_suffix_re, pfile):
                        new_parallel.append(pfile)
            parallel = new_parallel

        # strip the A off the end
        parallel = [ f[:-1] for f in parallel ] 

        # Record number of a files
        self.A_file_count = len(parallel)
        
        # Add directory name before each file
        parallel = [ dir_name + os.sep + p for p in parallel ]

        # Sort in ascending order by processor number.
        parallel.sort() 

        cpus = psutil.cpu_count(logical=False)
        if processors is None:
            # If processors not set, limit to number of cpus
            processors = cpus
        if processors > cpus:
            # If more processors requested than cpus, limit to cpus
            processors = cpus 
        if processors > len(parallel):
            # If more processors requested than "A" files, limit to number of "A" files
            processors = len(parallel)

        self.__parent_conns = []
        if len(parallel) > 1:
            self.__filename = file_name
            
            if parallel_read:
                # Split files into even groups between processors
                parallel_count = len(parallel)
                parallel_groups = [ [] for i in range(processors) ]
                cur = 0
                i = 0
                while cur < parallel_count:
                    parallel_groups[i].append( parallel[cur] ) 
                    cur += 1
                    i += 1
                    if i == processors:
                        i = 0

                for mili_num, group in enumerate(parallel_groups):
                    mili = Mili()
                    self.__milis.append(mili)
                    parent_conn, child_conn = mp.Pipe()
                    pr = mp.Process(target=self.__child_read, args=(mili, child_conn, group, mili_num,))
                    self.__parent_conns.append([parent_conn, mili_num])
                    pr.start()
            else:
                i = 0
                for p in parallel:
                    mili = Mili()
                    self.__milis.append(mili)
                    labeltomili, mesh_objects = mili.read(p, mili.__labeltomili, i)
                    if(is_first):
                        is_first=False
                        self.__number_of_state_maps = mili.__number_of_state_maps
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
                        child_number, mesh_objects, labels, number_of_state_maps = conn.recv()
                        if self.__number_of_state_maps is None:
                            self.__number_of_state_maps = number_of_state_maps
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
    def __child_read(self, mili, conn, file_names, i):
        # Read the file

        if len(file_names) == 1:
            file_name = file_names[0]
            mili.__labeltomili, mesh_objects = mili.read(file_name, mili.__labeltomili, i)
            all_labels = mili.__labels
        else:
            j = 0
            is_first = True
            all_labels = {}
            for file_name in file_names:
                nested_mili = Mili()
                mili.__milis.append(nested_mili)
                nested_mili.__labeltomili, mesh_objects = nested_mili.read(file_name, nested_mili.__labeltomili, j)
                if is_first:
                    is_first = False
                    mili.__number_of_state_maps = nested_mili.__number_of_state_maps 
                for mesh_obj in mesh_objects:
                    if mesh_obj not in mili.__mesh_object_class_datas:
                        mili.__mesh_object_class_datas[mesh_obj] = mesh_objects[mesh_obj]

                for class_name in nested_mili.__labels:
                    labmap = nested_mili.__labels[class_name]

                    if class_name in all_labels:
                        all_labels[class_name].update(nested_mili.__labels[class_name])
                    else:
                        all_labels[class_name] = labmap

                    for label_key in labmap:
                        mili.__labeltomili[class_name][label_key].append(j)
                        
                j += 1
                
        # Send back mesh information and labels
        conn.send([i, mili.__mesh_object_class_datas, all_labels, mili.__number_of_state_maps])
        
        # ## Wait for querys
        while True:
            # sleep for 5 nanoseconds to reduce contention, this speeds us up by like 3x (just the read goes from 1.7 -> 0.6s on my system)
            if conn.poll(5/1000000):
                query = conn.recv()

                if query == "End":
                    conn.close()
                    return
                if query[0] == "Error file":
                    mili.__error_file = query[1]
                    if len(self.__milis):
                        for m in mili.__milis:
                            m.__error_file = query[1]
                elif query[0] == "labels of material":
                    material = query[1]
                    conn.send(mili.labels_of_material(material))
                elif query[0] == "nodes of material":
                    material = query[1]
                    conn.send(mili.nodes_of_material(material))
                elif query[0] == "nodes of elem":
                    label, class_name, raw_data, global_ids = query[1]
                    conn.send(mili.nodes_of_elem(label, class_name, raw_data, global_ids))
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
                elif query[0] == "get nodes":
                    nodes = mili.getNodes()
                    conn.send(nodes)
                elif query[0] == "get header":
                    header = mili.__header
                    conn.send(header)
                else:
                    sv, class_name, material, label, state_numbers, modify, int_points, raw, answ, use_exact_int_point = query

                    if state_numbers is None:
                        state_numbers = [] 

                    answer = mili.query(sv, class_name, material, label, state_numbers, modify, int_points, raw, answ, use_exact_int_point)

                    if not answer:
                        conn.send("Fail")
                    else:
                        state_number_zero = state_numbers if not isinstance(state_numbers, list) else state_numbers[0]

                        if answer and sv in answer[state_number_zero]:
                            send_answer = self.__create_answer(answer, sv, material, label, class_name, state_numbers, modify, raw)
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
    def query(self, names, class_name, material=None, labels=None, state_numbers=None, modify=False, int_points=False, raw_data=True, res=None, use_exact_int_point=False):
        # default args are instantiated at function definition, not when called, this makes mutable types cache modifications between calls
        res = res if res is not None else defaultdict(dict)
        processed_integration_point = False
        # Parse Arguments
        if state_numbers is None:
            state_numbers = [i  for i in range(1, self.__number_of_state_maps +1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        if type(labels) is int:
            labels = [labels]

        if material:
            if len(self.__milis):
                labels = self.labels_of_material(material)
                if not labels or class_name not in labels:
                    return self.__error('There are no elements from class ' + str(class_name) + ' of material ' + str(material))
                labels = list(labels[class_name])
            else:
                labels = self.__material_to_labels(material, class_name, labels)

            if not labels or not len(labels):
                return self.__error('There are no elements from class ' + str(class_name) + ' of material ' + str(material))
        
        if labels and type(labels) is not list:
            return self.__error('labels must of a list of ints or an int')

        if class_name not in self.__mesh_object_class_datas:
            return self.__error('invalid class name')
        
        if not labels and not self.__parallel_mode and not len(self.__milis):
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name
            labels = list(self.__labels[class_name].keys())
        
        if not labels and not self.__parallel_mode and len(self.__milis):
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name
            labels = list(self.__labeltomili[(sup_class, class_name)].keys())

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

        # Create local list of int points
        selected_int_points = False
        if int_points:
            selected_int_points = [ip for ip in int_points]
        # Deal with parallel Mili file case
        if len(self.__milis):
            answ = defaultdict(dict)
            for sv in names:
                vector, variables = self.__parse_name(sv)
                if not vector:
                    sv_key = variables
                else:
                    sv_key = vector

                # turn sv and class into elem
                sup_class = self.__mesh_object_class_datas[class_name].superclass
                sup_class = Superclass(sup_class).name
                
                failcount = 0
                milis = set()
                mili_to_labels = defaultdict(list)
                if labels != None:
                    for label in labels:
                        mili = self.__labeltomili[class_name][label]
                        for m in mili:
                            mili_to_labels[m].append(label)
                            milis.add(m)
                    milis = list(milis)
                if not labels: milis = set([i for i in range(len(self.__milis))])
                if self.__parallel_mode:
                    mili_conns = []
                    for mili_index in milis:
                        mili_conn, i = self.__parent_conns[mili_index]
                        mili_conn.send([sv, class_name, material, mili_to_labels[mili_index], state_numbers, modify, int_points, True, answ, use_exact_int_point])
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
                        resp = self.__milis[mili_index].query(sv, class_name, material, mili_to_labels[mili_index], state_numbers, modify, int_points, True, answ, use_exact_int_point)
                        # answ is modified in-place, shouldn't need to check/append if something isn't found, should avoid in-place modifications if something isn't found (which is currently happening)
                if answ is not None and len(answ) == 0:
                    answ = None

            return self.__create_answer(answ, names, material, labels, class_name, state_numbers, modify, raw_data)

        for name in names:
            # Handle case of vector[component]
            vector, variables = self.__parse_name(name)
            if not vector: vector = variables

            # Get the State Variable object and Subrecord numbers for each name
            if self.__is_vec_array(vector, class_name):
                if 'stress' in name: elem_sets = self.__int_points['stress']
                elif 'strain' in name: elem_sets = self.__int_points['strain']
                elif vector in self.__int_points: elem_sets = self.__int_points[vector]

                subrecords = []
                for set_name in elem_sets.keys():
                    temp_sv, temp_subrecords = self.__state_variables[set_name]
                    subrecords += temp_subrecords
                #sv, subrecords = self.__state_variables[vector]
                if(not processed_integration_point):
                    if not selected_int_points:
                        selected_int_points = list(self.__is_vec_array(vector, class_name))
                    else:
                        possible_int_points = self.__is_vec_array(vector, class_name)
                        for i in range(len(selected_int_points)):
                            ip = selected_int_points[i]
                            if ip not in possible_int_points:
                                if use_exact_int_point is True:
                                    return None
                                old_ip = ip
                                _, ip = min(enumerate(possible_int_points), key=lambda x: abs(x[1] - ip))
                                self.__error(str(old_ip) + ' is not an integration point, but the closest is ' + str(ip))
                                selected_int_points[i] = ip
                    selected_int_points.append(len(self.__is_vec_array(vector, class_name)))
                processed_integration_point = True
            elif vector: 
                if vector not in self.__state_variables:
                    return self.__error('There is no variable ' + vector)
                sv, subrecords = self.__state_variables[vector]
            else:
                if name not in self.__state_variables:
                    return self.__error('There is no variable ' + name)
                sv, subrecords = self.__state_variables[name]

            # Get only subrecords/subrec num that match the requested class name
            subrecords = [(self.__srec_container.subrecs[s], s) for s in subrecords if self.__srec_container.subrecs[s].class_name == class_name]

            # get Super class name for requested class
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name

            # Determine the mo_ids we are interested in for these subrecords
            labels_we_have = list( set(labels) & set(self.__labels[class_name]) )
            
            if len(labels_we_have) == 0:
                return None
            
            _, child_variables = self.__parse_name(name)
            child_variables = [child_variables]
            if name in self.__state_variables and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR':
                child_variables = self.__state_variables[name][0].svars

            if selected_int_points:
                selected_int_points, num_int_points = selected_int_points[:-1], selected_int_points[-1:][0]
                vector, variables = self.__parse_name(name)
                if not vector: vector = variables
                possible_int_points = self.__is_vec_array(vector, class_name)
                if possible_int_points is not False:
                    possible_int_points = list(possible_int_points)
                else:
                    # If there are no possible int_points set selected_int_points to False to
                    # avoid errors
                    selected_int_points = False

            indices = {}
            srec_label_indexes = {} # { subrecord name --> { label : indexes } }

            sup_class_labels = self.__labels[class_name]
            # For each subrecord, determine which elements (labels) appear in that subrecord
            # and create a dictionary entry for each subrecord that contains the labels in that
            # subrecord and the indexes at which the data for that label appears in the subrecord
            for subrecord, sub in subrecords:
                mo_idx = 0
                found_mo_ids = []
                append_mo_ids = found_mo_ids.append
                for block_range in subrecord.mo_blocks:
                    start, end = block_range
                    for label in labels_we_have:
                        target_mo = sup_class_labels[label]
                        if target_mo >= start and target_mo <= end:
                            append_mo_ids([label, mo_idx + target_mo - start])
                    mo_idx += (end - start) + 1

                if len(subrecord.mo_blocks) == 0:
                    # Handle case for Global values, assume label 1
                    found_mo_ids = [[1,0]]

                srec_label_indexes[subrecord.name] = {}
                #
                sv_names = []
                sv_group_start = {}
                sv_group_start[0] = 0
                sv_group_len = {}
                group_idx = 0
                
                for sv in subrecord.svar_names:
                    sv_var = self.__state_variables[sv][0]
                    sv_group_len[group_idx] = max(1, len(sv_var.svars))
                    if group_idx: sv_group_start[group_idx] = sv_group_start[group_idx - 1] + sv_group_len[group_idx-1]
                    if len(sv_var.svars) > 0:
                        for sv_name in sv_var.svars:
                            sv_names.append(sv_name)
                    else:
                        sv_names.append(sv)
                    group_idx += 1
                    
                var_indexes = []
                for child in child_variables:
                    if child not in sv_names:
                        return self.__error(child + ' not a valid variable name')
                    for sv_group in subrecord.svar_names:
                        if sv_group == child: 
                            var_indexes.append([subrecord.svar_names.index(sv_group), 0])
                        sv = self.__state_variables[sv_group][0].svars
                        if child in sv:
                            var_indexes.append([subrecord.svar_names.index(sv_group), sv.index(child)])

                # Simplify some of the booleans used in looping below. Performing these checks for every
                # mo_index can add significant time in larger queries
                is_vector_type = name in self.__state_variables \
                                 and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR'
                is_object_ordered = subrecord.organization == DataOrganization.OBJECT.value
                len_sv_names = len(sv_names)
                #
                for mo_index in found_mo_ids:
                    label, mo_index = mo_index
                    if selected_int_points: indexes = {}
                    else: indexes = []
                    
                    for var_index in var_indexes:
                        var_index, var_in_group = var_index
                        if selected_int_points: var_index = var_in_group
                        if selected_int_points and var_index not in indexes: indexes[var_index] = {}
                        if selected_int_points:
                            
                            offset = mo_index * len_sv_names * num_int_points
                            
                            for int_point in selected_int_points:
                                index = offset + var_index + (len_sv_names * possible_int_points.index(int_point))
                                indexes[var_index][int_point] = index
                        else:
                            if is_object_ordered:
                                indexes.append(mo_index * len_sv_names + sv_group_start[var_index] + var_in_group)
                            else:
                                indexes.append(sv_group_start[var_index] * subrecord.mo_qty + sv_group_len[var_index] * mo_index + var_in_group)
                    srec_label_indexes[subrecord.name][label] = indexes

            # Get only subrecords that contain elements that are being queried
            subrecords = [ [srec, num] for srec,num in subrecords if srec_label_indexes[srec.name] != {} ]

            if subrecords == []:
                return None

            # Determine which states (of those requested) appear in each of the state files.
            # This way we can open each file only once and process all the states that appear in it
            # rather than opening a state file for each iteration
            state_file_dict = {}
            for state in state_numbers:
                if state < 1 or state > len(self.__state_maps):
                    return self.__error('There is no state ' + str(state))
                state_map = self.__state_maps[state-1]
                fname = self.__state_map_filename[state_map.file_number]
                try:
                    state_file_dict[fname].append(state)
                except KeyError:
                    state_file_dict[fname] = [ state ]
            
            # Before reading and querying results, generate string representations of each subrecord
            # and pre compile structs for unpacking the data
            subrec_unpack_funcs = {}
            for subrecord, sub in subrecords:
                subrec_name = subrecord.name
                str_repr = self.__tag + self.__set_string(subrecord)
                subrec_unpack_func = struct.Struct(str_repr).unpack
                subrec_unpack_funcs[subrec_name] = subrec_unpack_func

            for state_file_name, state_nums in state_file_dict.items():
                with open(state_file_name, 'rb') as state_file:
                    # Loop over all states that appear in this state file
                    for state in state_nums:
                        state_offset = self.__state_maps[state-1].file_offset+8

                        for subrecord, sub in subrecords:
                            state_file.seek(state_offset + subrecord.offset)
                            byte_array = state_file.read(subrecord.size)
                            #### Offset into byte array and parse out the values we want
                            var_data = subrec_unpack_funcs[subrecord.name](byte_array)

                            if modify:
                                return self.__variable_at_state(state, subrecord, srec_label_indexes[subrecord.name], name, var_data, sub, res, modify, selected_int_points)
                            else:
                                res = self.__variable_at_state(state, subrecord, srec_label_indexes[subrecord.name], name, var_data, sub, res, modify, selected_int_points)

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
                    label, class_name, raw_data, global_ids = data_send
                    val = serial_function(m, label, class_name, raw_data, global_ids)
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

            if len(labels) == 0:
                return self.__error("No labes of material '" + str(material) + "' found in database files" )

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
            if len(nodes) == 0: 
                return self.__error("No labels with material '" + str(material) + "' found in database files.")
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
    def nodes_of_elem(self, label, class_name, raw_data=True, global_ids=False):
        labels = None
        if len(self.__milis) > 1:
            labels = self.__get_children_info(None, None, "nodes of elem", [label, class_name, True, global_ids], Mili.nodes_of_elem)
            if not labels: 
                return self.__error("Class name '" + str(class_name) + "' or label '" + str(label) + "' not found in datamaste files.")
        else:
            if class_name not in self.__mesh_object_class_datas:
                return self.__error('Class name ' + class_name + ' not found')
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name

            if label not in self.__labels[class_name]:
                return self.__error('label ' + str(label) + ' not found')
            mo_id = self.__labels[ class_name][label]

            if class_name not in self.__connectivity:
                return self.__error('Class name ' + class_name + ' has no connectivity')
            labels = self.__connectivity[class_name][mo_id]

            if labels is not None and global_ids is True:
                # Convert local ids to globals ids
                global_labels = []
                labels_list = list(self.__labels["node"].values())
                keys_list = list(self.__labels["node"].keys())
                for l in labels:
                    pos = labels_list.index(l)
                    gid = keys_list[pos]
                    global_labels.append(gid)
                labels = global_labels

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
            state_numbers = [i for i in range(1, self.__number_of_state_maps+1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        if type(labels) is int:
            labels = [labels]
        if type(state_variable) is not str:
            return self.__error('state variable must be a string')

        for state in state_numbers:
            if state < 1 or state > len(self.__state_maps):
                return self.__error('There is no state ' + str(state))

            state_map = self.__state_maps[state-1]

            with open(self.__state_map_filename[state_map.file_number], 'rb+') as f:
                state_subrecord_start_offset = state_map.file_offset+8
                
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
                    # Read in subrecord
                    subrecord = self.__srec_container.subrecs[sub]
                    f.seek(state_subrecord_start_offset+subrecord.offset)
                    data = f.read( subrecord.size )
                    s = self.__set_string(subrecord)
                    var_data = list( struct.unpack(self.__tag + s, data) )

                    # Update value in subrecord
                    if int_points:
                        # indices has form: indices[sub][label][sv_name][int_point] = position in subrecord
                        vector, component = self.__parse_name(name)
                        if vector: n = vector
                        else: n = name

                        for label in labels:
                            for sv_name in indices[sub][label]:
                                int_point_to_index = indices[sub][label][sv_name]
                                for ip in int_points:
                                    var_data[int_point_to_index[ip]] = value[state][n][label][sv_name][ip]
                                        
                    else:
                        # indices has form: indices[sub][label] = list of positions in subrecord
                        for label in labels:
                            if type(value[state][name][label]) is not list:
                                value[state][name][label] = [ value[state][name][label] ]
                            for idx in range( len(indices[sub][label]) ):
                                var_data[indices[sub][label][idx]] = value[state][name][label][idx]
                    
                    # Write out updated subrecord
                    byte_array = struct.pack(self.__tag + s, *var_data)
                    f.seek(state_subrecord_start_offset + subrecord.offset)
                    f.write(byte_array)

                if 'post_modified' in self.__params:
                    file_number, directory = self.__params['post_modified']
                    f.seek(directory.offset_idx)
                    one = struct.pack(self.__tag + 'i', 1)
                    self.__params['post_modified'][0] = 1
                    f.write(one)

    def print_search(self):
        print(self.__mili_search)
    
    def search(self,mili_class_name,variable_name,item):
        return self.__mili_search.find(mili_class_name, variable_name, item)
        

'''
This function is an example of how a user could use the Mili reader
'''
def main():
    # You can run code here as well if you copy the library !
    # This is the definition of the query function
    """def query(self, variable_names, 
                       class_name, 
                       material=None, 
                       labels=None, 
                       state_numbers=None, 
                       modify=False, 
                       int_points=False, 
                       raw_data=True, 
                       res=None,
                       use_exact_int_point=False):
    """
    if len(sys.argv) > 1:
        
        params = parseParameters(sys.argv[1:]) 
        if params is None:
            sys.exit(1)
        if(params["combine"]):
            combiner = MiliCombiner(params["input_file"],params)
        else:
            f = 'parallel/d3samp6.plt'
            mili = Mili()
            mili.read(f)
            answer = mili.query(['nodpos'], class_name='node', material=None, labels=[1])
            #print(answer)
        
    f = 'parallel/d3samp6.plt'
    #f='/p/lustre1/depiero/2020_08_19_073746_4370645/'
    #f = "nickolai/ingrido_f.dyna.plt"
    #mili = Mili()
    #mili.read(f, parallel_read=True)
    #mili.read(f, parallel_read=True)
    
    #mili.read(f)
    #mili.print_search()
    #result = mili.search("node", "pen27s", 169)
    #print (result.name)
    #d = mili.getParams()
    # Leaving these as they giz a variety of examples
    #mat_id ='2'
    #label = 5
    
    #answer = mili.query(['nodpos'], class_name='node', material=None, labels=[1])
    #answer = mili.query('stress[sy]', 'brick', None, [5], [70], False, [2], False)
    #answer = mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
    #answer = mili.query(['matcgx'], 'mat', None, [1,2], [4], raw_data=False)
    #print (answer)   
    #print(mili.query('stress[eps]', 'beam',None, [5],None,raw_data=False))
    
    #print(answer)
    # mili.setErrorFile()
        
if __name__ == '__main__':
        main()
