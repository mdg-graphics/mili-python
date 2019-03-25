import struct
import os
from enum import Enum 
import time

"""
This is the 'meta' information for a statemap, but not the statemap itself.
"""
class StateMap:
    def __init__(self, file_number, file_offset, time, state_map_id):
        self.file_number = file_number
        self.file_offset = file_offset
        self.time = time
        self.state_map_id = state_map_id

"""
Directories dictate where information can be found in the .pltA file.
"""        
class Directory:
    def __init__(self, type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx):
        self.type_idx = type_idx
        self.modifier_idx1 = modifier_idx1
        self.modifier_idx2 = modifier_idx2
        self.string_qty_idx = string_qty_idx
        self.offset_idx = offset_idx
        self.length_idx = length_idx

"""
A state variable is the information being stored in subrecords throughout time
at different states.
"""
class StateVariable:
        
    """
    Initiliazer for a state variable.
    """
    def __init__(self, name, title, agg_type, data_type):
        self.name = name
        self.title = title
        self.agg_type = agg_type
        self.data_type = data_type
        self.list_size = 0
        self.order = 0
        self.dims = []
        self.svars = [] # list of string names included in this if it is vector
    
    """
    Returns the quantity of atoms based on the aggregate type.
    """
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

"""
A subrecord contains a number of state varaibles organized in a certain order.
"""
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
        self.size = 0 # set this mo_qty * size of variables 

"""
Each Mili object has a subrecord container holding all the subrecords.
"""
class SubrecordContainer:
    subrecs = []
    size = 0
    
"""
The mesh information needed is stored in this data type.
"""
class MeshObjectClassData:
        
    """
    Initializer for necccessary mesh information.
    """
    def __init__(self, short_name, long_name, superclass):
        self.short_name = short_name
        self.long_name = long_name
        self.superclass = superclass
        self.blocklist = BlockList(0, 0, [])
        
    """
    Adding a block to the meshobjectclassdata class
    """
    def add_block(self, start, stop):
        self.blocklist.blocks.append((start, stop))
        self.blocklist.block_qty += 1
        self.blocklist.obj_qty += start - stop + 1
    
"""
Each mesh object contains a blocklist data structure.
The blocklist has information about which objects are 
part of the mesh object.
"""
class BlockList:
    def __init__(self, obj_qty, block_qty, blocks):
        self.obj_qty = obj_qty
        self.block_qty = block_qty
        self.blocks = blocks # array of tuples with start, stop

"""
An attribute has a name and a value
e.g. name = "mo_id" value = "5"
Always print is used in case the value is 0 for displaying in repr
"""
class Attribute:
    def __init__(self, name, value, always_print=False):
        self.name = name
        self.value = value
        self.always_print = always_print
    
    def __repr__(self):
        ret = ""
        if self.value or self.always_print:
            return self.name + ": " + str(self.value) + "\n"
        return ret

"""
An item is an object that represents a component of an answer
Any number of its attributes may be None if they aren't filled
"""
class Item:
    def __init__(self, name=None, material=None, mo_id=None, label=None, class_name=None, modify=None, value=None):
        self.name = name
        self.material= material
        self.mo_id= mo_id
        self.label= label
        self.class_name= class_name
        self.modify= modify
        self.value= value
        self.always_print = False # always print the value
    
    def set(self, value):
        self.value = value
        self.always_print = True
    
    def __str__(self):
        attributes = []
        attvalues = [self.name, self.material, self.mo_id, self.label, self.class_name, self.modify, self.value]
        attnames = ["name", "material", "mo_id", "label", "class_name", "modify", "value"]
        
        for i in range(len(attvalues)):
            name, value = attnames[i], attvalues[i]
            if name == "value" and self.always_print: attributes.append(Attribute(name, value, True))
            else: attributes.append(Attribute(name, value))
            
        ret = ""
        for attribute in attributes:
            ret += str(attribute)
        return ret + "\n"

"""
Each state in a query has one StateAnswer
Each contains a list of items and a state number
"""
class StateAnswer:
    def __init__(self):
        self.items = []
        self.state_number=None
    
    def __str__(self):
        ret = "\nstate number: " + str(self.state_number) + "\n"
        for item in self.items:
            ret += str(item)
        return ret

"""
The encompassing object that is returned by a query.
This object contains either:
1. A list of items
2. A list of state answers
If the answer is state based (comes from a query with state_numbers),
the answer contains a list of state answers.
"""
class Answer:
    
    def __init__(self):
        self.state_answers=[]
        self.items = []
    
    def __str__(self):
        ret = ""
        
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
            
"""
Every direcotry has a type that dictates what informaiton
is in the directory.
"""
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
    
"""
The superclass tells what class an object belongs to.
"""
class Superclass(Enum):
    M_UNIT = 0
    M_NODE = 1
    M_TRUSS = 2
    M_BEAM = 3
    M_TRI = 4
    M_QUAD = 5
    M_TET = 6
    M_PYRAMID = 7
    M_WEDGE= 8
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

"""
Connections per superclass
"""
class ConnWords(Enum):
    M_UNIT = 0
    M_NODE = 0
    M_TRUSS = 4
    M_BEAM = 5
    M_TRI = 5
    M_QUAD = 6
    M_TET = 6
    M_PYRAMID = 7
    M_WEDGE= 8
    M_HEX = 10
    M_MAT = 0
    M_MESH = 0
    M_SURFACE = 0
    M_PARTICLE = 3
    M_TET10 = 12
    
"""
The numbering system used by Mili files to state type.
"""  
class DataType(Enum):    
    M_STRING = 1
    M_FLOAT = 2
    M_FLOAT4 = 3
    M_FLOAT8 = 4
    M_INT = 5
    M_INT4 = 6
    M_INT8 = 7
    
"""
The size of each data type in bytes.
"""
class ExtSize(Enum):    
    M_STRING = 1
    M_FLOAT = 4
    M_FLOAT4 = 4
    M_FLOAT8 = 8
    M_INT = 4
    M_INT4 = 4
    M_INT8 = 8
    
"""
The aggregate type describes what a state variable can be.
"""
class AggregateType(Enum):
    SCALAR = 0
    VECTOR = 1
    ARRAY = 2
    VEC_ARRAY = 3
    
"""
The organization of data in a subrecord is either ordered by variable 
or object.
"""
class DataOrganization(Enum):
    RESULT = 0
    OBJECT = 1
    
"""
A Mili object contains data structures to store objects such as directories
and subrecords. It also has a number of querying functions that a user can call
to access this data.
"""
class Mili:
    def __init__(self):
        self.__state_maps = []
        self.__directories = []
        self.__names = []
        self.__params = {} # maps param name: [value]
        self.__state_variables = {} # map to state variable, list of subrecords it is in
        self.__mesh_object_class_datas = {} #shortname to object
        self.__labels = {} # maps from label : elem_id
        self.__label_keys = []
        self.__int_points = {}
        self.__nodes = []
        self.__materials = {} # map from material number to list of element number
        self.__matname = {} # from material name to number
        self.__connectivity = {} # shortname : {mo_id : node}
        self.__dim = None # dimensions from mesh dimensions
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
    
    """
    Getter for params
    """
    def getParams(self):
        return self.__params
    
    """
    Getter for state maps
    """
    def getStateMaps(self):
        return self.__state_maps
    
    """
    Getter for directories
    """
    def getDirectories(self):
        return self.__directories
    
    """
    Getter for state variables
    """
    def getStateVariables(self):
        return self.__state_variables
    
    """
    Getter for labels
    """
    def getLabels(self):
        return self.__labels
    
    """
    Getter for materials
    """
    def getMaterials(self):
        return self.__materials
    
    """
    Reads the data for statemap locations from the Mili file.
    This doesn't read the .pltA file, which contains data for the actual
    state maps
    """
    def readStateMaps(self, f):
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
    
    """
    Reads the state variables and parameters from the Mili file.
    """
    def readStateVariablesAndParams(self, f, offset):
        type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}
        
        offset -= self.__null_termed_names_bytes
        f.seek(offset, os.SEEK_END)
        fmt = str(self.__null_termed_names_bytes) + "s"
        byte_array = struct.unpack(self.__tag + fmt, f.read(self.__null_termed_names_bytes))[0]
        #strings = str(byte_array)[2:].split('\\x00') # only works for Python3
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
                self.__params[name] = file_number, i
                f.seek(directory.offset_idx)
                byte_array = f.read(directory.length_idx)
                type = directory.modifier_idx1
                num_entries = directory.modifier_idx2
                                
                type_rep = type_to_str[DataType(type).name]
                type_value = ExtSize[DataType(type).name].value
                                
                if type_to_str[DataType(type).name] == 's':
                    self.__params[name] = struct.unpack(self.__tag + str(directory.length_idx/type_value) + type_rep, byte_array)[0].split(b'\x00')[0]
                else:
                    self.__params[name] = struct.unpack(self.__tag + str(directory.length_idx/type_value) + type_rep, byte_array)                                  
                
                if name == "mesh dimensions":
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    self.__dim = struct.unpack(self.__tag + str(directory.length_idx/4) + 'i', byte_array)[0]
                
                if "Node Labels" in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    ints = struct.unpack(self.__tag + str(directory.length_idx/4) + 'i', byte_array)
                    first, last, node_labels = ints[0], ints[1], ints[2:]
                    
                    self.__labels[('M_NODE', 'node')] = {}
                    for j in range (first, last):
                        self.__labels[('M_NODE', 'node')][j] = node_labels[j]
                         
                    # do this
                
                if "Element Label" in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    ints = struct.unpack(self.__tag + str(directory.length_idx/4) + 'i', byte_array)                    
                    first, total, ints = ints[0], ints[1], ints[2:]
                    
                    sup_class_idx = name.index("Scls-") + len("Scls-")
                    sup_class_end_idx = name.index("/", sup_class_idx, len(name))
                    class_idx = name.index("Sname-") + len("Sname-")
                    class_end_idx = name.index("/", class_idx, len(name))
                    
                    sup_class = name[sup_class_idx : sup_class_end_idx]
                    clas = name[class_idx : class_end_idx]
                    self.__labels[(sup_class, clas)] = {}
                    
                    for j in range(len(ints)):
                        if "ElemIds" in name:
                                #print label_keys[i]
                            self.__labels[(sup_class, clas)][self.__label_keys[j]] = ints[j] 
                        else:
                            self.__label_keys.append(ints[j])
                        
                    if "ElemIds" in name:
                        self.__label_keys = []
                
                if "MAT_NAME" in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    matname = struct.unpack(str(directory.length_idx) + 's', byte_array)[0].split(b'\x00')[0]
                    num = name[-1:]
                    if matname in self.__matname:
                        self.__matname[matname].append(int(num))
                    else:
                        self.__matname[matname] = [int(num)]
                    
                if "es_" in name:
                    f.seek(directory.offset_idx)
                    byte_array = f.read(directory.length_idx)
                    i_points = struct.unpack(self.__tag + str(directory.length_idx/4) + 'i', byte_array)  
                    first, total, i_points, num_i_ponts = i_points[0], i_points[1], i_points[2:len(i_points)- 1], i_points[len(i_points) - 1]    
                    index = name.find('es_')
                    self.__int_points[name[index:]] = [i_points, num_i_ponts]
                    # determine stress or strain
                    
                                              
            if directory.type_idx == DirectoryType.STATE_VAR_DICT.value:
                f.seek(directory.offset_idx)
                svar_words, svar_bytes = struct.unpack('2i', f.read(8))
                num_ints = (svar_words - 2)
                ints = struct.unpack(str(num_ints) + 'i', f.read(num_ints * 4)) # what is this
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
                    
                    if sv_name == "es_1a" or sv_name == "es_3a" or sv_name == "es_3c":
                        stresscount = straincount = 0
                        stress = strain = []
                        
                        if 'stress' in self.__state_variables: stress = self.__state_variables['stress'][0].svars
                        if 'strain' in self.__state_variables: strain = self.__state_variables['strain'][0].svars
                        
                        for p in sv_names:
                            if p in stress: stresscount += 1
                            if p in strain: straincount += 1
                                                    
                        if stresscount == 6:
                            if not "stress" in self.__int_points: self.__int_points["stress"] = {}
                            self.__int_points["stress"][sv_name] = self.__int_points[sv_name[:-1]]
                        
                        if straincount == 6:
                            if not "strain" in self.__int_points: self.__int_points["strain"] = {}
                            self.__int_points["strain"][sv_name] = self.__int_points[sv_name[:-1]]                                        
                        
    """
    Reads in directory information from the Mili file
    """
    def readDirectories(self, f, offset):
        number_of_strings = 0
        directory_length = 4 * 6
        state_map_length = 20
        int_long = 'i'
        if self.__directory_version > 2:
            directory_length = 8 * 6
            int_long = 'q'
        offset -=  state_map_length * self.__number_of_state_maps + directory_length * self.__number_of_directories # make this not a hard coded 6 eventually
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + self.__number_of_directories):
            byte_array = f.read(directory_length)
            type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx = \
                struct.unpack(self.__tag + '6' + int_long, byte_array)
            self.__directories.append(Directory(type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx))
            number_of_strings += string_qty_idx
        return offset
        
    """
    Reads in information such as nodes and element connections from the Mili fiel 
    """
    def readMesh(self, f):
        name_cnt = 0 
        for i in range(len(self.__directories)):
            directory = self.__directories[i]     
            
                
            if directory.type_idx == DirectoryType.CLASS_DEF.value:
                superclass = directory.modifier_idx2
                short_name = self.__names[name_cnt]
                long_name = self.__names[name_cnt + 1]
                mocd = MeshObjectClassData(short_name, long_name, superclass)
                self.__mesh_object_class_datas[short_name]= mocd
            
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
            
            if directory.type_idx == DirectoryType.NODES.value:
                f.seek(directory.offset_idx)
                short_name = self.__names[name_cnt]
                start, stop = struct.unpack('2i', f.read(8))
                num_coordinates = self.__dim * (stop - start + 1)
                floats = struct.unpack(str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                class_name = self.__names[name_cnt]
                sup_class = self.__mesh_object_class_datas[class_name].superclass
                sup_class = Superclass(sup_class).name
                self.__labels[sup_class, class_name] = {}
                
                for n in range(0, len(floats), self.__dim):
                    self.__nodes.append([floats[n:n+self.__dim]])
                    self.__labels[(sup_class, class_name)][n/self.__dim] = n/self.__dim
                
                self.__mesh_object_class_datas[short_name].add_block(start, stop)
            
            if directory.type_idx == DirectoryType.ELEM_CONNS.value:
                f.seek(directory.offset_idx)
                short_name = self.__names[name_cnt]
                self.__connectivity[short_name] = {}
                superclass, qty_blocks = struct.unpack('2i', f.read(8))
                # print short_name, superclass, qty_blocks
                elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))
                for j in range(0,len(elem_blocks),2):
                    self.__mesh_object_class_datas[short_name].add_block(elem_blocks[j], elem_blocks[j+1])
                    #print elem_blocks[j], elem_blocks[j+1]
               
                elem_qty = directory.modifier_idx2
                word_qty = ConnWords[Superclass(superclass).name].value
                conn_qty = word_qty - 2
                mat_offset = word_qty - 1
                # print elem_blocks, short_name, word_qty
                ebuf = struct.unpack(str(elem_qty * word_qty) + 'i', f.read(elem_qty * word_qty * 4))
#                 print elem_blocks
                index = 0
                
#                 print Superclass(superclass).name, word_qty, qty_blocks
                for j in range(qty_blocks):
                    off = elem_blocks[j * 2] -1
                    elem_qty = elem_blocks[j * 2 + 1] - elem_blocks[j * 2] + 1
                
                    mo_id = 1
                    for k in range(index, len(ebuf), word_qty):
                        self.__connectivity[short_name][mo_id] = []
                        mat = ebuf[k + conn_qty]
                        for m in range(0, conn_qty):
                            node = ebuf[k + m]
                            self.__connectivity[short_name][mo_id].append(node)
                        part = ebuf[k + mat_offset]
                        if mat not in self.__materials:
                            self.__materials[mat] = []
                        self.__materials[mat].append([mo_id, short_name])
                        mo_id += 1
                    index = word_qty * elem_qty
#                 print self.__materials
            name_cnt += directory.string_qty_idx
         
    """
    Reads in all the subrecords for a Mili file
    """     
    def readSubrecords(self, f):
        for i in range(len(self.__directories)):
            directory = self.__directories[i]          
            if directory.type_idx == DirectoryType.STATE_REC_DATA.value:
                srec_int_data = directory.modifier_idx1 - 4
                srec_c_data = directory.modifier_idx2
                f.seek(directory.offset_idx)
                srec_id, srec_parent_mesh_id, srec_size_bytes, srec_qty_subrecs = struct.unpack('4i', f.read(16))
                idata = struct.unpack(str(srec_int_data) + 'i', f.read(srec_int_data * 4))
                cdata = struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].split(b'\x00')
                
                int_pos = 0
                c_pos = 0
                self.__srec_container = SubrecordContainer()
                
                
                for k in range(srec_qty_subrecs):
                    org, qty_svars, qty_id_blks = idata[int_pos], idata[int_pos+1], idata[int_pos+2]
                    # for j in range(qty_id_blks):
                        
                    int_pos += 3
                    name, class_name = cdata[c_pos:c_pos+2]
                    c_pos += 2
                    svars = cdata[c_pos:c_pos+qty_svars]
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
                        sub.mo_qty = 1
                    
                    
                    lump_atoms = []
                    lump_sizes = []
                    lump_offsets = []
                    count = 0
                    sz = 0
                    
                    # Handle Aggregate Types
                    for sv in svars:
                        sv = self.__state_variables[sv][0]
                        for sv_sv in sv.svars:
                            self.__state_variables[sv_sv][1].append(k)
                                        
                    if org == DataOrganization.OBJECT.value:
                        for sv in svars:
                            #print sv
                            state_var = self.__state_variables[sv][0]
                            self.__state_variables[sv][1].append(k)

                            atom_size = ExtSize[DataType(state_var.data_type).name].value
                            atoms = state_var.atom_qty(self.__state_variables) # define this 
                            
                            ### stuff about surface here
    
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
                            ### stuff about surface here
                            total_atoms = atoms
                            
                            lump_atoms.append(sub.mo_qty * total_atoms)
                            lump_sizes.append(sub.mo_qty * total_atoms * ExtSize[DataType(state_var.data_type).name].value)
                        
                        lump_offsets.append(0)
                        for j in range(1, sub.qty_svars):
                            lump_offsets.append(lump_offsets[j-1] + lump_sizes[j-1])
                        
                        j = sub.qty_svars - 1
                        sub.size = lump_offsets[j] + lump_sizes[j]
                    

                    sub.offset = self.__srec_container.size
                    self.__srec_container.size += sub.size
                    self.__srec_container.subrecs.append(sub)
        
    """
    This function returns the string representation of a subrecord, to be used
    when interpreting the bytes during reading of state.
    """
    def set_string(self, subrecord):
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
        
    """
    This function calls the other reader functions one at a time, spanning the entire
    Mili file.
    """
    def read(self, file_name):
        # Handle case of multiple state files
        end_dir = file_name.rfind(os.sep)
        dir_name = os.getcwd()
        if end_dir != -1:
            dir_name += os.sep + file_name[:end_dir]
            file_name = file_name[end_dir + 1:]
        
        state_files = []
        for f in os.listdir(dir_name):
            if file_name + ".plt" in f:
                f_end = f.rfind('.plt') + 4
                if str.isdigit(f[f_end:]):
                    state_files.append(f)
        state_files.sort()
        
        if end_dir != -1:
            for i in range(len(state_files)):
                state_files[i] = dir_name + os.sep + state_files[i]
            file_name = dir_name + os.sep + file_name
        file_name += ".pltA"
        self.__filename = file_name
        self.__state_map_filename = state_files
        
    # Open file with 'b' to specify binary mode
        with open(file_name, 'rb') as f:
            ### Read Header ###
            header = f.read(16)
            mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')
            self.__header_version, self.__directory_version, self.__endian_flag, self.__precision_flag, self.__state_file_suffix_length, self.__partition_flag = \
               struct.unpack('6b', header[4:10])
            
            if self.__endian_flag == 1:
                self.__tag = ">"
            else:
                self.__tag = "<"
            
            ### Read Indexing ###
            offset = -16
            f.seek(offset, os.SEEK_END)
            self.__null_termed_names_bytes, self.__number_of_commits,  self.__number_of_directories, self.__number_of_state_maps = \
                struct.unpack('4i', f.read(16))
       
            ### Read State Maps ####
            offset = self.readStateMaps(f)
    
            ### Read Directories ###
            offset = self.readDirectories(f, offset)
            
            
            if self.__null_termed_names_bytes > 0:
                 ### DIRECTORY AND SV DATA #
                self.readStateVariablesAndParams(f, offset)
            
                ### MESH DATA ###
                self.readMesh(f)       
                
                
                ### SUBRECORD DATA ###
                self.readSubrecords(f)
    
    """
    Take name and turn it into vector and component names
    """
    def parse_name(self, name):
        vector, component = None, name
        if "[" in name:
            pos = name.find("[")
            vector = name[:pos]
            component = name[pos+1:len(name)-1]
        return [vector, component]
    
    """
    Create an answer from res dictionary
    This is what is returned by query
    """
    def create_answer(self, res, names, materials, labels, class_name, state_numbers, modify, raw_data):
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
                    item.set(res[state_number][name][label])
                    state.items.append(item)
                
        return answer
    
    """
    Return true if this class name combo is a vector array
    """
    def is_vec_array(self, name, class_name):
        if not name: return False
        
        if 'stress' in name:
            elem_sets = self.__int_points['stress']
            vars = self.__state_variables['stress'][0].svars
        elif 'strain' in name: 
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
        
    """
    Given a single state variable name, a number of states, and a number of label(s) return the requested value(s)
    """
    def variable_at_state(self, subrecord, labels, name, vars, sup_class, clas, sub, res, modify=False, int_points=False):
        mo_search_arr = []
        indices = {}
        values = {} # from label to value
                
        # Deal with names like vector[component]
        vector, variables = self.parse_name(name)
        variables = [variables]
                
        # These are the mo_ids we are interested in for this subrecord
        for label in labels:
            if label not in self.__labels[(sup_class, clas)]:
                print "label " + str(label) + " was not found in " + clas
            else:
                mo_search_arr.append([label, self.__labels[(sup_class, clas)][label]])
                      
        mo_index = 0
        mo_idx_found = []
                
        # Search subrecord blocks to find what index the mo_ids are at
        for range in subrecord.mo_blocks:
            start, end = range
            for mo_search in mo_search_arr:
                label, mo_search = mo_search
                if mo_search >= start or mo_search <= end:
                    mo_idx_found.append([label, mo_index + mo_search - start])
            mo_index += end - start
                
        indices[sub] = []
        
        # Deal with aggregate types and create list of state variable names
        if name in self.__state_variables and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR':
            variables = self.__state_variables[name][0].svars
       
        sv_names = []
        for sv in subrecord.svar_names:
            sv_var = self.__state_variables[sv][0]
            if len(sv_var.svars) > 0:
                for sv_name in sv_var.svars:
                    sv_names.append(sv_name)
            else:
                sv_names.append(sv)
        
        var_indexes = []
        for child in variables:
            if child not in sv_names:
                print child + " not a valid variable name"
                return
            var_indexes.append(sv_names.index(child))       
                
        # Add correct values given organizational structure and correct indexing    
        if int_points:
            int_points, num_int_points = int_points[:-1], int_points[-1:][0]
        
        for mo_index in mo_idx_found:
            if int_points: indexes = {}
            else: indexes = []
            label, mo_index = mo_index
            for var_index in var_indexes:
                if int_points: indexes[var_index] = {}
                if int_points:
                    offset = mo_index * len(sv_names) * num_int_points
                    for int_point in int_points:
                        index = offset + var_index * len(sv_names) + int_point - 1
                        indexes[var_index][int_point] = index
                else:
                    if subrecord.organization == DataOrganization.OBJECT.value:
                        indexes.append(mo_index * len(sv_names) + var_index)
                    else:
                        indexes.append(var_index * subrecord.mo_qty + mo_index)
         
            # 3 different aggregate types here - contructing the res
            if int_points:
                values[label] = {}
                v_index = 0
                for index in indexes.keys():
                    sv_name = sv_names[index]
                    values[label][sv_name] = {}
                    for int_point in int_points:
                        values[label][sv_name][int_point] = vars[indexes[index][int_point]]
                        indices[sub].append(indexes[index][int_point])
                    v_index += 1
            elif name in self.__state_variables and AggregateType(self.__state_variables[name][0].agg_type).name == 'VECTOR':
                values[label] = []
                for index in indexes:
                    values[label].append(vars[index])
                    indices[sub].append(index)
            else:
                values[label] = vars[indexes[0]]
                indices[sub].append(indexes[0])
        
        if modify:
            return [res, indices]
        
        return values
    
    """
    Turn a class name and element list into the appropriate label array
    """
    def getLabelsFromClassElems(self, class_name, elems):
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
            
    """
    Single query function that given inputs can call the other functions
    
    Arguments:
    1. names: short names for all the shortnames you are interested in
    2. class: class name interesteds in
    3. material: string name of material you are interested in
            Empty: Ignore material
    4. labels: labels that you are interested in
            Empty: Find all possible labels
    5. state_numbers: States you are interested in
            Empty: all states
    6. modify: whether or not this is part of a modfication call
    7. int_points: if this is a vectory array, which integration points
    8. raw_data: whether the user wants raw data or Answer object
            
    
    The following is the structure of the result that is passed to create answer
    res[state][name][label] = value
    """
    def query(self, names, class_name, material=None, labels=None, state_numbers=None, modify=False, int_points=False, raw_data=True):
        # Parse Arguments
        if not state_numbers:
            state_numbers = [i-1 for i in range(1, self.__number_of_state_maps + 1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        
        if type(labels) is int:
            labels = [labels]
            
        if material:
            if type(material) is not str:
                print "material must be string"
                return
            if material not in self.__matname:
                print "There is no " + material + " material"
                return
            elems = []
            for mat in self.__matname[material]:
                elems += self.__materials[mat]            
            labels_mat = self.getLabelsFromClassElems(class_name, elems)
            if labels:
                for label in labels:
                    if label not in labels_mat:
                        labels.remove(label)
            else:
                labels = labels_mat
                
            if not len(labels):
                print "There are no elements from class " + str(class_name) + " of material " + material + " with class " + class_name
                return
        if labels:
            if type(labels) is not list or type(labels[0]) is not int:
                print "labels must of a list of ints or an int", labels
                return
        if not labels:
            sup_class = self.__mesh_object_class_datas[class_name].superclass
            sup_class = Superclass(sup_class).name
            labels = self.__labels[(sup_class, class_name)]
        
        if type(names) is str:
            names = [names]
        if type(names) is not list or type(names[0]) is not str:
            print "state variables names must be a string or list of strings"
            return
        if modify and type(modify) is not bool:
            print "modify must be boolean"
            return
        if raw_data and type(raw_data) is not bool:
            print "raw data must be boolean"
            return
        if type(int_points) is int:
            int_points = [int_points]
        if int_points and type(int_points) is not list:
            print "int point must be an integer or list of integers"
            return
        if type(state_numbers) is not list or type(state_numbers[0]) is not int:
            print "state numbers must be an integer or list of integers"
            return
        
                
        # Run Correct Function
        res = {}
                
        for state in state_numbers:
            if state < 0 or state >= len(self.__state_maps):
                print "There is no state " + str(state)
                return
            state_map = self.__state_maps[state]
            res[state] = {}
                
            with open(self.__state_map_filename[state_map.file_number], 'rb') as f:
                f.seek(state_map.file_offset)
                byte_array = f.read(8)
                time, state_map_id = struct.unpack(self.__tag + 'fi', byte_array)
                
                for name in names:
                    # Handle case of vector[component]
                    vector, variables = self.parse_name(name)
                    if not vector: vector = variables
                    
                    if self.is_vec_array(vector, class_name):
                        if 'stress' in name: elem_sets = self.__int_points['stress']
                        elif 'strain' in name: elem_sets = self.__int_points['strain']
                        
                        for set_name in elem_sets.keys():
                            temp_sv, temp_subrecords = self.__state_variables[set_name]
                            temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
                            if temp_subrecord.class_name == class_name:
                                sv, subrecords = temp_sv, temp_subrecords
                                set_name_chosen = set_name
                        if not int_points:
                            int_points = list(self.is_vec_array(vector, class_name))
                        else:
                            possible_int_points = self.is_vec_array(vector, class_name)
                            for i in range(len(int_points)):
                                ip = int_points[i]
                                if ip not in possible_int_points:
                                    idx_ip, ip = min(enumerate(possible_int_points), key=lambda x: abs(x[1]-ip))
                                    print str(ip) + " is not an integration point, but the closest is " + str(ip)
                                int_points[i] = ip
                        int_points.append(len(self.is_vec_array(vector, class_name)))
                                                                    
                    elif vector:
                        if vector not in self.__state_variables:
                            print "There is no variable " + vector
                            return
                        sv, subrecords = self.__state_variables[vector]
                    else:
                        if name not in self.__state_variables:
                            print "There is no variable " + name
                            return
                        sv, subrecords = self.__state_variables[name]
                    
                    for sub in subrecords:
                        subrecord = self.__srec_container.subrecs[sub]
                        f.seek(subrecord.offset, 1)
                        byte_array = f.read(subrecord.size)
                        s = self.set_string(subrecord)

                        vars = struct.unpack(self.__tag + s, byte_array)
                        
                        if class_name == subrecord.class_name:
                            sup_class = self.__mesh_object_class_datas[subrecord.class_name].superclass
                            sup_class = Superclass(sup_class).name
                            
                            if modify:
                                return self.variable_at_state(subrecord, labels, name, vars, sup_class, subrecord.class_name, sub, res, modify, int_points)
                            else:
                                res[state][name] = self.variable_at_state(subrecord, labels, name, vars, sup_class, subrecord.class_name, sub, res, modify, int_points)
        
        return self.create_answer(res, names, material, labels, class_name, state_numbers, modify, raw_data)
    
    """
    Given a specific material. Find all mo_ids with that material and return
    their values.
    """
    def elements_of_material(self, material, raw_data=False):
        if material not in self.__matname:
            print "There is no material " + material
            return
        
        matnumbers = self.__matname[material]
        mo_ids = []
        for matnumber in matnumbers:
            mod_id_class = self.__materials[matnumber]
            mo_ids += [item[0] for item in mod_id_class]
            class_names = [item[1] for item in mod_id_class]
        
        if raw_data:
            return mo_ids
        
        answer = Answer()
        answer.set(None, None, mo_ids, None, class_names, None)
        
        return answer
    
    """
    Find nodes associated with a material number
    AND CLASSNAME FOR NOW
    """
    def nodes_of_material(self, material, class_name, raw_data=False):
        nodes = set()
        elements = self.elements_of_material(material)
        if not elements:
            print "There are no elements with material " + material
            return
        for item in elements.items:
            mo_id, entry_class_name = item.mo_id, item.class_name
            if entry_class_name == class_name:
                for node in self.__connectivity[class_name][mo_id]:
                    nodes.add(node)
        
        if raw_data:
            return list(nodes)
        
        answer = Answer()
        answer.set(None, None,None, list(nodes), "node", None)
                    
        return answer
    
    """
    Find nodes associated with an element number
    AND CLASSNAME FOR NOW
    """
    def nodes_of_elem(self, label, class_name, raw_data=False):
        if class_name not in self.__mesh_object_class_datas:
            print "Class name " + class_name + " not found"
            return
        sup_class = self.__mesh_object_class_datas[class_name].superclass
        sup_class = Superclass(sup_class).name
        
        if label not in self.__labels[(sup_class, class_name)]:
            print "label " + str(label) + " not found"
            return
        mo_id = self.__labels[(sup_class, class_name)][label]
        labels = self.__connectivity[class_name][mo_id]
        
        if raw_data:
            return labels
        
        answer = Answer()
        answer.set(None, None, None, labels, class_name, None)
              
        return answer

    """
    Modifies a state variable at a specific label and time state
    
    The order of the values list (only do this if you're crazy):
    first sort by name, then sort by label, then by component name, then by integration point
    
    It is simpler to modify variables one at a time
    """
    def modify_state_variable(self, state_variables, class_name, value, labels, state_numbers, int_points=False):
        res, indices = self.query(state_variables, class_name, None, labels, state_numbers, True, int_points)
        type_to_str = {'s' : 'M_STRING', 'f' : 'M_FLOAT', 'd' : 'M_FLOAT8', 'i' : 'M_INT', 'q' : 'M_INT8'}
        
        if not state_numbers:
            state_numbers = [i-1 for i in range(1, self.__number_of_state_maps + 1)]
        elif type(state_numbers) is int:
            state_numbers = [state_numbers]
        if type(labels) is int:
            labels = [labels]
        
        for state in state_numbers:
            if state < 0 or state >= len(self.__state_maps):
                print "There is no state " + str(state)
                return
            
            state_map = self.__state_maps[state]
        
            with open(self.__state_map_filename[state_map.file_number], 'r+') as f:
                f.seek(state_map.file_offset)
                byte_array = f.read(8)
                time, state_map_id = struct.unpack(self.__tag + 'fi', byte_array)
        
                for name in state_variables:
                    vector, variables = self.parse_name(name)
                    if not vector: vector = variables 
                    if self.is_vec_array(vector, class_name):
                        if 'stress' in name: elem_sets = self.__int_points['stress']
                        elif 'strain' in name: elem_sets = self.__int_points['strain']
                                            
                        for set_name in elem_sets.keys():
                            temp_sv, temp_subrecords = self.__state_variables[set_name]
                            temp_subrecord = self.__srec_container.subrecs[temp_subrecords[0]]
                            if temp_subrecord.class_name == class_name:
                                sv, subrecords = temp_sv, temp_subrecords
                                set_name_chosen = set_name
                        if not int_points:
                            int_points = list(self.is_vec_array(vector, class_name))
                        int_points.append(len(self.is_vec_array(vector, class_name)))                                              
                    elif vector:
                        if vector not in self.__state_variables:
                            print "There is no variable " + vector
                            return
                        sv, subrecords = self.__state_variables[vector]
                    else:
                        if name not in self.__state_variables:
                            print "There is no variable " + name
                            return
                        sv, subrecords = self.__state_variables[name]
                    
                    for sub in subrecords:
                        subrecord = self.__srec_container.subrecs[sub]
                        f.seek(subrecord.offset, 1)
                        s = self.set_string(subrecord)
                        
                        # Iterate through s, the string representation of the subrecord, adding
                        # offset along the way
                        for j in range(len(indices[sub])):
                            offset = 0
                            for i in range(indices[sub][j]):
                                char = s[i]
                                offset += ExtSize[type_to_str[char]].value
                            
                            # Seek to variable location and write in the byte array
                            f.seek(offset, 1)
                            byte_array = struct.pack(self.__tag + s[indices[sub][j]], value[j % (len(indices[sub]) / len(labels))])
                            f.write(byte_array)
                            f.seek(-offset - len(byte_array), 1)                        
                             

"""
This function is an example of how a user could use the Mili reader
"""    
def main():    
    
    mili = Mili()
    mili.read("d3samp6")
    #mili.read("states/d3samp6")
    
if __name__ == '__main__':
        main()
