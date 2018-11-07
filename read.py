import struct
import os
from enum import Enum 

class StateMap:
    def __init__(self, file_number, file_offset, time, state_map_id):
        self.file_number = file_number
        self.file_offset = file_offset
        self.time = time
        self.state_map_id = state_map_id
        
class Directory:
    def __init__(self, type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx):
        self.type_idx = type_idx
        self.modifier_idx1 = modifier_idx1
        self.modifier_idx2 = modifier_idx2
        self.string_qty_idx = string_qty_idx
        self.offset_idx = offset_idx
        self.length_idx = length_idx

class StateVariable:
    def __init__(self, name, title, agg_type, data_type):
        self.name = name
        self.title = title
        self.agg_type = agg_type
        self.data_type = data_type
        self.list_size = 0
        self.order = 0
        self.dims = []
        self.svars = [] # list of string names included in this if it is vector
    
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

class SubrecordContainer:
    subrecs = []
    size = 0
    #qty_subrecs = 0 this is just len(subrecs)
        
class StateRecord:
    def __init__(self, subrecords):
        self.subrecords = []
        self.size = 8 # integer + float
        
    def add_subrecord(self, subrecord):
        self.subrecords.append(subrecord)
        self.size += subrecord.size

class MeshObjectClassData:
    def __init__(self, short_name, long_name, superclass):
        self.short_name = short_name
        self.long_name = long_name
        self.superclass = superclass
        self.surface_sizes = None
        self.blocklist = BlockList(0, 0, [])
    
    def add_block(self, start, stop):
        self.blocklist.blocks.append((start, stop))
        self.blocklist.block_qty += 1
        self.blocklist.obj_qty += start - stop + 1

class BlockList:
    def __init__(self, obj_qty, block_qty, blocks):
        self.obj_qty = obj_qty
        self.block_qty = block_qty
        self.blocks = blocks # array of tuples with start, stop

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
  
class DataType(Enum):    
    M_STRING = 1
    M_FLOAT = 2
    M_FLOAT4 = 3
    M_FLOAT8 = 4
    M_INT = 5
    M_INT4 = 6
    M_INT8 = 7

class ExtSize(Enum):    
    M_STRING = 1
    M_FLOAT = 4
    M_FLOAT4 = 4
    M_FLOAT8 = 8
    M_INT = 4
    M_INT4 = 4
    M_INT8 = 8

class AggregateType(Enum):
    SCALAR = 0
    VECTOR = 1
    ARRAY = 2
    VEC_ARRAY = 3

class DataOrganization(Enum):
    RESULT = 0
    OBJECT = 1

class Mili:
    def __init__(self):
        self.state_maps = []
        self.directories = []
        self.names = []
        self.params = {} # maps param name: file index, entry index
        self.state_variables = {} # map to state variable, list of subrecords it is in
        self.mesh_object_class_datas = {} #shortname to object
        self.header_version = None
        self.directory_version = None
        self.endian_flag = None
        self.precision_flag = None
        self.state_file_suffix_length = None
        self.partition_flag = None
        self.tag = None
        self.null_termed_names_bytes = None
        self.number_of_commits = None
        self.number_of_directories = None
        self.number_of_state_maps = None
    
    def readStateMaps(self, f):
        offset = -16
        state_map_length = 20
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + self.number_of_state_maps):
            f.seek(-1 * state_map_length, 1)
            byte_array = f.read(state_map_length)
            file_number, file_offset, time, state_map_id = struct.unpack(self.tag + 'iqfi', byte_array)
            self.state_maps.append(StateMap(file_number, file_offset, time, state_map_id))
            f.seek(-1 * state_map_length, 1)
        return offset

    def readStateVariablesAndParams(self, f, offset):
        offset -= self.null_termed_names_bytes
        f.seek(offset, os.SEEK_END)
        fmt = str(self.null_termed_names_bytes) + "s"
        byte_array = struct.unpack(self.tag + fmt, f.read(self.null_termed_names_bytes))[0]
        #strings = str(byte_array)[2:].split('\\x00') # only works for Python3
        strings = byte_array.split(b'\x00')
        nnames = 0
        file_number = 0
        
        for i in range(len(self.directories)):
            directory = self.directories[i]
            for j in range(directory.string_qty_idx):
                name = strings[nnames]
                self.names.append(name)
                nnames += 1
    
            if directory.type_idx == DirectoryType.MILI_PARAM.value or directory.type_idx == DirectoryType.APPLICATION_PARAM.value or \
                self.directory_version >= 2 and directory.type_idx == DirectoryType.TI_PARAM.value:
                self.params[name] = file_number, i
    
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
                            if sv_name_inner in self.state_variables:
                                sv = self.state_variables[sv_name_inner]
                            else:
                                sv_name_inner, title = s[c_pos], s[c_pos + 1]
                                agg_type, data_type = ints[int_pos], ints[int_pos + 1]
                                int_pos += 2
                                c_pos += 2
                                sv = StateVariable(sv_name_inner, title, agg_type, data_type)
                                self.state_variables[sv_name_inner] = [sv, []]
                        state_variable.svars = sv_names
                    
                    self.state_variables[sv_name] = [state_variable, []]
    
    def readDirectories(self, f, offset):
        number_of_strings = 0
        directory_length = 4 * 6
        state_map_length = 20
        int_long = 'i'
        if self.directory_version > 2:
            directory_length = 8 * 6
            int_long = 'q'
        offset -=  state_map_length * self.number_of_state_maps + directory_length * self.number_of_directories # make this not a hard coded 6 eventually
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + self.number_of_directories):
            byte_array = f.read(directory_length)
            type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx = \
                struct.unpack(self.tag + '6' + int_long, byte_array)
            self.directories.append(Directory(type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx))
            number_of_strings += string_qty_idx
        return offset
    
    def readMesh(self, f):
        name_cnt = 0 
        for i in range(len(self.directories)):
            directory = self.directories[i]     
            
                
            if directory.type_idx == DirectoryType.CLASS_DEF.value:
                superclass = directory.modifier_idx2
                short_name = self.names[name_cnt]
                long_name = self.names[name_cnt + 1]
                mocd = MeshObjectClassData(short_name, long_name, superclass)
                self.mesh_object_class_datas[short_name]= mocd
            
            if directory.type_idx == DirectoryType.CLASS_IDENTS.value:
                f.seek(directory.offset_idx)
                short_name = self.names[name_cnt]
                superclass, start, stop = struct.unpack('3i', f.read(12))
                self.mesh_object_class_datas[short_name].add_block(start, stop)
            
            if directory.type_idx == DirectoryType.NODES.value:
                f.seek(directory.offset_idx)
                short_name = self.names[name_cnt]
                start, stop = struct.unpack('2i', f.read(8))
                self.mesh_object_class_datas[short_name].add_block(start, stop)
            
            if directory.type_idx == DirectoryType.ELEM_CONNS.value:
                f.seek(directory.offset_idx)
                short_name = self.names[name_cnt]
                superclass, qty_blocks = struct.unpack('2i', f.read(8))
                elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))
                for j in range(0,len(elem_blocks),2):
                    self.mesh_object_class_datas[short_name].add_block(elem_blocks[j], elem_blocks[j+1])
                
            name_cnt += directory.string_qty_idx
          
    def readSubrecords(self, f):
        for i in range(len(self.directories)):
            directory = self.directories[i]          
            if directory.type_idx == DirectoryType.STATE_REC_DATA.value:
                srec_int_data = directory.modifier_idx1 - 4
                srec_c_data = directory.modifier_idx2
                f.seek(directory.offset_idx)
                srec_id, srec_parent_mesh_id, srec_size_bytes, srec_qty_subrecs = struct.unpack('4i', f.read(16))
                idata = struct.unpack(str(srec_int_data) + 'i', f.read(srec_int_data * 4))
                cdata = struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].split(b'\x00')
                
                int_pos = 0
                c_pos = 0
                self.srec_container = SubrecordContainer()
                
                
                for i in range(srec_qty_subrecs):
                    org, qty_svars, qty_id_blks = idata[int_pos], idata[int_pos+1], idata[int_pos+2]
                    # for j in range(qty_id_blks):
                        
                    int_pos += 3
                    name, class_name = cdata[c_pos:c_pos+2]
                    c_pos += 2
                    svars = cdata[c_pos:c_pos+qty_svars]
                    c_pos += qty_svars
                    
                    superclass = self.mesh_object_class_datas[class_name].superclass
                    
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
                    
                    if org == DataOrganization.OBJECT.value:
                        for sv in svars:
                            state_var = self.state_variables[sv][0]
                            atom_size = ExtSize[DataType(state_var.data_type).name].value
                            atoms = state_var.atom_qty(self.state_variables) # define this 
                            
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
                            state_var = self.state_variables[sv][0]
                            atoms = state_var.atom_qty(self.state_variables)
                            ### stuff about surface here
                            total_atoms = atoms
                            
                            lump_atoms.append(sub.mo_qty * total_atoms)
                            lump_sizes.append(sub.mo_qty * total_atoms * ExtSize[DataType(state_var.data_type).name].value)
                        
                        lump_offsets.append(0)
                        for j in range(1, sub.qty_svars):
                            lump_offsets.append(lump_offsets[j-1] + lump_sizes[j-1])
                        
                        j = sub.qty_svars - 1
                        sub.size = lump_offsets[j] + lump_sizes[j]
                    
                    
                    sub.offset = self.srec_container.size
                    self.srec_container.size += sub.size
                    self.srec_container.subrecs.append(sub)
    
    def read(self, file_name):
    # Open file with 'b' to specify binary mode
        with open(file_name, 'rb') as f:
            ### Read Header ###
            header = f.read(16)
            mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')
            self.header_version, self.directory_version, self.endian_flag, self.precision_flag, self.state_file_suffix_length, self.partition_flag = \
               struct.unpack('6b', header[4:10])
            
            if self.endian_flag == 1:
                self.tag = ">"
            else:
                self.tag = "<"
            
            ### Read Indexing ###
            offset = -16
            f.seek(offset, os.SEEK_END)
            self.null_termed_names_bytes, self.number_of_commits,  self.number_of_directories, self.number_of_state_maps = \
                struct.unpack('4i', f.read(16))
       
            ### Read State Maps ####
            offset = self.readStateMaps(f)
    
            ### Read Directories ###
            offset = self.readDirectories(f, offset)
            
            
           
            if self.null_termed_names_bytes > 0:
                 ### DIRECTORY AND SV DATA #
                self.readStateVariablesAndParams(f, offset)
            
                ### MESH DATA ###
                self.readMesh(f)       
                
                
                ### SUBRECORD DATA ###
                self.readSubrecords(f)

def main():
    
    file_name = '../../Mili/MILI-toss_3_x86_64_ib-RZGENIE/d3samp6new.pltA'
    
    '''
    Could add more files later ?
    '''
    
    mili = Mili()
    mili.read(file_name)
                        
    print len(mili.directories)


if __name__ == '__main__':
        main()
