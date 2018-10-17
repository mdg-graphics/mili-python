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
    ###
      # Add here ->
    ###
    TI_PARAM = 9
    ###
      # QTY Must be the last item!!
    ###
    QTY_DIR_ENTRY_TYPES = 10
    
    
def main():
    offset = -16
    file_name = '../../Mili/MILI-toss_3_x86_64_ib-RZGENIE/d3samp6new.pltA'
    
    state_maps = []
    directories = []
    names = []
    d = {} # maps param name: file index, entry index
    sv = {} # state variable dictionary
    
  # Open file with 'b' to specify binary mode
    with open(file_name, 'rb') as f:
        header = f.read(16)
        ### Read Header ###
    
        mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')
        header_version, directory_version, endian_flag, precision_flag, state_file_suffix_length, partition_flag = \
        struct.unpack('6b', header[4:10])
      
        ### Read Indexing ###
   
        f.seek(offset, os.SEEK_END)
        null_termed_names_bytes, number_of_commits,  number_of_directories, number_of_state_maps = \
            struct.unpack('4i', f.read(16))
   
        ### Read State Maps ####
    
        if endian_flag == 1:
            tag = ">"
        else:
            tag = "<"
        
        offset = -16
        state_map_length = 20
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + number_of_state_maps):
            f.seek(-1 * state_map_length, 1)
            byte_array = f.read(state_map_length)
            file_number, file_offset, time, state_map_id = struct.unpack(tag + 'iqfi', byte_array)      
            state_maps.append(StateMap(file_number, file_offset, time, state_map_id))
            f.seek(-1 * state_map_length, 1)

    ### Read Directories ###
    
        number_of_strings = 0
        directory_length = 4 * 6
        int_long = 'i'
        if directory_version > 2:
            directory_length = 8 * 6
            int_long = 'q'
        offset -=  state_map_length * number_of_state_maps + directory_length * number_of_directories # make this not a hard coded 6 eventually
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + number_of_directories):
            byte_array = f.read(directory_length)
            type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx = \
                struct.unpack(tag + '6' + int_long, byte_array)
            directories.append(Directory(type_idx, modifier_idx1, modifier_idx2, string_qty_idx, offset_idx, length_idx))
            number_of_strings += string_qty_idx    
        
        if null_termed_names_bytes > 0:
            offset -= null_termed_names_bytes
            f.seek(offset, os.SEEK_END)
            test = f.read(null_termed_names_bytes)
            fmt = str(null_termed_names_bytes) + "s"
            byte_array = struct.unpack(tag + fmt, test)[0]
            #strings = str(byte_array)[2:].split('\\x00') # only works for Python3
            strings = byte_array.split(b'\x00')
            nnames = 0
            
            for i in range(len(directories)):
                directory = directories[i]
                for j in range(directory.string_qty_idx):
                    name = strings[nnames]
                    names.append(name)
                    nnames += 1

                if directory.type_idx == DirectoryType.MILI_PARAM.value or directory.type_idx == DirectoryType.APPLICATION_PARAM.value or \
                    directory_version >= 2 and directory.type_idx == DirectoryType.TI_PARAM.value:
                    d[name] = file_number, i
        
                if directory.type_idx == DirectoryType.STATE_VAR_DICT.value:
                    print(directory.offset_idx)
                    f.seek(directory.offset_idx)
                    svar_words, svar_bytes = struct.unpack('2i', f.read(8))
                    num_ints = (svar_words - 2)
                    ints = struct.unpack(str(num_ints) + 'i', f.read(num_ints * 4)) # what is this
                    s = struct.unpack(str(svar_bytes) + 's', f.read(svar_bytes))[0].split(b'\x00')
                    for i in range(0, len(s), 2):
                        sv[s[i]] = s[i+1]
	    
    #for directory in directories:
     # for string in range(directory.string_qty_idx):
      #  print (string)
      


if __name__ == '__main__':
        main()
