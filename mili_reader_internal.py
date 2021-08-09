
            
    ############################## Notes ###############################
    # replace read functions one at a time and test combine
    # import multiprocessing
    # manager = multiprocessing.Manager()
    # shared_dict = manager.dict()
    #
    # Directories:
    # bc we don't actually go to info dir is pointing to in readDirs
    # wait until do reach that data in other read functions (svar,srec)
    # then add dir to all_local_dirs so that we have short_names
    # three dictionaries:
    # # dictionary for local directories - uses offset in file as index
    # # dictionary for local directories from all processors for svars, srecs - uses short_name as index
    # # dictionary for global combined directories that we are committing in the end
    ####################################################################

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

    # global dictionaries or lists for names, params, svars, srecs, dirs

    # dir offsets are file specific
    # How do we get all necessary info to create new directory for combined - can get offset but type_idx?
    # should associate file number w directory entries in dictionary

    
    def ReadHeader(self, f, file_tag, header_array): # add file_tag to header_array
        # Precision flag should be same across procs ??
        header = f.read(16)
        mili_taur = struct.unpack('4s', header[:4])[0].decode('ascii')
        #assert(mili_taur == 'mili' or mili_taur == 'taur')
        
        header_array = struct.unpack('6b', header[4:10])
        header_version, dir_version, endian_flag, precision_flag, state_file_suffix_length, partition_flag = header_array

        if endian_flag == 1:
            file_tag = '>'
        else:
            file_tag = '<'        


    def ReadStateMaps(self, f, file_tag, this_proc_indexing_array, global_state_maps):
        offset = -16
        state_map_length = 20
        f.seek(offset, os.SEEK_END)
        number_of_state_maps = this_proc_indexing_array[3]
        for i in range(1, 1 + number_of_state_maps):
            f.seek(-1 * state_map_length, 1)
            byte_array = f.read(state_map_length)
            state_map = struct.unpack(file_tag + 'iqfi', byte_array)
            file_number, file_offset, time, state_map_id = state_map
            # check if an entry at file_number index exists already
            global_state_maps[file_number].append(state_map)
            f.seek(-1 * state_map_length, 1)
        return offset

        '''
    Reads all the string data for the mili file and associates the strings
    with the appropriate directory structures.
    '''
    def ReadNames(self, f, file_tag, offset, this_indexing_array, this_proc_dirs, dir_num_strings_dict, global_names):
        # 1. Keep track of which strings associated w which directories
        # 2. Keep list of all names

        # this_indexing_array = [null_termed_names_bytes, num_commits, num_dirs, num_state_maps]
        # this_proc_dirs = long list of each dirs 6 digits ... [d1,d2,d3,d4,d5,d6,d1,d2,...] vs [[d1,d2,d3,d4,d5,d6],[d1,d2,d3,d4,d5,d6],...]
        # dir_num_strings_dict[dir_offset] = (string_offset_idx, [string1, string2...])
        
        offset -= this_indexing_array[0]
        f.seek(offset, os.SEEK_END)
        fmt = str(this_indexing_array[0]) + 's'
        byte_array = struct.unpack(file_tag + fmt, f.read(this_indexing_array[0]))[0]
        
        if (sys.version_info > (3, 0)):
            strings = str(byte_array)[2:].split('\\x00')  # only works for Python3
        else:
            strings = byte_array.split(b'\x00')

        # Can we put all null termed names bytes together
        # iterate through directories: this_proc_dirs[4] to get offset_idx to use as key to dir_num_strings_dict
        num_dirs = this_indexing_array[2]
        for dir_num in range(1,num_dirs+1):
            dir_offset = this_proc_dirs[6 * (dir_num-1) + 4] # each dir has 6 ints in array, offset at index=4

            # Use dir_offset to find str
            string_offset = dir_num_strings_dict[dir_offset][0]
            string_count = this_proc_dirs[6 * (dir_num-1) + 3] # needs to be index 3 for this directory
            directory_strings = [ strings[sidx] for sidx in range( string_offset, string_offset + string_count) ]
            dir_num_strings_dict[dir_offset][1] = directory_strings

        # Add to global_names if not in there already
        if global_names is None:
            global_names = strings
        else:
            for string in strings:
                if string not in global_names:
                    global_names.append(string)


    def ReadParams(self, f, file_tag, offset, dir_type_dict, dir_num_strings_dict, this_proc_dirs, this_proc_params, this_proc_dim, this_proc_node_labels, this_proc_elem_labels, this_proc_matname, this_proc_i_points):
        # Index Keys
        # dir_index_key = {'offset_idx': 4, 'length_idx': 5}
        # this_proc_node_labels_key = {'first': 0, 'last': 1, 'node_labels': [2:]}
        # this_proc_elem_labels_key = {'first': 0, '_': 1, 'ints': [2:]}
        # this_proc_i_points_key = {'first': 0, '_': 1, 'i_points': i_points[2:len(i_points) - 1], 'num_i_points': i_points[len(i_points) - 1]}

        # global_params[name_type][name] =
        # # 's' = params
        # # 'mesh_dimensions = array of dims and use one at the end?
        # # 'node_labels' = dictionary of (superclass, name): {label, local_id}
        # # 'Element Label'
        # # # 'ElemIds'
        # # 'MAT_NAME' = [matname].append(ints)
        # # 'es_' = i_points = first, _, i_points, num_i_points - how to store - get rid of es_ and only use name as key ? 
        
        
        type_to_str = {'M_STRING' : 's', 'M_FLOAT' : 'f', 'M_FLOAT4' : 'f', 'M_FLOAT8' : 'd', 'M_INT' : 'i', 'M_INT4' : 'i', 'M_INT8' : 'q'}

        file_number = 0

        can_proc = [ DirectoryType.MILI_PARAM.value, DirectoryType.APPLICATION_PARAM.value, DirectoryType.TI_PARAM.value ]
        for type_id in can_proc:
            for dir_proc_array_offset in dir_type_dict[type_id]:
                dir_strings = dir_num_strings_dict[dir_proc_array_offset][1]
                name = dir_strings[0]
                param_offset = this_proc_dirs[dir_proc_array_offset] # index 4 of directory entry array
                
                f.seek(param_offset)

                byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1]) # go to index 5 which is dir's length idx
                param_type = this_proc_dirs[dir_proc_array_offset - 3] # go to index 1 which is dir's mod_idx1

                type_rep = type_to_str[DataType(param_type).name]
                type_value = ExtSize[DataType(param_type).name].value
        
                if type_to_str[DataType(param_type).name] == 's': # replace hard coded strings w variable equivalents
                    if (sys.version_info > (3, 0)):
                        this_proc_params[name] = [str(struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / type_value)) + type_rep, byte_array)[0])[2:].split('\\x00')]
                    else:
                        this_proc_params[name] = [struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / type_value)) + type_rep, byte_array)[0].split(b'\x00')[0]]
                else:
                    this_proc_params[name] = [struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / type_value)) + type_rep, byte_array)]

                for param in this_proc_params[name]:
                    if name not in global_params:
                        global_params[name] = this_proc_params[name]
                    else:
                        if param not in global_params[name]:
                            global_params[name].append(this_proc_params[name])

                if name == 'mesh dimensions':
                    f.seek(param_offset)
                    byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1])
                    this_dim = struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / 4)) + 'i', byte_array)[0]
                    # Do I need to keep track of dims from all?
                
                if 'Node Labels' in name:
                    f.seek(param_offset)
                    byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1])
                    ints = struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / 4)) + 'i', byte_array)
                    first, last, node_labels = ints[0], ints[1], ints[2:]

                    this_proc_labels[('M_NODE', 'node')] = {}
                    for j in range (first - 1, last):
                        this_proc_labels[('M_NODE', 'node')][node_labels[j]] = j + 1 # change GetLabels()

                if 'Element Label' in name:
                    f.seek(param_offset)
                    byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1])
                    ints = struct.unpack(file__tag + str(int(this_proc_dirs[dir_proc_array_offset + 1] / 4)) + 'i', byte_array)
                    this_proc_elem_labels = this_proc_elem_labels + ints #
                    this_proc_params = this_proc_params + ints
                    first, _, ints = ints[0], ints[1], ints[2:]

                    sup_class_idx = name.index('Scls-') + len('Scls-')
                    sup_class_end_idx = name.index('/', sup_class_idx, len(name))
                    class_idx = name.index('Sname-') + len('Sname-')
                    class_end_idx = name.index('/', class_idx, len(name))

                    sup_class = name[sup_class_idx : sup_class_end_idx]
                    clas = name[class_idx : class_end_idx]

                    this_proc_labels[(sup_class, clas)] = {}

                    for j in range(len(ints)):
                        if 'ElemIds' in name:
                            this_proc_labels[(sup_class, clas)][this_proc_label_keys[j]] = ints[j]
                        else:
                            this_proc_label_keys.append(ints[j])

                    if 'ElemIds' in name:
                        this_proc_label_keys = []

                if 'MAT_NAME' in name: # need to combine? all procs same?
                    f.seek(this_proc_dirs[dir_proc_array_offset])
                    byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1])
                    
                    if (sys.version_info > (3, 0)):
                        matname = str(struct.unpack(str(this_proc_dirs[dir_proc_array_offset + 1]) + 's', byte_array)[0])[2:].split('\\x00')[0]  # only works for Python3
                    else:
                        matname = struct.unpack(str(directory.length_idx) + 's', byte_array)[0].split(b'\x00')[0]
                    
                    num = name[-1:]
                    if matname in this_proc_matname:
                        this_proc_matname[matname].append(int(num))
                    else:
                        this_proc_matname[matname] = [int(num)]

                if 'es_' in name: # check if need to combine or all procs same
                    f.seek(param_offset)
                    byte_array = f.read(this_proc_dirs[dir_proc_array_offset + 1])
                    i_points = struct.unpack(file_tag + str(int(this_proc_dirs[dir_proc_array_offset] / 4)) + 'i', byte_array)
                    first, total, i_points, num_i_ponts = i_points[0], i_points[1], i_points[2:len(i_points) - 1], i_points[len(i_points) - 1]
                    index = name.find('es_')
                    this_proc_int_points[name[index:]] = [first, total, i_points, num_i_ponts] # what is diff between total and num_i_points
                    
                    # determine stress or strain



    def ReadSvars(global_svar_header, global_svar_s, global_svar_ints, global_svar_inners):
        for dir_proc_array_offset in dir_type_dict[DirectoryType.STATE_VAR_DICT.value]:
            f.seek(dir_proc_array_offset)
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

            while int_pos < len(ints):
                exists = False
                this_svar_s = []
                this_svar_ints = []
                ints_len = 0
                s_len = 0
                
                sv_name, title = s[c_pos], s[c_pos + 1]
                this_svar_s.extend(sv_name, title)

                agg_type, data_type = ints[int_pos], ints[int_pos + 1]
                this_svar_ints.extend(agg_type, data_type)

                if sv_name in global_svars_s:
                    exists = True
                #else:
                    # ?

                int_pos += 2
                c_pos += 2

                if agg_type == AggregateType.ARRAY.value:
                    order, dims = ints[int_pos], []
                    this_svar_ints.append(order)
                    int_pos += 1
                    for k in range(order):
                        dims.append(ints[int_pos])
                        int_pos += 1
                    this_svar_ints.extend(dims)
                    
                    if not exists:
                        global_svars_s[sv_name] = this_svar_s
                        global_svars_ints[sv_name] = this_svar_ints

                        # Add to sizes
                        this_svar_s = "".join(this_svar_s)
                        ints_len += len(this_svar_ints)
                        s_len += sys.getsizeof(this_svar_s)
                        
                        #global_svar_header[0] += ints_len
                        #global_svar_header[1] += s_len
                    # else:
                        # ?

                if agg_type == AggregateType.VECTOR.value or agg_type == AggregateType.VEC_ARRAY.value:
                    if agg_type == AggregateType.VEC_ARRAY.value:
                        order, dims = ints[int_pos], []
                        this_svar_ints.append(order)
                        int_pos += 1
                        for k in range(order):
                            dims.append(ints[int_pos])
                            int_pos += 1
                        this_svar_ints.extend(dims)

                        if not exists:
                            global_svars_s[sv_name] = this_svar_s
                            global_svars_ints[sv_name] = this_svar_ints

                            # Add to sizes
                            this_svar_s = "".join(this_svar_s)
                            ints_len += len(this_svar_ints)
                            s_len += sys.getsizeof(this_svar_s)

                            #global_svar_header[0] += ints_len
                            #global_svar_header[1] += s_len                        

                    svar_list_size = ints[int_pos]
                    #this_svar_ints.append(svar_list_size)
                    global_svar_ints.append(svar_list_size)
                    ints_len += 1
                    
                    int_pos += 1
                    sv_names = []
                    for j in range(svar_list_size):
                        # Is it possible that some lists of sv_names would be incomplete?
                        sv_names.append(s[c_pos])
                        c_pos += 1
                    #this_svar_s.extend(sv_names)
                    global_svar_s.extend(sv_names)
                    s_len += sys.getsizeof("".join(sv_names))
                    
                    for sv_name_inner in sv_names:
                        # If we are adding inner svars here, does that mean we might be adding incomplete entries?
                        
                        if sv_name not in global_svar_inners:
                            global_svar_inners[sv_name] = [sv_name_inner]
                        else:
                            if sv_name_inner not in global_svar_inners[sv_name]:
                                global_svar_inners[sv_name].append(sv_name_inner)
                            
                        if sv_name_inner in global_svar_s:
                            print("Do we need to make this a svar?")
                            #sv = self.__state_variables[sv_name_inner]
                        else:
                            sv_name_inner, title = s[c_pos], s[c_pos + 1]
                            agg_type, data_type = ints[int_pos], ints[int_pos + 1]
                            int_pos += 2
                            c_pos += 2

                            inner_s = [sv_name_inner, title]
                            inner_ints = [agg_type, data_type]

                            s_len += sys.getsizeof("".join(inner_s))
                            ints_len += len(inner_ints)

                            global_svar_s[sv_name].extend(inner_s)
                            global_svar_ints[sv_name].extend(inner_ints)

                # At the end of while loop add lens
                global_svar_header[0] += ints_len
                global_svar_header[1] += s_len

                
        
        
     '''
    Reads in directory information from the Mili file
    '''
    def ReadDirectories(self, f, offset, this_header_array, this_indexing_array, this_proc_dirs, dir_num_strings_dict, dir_type_dict, all_local_dirs):
        
        # Dir Byte Array Index Key
        # dir_index_key = {'type_idx' = 0, 'mod_idx1' = 1, 'mod_idx2' = 2, 'str_qty_idx' = 3, 'offset_idx' = 4, 'length_idx' = 5}

        local_dir_dict = dir_type_dict[this_dir_type][this_dir_offset] = dir_entry
        
        number_of_strings = 0
        directory_length = 4 * 6
        state_map_length = 20
        int_long = 'i'
        dir_version = this_header_array[2]
        if dir_version > 2:
            directory_length = 8 * 6
            int_long = 'q'

        num_state_maps = this_indexing_array[3]
        num_dirs = this_indexing_array[2]
        offset -= state_map_length * num_state_maps + directory_length * num_dirs  # make this not a hard coded 6 eventually
        f.seek(offset, os.SEEK_END)
        for i in range(1, 1 + num_dirs): # for num dirs
            byte_array = f.read(directory_length)

            # number of strings isn't in the byte array - create another var or just sum dir index ?
            this_byte_array = struct.unpack(self.__tag + '6' + int_long, byte_array)
            #this_proc_dirs = this_proc_dirs + this_byte_array

            # Dictionary to map offset to number of strings for that dir
            this_dir_offset = this_byte_array[4]
            dir_num_strings_dict[this_dir_offset][0] = number_of_strings # 1 - how to get short_name from directory

            # Dictionary to map type to list of dictionary offsets ?
            this_dir_type = this_byte_array[0]
            if this_dir_type not in dir_type_dict:
                dir_type_dict[this_dir_type][this_dir_offset] = this_byte_array
            else:
                dir_type_dict[this_dir_type][this_dir_offset].append(this_byte_array)
            this_dir_string_qty = this_byte_array[3]
            number_of_strings += this_dir_string_qty
    
        return offset

    def AddDirectory():
        # Have NestedDirectoriesDict[Dir_type][short_name] = 6 int dir entry
        # Update length_idx, string_idx
        # depending on type_idx change might need to change idx1 or idx2

        # Find type

        # Fill out entry array
        # # Calc size if needed (map back to what points to and calc size)

        # maybe just take all 6 parts of entry as args and create and add dir to final dir dictionary
        
        return 0

    def GetLabels(self, superclass, short_name): # label:id
        labels = self.this_proc_labels[(superclass, short_name)]
        return labels

    # M_MAT and M_MESH are same across all procs?
    def ReadMesh(f, this_proc_dirs, dir_type_dict, this_indexing_array, node_label_global_id_dict):
        # Index Key
        # CLASS_IDENTS: this_proc_mesh_index_key = {'superclass': 0, 'start': 1, 'stop': 2}

        # v1
        # dir_num_strings_dict[offset in this_proc_dirs] = (number of strings, strings)
        # this_proc_dirs[type_idx][local offset] = entry

        # v2
        # dir_dictionary[type_idx][short_name] = entry
        # USE ABOVE STRINGS DICT not dir_strings[type][short_name] = strings <-- how to access strings, actual strings written as combined list
        
                                    
        can_proc = [ DirectoryType.CLASS_DEF.value , 
                     DirectoryType.CLASS_IDENTS.value, 
                     DirectoryType.NODES.value, 
                     DirectoryType.ELEM_CONNS.value ]

        for type_id in can_proc:
            for dir_proc_array_offset in dir_type_dict[type_id]: # check dir_type_dict
                
                if type_id == DirectoryType.CLASS_DEF.value:
                    # mod_idx1 = mesh_id, mod_idx2 = superclass
                    #superclass = this_proc_dirs[dir_proc_array_offset-2] # should get the mod_idx2 of this directory

                    dir_strings = dir_num_strings_dict[dir_proc_array_offset][1]
                    short_name = dir_strings[0]

                    if short_name not in global_dir_dictionary[type_id]:
                        global_dir_dictionary[type_id][short_name] = this_proc_dirs[type_id][dir_proc_array_offset]
                    #else:
                        # what could need to be combined?

                    #if short_name not in global_mesh_dictionary[type_id]:
                    #    global_mesh_dictionary[type_id][short_name] = 

                if type_id == DirectoryType.CLASS_IDENTS.value:
                    f.seek(this_proc_dirs[type_id][dir_proc_array_offset])
                    dir_strings = dir_num_strings_dict[dir_proc_array_offset][1]
                    short_name = dir_strings[0]
                    
                    # Add superclass, start, stop
                    sup_class_start_stop =  = struct.unpack('3i', f.read(12))
                    superclass, start, stop = sup_class_start_stop

                    if short_name not in global_dir_dictionary[type_id]:
                        global_dir_dictionary[type_id][short_name] = this_proc_dirs[type_id][dir_proc_array_offset]
                    #else:
                        # Change mod_idx2 = class id count
                        
                    if short_name not in global_mesh_dictionary[type_id]:
                        global_mesh_dictionary[type_id][short_name] = sup_class_start_stop
                    #else:
                        # Do we need to remap the start,stop to global ids?

                    # Do we need superclass name or can we use number?
                    superclass = Superclass(superclass).name
                    if (superclass, short_name) not in this_proc_labels:
                        this_proc_labels[(superclass, short_name)] = {}
                        
                    id = len(this_proc_labels[(superclass, short_name)])
                    for label in range(start, stop + 1):
                        id += 1
                        this_proc_labels[(superclass, short_name)][label] = id

                    # 3 - Need to get local labels for this short_name? Or can we just get labels from start-stop?
                    # just create same structure that getLabels() returns? or change all?


                if type_id == DirectoryType.NODES.value:
                    f.seek(this_proc_dirs[type_id][dir_proc_array_offset])
                    dir_strings = dir_num_strings_dict[dir_proc_array_offset][1]
                    short_name = dir_strings[0]
                    
                    # Add start, stop
                    start, stop = struct.unpack('2i', f.read(8))                    

                    # 4 - need to do readParams for self.__dim
                    num_coordinates = this_proc_dim * (stop - start + 1)
                    floats = struct.unpack(str(num_coordinates) + 'f', f.read(4 * num_coordinates))
                    class_name = short_name

                    # Need to go through nodes and only add if not already there
                    # Get all dicts which has label:local_id dict using getLabels() - see #3 about whether or not to create same dict struct
                    #array_of_dicts = getLabels()
                    array_of_dicts = GetLabels()
                    node_label_local_id_dict = {}

                    # For label in labels - if not in new dict label:global_id then add to dict and add associated floats to NODES array
                    if ('M_NODE','node') not in array_of_dicts:
                        return 0
                    else:
                        node_label_local_id_dict = array_of_dicts[('M_NODE','node')]

                    if len(node_label_global_id_dict) < 1:
                        labels = list(node_label_local_id_dict)

                        for label in labels:
                            node_label_global_id_dict[label] = node_label_local_id_dict[label]
                        node_lobal_global_id_length = len(node_label_global_id_dict)

                        global_floats = floats # changes to global will change floats - bad?

                    elif len(node_label_global_id_dict) > 0:
                        # Remap
                        sorted_labels = node_label_global_id_dict.deepcopy()

                        floats_pos = 0 # keep track of local pos
                        id = len(sorted_labels) + 1
                        for label in node_label_local_id_dict:
                            if label not in node_label_global_id_dict:
                                sorted_labels[label] = id
                                id += 1

                                # Add floats
                                global_floats.extend(floats[floats_pos:floats_pos + self.dimension])
                                
                            # Increment floats_pos of local
                            floats_pos = floats_pos + dimension

                        ids_to_labels = {value:key for key,value in sorted_labels.items()}
                        sorted_ids_to_labels = collections.OrderedDict(sorted(ids_to_labels.items()))


                if type_id == DirectoryType.ELEM_CONNS.value:
                    # cannot combine this until have node mapping - skip these and have separate mesh routine for elem_conns
                    # or can put all of these together in a lpcal dictionary to read/collect and then combine all local w global dictionary
                    f.seek(this_proc_dirs[dir_proc_array_offset])
                    dir_strings = dir_num_strings_dict[dir_proc_array_offset][1]
                    short_name = directory.strings[0]

                    # this dict holds the local connectivities that we will combine later when we have node mapping
                    this_proc_connectivity[short_name] = {} # could it already have an entry with this short name?
                                    
                    superclass, qty_blocks = struct.unpack('2i', f.read(8))
                    elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))

                                        
                    if len(this_proc_mesh) == 0:
                        this_proc_mesh_offset_dict[short_name] = (len(this_proc_mesh), type_id)
                    elif len(this_proc_mesh) > 0:
                        this_proc_mesh_offset_dict[short_name] = (len(this_proc_mesh + 1), type_id)

                    this_proc_mesh = this_proc_mesh + superclass + qty_blocks + elem_blocks
                                    
                    #for j in range(0, len(elem_blocks), 2):
                    #    self.__mesh_object_class_datas[short_name].add_block(elem_blocks[j], elem_blocks[j + 1])

                    elem_qty = this_proc_dirs[dir_proc_array_offset - 2] # index for mod_idx2
                    word_qty = ConnWords[Superclass(superclass).name].value # still use these 
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
                            if short_name not in this_proc_materials[mat]: this_proc_materials[mat][short_name] = []
                            this_proc_materials[mat][short_name].append(mo_id)
                            mo_id += 1
                        index = word_qty * elem_qty

        ### IGNORE doesn't do node/float replacement in arrays                      
        def RemapIds():            
            array_of_dicts = GetLabels()
            classes = list(array_of_dicts)
            localmoid_label = {}
            for clas_pair in classes:
                sup_clas, clas = clas_pair
                if sup_clas not 'M_NODE' or 'M_MESH' or 'M_MAT':
                    if clas_pair not in global_labels:
                        global_labels[clas_pair] = array_of_dicts[clas_pair]
                    else:
                        for label_localmoid in array_of_dicts[clas_pair]:
                            label, mo_id = label_localmoid
                            if label not in global_labels[clas_pair]:
                                global_moid = len(global_labels[clas_pair]) + 1
                                global_labels[clas_pair][label] = global_moid

                if sup_clas is 'M_NODE':
                    localnodeid_label = {v: k for k,v in array_of_dicts[clas_pair].items()}
                    
                # Always add, when looping through classes
                localmoid_label[clas] = {v: k for k,v in array_of_dicts[clas_pair].items()}
                
            for short_name in this_proc_connectivity:
                for mo_id in this_proc_connectivity[short_name]:
                    # get global mo_id
                    label = localmoid_label[short_name][mo_id]
                    sup_clas = findSupClass() # this doesnt exist
                    global_moid = global_labels[(sup_clas, short_name)][label]

                    global_connectivity[short_name][global_moid] = []
                    # get global node ids to change connectivity id and add to global_labels if not there
                    for local_node_id in this_proc_connectivity[short_name][mo_id]:
                        # get label so can either add or get global node id
                        node_label = localnodeid_label[local_node_id]
                        global_node_id = len(global_labels[('M_NODE', 'node')]) + 1 # only use this if not already there, else find already existing global_id
                        if node_label not in global_labels[('M_NODE', 'node')]:
                            global_labels[('M_NODE', 'node')][node_label] = global_node_id
                        else:
                            global_node_id = global_labels[('M_NODE', 'node')][node_label]
                        
                        global_connectivity[short_name][global_moid].append(global_node_id)

                        
            
        # Remap local to global    
        def RemapConns(procnum_nodemoid_label_dict, global_unsorted_elem_labels, global_sorted_elem_labels, global_start_stop_blocks):
            # Call this after go through all the meshes bc need to have the complete new node mapping
            #### Keep track of local mapping for each proc from mo_id to label dict = {[proc_num]: {node_moid:label,...}, ...}
            # procnum_nodemoid_label_dict = {} # THIS SHOULD BE GLOBAL, keeps local mappings for each proc
            # global_unsorted_elem_labels = {}
            # global_sorted_elem_labels = {}
            # global_start_stop_blocks = {} # For start,stop in subrecords
            ####


            # START: elem connections combine, Create local array of dictionaries of elem label: local id
            # need to keep track of each element's neighbors (no overlap) and switch out local node ids for global node ids
            elems_label_local_id_dict = []
            array_of_dicts = getLabels()
            classes = list(array_of_dicts)
            for clas in classes:
                if clas[0] not 'M_NODE' or 'M_MESH' or 'M_MAT': # 0 index bc returns sup_class, class
                    elems_label_local_id_dict[clas[0]] = array_of_dicts[classes]

                    # We may need a struct for ebuf_dict[short_name] = new_ebuf
                    ebuf_dict = {}
                    # Dict for each part of ELEM_CONNS array (unpacked info) for each sup_class
                    # elem_conns_dict[short_name] = (sup_class, qty_blocks, elem_blocks (start,stop pairs))
                    elem_conns_dict = {}

                    # Create list of dictionaries that is local_id:elem_label or easier to calc later by finding index of a value and then getting index in dict.keys() ?
                    elems_local_id_label_rdict = []
                    # iterate through dictionaries
                    for dict in elems_label_local_id_dict:
                        # Get the class for the only entry that should be in each dict
                        elem_class = list(dict)
                        # Get the labels from the dict of dictionaries of label:id
                        elem_labels = list(dict(elem_class))
                        # Keep track of the start and stop for the block of global elem ids
                        start = -1
                        stop = -1
                        count = 1
                        for label in elem_labels:
                            # Add this label to local_id:label for this class
                            local_elem_id = dict[elem_class][label]
                            elem_local_id_label_rdict[elem_class][local_elem_id] = label

                            # Add this label to global[class]
                            if len(global_unsorted_elem_labels[elem_class][label]) < 1:
                                this_global_id = 1
                            else:
                                this_global_id = len(global_unsorted_elem_labels[elem_class][label]) + 1

                            # Assign global id to this label in global array
                            global_unsorted_elem_labels[elem_class][label] = this_global_id

                            # Make this start/stop block
                            if count is 1:
                                start = this_global_id
                            elif count is len(elem_labels):
                                stop = this_global_id

                                    
                        if elem_class in global_start_stop_blocks:
                            global_start_stop_blocks[elem_class].append(tuple((start,stop))) # Can append as a tuple or just know that they will alternate as start/stop and qty_blocks will = len/2
                        else:
                            global_start_stop_blocks[elem_class] = tuple((start,stop))

                        global_sorted_elem_labels[elem_class] = global_unsorted_elem_labels[elem_class]
                        global_sorted_elem_labels[elem_class].sort()


                    local_elem_labels = list(elems_label_local_id_dict) # labels
                    global_unsorted_elem_labels = global_unsorted_elem_labels + local_elem_labels

                    #@ this (no gaps between the labels we have - labels may be non consecutive)
                    global_sorted_elem_labels = global_unsorted_elem_labels                            
                    global_sorted_elem_labels.sort() ##### Q: Should add and sort every time OR make array the size of greatest number label and add to the end if we come across a number larger than the current largest?


                    local_label_values = elems_label_local_id_dict.values() # local ids

                    for i, value in enumerate(local_label_values):
                        node_local_id_label_rdict[value] = local_label_keys[i]
                        #@ or this (leave space for labels when they are non consecutive)
                        if local_label_keys[i] > len(global_sorted_labels):
                            difference = local_label_keys[i] - len(global_sorted_labels)
                            for i in difference:
                                global_sorted_labels.append(-1) # create place for possible insertion of global id for label at that index

                    procnum_nodemoid_label_dict[self.__mili_num] = node_local_id_label_rdict

                    ## To integrate / Assume:
                    ## Loop through directories again to get ELEM_CONNS directory offsets
                    ## For ELEM_CONNS directory in directories:            
                    elem_conns_offset_from_dir = this_proc_dict_array[dictionary_offset + number to get to data offset]
                    f.seek(elem_conns_offset_from_dir)
                    short_name = directory.strings[0] # ! Fix so get short_name from associated strings array
                    ##

                    #### 3) Array to replace ELEM_CONNS array - (superclass, qty_blocks, elem_blocks (start,stop), ebuf (node,mat,part))
                    if short_name not in elem_conns_label_global_id_dict:
                        elem_conns_label_global_id_dict[short_name] = {} # CHECK if short_name already exists
                    #elem_conns_label_global_id_length = 0
                    sup_class, qty_blocks = struct.unpack('2i', f.read(8))
                    local_elem_blocks = struct.unpack(str(2 * qty_blocks) + 'i', f.read(8 * qty_blocks))

                    # ! Need to calc these things from arrays:
                    elem_qty = directory.modifier_idx2
                    word_qty = ConnWords[Superclass(superclass).name].value
                    conn_qty = word_qty - 2
                    mat_offset = word_qty - 1

                    ebuf = struct.unpack(str(elem_qty * word_qty) + 'i', f.read(elem_qty * word_qty * 4))

                    new_elem_conns_array = [sup_class, qty_blocks, -1 (=start), -1 (=stop)] # ! I think start,stop be the same as in local_elem_blocks ?
                    new_ebuf = [] # easier to have separate array to add to new_elem_conns_array afterwards

                    # Go through elem_blocks to get chunk of ebuf we need
                    # Go through ELEM_CONNS and use label:local_id from getLabels() again
                    index = 0
                    for block_num in qty_blocks:
                        start = local_elem_blocks[block_num * 2]
                        stop = local_elem_blocks[block_num * 2 + 1]
                        this_block_ebuf = ebuf[start * word_qty?:(stop * word_qty?) +1] # ! Check how to iterate through ebuf

                        # If first proc then can copy without checking
                        # if len(elem_conns_label_global_id_dict) < 1:
                        # elem_conns_label_global_id_dict[short_name] = array_of_dicts[short_name]

                        for k in range(index, len(ebuf), word_qty):
                            mat = ebuf[k + conn_qty]
                            part = ebuf[k + mat_offset]
                            for m in range(0, conn_qty):
                                # Get the local node 
                                local_node_id = ebuf[k + m]

                                # Remap elem_label:local_id --> elem_label:global_id
                                mo_id = len(elem_conns_label_global_id_dict[short_name]) + 1
                                # Get label from local_node_id
                                node_label = node_local_id_label_rdict[local_node_id]
                                # Get global_node_id from node_label - ! Add check to make sure node_label_local_id_dict and node_local_id_label_dict exist... otherwise need to find
                                global_node_id = node_label_global_id_dict[node_label]

                                # Update Dictionary
                                elem_conns_label_global_id_dict[short_name][mo_id].append(global_node_id) # mo_id = elem mo_id

                                # Update global_sorted_labels

                                # Update new_ebuf (node,mat,part)
                                new_ebuf.append(global_node_id)

                            # After all conns add mat and part
                            new_ebuf.extend(mat, part)

                            # What to do w these
                            #if short_name not in self.__materials[mat]: self.__materials[mat][short_name] = []
                            #self.__materials[mat][short_name].append(mo_id)

                            mo_id += 1
                        index = word_qty * elem_qty

                    # For node in ebuf - node=local_id - use local_id to get label - use label to get global_id - replace node with global_id in new ebuf - mat and part stay the same

                    # Append/add new_ebuf to ebuf_dict[short_name]
                    if short_name in ebuf_dict:
                        ebuf_dict[short_name].append(new_ebuf)

                        # Also need to update elem_conns_dict info
                        # Q:? How do add/change elem_blocks
                        elem_conns_dict elem_blocks stop += number of new nodes added?
                    elif short_name not in ebuf_dict:
                        ebuf_dict[short_name] = new_ebuf

                        # And create short_name in elem_conns_dict
                        elem_conns_dict[short_name] = [sup_class, qty_blocks, local_elem_blocks] # ?





    # Problem: change so that srec header leads subrecord groups so don't have to go through all during commit
    def ReadSubrecords(elem_local_id_to_label, label_to_global_elem_id, global_srec_headers, global_subrec_idata, global_srec_cdata, global_srecs_size):
        # this_proc_subrec_index = {'srec_qty_subrecs': 3}
        # global_subrec_idata_dict[class_name] = {'org': 0, 'qty_svars': 1, 'qty_id_blks': 2, 'start': 3, 'stop': 4}
        # global_subrec_cdata_dict[class_name] = {'name': 0, 'class_name': 1, 'svars': 2}

                                    
        # Name matches
        type_id = DirectoryType.STATE_REC_DATA.value
        for dir_proc_array_offset in dir_type_dict[type_id]:
            #srec_int_data = this_proc_dirs[(dir_proc_array_offset - 3) - 4]  # mod_idx - 4 = (offset_idx index - 3) to get to mod_idx1 index, then - 4
            #srec_c_data = this_proc_dirs[(dir_proc_array_offset - 2)]
            f.seek(dir_proc_array_offset)

            this_srec_array = struct.unpack('4i', f.read(16))                          
            _,_,_, srec_qty_subrecs = this_srec_array
                                    
            idata = struct.unpack(str(srec_int_data) + 'i', f.read(srec_int_data * 4))
            #srec_idata = srec_idata + idata
                                    
            if (sys.version_info > (3, 0)):
                cdata = str(struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0])[2:].split('\\x00')  # only works for Python3
            else:
                cdata = struct.unpack(str(srec_c_data) + 's', f.read(srec_c_data))[0].split(b'\x00')
            #srec_cdata = srec_cdata + cdata
    
            int_pos = 0
            c_pos = 0

            for subrec_num in range(srec_qty_subrecs):
                # Check if srec name already exists
                # Would need to combined strings, svars
                # mod_idx1 = count of all integer data contained in all subrecs + surface flag
                # mod_idx2 = num characters to read of the svar names that are associated with this srec


                org, qty_svars, qty_id_blks = idata[int_pos], idata[int_pos + 1], idata[int_pos + 2]

                int_pos += 3
                name, class_name = cdata[c_pos:c_pos + 2]
                c_pos += 2
                svars = cdata[c_pos:c_pos + qty_svars]
                c_pos += qty_svars

                if class_name not in global_subrec_idata_dict:
                    # If doesn't exist in dict
                    global_subrec_headers[class_name] = this_srec_array
                    global_subrec_cdata[class_name] = cdata

                    # Change local ids to global ids
                    # Could this change the number of blocks / the global ids end up being split and no longer contiguous
                    global_subrec_idata_dict[class_name] = idata
                    
                #class_name_keys = list(global_subrec_idata_dict)
                elif class_name in global_subrec_idata_dict:
                    # For the number of global blocks:
                    qty_global_id_blocks = global_subrec_idata_dict[class_name][SubrecIdata.QTY_ID_BLKS]

                    start, stop = idata[int_pos], idata[int_pos+1]
                    int_pos += 2
       
                    # Global cdata array - to get svars get from index 2 to end
                    this_subrec_global_cdata = global_subrec_cdata_dict[class_name]
                    this_subrec_global_svars = this_subrec_global_cdata[SubrecCdata.SVARS:]

                    # Check that num svars matches
                    expected_num_svars = this_subrec_global_cdata[SubrecIdata.QTY_SVARS]
                    if expected_num_svars is not len(this_subrec_global_svars):
                        print("Num svar mismatch")

                    # Global idata array
                    this_subrec_global_idata = global_subrec_idata_dict[class_name]

                    combined = False # combine blocks
                    for block_num in range(qty_global_id_blocks):
                        # change start and stop if needed
                        if block_num > 1:
                            start, stop = idata[int_pos], idata[int_pos+1]
                            int_pos += 2                            
                                        
                        for svar in svars:      
                            if svar not in this_subrec_global_svars:
                                this_subrec_global_svars.append(svar)
                                global_subrec_cdata_dict[class_name].append(svar)
                                global_subrec_cdata_dict[class_name][qty_svars_index] += 1

                        # Get current global (from global array) start and stop ids
                        curr_start = this_subrec_global_idata[SubrecIdata.START + 2*(block_num-1)]
                        curr_stop = this_subrec_global_idata[SubrecIdata.STOP + 2*(block_num-1)]
                                        
                        # Change ids to global so can compare (still local group)
                        start_stop = [e for e in range(start,stop+1)]
                        for elem in start_stop:
                            local_id = elem
                            label = elem_local_id_to_label[class_name][local_id]
                            global_id = label_to_global_elem_id[class_name][label]
                            start_stop[elem] = global_id

                        # This might not work
                        # Go through and see if there is overlap between starts and stops?
                        if start_stop[0] <= (curr_stop+1) and start_stop[0] >= curr_start:
                            # Combine because the blocks are touching
                            curr_stop = start_stop[-1]
                            combined = True
                        elif start_stop[-1] >= (curr_start-1) and start_stop[-1] <= curr_stop:
                            # Combine
                            curr_start = start_stop[0]
                            combined = True

                    if not combined:
                        # Add block and +1 for qty_blocks
                        global_subrec_idata_dict[class_name].extend(start,stop)
                        global_subrec_idata_dict[class_name][2] += 1                                           

        
    def CalcSrecSizes(global_subrec_cdata_dict, global_subrec_idata_dict, global_modidx2_count, global_dir_dict):
            # Create one global STATE_REC_DIR even if size too big bc can change size later?

            # How is Xmilics calculating the size, why can't we just calc bytes do we need to know what the data is?
            # for subrecs
                # add idata to collective idata array
                # add cdata to collective cdata array
                # add sizes of each to separate idata and cdata counts
                                
            # 5 - should wait to calculate sizes after everything has been combined to avoid repeats? 
            subrecs = list(global_subrec_cdata_dict) # idata and cdata dicts should have the same keys                
            for subrec in subrecs:
                this_srec_idata = global_subrec_idata_dict[subrec]
                this_srec_cdata = global_subrec_cdata_dict[subrec]                                                             
                this_srec_svars = this_srec_cdata[2:]
                                                                              
                for svar in this_srec_svars:
                    svar_int_pos = 0
                    svar_s_pos = 0
                    svar_agg_type = global_svar_ints_array[svar][svar_int_pos]
                    svar_int_pos += 1
                    if svar_agg_type is VECTOR:
                        svar_order = globa_svar_ints_array[svar][2] # order = index 2
                        svar_list_size = global_svar_ints_array[svar][svar_order + 3] # index of list_size but check this
                        sv_sv_list = global_svar_s_array[svar][2:2+svar_list_size]
                        for sv_sv in sv_sv_list:
                            if sv_sv not in this_srec_svars:
                                this_srec_svars.append(sv_sv)
                    elif svar_agg_type is VEC_ARRAY:
                        #same as above?

                org = this_srec_idata[0]
                sub_size = 0 
                if org is RESULT_TYPE:
                    qty_id_blocks = this_srec_idata[2]
                    for block_num in qty_id_blocks:
                        # Get the (start stop) pair
                        start = this_srec_idata[3 + 2*block_num]
                        stop = this_srec_idata[4 + 2*block_num]
                        mo_qty = stop-start + 1

                        for sv in this_srec_svars:
                            ints = global_svar_ints_array[sv] # need to check if exists
                            s = global_svar_s_array[sv]
                            atoms = atom_qty(ints, s) # fix atom_qty
                            size = ExtSize[DataType(ints[1]).name].value # is data type an int or string, data_type index

                            lump_atoms.append(mo_qty * atoms)
                            lump_sizes.append(mo_qty * atoms * size)

                        lump_offsets.append(0)
                        for j in range(1,this_srec_idata[1]): # qty_svars index
                            lump_offsets.append(lump_offsets[j-i] + lump_sizes[j-1])

                        j = this_srec_idata[1] - 1 # qty_svars index
                        sub_size = lump_offsets[j] + lump_sizes[j]

                elif org is OBJECT_TYPE:
                    qty_id_blocks = this_srec_idata[2]
                    for block_num in qty_id_blocks:
                        # Get the (start stop) pair
                        start = this_srec_idata[3 + 2*block_num]
                        stop = this_srec_idata[4 + 2*block_num]
                        mo_qty = stop-start + 1

                        for sv in this_srec_svars:
                            ints = global_svar_ints_array[sv]
                            s = global_svar_s_array[sv]
                            atoms = atom_qty(ints, s)
                            size = ExtSize[DataType(ints[1]).name].value # is data type an int or string, data_type index

                            total_atoms = atoms
                            count += total_atoms
                            sz += total_atoms * size

                        lump_atoms.append(count)
                        lump_sizes.append(sz)

                        # see note about directories at top
                        srec_name = this_srec_cdata[0]
                        directory = global_dir_dict[srec_name]
                        dir_type = directory[0]
                        #superclass = global_directory_strings[dir_type][
                        if superclass is not Superclass.M_SURFACE.value:
                            sub_size = sz * mo_qty
                        else:
                            sub_size = sz

                sub_offset = global_srecs_size
                global_srecs_size += sub_size # give size to commit

                                                                              
                            

    ####### Commits #######

    # struct.pack()
    # write()
    # flush()


    def CommitNonState():
        # Instead of commit non state calling other commit functions like xmilics, have it do some prep things and then other commits called later so this doesn't need to have all the arrays
        # filename/create file
        # file count
        # write header
        # bit - maxsize                                                                          


    def CommitSvars(filename, curr_index, sizes, svar_header, svar_ints, svar_s):
        # Assume we are calling this bc we already have info - not checking for more than header
        # Need to have size of data
        # 1 - svar_hdr ( qty int words idx, qty bytes idx), svar_words, svar_bytes
        # 2 - add directory entry?
        # 3 - write out svar_i_ios then svar_c_ios

        # Need to check if svar_bytes is larger than single file size ?
        # svar_bytes = svar_header[1]
        # if svar_bytes > FILE_MAX

        # Go through global svar dictionary
        # get the class_name to access directory
        # calc what offset will be
        # update directory
        # add svar ints and s to long arrays before writing all out

        svar_byte_len = svar_header[0] + svar_header[1]
        AddDirectory(SVAR, _, _, str_qty_idx, curr_index, svar_byte_len)

        with open(filename, 'a') as a_file:
            # append becuse will not be first write to file
            a_file.seek(curr_index)

            a_file.write(svar_header)

            svar_ints = struct.pack('ints_size i', *svar_ints) # * to unpack not a pointer
            a_file.write(svar_ints)

            svar_s = struct.pack('s_size s', *svar_s) # ('[endian < or >] num_items type') use * to unpack svar_s
            a_file.write(svar_s)

        # update directory = [type_idx  mod_idx1  mod_idx2  string_qty  offset  length]
        # update curr_index - maybe return curr_index ?

        return OK


    def CommitSrecs(filename, curr_index, srec_idata, srec_cdata, global_srecs_size): # find how fam.qty_meshes calculated
        # commit_max
        # qty_srecs can come from global list of srec names

        # Problem: State rec header = srec_id, mesh_id, size, qty_subrecs ( part of idata so add in combine? ) + for sizes have dictionary instead of single ciuong for global srec size
        #          state rec id = state_maps[3]
        #          Each state record has specific idata, cdata that comes after header
        #              Redo srecs combined, need to be separated
        #              state_record_idata_dict[state_rec ?] = state_rec_header, idata for each subrec
        #              state_record_cdata_dict[state_rec ?] = cdata for each subrec including svar names,

        # When going through srecs in ReadSrecs() - take note of file # so we can find state map id?

        # Do we need to check sizes to see if need more than one directory bc might be on multiple files?

        # mod_idx1 = count integer data + usrface flag
        # mod_idx2 = # characters to read of the state variables assoc w this srec ? calculated in CalcSrecSizes()

        all_i_size, all_c_size, total_size = CalcSrecSizes(global_srec_cdata, global_srec_idata)
        
        all_srec_arrays = []
        formatting_string = None
        for subrec_name in srec_idata:
            # i_size = get size idata
            # c_size = get size cdata
            formatting_string.append(i_size + 'i' + c_size + 'c')
            all_srec_arrays.extend(srec_idata[subrec_name], srec_cdata[subrec_name])

        # Create srec_header 

        with open(filename, 'a') as a_file:
            a_file.seek(curr_index)
            
            # pack and write state rec header
            header_item_size = ExtSize[DataType(srec_header[0]).name].value
            struct.pack(len(srec_header) + 'i', srec_header)

            # in loop add idata,cdata for each subrec OR create long array in loop above and write in one 
            srec_idata = struct.pack('_ idata_size i', *srec_idata)

            srec_cdata = struct.pack('_ cdata_size s', *srec_cdata)
            a_file.write(all_srec_arrays)

        AddDirectory(SREC, mod_idx1, mod_idx2, str_qty, offset, srec_byte_len)
                
        # update curr_index

        return OK


    def CommitDir(filename, curr_index, sizes, dir_header, names, dirs):
        # We will have a global dictionary of all the directories, but should the directories we commit be condensed down as much as possible?
        # dir_header = names_len, commit_count, qty_entries, qty_states

        # directory entries in dictionary
        # loop through all dirs and add to single long array

        # write 1. names...name_data?, 2. entries, 3. header
        with open(filename, 'a') as a_file:
            a_file.seek(curr_index)

            a_file.write(names)

            a_file.write(dirs)

            a_file.write(dir_header)

        # update curr_index

        return OK







    class MiliFamily:
        # Set to None, [], {}, or should actually be init     

        def __init__(self):
            self.pid = None # getpid()
            self.my_id = None # family id
            self.root = None # family root string
            self.root_len = None
            self.path = None# path to directory where family is/will be
            self.file_root = None
            self.aFile = None # filename ?
            self.access_mode = None
            self.num_procs = None
            #self.post_modified = post_modified
            #self.visit_file_on = visit_file_on
            self.db_type = None # Database_type
            #self.lock_file_descriptor = lock_file_descriptor # lockf() ?
            #self.st_suffix_width = st_suffix_width
            self.non_state_ready = None # Bool_type
            #self.hide_states = hide_states # Bool_type
            self.char_header = None # DIR_VERSION_IDX, ENDIAN_IDX, HDR, ST_FILE
            #self.swap_bytes = swap_bytes # swap_bytes()
            #self.precision_limit = precision_limit # Precision_limit_type, set by read_header()
            #self.partition_scheme = partition_scheme
            #self.states_per_file = states_per_file # qty of states per family state_data file
            #self.bytes_per_file_limit = bytes_per_file_limit # compare to bytes_per_file_limit
            #self.active_family = active_family # Bool_type
            #self.written_st_qty = written_st_qty
            #self.commit_count = commit_count # p_fd
            #self.cur_file = cur_file # FILE*
            #self.cur_file_size = cur_file_size
            #self.cur_index = cur_index # current family id index
            #self.file_count = file_count # call prep_for_new_data() to update - is this the number of files it takes to hold all info?
            self.next_free_byte = None # dependent on cur_file_size
            self.cur_st_file = None # FILE*, seek_state_file()
            self.cur_st_file_mode = None
            self.cur_st_file_size = None # should not be larger than fam.bytes_per_file_limit, ABSOLUTE_MAX_FILE_SIZE
            #self.cur_st_index = cur_st_index # ST_FILE_SUFFIX()
            #self.st_file_count = st_file_count # file count in active db, includes partial files
            #self.st_file_index_offset = st_file_index_offset
            #self.cur_st_offset = cur_st_offset
            #self.file_st_qty = file_st_qty # compare to fam.states_per_file
            self.cur_srec_id = None
            self.state_qty = None
            self.state_closed = None
            self.state_dirty = None
            self.file_map = None # State_file_descriptor*, has state_qtys
            self.state_map = None # State_descriptor*
            # Directory data
            self.directory = None # File_dir*
            # Parameter data
            self.param_table = None # Hash_table*
            # Mesh data
            self.mesh_type = None # Mesh_type*
            self.dimensions = None
            self.meshes = None # Mesh_descriptor**
            self.qty_meshes = None
            # State record descriptor data
            self.srecs = None # Srec**
            self.srec_meshes = None # Mesh_descriptor**
            self.qty_srecs = None
            self.commit_max = None
            # State variable table and I/O stores
            self.svar_table = None # Hash_table*
            self.svar_c_ios = None # IO_mem_store*
            self.svar_i_ios = None # IO_mem_store*
            self.svar_hdr = None # int*
            # Subrecord table
            self.subrec_table = None # Hash_table*

            # I/O routines
            #def readFuncs():
            #    return int
            #def state_read_funcs():
            #    return int
            #def write_funcs():
            #    return int
            #def state_write_funcs():
            #    return int





    class StateFileDescriptor():
        def __init__(self):
            self.state_qty = None

    class StateDescriptor():
        def __init__(self):
            self.file = None # int
            self.offset = None # LONGLONG
            self.time = None # float
            self.srec_format = None # int

    class IO_mem_buffer():
        def __init__(self):
            self.data = []
            self.used = None # size_t
            self.output = None # size_t
            self.invalid = [] # Block_list

    class IO_mem_store():
        def __init__(self):
            self.IO_mem_buffer = None
            self.type = None # int
            self.current_index = None # size_t
            self.current_output_index = None # size_t
            self.traverse_index = None # size_t
            self.traverse_remain = None # size_t
            self.traverse_next = None # string?

    class File_dir():
        def __init__(self):
            self.commit_count = None # int
            self.qty_entries = None # int
            self.dir_entries = None # Dir_entry 
            self.qty_names = None # int
            self.names = [] # char**
            self.name_data = None # IO_mem_store

    class MeshType(Enum):
        UNDEFINED = 0
        #UNSTRUCTURED

    class MeshDescriptor():
        def __init__(self):
            self.name = None # string?

    class Srec():
        def __init__(self):
            self.qty_subrecs = None # int
            self.subrecs = None # Sub_srec**
            self.size = None # LONGLONG
            self.status = None # Db_object_status

    class BlockList():
        def __init__(self):
            self.object_qty = None # int
            self.block_qty = None # int
            self.blocks = [] # Int_range

    class IntRange():
        def __init__(self):
            self.prev = None # IntRange
            self.next = None # IntRange
            self.start = None # int
            self.stop = None # int

    class DirEntry():
        def __init__(self):
            self.entry = None # 6 numbers so can be array of 6 ints?
    
    ############ Create the database / create_family() ##############

    def CreateMiliFamily(fam):

        fam.st_suffix_width = DEFAULT_SUFFIX_WIDTH
        fam.partition_scheme = DEFAULT_PARTITION_SCHEME
        fam.states_per_file  = 0
        fam.bytes_per_file_limit = 0
        fam.post_modified = 0

        fam.st_file_count = 0
        fam.file_count = 0
        fam.cur_index = -1
        fam.commit_max = -1

        fam.ti_cur_index = -1
        fam.ti_file_count = 0

        # Create first file and its directory - non_state_file_open()
        NonStateFileOpen()

        # If fam.param_table is not None, init_header(fam) and mc_init_metadata(fam.my_id)
        # Q: what are the hash tables created for?

        fam.file_count += 1
        InitHeader()

        mcInitMetadata()

        return OK


    def NonStateFileOpen(fam, index, mode):
        with open(fname, 'w+') as p_file:


        fam.next_free_byte = fam.cur_file_size

        # Set current file index and manage directory
        fam.cur_index = index

        fam.directory = RENEW_N # Q: figure out renew(file_dir, fam.dir, fcount, 1, "file dir head")



    def InitHeader(fam):
        # fam.char_header gets parts from attributes of fam, Q: is header a struct?    

        # fam.char_header[HDR_VERSION_IDX], fam.char_header[DIR_VERSION_IDX]

        # fam.swap_bytes, set fam.char_header[ENDIAN_IDX], fam.
        return OK

    def WriteHeader(fam):
        # Write character header to A file    

        # Open file w read/write access
        NonStateFileOpen(fam, 0, 'a')

        # Check for read/write access()

        write()

        NonStateFileClose(fam)    



    def DefSvars(fam, qty, names, svar_ints, svar_s):
        # see where names and qty come from
        # should have complete list of names and titles

        # Go though the names...what are titles?
        for name in names:
            # valid_svar_data() --> check to see if svar?

            # add name to family names
            fam

            # add title?

            # add agg_type

            # add data_type
            fam.svar_c_ios
            fam.svar_i_ios

        return OK    


    ############## mc_flush() ################       

    def CommitNonState(fam):
        p_fname = []
        fd = 0
        rval = 0 # ReturnValue type
        write_cnt = 0

        rval = PrepForNewData(fam, NON_STATE_DATA) # NON_STATE_DATA is constant?

        if rval is not OK: # OK is constant?
            return rval

        if(fseek(fam.cur_file, 0L, SEEK_END) is not 0): # need to get current file for seek
            return -1

        if(fam.svar_table is not NULL):
            rval = CommitSvars(fam)
            if rval is not OK:
                return rval

        if(fam.qty_srecs is not 0):
            rval = CommitSrecs(fam)
            if rval is not OK:
                return rval


        rval = CommitDir(fam)
        if rval is not OK:
            return rval


        fd = fam.lock_file_descriptor

        p_fname = NEW_N(char, M_MAX_NAME_LEN, "Lock file NS name buffer")
        if p_fname:
            return ALLOC_FAILED

        MakeFnam(NON_STATE_DATA, fam, fam.file_count - 1, p_fname)

        rval = GetNameLock(fam, NON_STATE_DATA)
        #if rval is not OK:

        if filelock_enable:
            write_ct = write(fd, p_fname, M_MAX_NAME_LEN)
            if write_ct is not M_MAX_NAME_LEN:
                return SHORT_WRITE


        # LOCK

        fam.commit_count += 1
        fam.non_state_ready = FALSE

        return rval


    def PrepForNewData(MiliFamily fam, int ftype): # types for clarification, need to change
        amode = fam.access_mode
        index = 0
        rval = -1
        #open_next

        return OK



