#!/usr/bin/env python3
import sys
import os


parameters = ["-o","-c","-start","-stop","-append","-proc","-procs","-debug"]

def notParameter(p):
    return (not p in parameters)

def isParameter(p):
    return (p in parameter)
    

def usage(message):
    if message == None:
        print("HELP:")
    else:
        print(f"\nERROR: {message} \n")

    print("""
USAGE:
    para_mrl.py -i <input_base_name> [-o <output_base_name>]

OPTIONS:
    
    [-o] <output file name>
    ###   Create output file using this name              ###
    ###   default output file name: {input_base_name}_c   ###

    [-start] <state to begin at>
    ###   Limit states per file with non-zero           ###
    ###   starting state. When appending this is set    ###
    ###   to the last time of output file if starting   ###
    ###   state is than the end state of the            ###
    ###   output database.                              ###

    [-stop] <state to end at>
    ###   Limit states per file with an ending state   ###
    ###   This is ignored if less than the last        ###
    ###   state in the output file when appending      ###

    [-append] <state to end at>
    ###   Append a timestep to an existing database.    ###
    ###   This is overridden if a start or stop state   ###
    ###   are defined.                                  ###
    
    [-proc] <processor list to combine>
    ###   Single Processor Format = -proc 10            ###
    ###   OR Tuple Format (NO SPACES!) = -proc 1-5,10   ###
    """)


def parseParameters(params):
    """Parses command line arguments.
    
    Parameters:
        params (str): list of command line arguments (sys.argv[1:])

    Returns:
        dict: dictionary containing all arguments for combining
    """
    if "-help" in params or "-h" in params or "--help" in params:
        usage(None)
        sys.exit(0)
    if len(params) < 2:
        usage("Too few arguments")
        sys.exit(1)
    
    input_file_name = ""
    directory_name = None
    output_file_name = ""
    debug = False

    procs = []
    pad_count = 0
    start_state = None
    stop_state = None
    start_stop = False
    append = False
    append_arg = None
    combine = False

    length = len(params)
    i = 0
    while i < length:

        parameter = params[i]

        if parameter == "-i":
            if i+1 < length and notParameter(params[i+1]):
                input_file_name = params[i+1]
                i = i+1
                pad_count = find_pad_count(input_file_name)

                # Find directory of input_file_name
                loc = input_file_name.rfind("/") + 1
                if loc != -1: 
                    directory_name = input_file_name[:loc-1]
                    input_file_name = input_file_name[loc:]
                else:
                    directory_name = "."
            else:
                usage("-i flag given, but no input file specified.")
                return None

        elif parameter == "-c":
            combine = True
            i = i+1
        elif parameter == "-o":
            if i+1 < length and notParameter(params[i+1]):
                output_file_name = params[i+1]
                i = i+1
            else:
                usage("-o flag given, but no output file specified.")
                return None

        elif parameter == "-start":
            if i+1 < length and notParameter(params[i+1]):
                start_state = params[i+1]
                start_stop = True

                if not start_state.isdigit():
                    usage(f"The given start state is incorrect {start_state}")
                    return None

                if start_state < 0:
                    start_state = 1

                i = i+1
            else:
                usage("-start flag given, but no start state specified.")
                return None

        elif parameter == "-stop":
            if i+1 < length and notParameter(params[i+1]):
                stop_state = params[i+1]
                start_stop = True

                if not stop_state.isdigit():
                    usage(f"The given stop state is incorrect {stop_state}")
                    return None

                if stop_state < 0:
                    stop_state = 0

                i = i+1
            else:
                usage("-stop flag given, but no stop state specified.")
                return None

        elif parameter == "-append" and not start_stop:
            if i+1 < length and notParameter(params[i+1]):
                append = True
                stop_state = params[i+1]
                if not stop_state.isdigit():
                    usage(f"The given stop state is incorrect {stop_state}")
                    return None

                if stop_state < 0:
                    stop_state = 0

                start_state = stop_state
                i = i+1

        elif parameter == "-proc" or parameter == "-procs":
            if i+1 < length and notParameter(params[i+1]):
                processor_list = params[i+1]
                processor_list = processor_list.split(",")
                for processor in processor_list:
                    if "-" in processor:
                        start_proc, end_proc = processor.split("-")                    
                        if start_proc.isdigit() and end_proc.isdigit():
                            
                            for p in range(int(start_proc),int(end_proc)+1):
                                procs.append(p)
                        else:
                            usage(f"Invalid processor specified: {processor}") 
                            return None
                    else:
                        if processor.isdigit():
                            procs.append(int(processor))
                        else:
                            usage(f"Invalid processor specified: {processor}") 
                            return None
                i = i+1
            else:
                usage("-proc flag given, but no processor list specified.")
                return None

        elif parameter == "-debug":
            debug = True

        else:
            # Invalid parameter
            usage(f"Invalid parameter {parameter}")
            return None

        i = i+1

    # If no output file is specified, use input file name with "_c" appended    
    if output_file_name == "":
        loc = input_file_name.rfind("/") + 1
        if loc != -1: 
            output_file_name = input_file_name[loc:]
        else:
            output_file_name = input_file_name
        output_file_name = output_file_name + "_c"

    # Package parameters into dictionary
    params = {
        "input_file"  : input_file_name,
        "input_dir"   : directory_name,
        "pad_count"   : pad_count,
        "output_file" : output_file_name,
        "start_state" : start_state,
        "stop_state"  : stop_state,
        "append"      : append,
        "processors"  : procs,
        "debug"       : debug,
        "combine"     : combine
    }

    return params


def find_pad_count(input_file_name):
    """Finds the count of integers to fill the file name.
        
    Parameters:
        input_file_name (str): name of input file

    Returns:
        pad_count (int): Number of 0s padding "A" file names
    """
    pad_count = 0
    directory_name = None

    # Find directory of input_file_name
    loc = input_file_name.rfind("/") + 1
    if loc != -1: 
        directory_name = input_file_name[:loc]
        input_file_name = input_file_name[loc:]
    else:
        directory_name = "."

    files = os.listdir(directory_name)

    for f in files:
        if not os.path.isdir(f):
            if f.startswith(input_file_name) and f[-1] == 'A' and f[-2] != '_':
                pos = len(input_file_name)
                filename_length = len(f)
                while f[pos].isdigit() and pos < filename_length-1:
                    pad_count = pad_count+1
                    pos = pos+1
                if pad_count > 0 and f[pos] == 'A':
                    break

    return pad_count
            

def gather_A_files(params):
    """Gathers the names of all "A" files in the directory of input_file_name.

    Parameters:
        params (dict): Dictionary containing parameters for combining

    Returns:
        A_files (list): List of all "A" files in directory for the parameter
            input_file_name
    """
    input_file_name = params["input_file"]
    ifn_len = len(input_file_name)
    directory_name = params["input_dir"]
    A_files = []

    # Scan directory for A files
    try:
        with os.scandir(directory_name) as directory:
            for entry in directory:
                if entry.is_file():
                    if entry.name.startswith(input_file_name)                     \
                            and entry.name[-1] == 'A'                             \
                            and (entry.name[-2] >= '0' and entry.name[-2] <= '9') \
                            and (entry.name[-2] != '_')                           \
                            and (entry.name[ifn_len] != '_' and entry.name[ifn_len+1] != 'c'):
                            
                        # Check that all characters in entry after the length of the
                        # input_file_name have the form input_file_name[0-9]*A
                        found_A = False

                        for i in range(ifn_len, len(entry.name)):
                            if entry.name[i] == 'A' and i == len(entry.name)-1:
                                found_A = True
                                continue

                            if entry.name[i] < '0' or entry.name[i] > '9':
                                break

                        if found_A:
                            A_files.append(entry.name)
    except OSError:
        A_files = []
        return A_files
    
    # Loop over found A files and prepend directory path
    if directory_name is not None and directory_name != '.':
        A_files = [ directory_name+os.sep+fname for fname in A_files ]

    return A_files


if __name__ == "__main__":
    pass
