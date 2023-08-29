# MDG Mili Python Reader
[![pipeline status](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/pipeline.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)
[![coverage report](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/coverage.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)

# Table of Contents

- [Developer Guide](#developer-guide)
- [Installation](#installation)
- [Importing the Reader Module](#importing-the-reader-module)
- [Opening a Database](#opening-a-database)
- [Parallel vs. Serial Database Results](#parallel-vs-serial-database-results)
- [Querying Results](#querying-results)
- [Derived Variables](#derived-variables)
- [Modifying Results](#modifying-results)
- [MiliDatabase Member Functions](#milidatabase-class-member-functions)
    - [mesh_dimensions](#mesh_dimensions)
    - [state_maps](#state_maps)
    - [times](#times)
    - [nodes](#nodes)
    - [labels](#labels)
    - [parameter](#parameter)
    - [parameters](#parameters)
    - [class_names](#class_names)
    - [mesh_object_classes](#mesh_object_classes)
    - [connectivity](#connectivity)
    - [nodes_of_elems](#node_of_elems)
    - [queriable_svars](#queriable_svars)
    - [classes_of_state_variable](#classes_of_state_variable)
    - [material_numbers](#material_numbers)
    - [materials_of_class_name](#materials_of_class_name)
    - [class_labels_of_material](#class_labels_of_material)
    - [all_labels_of_material](#all_labels_of_material)
    - [nodes_of_material](#nodes_of_material)
- [Development Nodes](#development-notes)


# Developer Guide

### Style Guide
The current style guide for mili-python is [PEP8](https://peps.python.org/pep-0008/), except instead of indentation being 4 spaces we use 2 spaces.

### Virtual Environment

You will need to create a local virtual environment for testing and development. To do this run:
```
cd mili-python
source .venv.sh
```
This will create a virtual environment called `.venv-mili-python-3.8.2` on Toss3 and `.venv-mili-python-3.9.12` on Toss4 with all the required dependencies for mili-python and activate it. To deactivate the virtual environment, run the command `deactivate` and to activate the environment run `source .venv-mili-python-3.8.2/bin/activate`.

### Testing

To run the test suite locally, cd into the directory `mili-python/src` and run: `python3 -m unittest discover tests`

### Deployment

The mili-python reader is distributed through the WCI Nexus repository.

Nexus documentation:
- Link to nexus docs: https://wci-svc-doc.llnl.gov/repo/nexus/
- Setup for python: https://wci-svc-doc.llnl.gov/repo/setup_proxy/#python-pypi
- Publishing python packages: https://wci-svc-doc.llnl.gov/repo/publishing/#python-pypi

Before continuing take a look at the above documentation and perform the steps described in the `Setup for python` link.

To generate the python `whl` file that is distributed run the following:
```
cd mili-python/
version=$(cat src/mili/__init__.py | grep 'version' | grep -Eo "[[:digit:]]+,[[:digit:]]+,[[:digit:]]+" | tr , . )
git tag -a v${version} -m "version $version"
git push origin v${version}
python3 -m build .
```

This will generate a `./dist` directory in the top level of the `mili-python` repository that contains the `whl` file, as well as the `sdist` (not currently distributed).

To upload the `whl` file to the nexus repository you will need to use twine as shown below:
```
pip3 install twine
python3 -m twine upload -r pypi-wci <whl-file>
```
> **NOTE**: The password requested is your AD password.

# Installation

There are currently two methods of installing the mili-python module. The recommeded method is installing from the WCI Nexus Mirror. However, if this does not work user may also install from source using the Git repository.

### From the WCI Nexus Mirror
```
module load python/3.9.12
python -m venv <venv_name>
source <venv_name>/bin/activate
pip install --upgrade pip
pip install --upgrade --no-cache --https://wci-repo.llnl.gov/repository/pypi-group/simple mili
```
- Make sure you're using python > 3.7:
- Create a python virtual environment to install packages into locally:
- Activate the environment:
- Upgrade pip (numpy > 1.20.0 will fail to build with the base RZ pip):
- Install the mili python package into the local venv from the WCI Nexus Mirror:

> **Note:** Using `--find-links=<url>` will pull dependencies from the WCI Nexus Mirror, which should contain sufficient requirements to install mili-python and should be available on OCF and SCF.
> **Note:** Using `--upgrade` will upgrade any already-installed copies of the mili module in the venv.

If you want to install the packages into your ~/.local/ python cache so the module is usable with the system python install, try instead not creating and activating a virtual environment and instead (untested and may not work):
```
module load python/3.9.12
python -m pip install --upgrade pip --user
python -m pip install --upgrade --user --no-cache --find-links=https://wci-repo.llnl.gov/repository/pypi-group/simple mili
```

### From Gitlab Repository
```bash
git clone ssh://git@rzgitlab.llnl.gov:7999/mdg/mili/mili-python.git
module load python/3.9.12
cd mili-python
source .venv.sh
```

# Importing the reader module
```python
from mili import reader
# The reader module has docstrings
help(reader)
help(reader.MiliDatabase)
help(reader.MiliDatabase.query)
```

# Opening a Database
You can open a Mili database using the function `open_database`.

```python
def open_database( base : os.PathLike, procs = [], suppress_parallel = False, experimental = False, **kwargs ):
  """
   Open a database for querying. This opens the database metadata files and does additional processing to optimize query
   construction and execution. Don't use this to perform database verification, instead prefer AFileIO.parse_database()
   The object returned by this function will have the same interface as an MiliDatabase object, though will return a list
   of results from the specified proc files instead of a single result.
  Args:
   base (os.PathLike): the base filename of the mili database (e.g. for 'pltA', just 'plt', for parallel
                            databases like 'dblplt00A', also exclude the rank-digits, giving 'dblplt')
   procs (Optional[List[int]]) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Optional[Bool]) : optionally return a serial database reader object if possible (for serial databases).
                                        Note: if the database is parallel, suppress_parallel==True will return a reader that will
                                        query each processes database files in series.
   experimental (Optional[Bool]) : optional developer-only argument to try experimental parallel features
  """
```

### Examples
```python
from mili import reader

# This will open a serial or uncombined database.
db = reader.open_database('path-to-mili-files.plt')

# To use the experimental (most performant) parallel implementation, set experimental to True
# WARNING: Do not do this on a login node.
db = reader.open_database('path-to-mili-files.plt', experimental=True)
```

# Parallel vs. Serial Database Results
The result of `open_database` isn't always a `MiliDatabase` object, but all objects returned by `open_database` have the same interface as `MiliDatabase` (parallel operation uses various wrapper classes to dispatch read/write operations in parallel).

When calling a `MiliDatabase` function on a parallel database the results for each rank are returned in a top level list. For example:

```python
# For a serial database
lbls = db.labels('node')
# lbls = array([1, 2, 3, 4, 5, 6, 7, 8, ...])

# For a parallel database
lbls = db.labels('node')
# lbls = [ <proc_1_labels>, <proc_2_labels>, <proc_3_labels>, ...]
```

# Querying Results
Querying results from mili database can be done using the `query` function:
```python
def query( self,
           svar_names : Union[List[str],str],
           class_sname : str,
           material : Optional[Union[str,int]] = None,
           labels : Optional[Union[List[int],int]] = None,
           states : Optional[Union[List[int],int]] = None,
           ips : Optional[Union[List[int],int]] = None,
           write_data : Optional[Mapping[int, Mapping[str, Mapping[str, npt.ArrayLike]]]] = None,
           **kwargs ):
    '''
    Query the database for svars, returning data for the specified parameters, optionally writing data to the database.
    The parameters passed to query can be identical across a parallel invocation of query, since each individual
      database query object will filter only for the subset of the query it has available.

    : param svar_names : short names for state variables being queried
    : param class_sname : mesh class name being queried
    : param material : optional sname or material number to select labels from
    : param labels : optional labels to query data about, filtered by material if material if material is supplied, default is all
    : param states: optional state numbers from which to query data, default is all
    : param ips : optional for svars with array or vec_array aggregation query just these components, default is all available
    : param write_data : optional the format of this is identical to the query result, so if you want to write data, query it first to retrieve the object/format, then modify the values desired, then query again with the modified result in this param
    '''
```

Here are a few example queries:
```python
from mili import reader
db = reader.open_database( 'base_filename', suppress_parallel = True )

# Query the result nodpos[ux] for the node element class
# Because labels is None, this will return the result for all nodes.
# Because states is None, this will return the result for all states in the database.
result = db.query('nodpos[ux]', 'node')

# Query the result sx for the brick element class.
# Only query the label 228 at state number 10
result = db.query( 'sx', 'brick', labels = [228], states = [10])

# Query the result refrcx for the node element class
# Get this result for element labels 6,7,8 at states 1,3,3
result = db.mili.query( 'refrcx', 'node', labels = [6,7,8], states = [1,2,3])

# Query the result eps for the shell element class.
# Get the result for label 1 at state number 2.
# This result has multiple integration points, Only get integration point 1.
result = db.query( 'eps', 'shell', labels = [1], states = [2], ips = [1])

# Query the result sx for the brick element class
# Get result for all brick elements that are material number 2
# Only get state 37
result = db.query('sx', 'brick', material = 2, states = 37 )

# Query the result sx for the brick element class
# Get result for all brick elements that have material name 'es_12'
# Only get state 37
result = db.query('sx', 'brick', material = 'es_12', states = 37 )
```

---
### Understanding the results:

Lets take a look at how the returned results are formatted using the query below.
```python
result = db.query('nodpos[ux]', 'node')
```

The return format for the above examples is:
```python
result = {
    'nodpos[ux]' : {
        'layout' : {
            'states'  : <numpy_array_of_states>
            'labels' : <numpy_array_of_labels>
        }
        'data': <numpy_multidim_array_of_data>
    },
}
```
**Note:** for parallel databases, the result data for each rank is contained in a top-level list.

### Getting data out of the result dictionary

The data array is indexed with tuples of the form `(state_index, label_index, scalar_index)`. To be clear the `state_index` and `label_index` are the indices of the state and label in the list of states and list of labels passed to query function call, which is why these are also returned in the 'layout' data since those query arguments are optional. Thus to find the indices for state `T` and node label `N`, we need to:
```python
sidx = np.where(T == result['nodpos[ux]']['layout']['states'])
nidx = np.where(N == result['noppos[ux]']['layout']['labels'])
# the above return size-1 numpy arrays (assuming they succeed)
N_data_at_T = result['nodpos[ux]']['data'][sidx[0], nidx[0],:]
```
**Note**: if you only need data for a single node or for single state, format the query so only that data is returned rather than querying large amounts of data and indexing into it as above.

### Result Dictionary Utility Functions

There are some additional functions that users will find helpful in reorganizing the data into an easier to use format. These are the `combine` function and the `results_by_element` function.

#### The `combine` function

The `combine` function converts the list of result dictionaries returned when querying an uncombined Mili database and merges all the data into a single dictionary.

```python
from mili.reader import combine

result = db.query("sx", "brick", states=[40])
combined = combine(result)

# OR

combined = combine( db.query("sx", "brick", states=[40]) )
```

#### The `results_by_element` function

The `results_by_element` function extracts all the data from a result dictionary (For both serial and parallel database) and reorganizes the data into a new dictionary with the form `<variable-name> : { <element-id> : <numpy_array_of_data> }`.

```python
from mili.reader import results_by_element

result = db.query("stress", "brick")
element_data = results_by_element( result )

# OR

element_data = results_by_element( db.query("stress", "brick") )

stress_brick_101 = element_data['stress'][101]  # [[ STATE 1: sx, sy, sz, sxy, syz, szx ],
                                                #  [ STATE 2: sx, sy, sz, sxy, syz, szx ]
                                                #  [ STATE 3: sx, sy, sz, sxy, syz, szx ]
                                                #  ... ]

# NOTE: State 0 is not the first state in the problem, but the first state
#       that you queried. That may be the first state in the problem or a different
#       state depending on the arguments you passed to the query
stress_brick_20_state_0 = element_data['stress'][101][0]

# Integration Point Data

# The shell elements have integration points 1 and 2 for stress
element_data = results_by_element( db.query("sy", "shell") )

sy_shell_1 = element_data['sy'][1]  # [[ STATE 1: sy_int_point1, sy_int_point2 ],
                                    #  [ STATE 2: sy_int_point1, sy_int_point2 ]
                                    #  [ STATE 3: sy_int_point1, sy_int_point2 ]
                                    #  ... ]
```

---
##### NOTES:

  - Mixed-data subrecords (subrecords containing more than 1 datatype (e.g. mixed double/float, int/double, etc)) are functional but have received less testing and performance evaluation, so some performance degredation is expected when querying variables from these subrecords.

# Derived Variables

mili-python supports many derived variables that can be queried using the `query` method. To see the derived variables that are supported in mili-python use the function `supported_derived_variables` as show below:

```python
print( db.supported_derived_variables() )
```

> **NOTE:** Not all of these variables are necessarily able to be calculated for every database. This list contains all the derived variables that mili-python can calculate if the required primal variables exist for your specific database.

To determine the derived variables that can be calculated for your specific database you can use the functions `derived_variables_of_class` or `classes_of_derived_variable`.  The function `derived_variables_of_class` takes an element class
name as the argument and returns a list of the derived variables that can be calculated for the class. The function `classes_of_derived_variable` takes a derived variable name as the argument and returns a list of the classes for which the derived variable can be calculated.

```python
classes_eff_stress = db.classes_of_derived_variable("eff_stress")  # ["brick", "beam", "shell"]

nodal_derived_variables = db.derived_variables_of_class("node")  # ["disp_x", "disp_y", ...]
```

> **NOTE:** These functions do not guarantee that the derived variable can be calculated for **ALL** elements of the specified element class. They are checking that all the primal variables required to calculate the specific derived variable exist within the database for a given class (It does not check which specific elements of that class the primals exist for).

Querying these derived variables works the same as querying any primal variable. The following are several examples of queries for derived results.

```python
result = db.query('disp_x', 'node')

# Query the pressure for the brick element class.
result = db.query( 'pressure', 'brick', labels = [228], states = [10])

# Query the result disp_mag for the node element class
result = db.mili.query( 'disp_mag', 'node', labels = [6,7,8], states = [1,2,3])

# Query the 1st principle stress for the shell element class.
result = db.query( 'prin_stress1', 'shell', labels = [1], states = [2], ips = [1])
```

> **NOTE**: When calculating rate-based results, numerical difference methods are used to approximate the time derivatives of primal variables. These results include nodal velocity, nodal acceleration, and effective plastic strain rate. There are a few limitations and assumptions for these results:
>- The accuracy of the results is highly dependent on the size of the time step. Smaller time steps provide more accurate results.
>- The calculations assume that the time step is the same on each side of the calculation time ( t^(n-1) and t^(n+1) ).  Significant differences in dt will result in more error.
>- Results for the first and last states use forward and backward difference methods, which are less accurate than the central difference method used for the other states.  The exception is that nodal velocity uses backward difference for all states (except state 1), which is consistent with the griz calculation.  The nodal velocity at state 1 is set to zero.
>- When possible, have the analysis code output primal variables for rates instead of calculating derived variables.  They will almost always be more accurate, and will never be less accurate.

# Modifying Results

```python
# For a Serial Database
db = reader.open_database( 'base_filename', suppress_parallel = True )
nodpos_ux = db.query( 'nodpos[ux]', 'node' )
# modify nodpos_ux
nodpos_ux = db.query( 'nodpos[ux]', 'node', write_data = nodpos_ux )

# For a Parallel Database
db = reader.open_database( 'base_filename', suppress_parallel = True )
nodpos_ux = db.query( 'nodpos[ux]', 'node' )
# This merges all the individual processor result dictionaries into a single dictionary
nodpos_ux = reader.combine(nodpos_ux)
# modify nodpos_ux
nodpos_ux = db.query( 'nodpos[ux]', 'node', write_data = nodpos_ux )

# When using the results_by_element function
nodpos_ux = db.query( 'nodpos[ux]', 'node' )
nodpos_by_element = results_by_element( nodpos_ux )
# Modify nodpos_by_element
writeable_nodpos = writeable_from_results_by_element(nodpos_ux, nodpos_by_element)
nodpos_ux = db.query( 'nodpos[ux]', 'node', write_data = writeable_nodpos )
```

Will write modified data back to the database. The `write_data` must have the same format as the result data for an identical query. In practice it is best to simply process a query, modify the results, and then submit the same query supplying the modified results as the `write_data` argument.

> **Note:** in parallel, the result format has an additonal top-level list. This is not currently accounted for in write-mode. A user must currently produce a `write_data` argument that contains the union of the set of data to be written to the parallel database. Thus if some data was queried from rank 0 databases and some from rank 1, the results dictionaries for the first two top-level results in the list must be merged appropriately to allow the opertional to succeed. This can be done using the `combine` function references above.

> **Note:** minimal enforcement/checking of the `write_data` structure is currently done and malformed `write_data` *could* possibly (unlikely) cause database corruption, use at your own discretion. Create backups, test on smaller databases first, etc. A python expection is the most likely outcome here, but caution is best.

# MiliDatabase Class member functions

## mesh_dimensions
```python
def mesh_dimensions(self) -> int
```
**Description**: Get the mesh dimensions
**Arguments**:
    - `None`
**Returns**:
    - `int`: The dimensions (2 or 3).

### Example
```python
dims = db.mesh_dimensions()
# 3
```

## state_maps
```python
def state_maps(self) -> List[StateMap]
```
**Description**: Getter for the state maps
**Arguments**:
    - `None`
**Returns**:
    - `List[StateMap]`: The list of state map objects for the database.

### Example
```python
state_maps = db.state_maps()
```

## times
```python
def times( self, states : Optional[Union[List[int],int]] = None )
```
**Description**: Get the times for specific states numbers in the database.
**Arguments**:
    - `states`: The List of states to get the times for. None means all states.
**Returns**:
    - `numpy.ndarray`: List of state times.

### Example
```python
times = db.times()
print(times)
"""
array([0.00000000e+00, 9.99999975e-06, 1.99999995e-05, 2.99999992e-05,
       3.99999990e-05, 4.99999987e-05, 5.99999985e-05, 7.00000019e-05,
       7.99999980e-05, 9.00000014e-05, 9.99999975e-05, 1.10000001e-04,
       1.19999997e-04, 1.30000000e-04, 1.40000004e-04, 1.50000007e-04,
       1.59999996e-04, 1.69999999e-04, 1.80000003e-04, 1.90000006e-04,
       1.99999995e-04, 2.09999998e-04, 2.20000002e-04, 2.30000005e-04,
       2.39999994e-04, 2.50000012e-04, 2.60000001e-04, 2.69999990e-04,
       2.80000007e-04, 2.89999996e-04, 3.00000014e-04, 3.10000003e-04,
       3.19999992e-04, 3.30000010e-04, 3.39999999e-04, 3.49999988e-04,
       3.60000005e-04, 3.69999994e-04, 3.80000012e-04, 3.90000001e-04,
       3.99999990e-04, 4.10000008e-04, 4.19999997e-04, 4.29999985e-04,
       4.40000003e-04, 4.49999992e-04, 4.60000010e-04, 4.69999999e-04,
       4.79999988e-04, 4.90000006e-04, 5.00000024e-04, 5.09999983e-04,
       5.20000001e-04, 5.30000019e-04, 5.39999979e-04, 5.49999997e-04,
       5.60000015e-04, 5.69999975e-04, 5.79999993e-04, 5.90000011e-04,
       6.00000028e-04, 6.09999988e-04, 6.20000006e-04, 6.30000024e-04,
       6.39999984e-04, 6.50000002e-04, 6.60000020e-04, 6.69999979e-04,
       6.79999997e-04, 6.90000015e-04, 6.99999975e-04, 7.09999993e-04,
       7.20000011e-04, 7.30000029e-04, 7.39999989e-04, 7.50000007e-04,
       7.60000024e-04, 7.69999984e-04, 7.80000002e-04, 7.90000020e-04,
       7.99999980e-04, 8.09999998e-04, 8.20000016e-04, 8.29999975e-04,
       8.39999993e-04, 8.50000011e-04, 8.59999971e-04, 8.69999989e-04,
       8.80000007e-04, 8.90000025e-04, 8.99999985e-04, 9.10000002e-04,
       9.20000020e-04, 9.29999980e-04, 9.39999998e-04, 9.50000016e-04,
       9.59999976e-04, 9.69999994e-04, 9.80000012e-04, 9.89999971e-04,
       1.00000005e-03])
"""
```

## nodes
```python
def nodes(self)
```
**Description**: Getter function for the initial nodal positions.
**Arguments**:
    - `None`
**Returns**:
    - `numpy.ndarray`: The initial nodal positions.

### Example
```python
nodes = db.nodes()
print(nodes)
"""
array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 9.6592581e-01,  2.5881904e-01,  1.0000000e+01],
       [ 8.6602539e-01,  5.0000000e-01,  0.0000000e+00],
       [ 7.0710677e-01,  7.0710677e-01,  1.0000000e+01],
       [ 4.9999997e-01,  8.6602545e-01,  0.0000000e+00],
       [ 2.5881907e-01,  9.6592581e-01,  1.0000000e+01],
       [-4.3711388e-08,  1.0000000e+00,  0.0000000e+00],
       [ 1.0100000e+00,  0.0000000e+00,  1.0000000e+00],
       [ 8.7468565e-01,  5.0500000e-01,  1.0000000e+00]], dtype=float32)
"""
```

## labels
```python
def labels(self, class_name: Optional[str] = None)
```
**Description**: Get the element labels for an element class.
**Arguments**:
    - `class_name`:  The class name to retrieve labels for. If None, returns labels for all classes.
**Returns**:
    - `Union[ Dict[str,numpy.ndarray], numpy.ndarray ]`: If a class name is specified returns a list containing all element labels. If no class name is specified then returns a dictionary containing a list of the element labels for each class.

### Example
```python
labels = db.labels('shell')
print(labels)
"""
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
"""
```

## parameter
```python
def parameter(self, name: str, default: Optional[Any] = None) -> Any
```
**Description**: Getter for a single Mili parameter.
**Arguments**:
    - `name`: The name of the parameter to retrieve.
    - `default`: An optional value specifying what to return if the parameter does not exist. Defaults to None.
**Returns**:
    - `Any`: The value of the specified parameter if it exists, else default value.

### Example
```python
db_job_id = db.parameter('job_id')
```

## parameters
```python
def parameters(self) -> Dict[str,Any]
```
**Description**: Getter function for the Mili database parameters dictionary.
**Arguments**:
    - `None`
**Returns**:
    - `Dict[str,Any]`: Parameters dictionary.

### Example
```python
params = db.parameters()
```

## class_names
```python
def class_names(self) -> List[str]
```
**Description**: Getter function for the element class names.
**Arguments**:
    - `None`
**Returns**:
    - `List[str]`: List of element class names.

### Example
```python
class_names = db.class_names()
print(class_names)
"""
['glob', 'mat', 'node', 'beam', 'brick', 'shell', 'cseg']
"""
```

## mesh_object_classes
```python
def mesh_object_classes(self) -> Dict[str,MeshObjectClass]
```
**Description**: Getter function mesh object class dictionary.
**Arguments**:
    - `None`
**Returns**:
    - `Dict[str,MeshObjectClass]`: The Mesh object classes in the database.

### Example
```python
mo_classes = db.mesh_object_classes()
print(mo_classes.keys())
print(mo_classes['brick'])
"""
['glob', 'mat', 'node', 'beam', 'brick', 'shell', 'cseg']
MeshObjectClass(mesh_id=0, short_name='brick', long_name='Bricks', sclass=<Superclass.M_HEX: 9>, elem_qty=36, idents_exist=True)
"""
```

## connectivity
```python
def connectivity( self, class_name : Optional[str] = None )
```
**Description**: Get the element connectivity for a specified element class.
**Arguments**:
    - `class_name`: The name of the element class to get the connectivity for. If none, gets connectivity for all classes.
**Returns**:
    - `Union[ Dict[str,numpy.ndarray], numpy.ndarray ]`: The element connecivity. If class name is specified then returns a numpy.ndarray of integers. If class name is None, then dictionary is returned with keys being class_names and values being numpy.ndarray.

### Example
```python
conns = db.connectivity()
print(conns.keys())
print(conns['brick'])
"""
['beam', 'brick', 'shell', 'cseg']
array([[ 64,  80,  84,  68,  65,  81,  85,  69],
       [ 80,  96, 100,  84,  81,  97, 101,  85],
       [ 68,  84,  88,  72,  69,  85,  89,  73],
       [ 84, 100, 104,  88,  85, 101, 105,  89],
       [ 72,  88,  92,  76,  73,  89,  93,  77],
       [ 88, 104, 108,  92,  89, 105, 109,  93],
       [ 65,  81,  85,  69,  66,  82,  86,  70],
       [ 81,  97, 101,  85,  82,  98, 102,  86],
       [ 69,  85,  89,  73,  70,  86,  90,  74],
       [ 85, 101, 105,  89,  86, 102, 106,  90],
       [ 73,  89,  93,  77,  74,  90,  94,  78],
       [ 89, 105, 109,  93,  90, 106, 110,  94],
       [ 66,  82,  86,  70,  67,  83,  87,  71],
       [ 82,  98, 102,  86,  83,  99, 103,  87],
       [ 70,  86,  90,  74,  71,  87,  91,  75],
       [ 86, 102, 106,  90,  87, 103, 107,  91],
       [ 74,  90,  94,  78,  75,  91,  95,  79],
       [ 90, 106, 110,  94,  91, 107, 111,  95],
       [ 96, 112, 116, 100,  97, 113, 117, 101],
       [112, 128, 132, 116, 113, 129, 133, 117],
       [100, 116, 120, 104, 101, 117, 121, 105],
       [116, 132, 136, 120, 117, 133, 137, 121],
       [104, 120, 124, 108, 105, 121, 125, 109],
       [120, 136, 140, 124, 121, 137, 141, 125],
       [ 97, 113, 117, 101,  98, 114, 118, 102],
       [113, 129, 133, 117, 114, 130, 134, 118],
       [101, 117, 121, 105, 102, 118, 122, 106],
       [117, 133, 137, 121, 118, 134, 138, 122],
       [105, 121, 125, 109, 106, 122, 126, 110],
       [121, 137, 141, 125, 122, 138, 142, 126],
       [ 98, 114, 118, 102,  99, 115, 119, 103],
       [114, 130, 134, 118, 115, 131, 135, 119],
       [102, 118, 122, 106, 103, 119, 123, 107],
       [118, 134, 138, 122, 119, 135, 139, 123],
       [106, 122, 126, 110, 107, 123, 127, 111],
       [122, 138, 142, 126, 123, 139, 143, 127]], dtype=int32)
"""
```

## nodes_of_elems
```python
def nodes_of_elems( self, class_sname, elem_labels )
```
**Description**: Get a list of nodes associated with elements by label.
**Arguments**:
    - `class_sname`: The element class name.
    - `elem_labels`: The list of element labels.
**Returns**:
    - `numpy.ndarray, numpy.ndarray`: The list of element labels retrieved and the nodes associated with those labels.

### Example
```python
nodes = db.nodes_of_elems( 'brick', [1,2] )
"""
array([
        [ 65,  81,  85,  69,  66,  82,  86,  70],
        [ 81,  97, 101,  85,  82,  98, 102,  86]
      ], dtype=int32),
array([[1], [2]], dtype=int32)
"""
```

## queriable_svars
```python
def queriable_svars(self, vector_only = False, show_ips = False)
```
**Description**: Get a list of queriable svar names for the database.
**Arguments**:
    - `vector_only`: Only return vector state variables.
    - `show_ips`: Return all integration points for each state variable.
**Returns**:
    - `List[str]`: List of state variable names.

### Example
```python
svars = db.queriable_svars()
```

## classes_of_state_variable
```python
def classes_of_state_variable(self, svar: str)
```
**Description**: Get the element class names for which a state variable exists.
**Arguments**:
    - `svar`:  The name of the state variable.
**Returns**:
    - `List[str]`: The names of the element classes for which the result exists.

### Example
```python
classes = db.classes_of_state_variable('sx')
print(classes)
"""
['shell', 'beam', 'brick']
"""
```

## material_numbers
```python
def material_numbers(self)
```
**Description**: Get the material numbers in the database.
**Arguments**:
    - `None`
**Returns**:
    - `List[int]`: The material numbers in the database.

### Example
```python
mat_nums = db.material_numbers()
print(mat_nums)
"""
[1, 2, 3, 4, 5]
"""
```

## materials_of_class_name
```python
def materials_of_class_name( self, class_name: str )
```
**Description**: Get List of materials for all elements of a given class name.
**Arguments**:
    - `class_name`: The Element class name.
**Returns**:
    - `numpy.ndarray`: The material number for each element of the specified element class.

### Example
```python
mats = db.materials_of_class_name( 'cseg' )
print(mats)
"""
array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int32)
"""
```

## class_labels_of_material
```python
def class_labels_of_material( self, mat, class_name )
```
**Description**: Convert a material name into labels of the specified class (if any).
**Arguments**:
    - `mat`: The material name or number.
    - `class_name`: The Element class name.
**Returns**:
    - `numpy.ndarray`: The labels of the specified element class that are the specified material.

### Example
```python
lbls = db.class_labels_of_material( 4, 'cseg' )
print(lbls)
"""
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
"""
```

## all_labels_of_material
```python
def all_labels_of_material( self, mat )
```
**Description**: Given a specific material. Find all labels with that material and return their values.
**Arguments**:
    - `mat`: The material name or number.
**Returns**:
    - `Dict[str,numpy.ndarray]`: The labels of the specified material.

### Example
```python
lbls = db.all_labels_of_material( 1 )
print(lbls)
"""
{'beam': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46], dtype=int32)}
"""
```

## nodes_of_material
```python
def nodes_of_material( self, mat )
```
**Description**: Get a list of nodes associated with a specific material number.
**Arguments**:
    - `mat`: The material name or number.
**Returns**:
    - `numpy.ndarray`: The list of nodes associated with the material number.

### Example
```python
nodes = db.nodes_of_material( 5 )
"""
array([ 65,  69,  73,  77,  81,  85,  89,  93,  97, 101, 105, 109, 113,
        117, 121, 125, 129, 133, 137, 141], dtype=int32)
"""
```

# Development Notes

# License
----------------

Mili Python Reader is distributed under the terms of both the MIT license.

All new contributions must be made under both the MIT license.

See [LICENSE-MIT](https://github.com/mdg/mili-python/LICENSE)

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-838121
