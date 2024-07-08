===============
Examples
===============


Opening a Database
========================

You can open a Mili database using the function :ref:`open_database<open database>` as show in the examples below:

.. code-block:: python
    :linenos:

    from mili import reader
    from mili.milidatabase import MiliDatabase

    # To open a serial database
    db: MiliDatabase = reader.open_database('path-to-mili-files.plt')

    # To open a parallel database in serial
    db: MiliDatabase = reader.open_database('path-to-mili-files.plt', suppress_parallel=True)

    # To open a parallel database in parallel
    db: MiliDatabase = reader.open_database('path-to-mili-files.plt')

    # The MiliDatabase class also supports the context management protocol
    with reader.open_database("path-to-mili-files.plt") as mili_db:
        pass


*WARNING*: If on an LLNL system, you do not want to open a parallel database in parallel on a login node.
This uses a lot of resources and will negatively impact machine performance for other users.


Quering Results
========================

.. code-block:: python
    :linenos:

    from mili import reader
    from mili.milidatabase import MiliDatabase

    db: MiliDatabase = reader.open_database('path-to-mili-files.plt')

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


Understanding the results
--------------------------

Lets take a look at how the returned results are formatted using the query below.

.. code-block:: python

    result = db.query('nodpos[ux]', 'node')

The return format for the above example is:

.. code-block:: python

    result = {
        'nodpos[ux]' : {
            'layout' : {
                'states'  : <numpy_array_of_states>
                'labels' : <numpy_array_of_labels>
            }
            'data': <numpy_multidim_array_of_data>
        },
    }



Getting data out of the result dictionary
--------------------------------------------

The data array is indexed with tuples of the form :code:`(state_index, label_index, scalar_index)`.
To be clear the :code:`state_index` and :code:`label_index` are the indices of the state and label in
the list of states and list of labels passed to query function call, which is why
these are also returned in the 'layout' data since those query arguments are optional.
Thus to find the indices for state :code:`T` and node label :code:`N`, we need to:

.. code-block:: python

    sidx = np.where(T == result['nodpos[ux]']['layout']['states'])
    nidx = np.where(N == result['noppos[ux]']['layout']['labels'])
    # the above return size-1 numpy arrays (assuming they succeed)
    N_data_at_T = result['nodpos[ux]']['data'][sidx[0], nidx[0],:]

*Note*: if you only need data for a single node or for a single state, format the query
so only that data is returned rather than querying large amounts of data and indexing into it as above.


Result dictionary utility function
---------------------------------------

There are some additional functions that users will find helpful in reorganizing the data
into an easier to use format. One of these is the `results_by_element` function.

The :code:`results_by_element` function extracts all the data from a result dictionary
(For both serial and parallel database) and reorganizes the data into a new dictionary
with the form :code:`<variable-name> : { <element-id> : <numpy_array_of_data> }`.

.. code-block:: python

    from mili.reader import results_by_element

    result = db.query("stress", "brick")
    element_data = results_by_element( result )

    # OR

    element_data = results_by_element( db.query("stress", "brick") )

    stress_brick_101 = element_data['stress'][101]
    # [[ STATE 1: sx, sy, sz, sxy, syz, szx ],
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

    sy_shell_1 = element_data['sy'][1]
    # [[ STATE 1: sy_int_point1, sy_int_point2 ],
    #  [ STATE 2: sy_int_point1, sy_int_point2 ]
    #  [ STATE 3: sy_int_point1, sy_int_point2 ]
    #  ... ]


Pandas Dataframes
-------------------------

Optionally the results of the query method can be returned as a Pandas DataFrame.
If the keyword argument :code:`as_dataframe` is set to :code:`True` then the result dictionary
will store the data for each state variable as a DataFrame.

.. code-block:: python

    result = db.query( 'sx', 'brick', as_dataframe=True )

    """
    results will have the format:
    {
        <svar-name> : <dataframe>,
        <svar-name> : <dataframe>,
        ...
    }
    """

    sx_df = result['sx']
    # sx_df.columns are the labels
    # sx_df.index are the state numbers

    sx_brick_12_state_44 = sx_df[12][44]


*WARNING*: Querying vector state variables, such as stress or strain with :code:`as_dataframe=True`
can take a very long time, especially in parallel. It is not recommended that you query
vector state variables as dataframes. Vector state variable data is stored in 3 dimensional
array (num_states by num_labels by num_components) and Pandas DataFrames are not very
efficient with 3 dimensional data. There are significant overheads associated with converting
the data to a DataFrame as well as performing the data serialization that is required when using mili-python in parallel.

*NOTE*: The DataFrames can be converted back to the dictionary format if necessary (such as when overwriting data)
using the :code:`dataframe_to_result_dictionary` function. Existing result dictionaries can be converted to
dataframes using the :code:`result_dictionary_to_dataframe` function.


Derived Results
========================

mili-python supports many derived variables that can be queried using the :code:`query` method.
To see the derived variables that are supported in mili-python use the function :code:`supported_derived_variables` as show below:

.. code-block:: python

    print( db.supported_derived_variables() )

*NOTE:* Not all of these variables are necessarily able to be calculated for every database.
This list contains all the derived variables that mili-python can calculate if the required
primal variables exist for your specific database.

To determine the derived variables that can be calculated for your specific database you
can use the functions :code:`derived_variables_of_class` or :code:`classes_of_derived_variable`.
The function :code:`derived_variables_of_class` takes an element class name as the argument and
returns a list of the derived variables that can be calculated for the class. The function
:code:`classes_of_derived_variable` takes a derived variable name as the argument and returns
a list of the classes for which the derived variable can be calculated.

.. code-block:: python

    classes_eff_stress = db.classes_of_derived_variable("eff_stress")
    # ["brick", "beam", "shell"]

    nodal_derived_variables = db.derived_variables_of_class("node")
    # ["disp_x", "disp_y", ...]

*NOTE:* These functions do not guarantee that the derived variable can be calculated
for *ALL* elements of the specified element class. They are checking that all the primal
variables required to calculate the specific derived variable exist within the database
for a given class (It does not check which specific elements of that class the primals exist for).

Querying these derived variables works the same as querying any primal variable.
The following are several examples of queries for derived results:

.. code-block:: python

    result = db.query('disp_x', 'node')

    # Query the pressure for the brick element class.
    result = db.query( 'pressure', 'brick', labels = [228], states = [10])

    # Query the result disp_mag for the node element class
    result = db.mili.query( 'disp_mag', 'node', labels = [6,7,8], states = [1,2,3])

    # Query the 1st principle stress for the shell element class.
    result = db.query( 'prin_stress1', 'shell', labels = [1], states = [2], ips = [1])

*NOTE*: When calculating rate-based results, numerical difference methods are used to approximate the
time derivatives of primal variables. These results include nodal velocity, nodal acceleration,
and effective plastic strain rate. There are a few limitations and assumptions for these results:

* The accuracy of the results is highly dependent on the size of the time step. Smaller time steps provide more accurate results.
* The calculations assume that the time step is the same on each side of the calculation time ( t^(n-1) and t^(n+1) ). Significant differences in dt will result in more error.
* Results for the first and last states use forward and backward difference methods, which are less accurate than the central difference method used for the other states. The exception is that nodal velocity uses backward difference for all states (except state 1), which is consistent with the griz calculation. The nodal velocity at state 1 is set to zero.
* When possible, have the analysis code output primal variables for rates instead of calculating derived variables. They will almost always be more accurate, and will never be less accurate.


Modifying Results
========================

.. code-block:: python

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

Will write modified data back to the database. The :code:`write_data` must have the same format as
the result data for an identical query. In practice it is best to simply process a query,
modify the results, and then submit the same query supplying the modified
results as the :code:`write_data` argument.

*Note:* minimal enforcement/checking of the :code:`write_data` structure is currently done and
malformed :code:`write_data` *could* possibly (unlikely) cause database corruption, use at your
own discretion. Create backups, test on smaller databases first, etc. A python expection
is the most likely outcome here, but caution is best.


Appending States
========================

:code:`mili-append` is a script that provides users with a framework for adding additional states to a database.
New states and data can be appended to the end of an existing database or added to a new database (A copy of the
existing database with just the new states added). Users can specify the states and data to be appended by creating
the dictionary :code:`append_states_spec` in a separate python file. This file is then the input to the tool. A
complete list of available fields in the :code:`append_states_spec` dictionary as well as an example of
defining the dictionary is shown below:

Available fields in :code:`append_states_spec`:

.. list-table:: Fields
    :header-rows: 1

    * - Field
      - Type
      - Description
      - Required
    * - database_basename
      - str
      - The name of the input database.
      - Always
    * - output_type
      - List[str]
      - List of outputs. Options include: "mili".
      - Always
    * - output_mode
      - str
      - "write" or "append".
      - Always
    * - output_basename
      - str
      - The name of the new database if output_mode is "write".
      - When :code:`output_mode` is "write"
    * - states
      - int
      - The number of states to add.
      - Always
    * - state_times
      - List[float]
      - The new state times to append to the database.
      - If :code:`time_inc` is not provided
    * - time_inc
      - float
      - The increment for each new state to add.
      - If :code:`state_times` is not provided|
    * - state_variables
      - dict
      - The state variable data to append
      - Always
    * - limit_states_per_file
      - int
      - Limit the number of states in generated state files
      - False
    * - limit_bytes_per_file
      - int
      - Limit the number of bytes in generated state files
      - False

The format of the `state_variables` dictionary is:

.. code-block:: python

    "state_variables": {

    "class-name-1": {
        "state-variable-1": {
        # Labels is a list of integers
        "labels": [1,2,3,...],
        # int_point specifies the integration point you are writing out data for (Only needed for results with multiple integration points)
        # If this value is omitted for a state variable that has multiple integration points, then "data" must have values for all integration points
        # that exist
        "int_point": 2,
        # Data is a 2d-array that is num_states_to_append by (num_elems * values_per_element)
        # Data can also be a 1d-array of length num_states_to_append * num_elems * values_per_element
        "data": [
            [1.0, 2.0, 3.0, ...],
            [1.0, 2.0, 3.0, ...],
            ...
        ]
        },
        "state-variable-2": {
        "labels": [10, 11, 12],
        "data": [
            [1.0, 2.0, 3.0, ...],
            [1.0, 2.0, 3.0, ...],
            ...
        ]
        }
    },

    "class-name-2": {
        # ...
    }
    # ...
    }

An example of the :code:`append_states_spec` and how to run the tool:

.. code-block:: python

    # example_input.py
    import numpy as np
    from mili.reader import open_database

    """
    The code below demonstrates one of the main advantages of having this defined in a python file
    that the tool can then import. It allows users to write their own code to read in/setup the new
    data that will be appended. This means that users can get data from multiple different sources or
    file formats as long as they format it correctly. Below are examples of reading in the data from a txt
    file using numpy, hard coding your own values, and reading in data from another mili database.
    """
    stress_data = np.loadtxt("example_stress_data.txt", comments="#", dtype=object)
    brick_labels = stress_data[:,1]
    brick_data = np.array( [stress_data[:,3]] )

    node_labels = [1,2,3,4,5]
    node_data = [ [1.0, 2.0, 3.0, 4.0, 5.0] ]

    different_database = open_database("some-other-database-name")
    shell_results = different_database.query("ey", "shell", states=[55], labels=[1,2,3], ips=[1])
    shell_labels = shell_results['ey']['layout']['labels']
    shell_data = np.reshape( shell_result['ey']['data'], (1,len(shell_labels)) )

    # The name of the dictionary must be append_states_spec
    append_states_spec = {
        "database_basename": "dblplt",    # The name of the database we are appending data to.
        "output_type": ["mili"],          # Output is a mili database. "mili" is currently the only option but we will likely support for outputs in the future
        "output_mode": "write",           # Write out a new database with 0 states and append the new states/data to it.
        "output_basename": "new_dblplt",  # The new database to write. Required because output_mode is "write"

        "states": 1,        # We want to append 1 states to the database
        "time_inc": 0.001,  # New time will increment 0.001 from the last state of the input database

        # The state variable data to append to the database
        "state_variables": {
            "node": {
                "evec_dx": {
                    "labels": node_labels,
                    "data": node_data
                }
            },
            "brick": {
                "sx": {
                    "labels": brick_labels,
                    "data": brick_data
                }
            },
            "shell": {
                "ey": {
                    "labels": shell_labels,
                    "data": shell_data
                }
            }
        }
    }

Usage
---------

:code:`mili-append` is a script that is installed with mili-python. To run with :code:`example_input.py` shown above:

.. code-block:: bash

    mili-append -i example_input.py

*NOTE*: This assumes you have installed mili locally or in an active virtual environment.

Using the Append States Tool in a Script
-------------------------------------------

It is also possible to use the append states tool from within a script as shown below. The dictionary passed
to the :code:`AppendStatesTool` object must still have the same format as :code:`append_states_spec`.

.. code-block:: python

    from mili.append_states import AppendStatesTool

    append_states_spec = {
    # ...
    }

    append_tool = AppendStatesTool(append_states_spec)
    append_tool.write_states()


Adjacency Queries
========================

The mili-python reader provides some support for querying element adjacencies through the :code:`AdjacencyMapping`
wrapper. The current list of supported adjacency queries includes:

* Querying all elements/nodes within a specified radius of a given element using the function:code: `mesh_entities_within_radius`.
* Querying all elements/nodes within a specified radius of a given 3d coordinate using the function :code:`mesh_entities_near_coordinate`.
* Querying all elements associated with a set of specific nodes using the :code:`elems_of_nodes` function.
* Querying the nearest node to a 3d coordinate using the :code:`nearest_node` function.
* Querying the nearest element to a 3d coordinate using the :code:`nearest_element` function.
* Querying the neighboring elements for a given element using the :code:`neighbor_elements` function.

The function :code:`mesh_entities_within_radius` computes the centroid of the element you have specified using the nodal
coordinates for that element at the specified state. The reader then gathers all nodes within the specified radius
of that centroid and returns all elements that are associated with those nodes. This function also takes the optional
arguement :code:`material` that limits the search to specific material name(s) or number(s).

.. code-block:: python

    from mili import adjacency
    adj = adjacency.AdjacencyMapping(db)

    # Gathers the elements within a radius of 0.30 length units from shell 6 at state 1
    adjacent_elements = adj.mesh_entities_within_radius("shell", 6, 1, 0.30, material=None)
    adjacent_elements = adj.mesh_entities_within_radius("shell", 6, 1, 0.30, material=2)
    adjacent_elements = adj.mesh_entities_within_radius("shell", 6, 1, 0.30, material=[1,2])

    """
    The format of the returned dictionary is shown below:

    adjacent_elements = {
        "shell": [3,4,5,6,9,11],
        "beam": [5,6,37,42],
        "cseg": [2,3,5,6,8,9],
        "node": [1,2,3,4,5,6,7,12,13,14,15,16],
    }
    """

The function :code:`mesh_entities_near_coordinate` gathers all nodes within the specified radius of the given 3d
coordinate and returns all elements that are associated with those nodes. This function also takes the optional
arguement :code:`material` that limits the search to specific material name(s) or number(s).

.. code-block:: python

    from mili import adjacency
    adj = adjacency.AdjacencyMapping(db)

    # Gathers the elements within a radius of 0.30 length units from the given coordinate at state 1
    adjacent_elements = adj.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=None)
    adjacent_elements = adj.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=2)
    adjacent_elements = adj.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=[3,4])

    """
    The format of the returned dictionary is shown below:

    adjacent_elements = {
        "shell": [3,4,5,6,9,11],
        "beam": [5,6,37,42],
        "cseg": [2,3,5,6,8,9],
        "node": [1,2,3,4,5,6,7,12,13,14,15,16],
    }
    """

The function :code:`elems_of_nodes` gathers all elements associated with a set of nodes. This function also takes the
optional arguement :code:`material` that limits the search to specific material names or numbers.

.. code-block:: python

    from mili import adjacency

    adj = adjacency.AdjacencyMapping(db)

    elems = adj.elems_of_nodes(120)
    # or
    elems = adj.elems_of_nodes([1,2,3])

The dictionary returned by :code:`elems_of_nodes` has the same format as that returned by :code:`mesh_entities_within_radius`

The function :code:`nearest_node` finds the closest node to a given 3d coordinate at a specified time step.

.. code-block:: python

    from mili import adjacency
    adj = adjacency.AdjacencyMapping(db)

    # Get the closest node and its distance from the point (0.0, 0.0, 0.0) at state 1.
    nearest_node, distance = adj.nearest_node( [0.0, 0.0, 0.0], 1)

The function :code:`nearest_element` finds the element whose centroid is closest to a given 3d coordinate at a specified time step.

.. code-block:: python

    from mili import adjacency
    adj = adjacency.AdjacencyMapping(db)

    # Get the closest element (class name and label) and its distance from the point (0.0, 0.0, 0.0) at state 1.
    class_name, label, distance = adj.nearest_element( [0.0, 0.0, 0.0], 1)

The function :code:`neighbor_elements` finds all elements that neighbor the specified element. A neighbor element
is defined as any element that shares a node with the specified element. The :code:`neighbor_radius`
argument can be used to specify the number of steps out from the element to perform while gathering the
neighbor elements. The :code:`material` argument can be used to limit the elements that are returned to
specified material names or numbers.

.. code-block:: python

    from mili import adjacency
    adj = adjacency.AdjacencyMapping(db)

    neighbors = adj.neighbor_elements("brick", 1, neighbor_radius=2, material=2)
    """
    The format of the returned dictionary is shown below:

    neighbors = {
        "brick": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18,
                19, 21, 23, 25, 27, 29, 31, 33, 35]
    }
    """