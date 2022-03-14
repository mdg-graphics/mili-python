# MDG Mili Python Reader
[![pipeline status](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/pipeline.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)
[![coverage report](https://rzlc.llnl.gov/gitlab/mili/mili-python/badges/master/coverage.svg)](https://rzlc.llnl.gov/gitlab/mili/mili-python/-/commits/master)

---
##### Installation:

  - Make sure you're using python > 3.7:

  `module load python/3.7.2`

  - Create a python virtual environment to install packages into locally:

  `python -m venv <venv_name>`

  - Activate the environment:

  `source <venv_name>/bin/activate`

  - Upgrade pip (numpy > 1.20.0 will fail to build with the base RZ pip):

  `pip install --upgrade pip`

  - Install the mili python package into the local venv:

  `pip install --no-cache --find-links=https://www-lc.llnl.gov/python/wheelhouse --find-links=/collab/usr/gapps/mdg/python/wheels/ mili`

  **Note:** Using `--find-links=<url>` will pull dependencies from the LC python wheelhouse, which should contain sufficient requirements and should be available on OCF and SCF

  This should install `mili` in the python venv.

  If you want to install the packages into your ~/.local/ python cache so the module is usable with the system python install, try instead not creating and activating a virtual environment and instead (untested and may not work):
  ``` 
  module load python/3.7.2
  python -m pip install --upgrade pip --user
  python -m pip install --user --no-cache --find-links=https://www-lc.llnl.gov/python/wheelhouse --find-links=/collab/usr/gapps/mdg/wheels/ mili
  ```

---
##### Getting started:

  The reader module has docstrings, so to get started try:
  ```
  from mili import reader
  help(reader)
  help(reader.open_database)
  help(reader.MiliDatabase)
  help(reader.MiliDatabase.query)
  ```
  **Note:** the result of `open_database` isn't always a `MiliDatabase` object, but all objects returned by `open_database` have the same interface as `MiliDatabase` (parallel operation uses various wrapper classes to dispatch read/write operations in parallel).

  Here is a minimal query example:
  ```
  db = reader.open_database( 'base_filename', suppress_parallel = True )
  result = db.query( 'nodpos[ux]', 'node' )
  ```
---
##### Understanding the results:

  The return format for the above exmples is:
  ```
  result = { 'nodpos[ux]' : 
             { 'layout' : { 'state' : <numpy_array_of_states>
                            <srec_with_nodpos> : <numpy_array_of_labels>,
                            ...
                            <srec_with_nodpos> : <numpy_array_of_labels> } 
             },
             { 'data' : { <srec_with_nodpos> : <numpy_multidim_array_of_data>,
                           ...
                          <srec_with_nodpos> : <numpy_multidim_array_of_data> }
             }
           }
  ```
  **Note:** for parallel databases, the result data for each rank is contained in a top-level list.

  **Note:** often only a single `<srec>` will be returned, but since mili allows multiple subrecords to contain the same state variables, we return all which match the given query criteria.

  The data array is indexed with tuples of the form `(state_index, label_index, scalar_index)`. To be clear the `state_index` and `label_index` are the indices of the state and label in the list of states and list of labels passed to query function call, which is why these are also returned in the 'layout' data since those query arguments are optional. Thus to find the indices for state `T` and node label `N`, we need to:
  ```
  sidx = np.where(T == result['nodpos[ux]']['layout']['state'])
  nidx = np.where(N == result['noppos[ux]']['layout'][<srec>])
  # the above return size-1 numpy arrays (assuming they succeed)
  N_data_at_T = result['nodpos[ux]']['data'][<srec>][sidx[0], nidx[0],:]
  ```
  **Note**: if you only need data for a single node or for single state, format the query so only that data is returned rather than querying large amounts of data and indexing into it as above.

---
##### Modifying databases:

  ```
  db = reader.open_database( 'base_filename', suppress_parallel = True )
  nodpos_ux = db.query( 'nodpos[ux]', 'node' )
  # modify nodpos_ux
  nodpos_ux = db.query( 'nodpos[ux]', 'node', write_data = nodpos_ux )
  ```

  Will write modified data back to the database. The `write_data` must have the same format as the result data for an identical query. In practice it is best to simply process a query, modify the results, and then submit the same query supplying the modified results as the `write_data` argument.

  **Note:** in parallel, the result format has an additonal top-level list. This is not currently accounted for in write-mode. A user must currently produce a `write_data` argument that contains the union of the set of data to be written to the parallel database. Thus if some data was queried from rank 0 databases and some from rank 1, the results dictionaries for the first two top-level results in the list must be merged appropriately to allow the opertional to succeed. This is cumbersome and not reccomended. Opening individual parallel databases in serial mode is supported, and may present a cleaner solution until development accomodates this use-case.

  **Note:** minimal enforcement/checking of the `write_data` structure is currently done and malformed `write_data` *could* possibly (unlikely) cause database corruption, use at your own discretion. Create backups, test on smaller databases first, etc. A python expection is the most likely outcome here, but caution is best.

---
##### NOTES:

  - Mixed-data subrecords (subrecords containing more than 1 datatype (e.g. mixed double/float, int/double, etc)) are functional but have received less testing and performance evaluation, so some performance degredation is expected when querying variables from these subrecords.


--- 
##### Development Notes:
  
  