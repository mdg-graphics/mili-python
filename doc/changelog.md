# Changelog

All notable changes to Mili-python will be documented in this file.

## [Unreleased]

### Added

### Fixed

### Changed

### Removed

## [v0.8.2] - 2025-2-21

### Fixed
- Fixed issue where some functions would return List[np._str] for uncombined databases. Now return List[str].
- Fixed typing of numpy arrays in `QueryLayout` and `QueryDict`

### Changed
- Updated the values of the `ResultModifer` enum to be strings instead of integers and removed the `string_repr` function. The string versions of each modifier can now be accessed with `ResultModifier.[MAX|MIN|AVERAGE|CUMMIN|CUMMAX].value`

## [v0.8.1] - 2025-1-31

### Added

- Added the derived variables `surfstrainx`, `surfstrainy`, `surfstrainz`, `surfstrainxy`, `surfstrainyz`, `surfstrainzx` for calculating the strains for faces of a hex element.
- Added the function `MiliDatabase.faces` to get the faces of a hex element.
- Added improved type hinting to the dictionary returned by the `MiliDatabase.query` method.

### Fixed

- Fixed import error with the `adjacency` module.

## [v0.8.0] - 2024-12-12

### Added

- Added support for negative indexing to the `MiliDatabase.query` method argument `states`. The state `-1` can be used to get the last state, `-2` the second to last state, etc.
- Multiple new fields have been added to the dictionary returned by the `MiliDatabase.query` method. These include:
  - `class_name`: The name of the element class that was queried.
  - `title`: The title (long name) of the state variable that was queried.
  - `times` (added to `layout` dictionary): The times for each state in the query.
  - `components` (added to `layout` dictionary): The components of the state variable that were queried.
- Added support for plotting results using matplotlib with the new `MatPlotLibPlotter` object.

## [v0.7.10] - 2024-11-11

### Fixed
- Fixed a bug with the `MiliDatabase.times()` argument `states` where it was expecting the index of the state not the state number itself.
- Fixed bugs in the Nodal velocity and acceleration derived results that caused exceptions related to mismatched shape/dimensions.

## [v0.7.9] - 2024-9-26

### Added

- Added the function `AdjacencyMapping.compute_centroid` to calculate the centroid of an element.
- Added support for calculate `min`, `max`, `average`, `cummin`, and `cummax` for queried results.

## [v0.7.8] - 2024-9-5

### Fixed

- Fixed bug in `Milidatabase.nodes` method causing incorrect number of nodal coordinates to be returned in some cases.

## [v0.7.7] - 2024-8-28

- Updated the functions `AdjacencyMapping.nearest_element` and `AdjacencyMapping.nearest_node` to support filtering by material name/number.
- Added function `state_variable_titles` to get titles of state variables.

### Fixed

### Changed

- Removed numpy 2.0 version limit

### Removed

## [v0.7.6] - 2024-8-8

## Added
- Added derived variable `area` to calculate the area of a quad element.

## [v0.7.5] - 2024-8-1

## Added
- Added typing overloads for the functions `MiliDatabase.connectivity` and `MiliDatabase.labels`.

## [v0.7.4] - 2024-7-23

### Fixed
- Fixed issue in `query` method relating to writing out data for specific element labels. In cases where labels were valid, but ordered differently from what mili was expecting, data would be written out to the incorrect elements. This affected both the `query` method and the `AppendStatesTool`.

## [v0.7.3] - 2024-6-28

### Added

- Added function `state_variables_of_class` to get the state variables that can be queried for a given element class.

### Fixed

### Changed

- The functions `mesh_entities_within_radius`, `mesh_entities_near_coordinate`, `elems_of_nodes`, and `neighbor_elements` of the `AdjacencyMapping` class now support filtering based on multiple materials. Previously only allowed limiting the results to 1 material
- The functions `mesh_entities_within_radius` and `mesh_entities_near_coordinate` of the `AdjacencyMapping` now return the `nodes` in addition to the elements within the radius.
- Updated function `writeable_from_results_by_element` to be less strict about data shape for scalar variables.

### Removed

## [v0.7.2] - 2024-6-19

### Added

- Support for Python 3.12.

### Fixed

- Fixed bug causing exception when querying multiple global results.
- Fixed bug causing exceptions in the query method to not be thrown.

### Changed

- Removed all dynamically added methods from the `MiliDatabase` class and replace with a static interface to better support static analysis.

### Removed

## [v0.7.1] - 2024-6-5

### Added

- The `MiliDatabase` class now supports the context management protocol and can be used with the `with` statement as a context manager.

## [v0.7.0] - 2024-5-15

### Added

- `components_of_vector_svar` method to get the component state variable names for a vector state variable.
- Added the keywords `limit_states_per_file` and `limit_bytes_per_file` to the append states tool to support contraining the size/count of state files when appending states to a database.

### Fixed

- Fixed bug where exception was not thrown when trying to query an invalid state variable, element class combination.

### Changed

- The `material_numbers` method now returns a `numpy.ndarray` rather than `List[int]`.
- The `int_points_of_state_variable` method now returns a `numpy.ndarray` rather than `List[int]`.
- The `connectivity` method now returns element LABELS instead of element IDS.
- There is no longer a difference in the format of the results returned by the `MiliDatabase` functions for a serial database vs. an uncombined database. The uncombined results are merged into a serial format now.
- The following `MiliDatabase` methods are now hidden: `subrecords`, `state_variables`, `mesh_object_classes`, `int_points`, `parameters`, `parameter`. These are functions that users should never need to use as they access the internal structures in Mili.

### Removed

- The `PoolWrapper` parallel implementation has been removed and the `ServerWrapper` parallel implementation has replaced it as the default parallel mode.

## [v0.6.7] - 2024-4-22

### Fixed

- Fixed bug where combining the derived result `element_volume` caused an exception.

## [v0.6.6] - 2024-4-17

### Fixed

- Fixed bug where querying multiple derived variables in a single query only returned the results for one of the variables.

## [v0.6.5] - 2024-3-12

### Added

- Added the function `neighbor_elements` to gather all elements that neighbor a specified element.

### Fixed

- Fixed issue in the function `nodes_of_elems` in parallel where no results were returned when queried element labels were split across multiple processors.
- Fixed the functions `AdjacencyMapping.elems_of_nodes`, `AdjacencyMapping.mesh_entities_within_radius` and `AdjacencyMapping.mesh_entities_near_coordinate` so that they return data in a merged format rather than the data from each processor.
- Fixed bug in `AdjacencyMapping.elems_of_nodes` where filtering by material returned the incorrect elements for element classes with more than one material. This affected the material filtering in `AdjacencyMapping.mesh_entities_within_radius` and `AdjacencyMapping.mesh_entities_near_coordinate`.

## [v0.6.4] - 2024-02-15

### Added

- Added `material` keyword argument to the function `mesh_entities_within_radius` and `mesh_entities_near_coordinate` to allow users to limit the search to elements of the specified material name or number.
- Add the derived variable `element_volume` for computing the volume of hex and tet elements.

## [v0.6.3] - 2024-01-24

### Added

- Added the derived variables `mat_cog_disp_[xyz]` to calculate material center of gravity displacement.

### Changed

- Changed `dill` dependency requirement from `==0.3.4` to `~=0.3`.
- Capped `numpy` dependency requirement to `<2` to prevent any possible errors from new numpy release.
- Various minor internal changes/bugfixes to `GrizInterface`.

## [v0.6.2] - 2023-12-20

### Added

- Add feature to query elements within a given radius of a 3d coordinate using the `mesh_entities_near_coordinate` function.
- Update the query function to allow data to be returned as a Pandas DataFrame using the `as_dataframe` keyword argument. It is recommended that this feature only be used when querying scalars. When querying vector results such as `nodpos`, `stress`, or `strain` with `as_dataframe=True` performance is significantly worse than querying with the default dictionary structure.

### Fixed

- Fix import error causing exception when Adjacency module was imported before importing the Reader module.
- Improved shared memory used to pass arguments to sub processes when running in parallel using `experimental=True`. This should help prevent the out of memory errors some users were experiencing when trying to overwrite large amounts of data.

## [v0.6.1] - 2023-11-27

### Fixed

- Fixed issues in append states tool when trying to write out stress/strain components when stress_in|mid|out or strain_in|out exist

## [v0.6.0] - 2023-11-16

### Added

- Added support for writing new states to existing Mili databases.

## [v0.5.1] - 2023-11-2

### Added

- Added the additional adjacency queries `nearest_node` and `nearest_element` to get the closest node/element to a specified 3d coordinate.

## [v0.5.0] - 2023-10-12

### Added

- Updates experimental parallel implementation to use shared memory to pass commands and arguments to subprocesses.
- Performance improvements to derived calculations for Principal stresses and strains.
- Performance improvement to the `combine` and `writeable_from_result_by_element` functions.

### Fixed

- Fixed bug causing out of memory issues when trying to overwrite variables in large databases.
- Fixed bug causing mili-python to crash when trying to overwrite a variable for a set of elements that are split over multiple subrecords.

## [v0.4.5] - 2023-9-22

### Added

- Performance improvements to multiple derived variable calculations: principal strains, principal deviatoric strains, and max shear stress.

## [v0.4.4] - 2023-9-14

### Added

- Added support for Nodal Tangential Traction Magnitude derived variable `nodtangmag`.
- Added support for querying element adjacency through the AdjacencyMapping object:
    - The function `mesh_entities_within_radius` allows users to query all elements within a specified radius of a given element.
    - The function `elems_of_nodes` will get all elements associated with a set of nodes.

### Fixed

- Fixed bug in `times()` function where an exception was being incorrectly thrown when a list of integers was passed as the argument.
- Fixed bug causing the reader to crash when trying to reader older versions of mili databases.
- Removed numpy dependency ceiling when using Python 3.8 or later.
- Added better support for Exception handling in parallel.
    - Added Exception when attempting to query or modify a state variable that doesn't exist.

## [v0.4.3] - 2023-7-6

### Added

- Added `combine` function to merge data from a parallel query into a single dictionary.
- Added `results_by_element` function that reorganizes the data from the results dictionary returned by the query function into a new dictionary with the format `{<variable-name> : { <element-label> : <numpy-array-of-data> } }`.
- Added `writeable_from_results_by_element` function that converts the `results_by_element` dictionary format back into a format that can be written to the Mili database by the query function.
- Added the function `derived_variables_of_class` and `classes_of_derived_variable` to help determine which derived results can be calculated for a given element class.

### Fixed

- Fixed bug preventing users from writing data to an uncombined database.
- Fixed bug causing data written back to an uncombined database to be ordered incorrectly.
- Fixed issue where functions expecting a material number argument wouldn't return data when a material number was passed in as a string ( `"1"` instead of `1` ).
- Fixed issue where functions expecting a material number argument would throw an exception when the argument had a numpy integer type instead of the python integer type.

## [v0.4.2] - 2023-6-8

### Added

- Added the derived variable `triaxiality`.

## [v0.4.1] - 2023-5-25

### Fixed

- Fixed bug causing incomplete data to be returned from the query when that data was split across multiple subrecords
- Added missing `py.typed` package data to Wheel file

## [v0.4.0] - 2023-4-20

### Added

- Derived variables are now supported in mili-python. Huge shoutout to Zeigle, Alex for his efforts to make this happen. These derived results can be queried using the existing `query` function. To see all the currently supported derived variables in mili-python you can use the function `supported_derived_variables()` of the `MiliDatabase` class. Some notable derived variables that were added include prin_stress1, prin_stress3, and eff_stress, but there are many others as well.
- The structure of the result dictionary returned by the `query` function has been updated. Subrecord names are no longer in the dictionary. The new format is shown below. Upgrading to this new version will break existing scripts that were written to process the previous result dictionary so users will need to update their scripts after switching to version `0.4.0`.
```
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

### Fixed

- Fixed bug querying vector array state variables.

## [v0.3.2] - 2023-3-16

### Fixed

- Fixed a bug where mili-python threw and Exception when trying to modify data for multiple states
- Fixed a bug causing mili-python to fail when mili file version is 3 but no T-files were present

## [v0.3.1] - 2023-2-9

### Fixed

- Various minor bugfixes mostly related to handling parameters from mili databases.

## [v0.3.0] - 2022-12-14

### Added

- Support for the upcoming mili database format v3.
- Reworked mili database A/T file parsing layer to improve debugging, including incremental parsing verification and logging (using the default python logger, subsequent releases will likely use a custom logging object instead).
- More complete internal mili database metadata representation.

### Fixed

- Fixed a bug with querying arbitrarily nested vector/vector-array state variables.
