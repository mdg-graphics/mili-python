# Changelog

All notable changes to Mili-python will be documented in this file.

## [Unreleased]

### Added

### Fixed

### Changed

### Removed

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
