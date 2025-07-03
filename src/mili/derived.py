"""Derived Variables Module.

Supports calculating various derived values from state variables in Mili database
through the DerivedExpressions class.

SPDX-License-Identifier: (MIT)
"""

from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, List, Union, Dict, Optional, Callable, Any, Tuple
from typing_extensions import TypedDict, NotRequired
import numpy as np
from numpy.typing import NDArray
from itertools import groupby

from mili.datatypes import Superclass, QueryDict, QueryLayout
from mili.mdg_defines import DerivedVariables, NodalStateVariables, StressStrainStateVariables, MaterialStateVariables, EntityType

if TYPE_CHECKING:
  from mili.miliinternal import _MiliInternal

class DerivedSpec(TypedDict):
  title: str
  primals: List[str]
  primals_class: List[Optional[str]]
  supports_batching: bool
  compute_function: Callable[...,Any]
  only_sclasses: NotRequired[List[Superclass]]

class QueryArgs(TypedDict):
  svar_names: List[str]
  class_sname: List[str]
  labels: NDArray[np.int32]
  states: NDArray[np.int32]
  ips: NDArray[np.int32]
  result_class_name: str
  elem_node_map: NDArray[np.int32]

class DerivedExpressions:
  """A wrapper class for _MiliInternal that calculates derived results using the primal
  state variables found in a mili database.

  Args:
      db (_MiliInternal): The _MiliInternal interface object

  """
  def __init__(self, db: _MiliInternal):
    self.db = db

    self.__derived_expressions: Dict[str,DerivedSpec] = {
      DerivedVariables.X_DISPLACEMENT.value: DerivedSpec(
        title = "X Displacement",
        primals = [NodalStateVariables.X_POSITION.value],  # The primals needed to compute the derived result
        primals_class =  [None],  # The element class of each primal, None = same as requested class_name.
        supports_batching = False,
        compute_function = self.__compute_node_displacement
      ),
      DerivedVariables.Y_DISPLACEMENT.value: DerivedSpec(
        title = "Y Displacement",
        primals = [NodalStateVariables.Y_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_displacement
      ),
      DerivedVariables.Z_DISPLACEMENT.value: DerivedSpec(
        title = "Z Displacement",
        primals = [NodalStateVariables.Z_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_displacement
      ),
      DerivedVariables.DISPLACEMENT_MAGNITUDE.value: DerivedSpec(
        title = "Displacement Magnitude",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [None, None, None],
        supports_batching = False,
        compute_function = self.__compute_node_displacement_magnitude
      ),
      DerivedVariables.RADIAL_DISPLACEMENT_MAGNITUDE_XY.value: DerivedSpec(
        title = "Radial Displacement Magnitude XY",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value],
        primals_class = [None, None],
        supports_batching = False,
        compute_function = self.__compute_node_radial_displacement
      ),
      DerivedVariables.X_VELOCITY.value: DerivedSpec(
        title = "X Velocity",
        primals = [NodalStateVariables.X_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_velocity
      ),
      DerivedVariables.Y_VELOCITY.value: DerivedSpec(
        title = "Y Velocity",
        primals = [NodalStateVariables.Y_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_velocity
      ),
      DerivedVariables.Z_VELOCITY.value: DerivedSpec(
        title = "Z Velocity",
        primals = [NodalStateVariables.Z_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_velocity
      ),
      DerivedVariables.X_ACCELERATION.value: DerivedSpec(
        title = "X Acceleration",
        primals = [NodalStateVariables.X_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_acceleration
      ),
      DerivedVariables.Y_ACCELERATION.value: DerivedSpec(
        title = "Y Acceleration",
        primals = [NodalStateVariables.Y_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_acceleration
      ),
      DerivedVariables.Z_ACCELERATION.value: DerivedSpec(
        title = "Z Acceleration",
        primals = [NodalStateVariables.Z_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_node_acceleration
      ),
      DerivedVariables.VOLUMETRIC_STRAIN.value: DerivedSpec(
        title = "Volumetric Strain",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value],
        primals_class = [None, None, None],
        supports_batching = False,
        compute_function = self.__compute_vol_strain
      ),
      DerivedVariables.PRINCIPAL_STRAIN_1.value: DerivedSpec(
        title = "Principal Strain 1",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_strain
      ),
      DerivedVariables.PRINCIPAL_STRAIN_2.value: DerivedSpec(
        title = "Principal Strain 2",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_strain
      ),
      DerivedVariables.PRINCIPAL_STRAIN_3.value: DerivedSpec(
        title = "Principal Strain 3",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_strain
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_1.value: DerivedSpec(
        title = "Principal Deviatoric Strain 1",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_dev_principal_strain
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_2.value: DerivedSpec(
        title = "Principal Deviatoric Strain 2",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_dev_principal_strain
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_3.value: DerivedSpec(
        title = "Principal Deviatoric Strain 3",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_dev_principal_strain
      ),
      # Alternate calculation methods used to check for possibility of errors.
      DerivedVariables.PRINCIPAL_STRAIN_1_ALT.value: DerivedSpec(
        title = "Principal Strain 1 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_STRAIN_2_ALT.value: DerivedSpec(
        title = "Principal Strain 2 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_STRAIN_3_ALT.value: DerivedSpec(
        title = "Principal Strain 3 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_1_ALT.value: DerivedSpec(
        title = "Principal Deviatoric Strain 1 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_dev_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_2_ALT.value: DerivedSpec(
        title = "Principal Deviatoric Strain 2 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_dev_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_DEV_STRAIN_3_ALT.value: DerivedSpec(
        title = "Principal Deviatoric Strain 3 (alt)",
        primals = [StressStrainStateVariables.X_STRAIN.value,
                   StressStrainStateVariables.Y_STRAIN.value,
                   StressStrainStateVariables.Z_STRAIN.value,
                   StressStrainStateVariables.XY_STRAIN.value,
                   StressStrainStateVariables.YZ_STRAIN.value,
                   StressStrainStateVariables.ZX_STRAIN.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_dev_principal_strain_alt
      ),
      DerivedVariables.PRINCIPAL_STRESS_1.value: DerivedSpec(
        title = "Principal Stress 1",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_stress
      ),
      DerivedVariables.PRINCIPAL_STRESS_2.value: DerivedSpec(
        title = "Principal Stress 2",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_stress
      ),
      DerivedVariables.PRINCIPAL_STRESS_3.value: DerivedSpec(
        title = "Principal Stress 3",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_stress
      ),
      DerivedVariables.EFFECTIVE_STRESS.value: DerivedSpec(
        title = "Effective Stress",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_effective_stress
      ),
      DerivedVariables.PRESSURE.value: DerivedSpec(
        title = "Pressure",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value],
        primals_class = [None, None, None],
        supports_batching = False,
        compute_function = self.__compute_pressure
      ),
      DerivedVariables.PRINCIPAL_DEV_STRESS_1.value: DerivedSpec(
        title = "Principal Deviatoric Stress 1",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_dev_stress
      ),
      DerivedVariables.PRINCIPAL_DEV_STRESS_2.value: DerivedSpec(
        title = "Principal Deviatoric Stress 2",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_dev_stress
      ),
      DerivedVariables.PRINCIPAL_DEV_STRESS_3.value: DerivedSpec(
        title = "Principal Deviatoric Stress 3",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = True,
        compute_function = self.__compute_principal_dev_stress
      ),
      DerivedVariables.MAX_SHEAR_STRESS.value: DerivedSpec(
        title = "Maximum Shear Stress",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_max_shear_stress
      ),
      DerivedVariables.TRIAXIALITY.value: DerivedSpec(
        title = "Triaxiality",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_triaxiality
      ),
      DerivedVariables.NORMALIZED_PRESSURE.value: DerivedSpec(
        title = "Normalized Pressure",
        primals =  [StressStrainStateVariables.X_STRESS.value,
                    StressStrainStateVariables.Y_STRESS.value,
                    StressStrainStateVariables.Z_STRESS.value,
                    StressStrainStateVariables.XY_STRESS.value,
                    StressStrainStateVariables.YZ_STRESS.value,
                    StressStrainStateVariables.ZX_STRESS.value],
        primals_class = [None, None, None, None, None, None],
        supports_batching = False,
        compute_function = self.__compute_normalized_pressure
      ),
      DerivedVariables.EPS_RATE.value: DerivedSpec(
        title = "Equiv. Plastic Strain Rate",
        primals = [StressStrainStateVariables.EQUIV_PLASTIC_STRAIN.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_plastic_strain_rate
      ),
      DerivedVariables.TANGENTIAL_TRACTION_MAGNITUDE.value: DerivedSpec(
        title = "Nodal Tangential Traction Magnitude",
        primals = [NodalStateVariables.X_TANGENTIAL_TRACTION.value,
                   NodalStateVariables.Y_TANGENTIAL_TRACTION.value,
                   NodalStateVariables.Z_TANGENTIAL_TRACTION.value],
        primals_class = [None, None, None],
        supports_batching = False,
        compute_function = self.__compute_nodal_tangential_traction_magnitude
      ),
      DerivedVariables.MAT_COG_DISP_X.value: DerivedSpec(
        title = "Material Center of Gravity X Displacement",
        primals = [MaterialStateVariables.CENTER_OF_GRAVITY_X_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_material_cog_displacement
      ),
      DerivedVariables.MAT_COG_DISP_Y.value: DerivedSpec(
        title = "Material Center of Gravity Y Displacement",
        primals = [MaterialStateVariables.CENTER_OF_GRAVITY_Y_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_material_cog_displacement
      ),
      DerivedVariables.MAT_COG_DISP_Z.value: DerivedSpec(
        title = "Material Center of Gravity Z Displacement",
        primals = [MaterialStateVariables.CENTER_OF_GRAVITY_Z_POSITION.value],
        primals_class = [None],
        supports_batching = False,
        compute_function = self.__compute_material_cog_displacement
      ),
      DerivedVariables.ELEMENT_VOLUME.value: DerivedSpec(
        title = "Element Volume",
        primals = [NodalStateVariables.NODAL_POSITION.value],
        primals_class = [EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_element_volume,
        only_sclasses = [Superclass.M_HEX, Superclass.M_TET]
      ),
      DerivedVariables.AREA.value: DerivedSpec(
        title = "Quad Area",
        primals = [NodalStateVariables.NODAL_POSITION.value],
        primals_class = [EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_quad_area,
        only_sclasses = [Superclass.M_QUAD],
      ),
      DerivedVariables.CENTROID.value: DerivedSpec(
        title = "Centroid Position",
        primals = [NodalStateVariables.NODAL_POSITION.value],
        primals_class = [EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_centroid,
      ),
      DerivedVariables.SURFACE_STRAIN_X.value: DerivedSpec(
        title = "Surface Strain X",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      DerivedVariables.SURFACE_STRAIN_Y.value: DerivedSpec(
        title = "Surface Strain Y",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      DerivedVariables.SURFACE_STRAIN_Z.value: DerivedSpec(
        title = "Surface Strain Z",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      DerivedVariables.SURFACE_STRAIN_XY.value: DerivedSpec(
        title = "Surface Strain XY",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      DerivedVariables.SURFACE_STRAIN_YZ.value: DerivedSpec(
        title = "Surface Strain YZ",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      DerivedVariables.SURFACE_STRAIN_ZX.value: DerivedSpec(
        title = "Surface Strain ZX",
        primals = [NodalStateVariables.X_POSITION.value,
                   NodalStateVariables.Y_POSITION.value,
                   NodalStateVariables.Z_POSITION.value],
        primals_class = [EntityType.NODE.value,
                         EntityType.NODE.value,
                         EntityType.NODE.value],
        supports_batching = False,
        compute_function = self.__compute_surface_strain,
        only_sclasses = [Superclass.M_HEX],
      ),
      # TODO: Add more primals here
    }

  def supported_variables(self) -> List[str]:
    """Return a list of derived expressions that are supported.

    NOTE: This does not mean all derived variables can be calculated for a given simulation.
          Only that mili-python can caclulate them if all required inputs exist.
    """
    return list(self.__derived_expressions.keys())

  def derived_variable_titles(self) -> Dict[str,str]:
    """Return dictionary containing the title for each derived variable."""
    return { var: spec["title"] for var, spec in self.__derived_expressions.items() }

  def __variable_exists_for_class(self, variable: str, class_name: str) -> bool:
    primal_exists = class_name in self.db.classes_of_state_variable(variable)
    try:
      derived_exists = class_name in self.db.classes_of_derived_variable(variable)
    except:
      derived_exists = False
    return primal_exists or derived_exists

  def derived_variables_of_class(self, class_name: str) -> List[str]:
    """Return list of derived variables that can be calculated for a given class."""
    derived_list = []
    if class_name in self.db.class_names():
      class_def = self.db.mesh_object_classes()[class_name]
      queriable_state_variables = self.db.queriable_svars()
      for var_name, specs in self.__derived_expressions.items():
        if 'only_sclasses' in specs:
          if class_def.sclass not in specs['only_sclasses']:
            continue
        primals_found = []
        for req_primal, req_primal_class in zip(specs['primals'], specs['primals_class']):
          # Check that primal exists
          if req_primal in queriable_state_variables or req_primal in self.__derived_expressions:
            # Check that primal exists for required element class
            req_primal_class = class_name if req_primal_class is None else req_primal_class
            if self.__variable_exists_for_class(req_primal, req_primal_class):
              primals_found.append(True)
        # Check if all primals were found
        if len(primals_found) == len(specs['primals']) and all(primals_found):
          derived_list.append(var_name)
    return derived_list

  def classes_of_derived_variable(self, var_name: str) -> List[str]:
    """Return list of element classes for which the specified derived variable can be calculated."""
    if var_name not in self.__derived_expressions:
      raise KeyError(f"The derived result '{var_name}' does not exist")
    derived_spec = self.__derived_expressions[var_name]
    classes_of_derived = []
    element_class_data = self.db.mesh_object_classes()

    if all( [primal_class is None for primal_class in derived_spec['primals_class']] ):
      # CASE 1: All primals must exist for same element class as derived result
      for class_name, class_def in element_class_data.items():
        if "only_sclasses" in derived_spec:
          if class_def.sclass not in derived_spec["only_sclasses"]:
            continue
        primals_found = [ self.__variable_exists_for_class(primal, class_name) for primal in derived_spec['primals'] ]
        if all(primals_found):
          classes_of_derived.append(class_name)
    else:
      # CASE 2: primals must exists for class different from derived result
      for class_name, class_def in element_class_data.items():
        if "only_sclasses" in derived_spec:
          if class_def.sclass not in derived_spec["only_sclasses"]:
            continue
        primals_found = [self.__variable_exists_for_class(primal, primal_class) for primal, primal_class  # type: ignore  # mypy thinks primal_class might be None, but won't be.
                         in zip(derived_spec['primals'],derived_spec["primals_class"])]
        if all(primals_found):
          classes_of_derived.append(class_name)

    return classes_of_derived

  def find_batchable_queries(self, result_names: List[str]) -> List[List[str]]:
    """Determine if any derived queries can be batched based on the result names."""
    groups = []
    result_names = sorted(result_names)
    for res in result_names:
      if res in self.__derived_expressions:
        groups.append( (res, self.__derived_expressions[res]["compute_function"].__name__, self.__derived_expressions[res].get("supports_batching", False)) )
    grouped_by_compute_function = [list(g) for _,g in groupby(groups, lambda x: x[1])]
    final_groups: List[List[str]] = []
    for group in grouped_by_compute_function:
      if all(g[2] for g in group):
        final_groups.append([g[0] for g in group])
      else:
        for g in group:
          final_groups.append([g[0]])
    return final_groups

  def __init_derived_query_parameters(self,
                                      result_names: List[str],
                                      class_name: str,
                                      labels: NDArray[np.int32],
                                      states: NDArray[np.int32],
                                      ips: NDArray[np.int32]) -> Tuple[List[str],str,NDArray[np.int32],NDArray[np.int32],NDArray[np.int32]]:
    """
    Parse the query parameters and normalize them to the types expected by the rest of the query operation,
        throws TypeException and/or ValueException as appropriate. Does not throw exceptions in cases where
        the result of the argument deviation from the norm would be expected to be encountered when operating in parallel.
    """
    # Check that derived results are supported
    for result_name in result_names:
      if result_name not in self.__derived_expressions:
        raise ValueError(f"The derived result '{result_name}' is not supported.")

    # Check that class name exists in problem
    if class_name not in self.db.class_names():
      raise ValueError(f"The class name '{class_name}' does not exist.")

    # NOTE: Valid state range was already checked by MiliDatabase.
    if not isinstance( states, np.ndarray ):
      raise TypeError( f"'states' must be None, an integer, or a list of integers" )

    # Get labels that are available on the current processor
    available_labels = self.db.labels().get( class_name, np.empty([0],np.int32) )
    labels = np.intersect1d( labels, available_labels )

    if not isinstance(ips, (list,np.ndarray)):
      raise TypeError( 'comp must be an integer or list of integers' )
    # Ensure not duplicate integration points
    ips = np.unique( np.array( ips, dtype=np.int32 ))

    return result_names, class_name, labels, states, ips

  def __parse_derived_result_spec(self, result_names: List[str], class_name: str,
                                  **kwargs: Any) -> Tuple[List[str],List[str],Dict[str,Any],bool,Callable[...,Any]]:
    """Parse the derived result specification, perform error handling, set up kwargs (if necessary)."""
    # There will always be at least one result name. If there are more than one result names then
    # the query specification will be the same for all so we can just use the first result
    result_name = result_names[0]

    required_variables = self.__derived_expressions[result_name]['primals']
    primal_class_names = self.__derived_expressions[result_name]['primals_class']
    primal_classes = [ pclass if pclass is not None else class_name for pclass in primal_class_names]
    compute_function = self.__derived_expressions[result_name]['compute_function']
    supports_batching = self.__derived_expressions[result_name]['supports_batching']

    # Check that the element classes exists in problem.
    for primal_class in primal_classes:
      if primal_class not in self.db.class_names():
        raise ValueError(f"The primal class name '{primal_class}' does not exist.")

    # Check that all required primals exist for the desired element class.
    primals_found_for_class = []
    for primal, pclass in zip(required_variables, primal_classes):
      primals_found_for_class.append( self.__variable_exists_for_class(primal, pclass) )
    if not all( primals_found_for_class ):
      raise ValueError((f"The required variables do not all exist for the required_classes\n"
                        f"required_variables = {required_variables}\n"
                        f"required_classes = {primal_classes}"))

    # Check that computation is supported for this element superclass
    only_sclasses = self.__derived_expressions[result_name].get("only_sclasses", None)
    if only_sclasses is not None:
      # This derived result can only be calculated for a subset of element superclasses
      class_def = self.db.mesh_object_classes().get(class_name)
      if class_def is not None:
        if class_def.sclass not in only_sclasses:
          raise ValueError((f"The derived result '{result_name}' is not supported for the '{class_def.sclass.name}' element superclass\n"
                            f"This result is supported for: {only_sclasses}"))

    # Handle keyword arguments
    # Use function reflection to get list of arguments for the derived compute function
    function_signature = inspect.signature(compute_function)
    function_arguments = set(function_signature.parameters.keys())

    # Check that user has not provide any extra arguments.
    for keyword in kwargs:
      if keyword not in function_arguments:
        raise ValueError(f"Unexpected keyword argument '{keyword}' was provided.")

    # Check that any keyword arguments have been provided.
    consistant_arguments = set(['self', 'result_name', 'result_names', 'primal_data', 'query_args'])
    keyword_arguments = function_arguments - consistant_arguments
    if keyword_arguments:
      for keyword in keyword_arguments:
        if keyword not in kwargs:
          # Check for default value
          if function_signature.parameters[keyword].default == inspect.Parameter.empty:
            raise ValueError((f"For the derived result '{result_name}' you must "
                              f"provide the key word argument '{keyword}'"))

    return required_variables, primal_classes, kwargs, supports_batching, compute_function

  def __generate_elem_node_map(self, elem_node_association: Tuple[NDArray[np.int32],NDArray[np.int32]]) -> Tuple[NDArray[np.int32],NDArray[np.int32]]:
    """Generate masks for each element that maps to nodal data.

    Args:
      elem_node_association: The output of the nodes_to_elem MiliDatabase function.
    """
    elem_node_map = []
    nodes_by_elem, elem_order = elem_node_association

    # Get list of unique nodes that need to be queried
    labels_to_query = np.unique( nodes_by_elem.flatten() )

    # Generate mask for each elements nodal results
    sorter = np.argsort(labels_to_query)
    for _, associated_nodes in zip( elem_order, nodes_by_elem ):
      # We use searchsorted to maintain node order
      elem_node_map.append( sorter[np.searchsorted(labels_to_query, associated_nodes, sorter=sorter)] )
    return labels_to_query, np.array(elem_node_map)

  def __query_required_primals(self,
                               required_primals: List[str],
                               primal_classes: List[str],
                               class_name: str,
                               labels : NDArray[np.int32],
                               states : NDArray[np.int32],
                               ips: NDArray[np.int32]) -> Tuple[Dict[str,QueryDict],QueryArgs]:
    """Query all the required primals for a derived calculation."""
    query_args = QueryArgs(
      svar_names = required_primals,
      class_sname = primal_classes,
      labels = labels,
      states = states,
      ips = ips,
      result_class_name = class_name,
      elem_node_map = np.empty([0], dtype=np.int32)
    )

    # Group required primals by class_name
    unique_classes = list(set(primal_classes))
    primals_grouped_by_class: List[Tuple[List[str],str]] = []
    for cname in unique_classes:
      idxs_of_primals = np.where( np.isin(primal_classes, cname) )
      grouped_primals: List[str] = list(np.array(required_primals)[idxs_of_primals])
      primals_grouped_by_class.append( (grouped_primals, cname) )

    # query each group of states variables for each class
    primal_data = {}
    for primal_names, primal_class_name in primals_grouped_by_class:
      # By default use the labels requested for the derived result
      labels_to_query = labels

      # Special case:
      if primal_class_name != class_name:
        # Element class of derived result does not match element class of one or more required primal
        # so we need to map the labels from one super class to another. Most commonly this
        # will be getting all the nodes that make up each of the requested elements.
        if primal_class_name == 'node':
          associated_nodes = self.db.nodes_of_elems( class_name, labels )
          labels_to_query, elem_node_map = self.__generate_elem_node_map( associated_nodes )
          query_args['elem_node_map'] = elem_node_map
        else:
          # Don't know if there are any other cases, skip for now...
          continue

      # Do the query
      primal_data.update( self.db.query(primal_names, primal_class_name, None, labels_to_query, states, ips) )

    return primal_data, query_args

  def query(self,
            result_names: List[str],
            class_name: str,
            labels : NDArray[np.int32],
            states : NDArray[np.int32],
            ips : NDArray[np.int32],
            **kwargs: Any) -> Dict[str,QueryDict]:
    """General derived result function.

    Args:
      result_names (Union[List[str],str]): The list of derived results to compute.
      class_sname (str): The element class name being queried (e.g. brick. shell, node).
      labels (NDArray[np.int32]): Labels to query data for.
      states (NDArray[np.int32]): State numbers from which to query data.
      ips (NDArray[np.int32]): Integration point to query for vector array state variables.
    """
    # Validate incoming data
    result_names, class_name, labels, states, ips = self.__init_derived_query_parameters(result_names, class_name, labels, states, ips)

    # Group together derived variables that can be calculated with a single call
    batchable_results = self.find_batchable_queries( result_names )

    derived_result = {}
    for group in batchable_results:
      # Get specifications for this derived result
      required_primals, primal_classes, kwargs, supports_batching, compute_function = self.__parse_derived_result_spec( group, class_name, **kwargs )

      # Gather the required primal data
      primal_data, query_args = self.__query_required_primals( required_primals, primal_classes, class_name, labels, states, ips )

      # Call function to derive result
      if supports_batching:
        derived_result.update( compute_function( group, primal_data, query_args, **kwargs ) )
      else:
        derived_result.update( compute_function( group[0], primal_data, query_args, **kwargs ) )

    return derived_result

  def __initialize_result_dictionary(self,
                                     result_names: Union[str,List[str]],
                                     primal_data: Dict[str,QueryDict],
                                     class_name: str ) -> Dict[str,QueryDict]:
    """Initialize the empty dictionary structure to store the derived results in."""
    primal = list(primal_data.keys())[0]
    derived_result = {}
    if isinstance(result_names, str):
      result_names = [result_names]
    for result_name in result_names:
      derived_result[result_name] = QueryDict(
        class_name = class_name,
        source = 'derived',
        data = np.empty_like( primal_data[primal]['data'] ),
        title = self.__derived_expressions[result_name]['title'],
        layout = primal_data[primal]['layout']
      )
      derived_result[result_name]['layout']['components'] = [result_name]
    return derived_result

  def __get_nodal_reference_positions( self, components: Union[List[str],str], reference_state: int, labels: NDArray[np.int32] ) -> NDArray[np.floating]:
    """Load the nodal reference positions for the specified state and nodes."""
    if isinstance(components, str):
      components = [components]
    qty_components: int = len(components)

    result_idxs = np.where( np.isin(['ux','uy','uz'], components) )[0]

    # Determine what labels we have
    labels_of_class = self.db.labels().get( 'node', np.empty([0], dtype=np.int32) )
    ordinals = np.where( np.isin( labels_of_class, labels ) )[0]
    qty_nodes = len(ordinals)

    reference_data: NDArray[np.floating]
    if reference_state == 0:
      # Use initial nodal positions
      initial_positions = self.db.nodes()
      # Get x,y,z primal values for specified elements
      reference_data = (initial_positions[ordinals])[:,result_idxs]
    else:
      # Query specific state from reference data
      data_query = self.db.query( components, 'node', None, labels, reference_state)

      # Create array with the reference nodal coordinates for each node
      reference_data = np.zeros((qty_nodes,qty_components))
      for idx, comp in enumerate(components):
        reference_data[:,idx:idx+1] = data_query[comp]['data'][0]

    return reference_data

  def __compute_node_displacement(self,
                                  result_name: str,
                                  primal_data: Dict[str,QueryDict],
                                  query_args: QueryArgs,
                                  reference_state: int = 0) -> Dict[str,QueryDict]:
    """Calculate the derived result 'disp_x', 'disp_y', or 'disp_z'."""
    labels = query_args['labels']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['disp_x', 'disp_y', 'disp_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( required_primal, reference_state, labels )

    # Compute the displacement
    disp = primal_data[required_primal]['data'] - reference_data
    derived_result[result_name]['data'] = disp

    return derived_result

  def __compute_node_displacement_magnitude(self,
                                  result_name: str,
                                  primal_data: Dict[str,QueryDict],
                                  query_args: QueryArgs,
                                  reference_state: int = 0) -> Dict[str,QueryDict]:
    """Calculate the derived result 'disp_mag'."""
    labels = query_args['labels']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( ['ux', 'uy', 'uz'], reference_state, labels )

    # Compute the displacement components
    dx = primal_data['ux']['data'] - reference_data[:,:1]
    dy = primal_data['uy']['data'] - reference_data[:,1:2]
    dz = primal_data['uz']['data'] - reference_data[:,2:]

    derived_result[result_name]['data'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return derived_result

  def __compute_node_radial_displacement(self,
                                  result_name: str,
                                  primal_data: Dict[str,QueryDict],
                                  query_args: QueryArgs,
                                  reference_state: int = 0) -> Dict[str,QueryDict]:
    """Calculate the derived result 'disp_rad_mag_xy'."""
    labels = query_args['labels']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Get reference nodal positions
    reference_data = self.__get_nodal_reference_positions( ['ux', 'uy', 'uz'], reference_state, labels )

    # Compute the displacement components
    dx = primal_data['ux']['data'] - reference_data[:,:1]
    dy = primal_data['uy']['data'] - reference_data[:,1:2]

    derived_result[result_name]['data'] = np.sqrt(dx**2 + dy**2)

    return derived_result

  def __compute_node_velocity(self,
                              result_name: str,
                              primal_data: Dict[str,QueryDict],
                              query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'vel_x', 'vel_y', or 'vel_z'."""
    labels = query_args['labels']
    states = query_args['states']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['vel_x', 'vel_y', 'vel_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    mask = (states != 1)  # Boolean list used to exclude 1st state (if requested)
    states_prev = states[mask] - 1  # list of previous states

    derived_result[result_name]['data'][~mask] = 0  # velocity at 1st state set to zero
    if len(states_prev) > 0:
      # Query data for the previous states
      previous_data = self.db.query( required_primal, 'node', None, labels, states_prev)[required_primal]['data']

      # Compute the displacements
      disp = primal_data[required_primal]['data'][mask] - previous_data

      # Compute dt
      time_prev = self.db.times()[states_prev-1]  # offset because state 1 = index 0 for these arrays
      time_curr = self.db.times()[states[mask]-1]
      dt_inv = 1.0 / (time_curr - time_prev)
      dt_inv = np.expand_dims(dt_inv, 1)  # add dummy dimension to convert from scalar to 2D
      dt_inv = np.repeat(dt_inv, disp.shape[1], axis=1)  # Repeat values so we have the same shape as disp
      dt_inv = np.expand_dims(dt_inv, 2)  # add dummy dimension to convert from 2d to 3D

      derived_result[result_name]['data'][mask] = disp * dt_inv

    return derived_result

  def __compute_node_acceleration(self,
                                  result_name: str,
                                  primal_data: Dict[str,QueryDict],
                                  query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'acc_x', 'acc_y', or 'acc_z'."""
    labels = query_args['labels']
    states = query_args['states']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['acc_x', 'acc_y', 'acc_z'].index( result_name )
    required_primal = ['ux', 'uy', 'uz'][result_idx]

    # Create array masks (boolean lists) for operating on parts of the request
    max_st = len(self.db.state_maps())  # last state in the database
    mask_cent = (states != 1) & (states != max_st)  # mask for central diff. calculations
    mask_fwd = (states == 1)  # mask for forward diff. calculations
    mask_back = (states == max_st)  # mask for backward diff. calculations

    states_prev = states[mask_cent] - 1  # list of previous states
    states_next = states[mask_cent] + 1  # list of next states

    # Query data for the previous states (center diff. only)
    u_p_cent = self.db.query( required_primal, 'node', None, labels, states_prev)[required_primal]['data']

    # Query data for the next states (center diff. only)
    u_n_cent = self.db.query( required_primal, 'node', None, labels, states_next)[required_primal]['data']

    # Get the data for the current states  (center diff. only)
    u_c_cent = primal_data[required_primal]['data'][mask_cent]

    times = self.db.times()

    # Central Difference Calculation
    if any(mask_cent==True):
      delta_t = 0.5 * (times[states_next-1] - times[states_prev-1])  # Calculate average dt

      one_over_tsqr = 1 / (delta_t**2)
      one_over_tsqr = np.expand_dims(one_over_tsqr, 1)  # add dummy dimension to convert from scalar to 2D
      one_over_tsqr = np.repeat(one_over_tsqr, u_n_cent.shape[1], axis=1)  # Repeat values so we have the same shape as u_n_cent
      one_over_tsqr = np.expand_dims(one_over_tsqr, 2)  # add dummy dimension to convert from 2d to 3D

      accel = (u_n_cent - 2*u_c_cent + u_p_cent) * one_over_tsqr
      derived_result[result_name]['data'][mask_cent] = accel

    # Use backward differnce if the last state is requested
    # Requires 2 previous states, and assumes no duplicate states (i.e. 1 state)
    if any(mask_back==True):
      u_p = self.db.query( required_primal, 'node', None, labels, max_st-1)[required_primal]['data']  # previous state coordinates
      u_pp = self.db.query( required_primal, 'node', None, labels, max_st-2)[required_primal]['data']  # previous-previous state coordinates
      u_c = primal_data[required_primal]['data'][mask_back]  # current state coordinates

      delta_t = 0.5 * (times[-1] - times[-3])  # average time step
      one_over_tsqr = 1 / (delta_t**2)
      accel = (u_c - 2*u_p + u_pp) * one_over_tsqr

      derived_result[result_name]['data'][mask_back] = accel

    # Use forward differnce if the first state is requested
    # Requires 2 forward states, and assumes no duplicate states (i.e. 1 state)
    if any(mask_fwd==True):
      u_n = self.db.query( required_primal, 'node', None, labels, 2)[required_primal]['data']  # previous state coordinates
      u_nn = self.db.query( required_primal, 'node', None, labels, 3)[required_primal]['data']  # previous-previous state coordinates
      u_c = primal_data[required_primal]['data'][mask_fwd]  # current state coordinates

      delta_t = 0.5 * (times[2] - times[0])  # average time step
      one_over_tsqr = 1 / (delta_t**2)
      accel = (u_nn - 2*u_n + u_c) * one_over_tsqr

      derived_result[result_name]['data'][mask_fwd] = accel

    return derived_result

  def __compute_vol_strain(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'vol_strain'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Perform principal stress computation
    ex = primal_data['ex']['data']
    ey = primal_data['ey']['data']
    ez = primal_data['ez']['data']
    derived_result[result_name]['data'] = ex + ey + ez

    return derived_result

  def __compute_principal_strain(self,
                         result_names: List[str],
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_strain1', 'prin_strain2', 'prin_strain3'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_names, primal_data, class_name )

    # Perform principal stress computation
    ex = primal_data['ex']['data']
    ey = primal_data['ey']['data']
    ez = primal_data['ez']['data']
    exy = primal_data['exy']['data']
    eyz = primal_data['eyz']['data']
    ezx = primal_data['ezx']['data']

    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [ex, exy, ezx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [exy, ey, eyz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [ezx, eyz, ez])  # 3 x num_states x num_labels x 1 array
    strain = np.stack( [x_col, y_col, z_col])  # full 5-d Strain array

    # Strain is 3 x 3 x states x elems x ipts,
    # Shift to be states x elems x ipts x 3 x 3 so that first three indices match result_matrix.
    strain = np.moveaxis(strain, [0,1,2,3,4], [3,4,0,1,2])

    # Compute eigenvalues
    eigen_values = np.linalg.eigvalsh(strain)

    # Extract correct component
    for result_name in result_names:
      result_matrix = np.empty_like(ex)
      if result_name == "prin_strain1":
        result_matrix = np.max(eigen_values, axis=3)
      elif result_name =='prin_strain3':
        result_matrix = np.min(eigen_values, axis=3)
      elif result_name =='prin_strain2':
        result_matrix = eigen_values[:,:,:,1]  # get the 2nd value
      else:
        raise ValueError
      derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_principal_strain_alt(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt'."""
    """
    Uses an alternate calculation method based on the griz algorithm. This was used to help debug
    the code by checking to see if the methods matched.
    """
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    ex = primal_data['ex']['data']
    ey = primal_data['ey']['data']
    ez = primal_data['ez']['data']
    exy = primal_data['exy']['data']
    eyz = primal_data['eyz']['data']
    ezx = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain

    # Create 6 x num_states x num_labels x 1 array
    dev_strain = np.stack( [ex-e_hyd, ey-e_hyd, ez-e_hyd, exy, eyz, ezx] )
    # Calc J2
    J2 = -(dev_strain[0,:,:,:]*dev_strain[1,:,:,:] \
          + dev_strain[1,:,:,:]*dev_strain[2,:,:,:] \
          + dev_strain[0,:,:,:]*dev_strain[2,:,:,:]) \
            + dev_strain[3,:,:,:]**2 \
            + dev_strain[4,:,:,:]**2 \
            + dev_strain[5,:,:,:]**2

    # Calc J3
    J3 = -dev_strain[0,:,:,:] * dev_strain[1,:,:,:] * dev_strain[2,:,:,:] \
        - 2.0 * dev_strain[3,:,:,:] * dev_strain[4,:,:,:] * dev_strain[5,:,:,:] \
        + dev_strain[0,:,:,:] * dev_strain[4,:,:,:]**2 \
        + dev_strain[1,:,:,:] * dev_strain[5,:,:,:]**2 \
        + dev_strain[2,:,:,:] * dev_strain[3,:,:,:]**2

    # Calculate a limit check value for all J2!=0.
    nz_mask = J2 > 0.0  # mask for non-zero J2 values (J2 is never negative)
    limit_check = np.zeros_like(J2)  # default value is zero (for J2<=0)
    limit_check[nz_mask] = J2[nz_mask]

    # Break out elements that pass the limit check (non-zero J2)
    limit_mask = limit_check >= 1e-12  # elements that pass the limit check
    J2_slice = J2[limit_mask]
    J3_slice = J3[limit_mask]

    alpha = -0.5 * np.sqrt(27/J2_slice) * J3_slice/J2_slice

    # Limit alpha to the range -1 to 1
    alpha[alpha<0] = np.maximum(alpha[alpha<0], -1.0)
    alpha[alpha>0] = np.minimum(alpha[alpha>0], 1.0)

    # Calculate the load angle (in rad)
    angle = np.arccos(alpha) * (1/3)

    # Calculate an intermediate value
    value = 2 * np.sqrt(J2_slice * (1/3))

    # Modify the load angle for 2nd and 3rd principal stresses
    if result_name=='prin_strain2_alt':
        angle = angle - 2*np.pi * (1/3)
    elif result_name=='prin_strain3_alt':
        angle = angle + 2*np.pi * (1/3)

    # Calculate the requested principal stress (zero if limit_check failed)
    princ_strain = np.zeros_like(J2)
    princ_strain[limit_mask] = value * np.cos(angle)

    princ_strain[limit_mask] = princ_strain[limit_mask] + e_hyd[limit_mask]  # convert to principal strain

    derived_result[result_name]['data'] = princ_strain

    return derived_result

  def __compute_dev_principal_strain(self,
                         result_names: List[str],
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_names, primal_data, class_name )

    ex = primal_data['ex']['data']
    ey = primal_data['ey']['data']
    ez = primal_data['ez']['data']
    exy = primal_data['exy']['data']
    eyz = primal_data['eyz']['data']
    ezx = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain

    # Create 3 x 3 x num_states x num_labels x num_ipt array
    x_col = np.stack( [ex-e_hyd, exy, ezx])  # 3 x num_states x num_labels x num_ipt array
    y_col = np.stack( [exy, ey-e_hyd, eyz])  # 3 x num_states x num_labels x num_ipt array
    z_col = np.stack( [ezx, eyz, ez-e_hyd])  # 3 x num_states x num_labels x num_ipt array
    dev_strain = np.stack( [x_col, y_col, z_col])  # full 5-d strain array

    # Strain is 3 x 3 x states x elems x ipts,
    # Shift to be states x elems x ipts x 3 x 3 so that first three indices match result_matrix.
    dev_strain = np.moveaxis(dev_strain, [0,1,2,3,4], [3,4,0,1,2])

    # Compute eigenvalues
    eigen_values = np.linalg.eigvalsh(dev_strain)

    # Extract correct component
    for result_name in result_names:
      result_matrix = np.empty_like(ex)
      if result_name == "prin_dev_strain1":
        result_matrix = np.max(eigen_values, axis=3)
      elif result_name =='prin_dev_strain3':
        result_matrix = np.min(eigen_values, axis=3)
      elif result_name =='prin_dev_strain2':
        result_matrix = eigen_values[:,:,:,1]  # get the 2nd value
      else:
        raise ValueError
      derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_dev_principal_strain_alt(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt'."""
    """
    Uses an alternate calculation method based on the griz algorithm. This was used to help debug
    the code by checking to see if the methods matched.
    Can also calculate the derived result 'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt'.
    """
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    ex = primal_data['ex']['data']
    ey = primal_data['ey']['data']
    ez = primal_data['ez']['data']
    exy = primal_data['exy']['data']
    eyz = primal_data['eyz']['data']
    ezx = primal_data['ezx']['data']

    e_hyd = (1/3) * (ex + ey + ez)  # hydrostatic strain

    # Create 6 x num_states x num_labels x 1 array
    dev_strain = np.stack( [ex-e_hyd, ey-e_hyd, ez-e_hyd, exy, eyz, ezx] )

    # Calc J2
    J2 = -(dev_strain[0,:,:,:]*dev_strain[1,:,:,:] \
          + dev_strain[1,:,:,:]*dev_strain[2,:,:,:] \
          + dev_strain[0,:,:,:]*dev_strain[2,:,:,:]) \
            + dev_strain[3,:,:,:]**2 \
            + dev_strain[4,:,:,:]**2 \
            + dev_strain[5,:,:,:]**2

    # Calc J3
    J3 = -dev_strain[0,:,:,:] * dev_strain[1,:,:,:] * dev_strain[2,:,:,:] \
        - 2.0 * dev_strain[3,:,:,:] * dev_strain[4,:,:,:] * dev_strain[5,:,:,:] \
        + dev_strain[0,:,:,:] * dev_strain[4,:,:,:]**2 \
        + dev_strain[1,:,:,:] * dev_strain[5,:,:,:]**2 \
        + dev_strain[2,:,:,:] * dev_strain[3,:,:,:]**2

    # Calculate a limit check value for all J2!=0.
    nz_mask = J2 > 0.0  # mask for non-zero J2 values (J2 is never negative)
    limit_check = np.zeros_like(J2)  # default value is zero (for J2<=0)
    limit_check[nz_mask] = J2[nz_mask]

    # Break out elements that pass the limit check (non-zero J2)
    limit_mask = limit_check >= 1e-12  # elements that pass the limit check
    J2_slice = J2[limit_mask]
    J3_slice = J3[limit_mask]

    alpha = -0.5 * np.sqrt(27/J2_slice) * J3_slice/J2_slice

    # Limit alpha to the range -1 to 1
    alpha[alpha<0] = np.maximum(alpha[alpha<0], -1.0)
    alpha[alpha>0] = np.minimum(alpha[alpha>0], 1.0)

    # Calculate the load angle (in rad)
    angle = np.arccos(alpha) * (1/3)

    # Calculate an intermediate value
    value = 2 * np.sqrt(J2_slice * (1/3))

    # Modify the load angle for 2nd and 3rd principal stresses
    if result_name=='prin_strain2_alt' or result_name=='prin_dev_strain2_alt':
        angle = angle - 2 * np.pi * (1/3)
    elif result_name=='prin_strain3_alt' or result_name=='prin_dev_strain3_alt':
        angle = angle + 2 * np.pi * (1/3)

    # Calculate the requested principal stress (zero if limit_check failed)
    princ_strain = np.zeros_like(J2)
    princ_strain[limit_mask] = value * np.cos(angle)

    if 'dev' not in result_name:
      princ_strain[limit_mask] = princ_strain[limit_mask] + e_hyd  # convert to principal strain

    derived_result[result_name]['data'] = princ_strain

    return derived_result

  def __compute_principal_stress(self,
                         result_names: List[str],
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_stress1', 'prin_stress2', 'prin_stress3'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_names, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    # Create 3 x 3 x l x m x 1 array
    x_col = np.stack( [sx, sxy, szx])  # 3 x l x m x 1 array
    y_col = np.stack( [sxy, sy, syz])  # 3 x l x m x 1 array
    z_col = np.stack( [szx, syz, sz])  # 3 x l x m x 1 array
    stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array

    # Stress is 3 x 3 x states x elems x ipts,
    # Shift to be states x elems x ipts x 3 x 3 so that first three indices match result_matrix.
    stress = np.moveaxis(stress, [0,1,2,3,4], [3,4,0,1,2])

    # Compute eigenvalues
    eigen_values = np.linalg.eigvalsh(stress)

    # Extract component based on result(s)
    for result_name in result_names:
      result_matrix = np.empty_like(sx)
      if result_name == "prin_stress1":
        result_matrix = np.max(eigen_values, axis=3)
      elif result_name =='prin_stress3':
        result_matrix = np.min(eigen_values, axis=3)
      elif result_name =='prin_stress2':
        result_matrix = eigen_values[:,:,:,1]  # get the 2nd value
      else:
        raise ValueError

      derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_effective_stress(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'eff_stress'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    """
    NOTE:
    If sx == sy == sz then the pressure should be equal to -sx == -sy == -sz
    However -(1/3) * (sx + sy + sz) can produce values that are slightly off from
    sx, sy, and sz. So when we calculate dev_stress_x|y|z we would expect the values
    to be 0.0, but get results along the lines of 0.000X which leads to incorrect
    results for effective stress.
    """
    rtol = 0.0
    atol = 1e-15
    if np.allclose(sx, sy, rtol=rtol, atol=atol) and np.allclose(sy, sz, rtol=rtol, atol=atol):
      pressure = -sx
    else:
      pressure = -(1/3) * (sx + sy + sz)

    dev_stress_x = sx+pressure
    dev_stress_y = sy+pressure
    dev_stress_z = sz+pressure

    # Calculate the 2nd deviatoric stress invariant, J2
    J2 = 0.5 * (dev_stress_x**2 + dev_stress_y**2 + dev_stress_z**2) \
              + sxy*sxy + syz*syz + szx*szx

    derived_result[result_name]['data'] = np.sqrt(3*J2)

    return derived_result

  def __compute_pressure(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'pressure'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Perform pressure computation
    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    derived_result[result_name]['data'] = (-1/3) * (sx + sy + sz)

    return derived_result

  def __compute_principal_dev_stress(self,
                         result_names: List[str],
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_names, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)

    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [sx+pressure, sxy, szx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [sxy, sy+pressure, syz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [szx, syz, sz+pressure])  # 3 x num_states x num_labels x 1 array
    dev_stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array

    # Stress is 3 x 3 x states x elems x ipts,
    # Shift to be states x elems x ipts x 3 x 3 so that first three indices match result_matrix.
    dev_stress = np.moveaxis(dev_stress, [0,1,2,3,4], [3,4,0,1,2])

    # Compute eigenvalues
    eigen_values = np.linalg.eigvalsh(dev_stress)

    # Extract correct component
    for result_name in result_names:
      result_matrix = np.empty_like(sx)
      if result_name == "prin_dev_stress1":
        result_matrix = np.max(eigen_values, axis=3)
      elif result_name =='prin_dev_stress3':
        result_matrix = np.min(eigen_values, axis=3)
      elif result_name =='prin_dev_stress2':
        result_matrix = eigen_values[:,:,:,1]  # get the 2nd value
      else:
        raise ValueError
      derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_max_shear_stress(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'max_shear_stress'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)

    # Create 3 x 3 x num_states x num_labels x 1 array
    x_col = np.stack( [sx+pressure, sxy, szx])  # 3 x num_states x num_labels x 1 array
    y_col = np.stack( [sxy, sy+pressure, syz])  # 3 x num_states x num_labels x 1 array
    z_col = np.stack( [szx, syz, sz+pressure])  # 3 x num_states x num_labels x 1 array
    stress = np.stack( [x_col, y_col, z_col])  # full 5-d stress array

    result_matrix = np.empty_like(sx)
    # Stress is 3 x 3 x states x elems x ipts,
    # Shift to be states x elems x ipts x 3 x 3 so that first three indices match result_matrix.
    stress = np.moveaxis(stress, [0,1,2,3,4], [3,4,0,1,2])

    eigen_values = np.linalg.eigvalsh(stress)
    result_matrix = 0.5 * (np.max(eigen_values, axis=3) - np.min(eigen_values, axis=3))

    derived_result[result_name]['data'] = result_matrix

    return derived_result

  def __compute_triaxiality(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'triaxiality'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)
    dev_stress_x = sx+pressure
    dev_stress_y = sy+pressure
    dev_stress_z = sz+pressure

    # Calculate the 2nd deviatoric stress invariant, J2
    J2 = 0.5 * (dev_stress_x**2 + dev_stress_y**2 + dev_stress_z**2) \
              + sxy*sxy + syz*syz + szx*szx

    # Calculate the effective stress
    seff = np.sqrt(3*J2)

    derived_result[result_name]['data'] = -pressure/seff

    return derived_result

  def __compute_normalized_pressure(self,
                                    result_name: str,
                                    primal_data: Dict[str,QueryDict],
                                    query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'norm_press'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    sx = primal_data['sx']['data']
    sy = primal_data['sy']['data']
    sz = primal_data['sz']['data']
    sxy = primal_data['sxy']['data']
    syz = primal_data['syz']['data']
    szx = primal_data['szx']['data']

    pressure = -(1/3) * (sx + sy + sz)
    dev_stress_x = sx+pressure
    dev_stress_y = sy+pressure
    dev_stress_z = sz+pressure

    # Calculate the 2nd deviatoric stress invariant, J2
    J2 = 0.5 * (dev_stress_x**2 + dev_stress_y**2 + dev_stress_z**2) \
              + sxy*sxy + syz*syz + szx*szx

    # Calculate the effective stress
    seff = np.sqrt(3*J2)

    derived_result[result_name]['data'] = pressure/seff

    return derived_result

  def __compute_plastic_strain_rate(self,
                         result_name: str,
                         primal_data: Dict[str,QueryDict],
                         query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Calculate the derived result 'eps_rate'."""
    states = query_args['states']
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Get the additional query arguments
    labels = query_args['labels']
    ips = query_args['ips']
    primal_classes = query_args['class_sname'][0]

    # Create mask that excludes first and last states (only states compatable with center difference)
    max_st = len(self.db.state_maps())  # last state in the database
    mask_cent = (states != 1) & (states != max_st)
    mask_first = states == 1  # mask that includes only first state (if requested)
    mask_last = states == max_st  # mask that includes only last state (if requested)

    # Create lists of states n-1 and n+1
    states_prev = states[mask_cent] - 1
    states_next = states[mask_cent] + 1

    # Query data from states before and after desired states
    eps_prev_q = self.db.query('eps', primal_classes, None, labels, states_prev, ips)
    eps_next_q = self.db.query('eps', primal_classes, None, labels, states_next, ips)

    state_times = self.db.times()

    eps_curr = primal_data['eps']['data'][mask_cent]  # requested state eps
    eps_prev = eps_prev_q['eps']['data']
    eps_next = eps_next_q['eps']['data']
    time_prev = state_times[states_prev-1]  # offset because state 1 = index 0 for these arrays
    time_curr = state_times[states[mask_cent]-1]
    time_next = state_times[states_next-1]

    dt1_inv = 1.0 / (time_curr - time_prev)
    dt2_inv = 1.0 / (time_next - time_curr)

    # Add dummy dimensions so these 1D arrays can be multiplied by the 3D arrays.
    dt1_inv = np.expand_dims(dt1_inv, (1,2))
    dt2_inv = np.expand_dims(dt2_inv, (1,2))

    e0 = (eps_curr - eps_prev) * dt1_inv  # backward difference calculation
    e1 = (e0 + (eps_next - eps_curr) * dt2_inv) * 0.5  # average with forward diff. calc.

    derived_result[result_name]['data'][mask_cent] = e1
    derived_result[result_name]['data'][mask_first] = 0  # set first state to zero

    # Calculate last state separately using backward difference.  This probably isn't very common.
    if any(mask_last==True):
      states_last_prev = states[mask_last]-1  # 2nd to last state (includes duplicates)
      eps_last_prev_q = self.db.query('eps', primal_classes, None, labels, states_last_prev, ips)
      eps_curr = primal_data['eps']['data'][mask_last]
      eps_prev = eps_last_prev_q['eps']['data']
      time_prev = self.db.times()[states_last_prev-1]  # time arrays are all 1D
      time_curr = self.db.times()[states[mask_last]-1]

      dt1_inv = 1.0 / (time_curr - time_prev)
      dt1_inv = np.expand_dims(dt1_inv, (1,2))  # add dummy dimensions to convert to 3D
      e0 = (eps_curr - eps_prev) * dt1_inv  # backward difference calculation

      derived_result[result_name]['data'][mask_last] = e0

    return derived_result


  def __compute_nodal_tangential_traction_magnitude(self,
                                                    result_name: str,
                                                    primal_data: Dict[str,QueryDict],
                                                    query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Compute the derived result 'nodtangmag'."""
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Perform calculation
    nodtang_x = primal_data['nodtang_x']['data']
    nodtang_y = primal_data['nodtang_y']['data']
    nodtang_z = primal_data['nodtang_z']['data']
    derived_result[result_name]['data'] = np.sqrt(nodtang_x**2 + nodtang_y**2 + nodtang_z**2)

    return derived_result


  def __compute_material_cog_displacement(self,
                                          result_name: str,
                                          primal_data: Dict[str,QueryDict],
                                          query_args: QueryArgs,
                                          reference_state: int = 1) -> Dict[str,QueryDict]:
    """Compute the derived results mat_cog_disp_[x|y|z]. Material Center of Gravity Displacement."""
    labels = query_args['labels']
    class_sname = query_args['class_sname'][0]
    class_name = query_args['result_class_name']
    # Create dictionary structure for the final result
    derived_result = self.__initialize_result_dictionary( result_name, primal_data, class_name )

    # Determine which displacement we are calculating and what primal is needed.
    result_idx = ['mat_cog_disp_x', 'mat_cog_disp_y', 'mat_cog_disp_z'].index( result_name )
    required_primal = ['matcgx', 'matcgy', 'matcgz'][result_idx]

    reference_data = self.db.query(required_primal, class_sname, labels=labels, states=[reference_state])

    disp = primal_data[required_primal]['data'] - reference_data[required_primal]['data']
    derived_result[result_name]['data'] = disp

    return derived_result

  def __compute_element_volume(self,
                               result_name: str,
                               primal_data: Dict[str,QueryDict],
                               query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Compute the element volume of the M_HEX or M_TET superclass."""
    labels = query_args['labels']
    states = query_args['states']
    times = self.db.times(states)
    class_name = query_args['result_class_name']

    # Manually construct derived result dictionary since result element class
    # does not match primal data element class.
    derived_result = {
      result_name: QueryDict(
        class_name=class_name,
        source='derived',
        title=self.__derived_expressions[result_name]["title"],
        data=np.empty( (len(states), len(labels), 1), dtype=np.float32 ),
        layout=QueryLayout(
          states=states,
          labels=labels,
          components=[result_name],
          times=times,
        )
      )
    }

    elem_node_map = query_args["elem_node_map"]
    nodpos = primal_data["nodpos"]["data"]

    # Differentiate between M_HEX and M_TET Superclass.
    class_def = self.db.mesh_object_classes().get(class_name)
    if class_def is None:
      raise ValueError(f"Failed to find class data for {class_name}")

    if class_def.sclass == Superclass.M_HEX:
      a45 = nodpos[:, elem_node_map[:,3], 2] - nodpos[:, elem_node_map[:,4], 2]
      a24 = nodpos[:, elem_node_map[:,1], 2] - nodpos[:, elem_node_map[:,3], 2]
      a52 = nodpos[:, elem_node_map[:,4], 2] - nodpos[:, elem_node_map[:,1], 2]
      a16 = nodpos[:, elem_node_map[:,0], 2] - nodpos[:, elem_node_map[:,5], 2]
      a31 = nodpos[:, elem_node_map[:,2], 2] - nodpos[:, elem_node_map[:,0], 2]
      a63 = nodpos[:, elem_node_map[:,5], 2] - nodpos[:, elem_node_map[:,2], 2]
      a27 = nodpos[:, elem_node_map[:,1], 2] - nodpos[:, elem_node_map[:,6], 2]
      a74 = nodpos[:, elem_node_map[:,6], 2] - nodpos[:, elem_node_map[:,3], 2]
      a38 = nodpos[:, elem_node_map[:,2], 2] - nodpos[:, elem_node_map[:,7], 2]
      a81 = nodpos[:, elem_node_map[:,7], 2] - nodpos[:, elem_node_map[:,0], 2]
      a86 = nodpos[:, elem_node_map[:,7], 2] - nodpos[:, elem_node_map[:,5], 2]
      a57 = nodpos[:, elem_node_map[:,4], 2] - nodpos[:, elem_node_map[:,6], 2]
      a6345 = a63 - a45
      a5238 = a52 - a38
      a8624 = a86 - a24
      a7416 = a74 - a16
      a5731 = a57 - a31
      a8127 = a81 - a27

      px1 = (nodpos[:, elem_node_map[:,1], 1] * a6345 +
             nodpos[:, elem_node_map[:,2], 1] * a24   -
             nodpos[:, elem_node_map[:,3], 1] * a5238 +
             nodpos[:, elem_node_map[:,4], 1] * a8624 +
             nodpos[:, elem_node_map[:,5], 1] * a52   +
             nodpos[:, elem_node_map[:,7], 1] * a45)

      px2 = (nodpos[:, elem_node_map[:,2], 1] * a7416 +
             nodpos[:, elem_node_map[:,3], 1] * a31   -
             nodpos[:, elem_node_map[:,0], 1] * a6345 +
             nodpos[:, elem_node_map[:,5], 1] * a5731 +
             nodpos[:, elem_node_map[:,6], 1] * a63   +
             nodpos[:, elem_node_map[:,4], 1] * a16)

      px3 = (nodpos[:, elem_node_map[:,3], 1] * a8127 -
             nodpos[:, elem_node_map[:,0], 1] * a24   -
             nodpos[:, elem_node_map[:,1], 1] * a7416 -
             nodpos[:, elem_node_map[:,6], 1] * a8624 +
             nodpos[:, elem_node_map[:,7], 1] * a74   +
             nodpos[:, elem_node_map[:,5], 1] * a27)

      px4 = (nodpos[:, elem_node_map[:,0], 1] * a5238 -
             nodpos[:, elem_node_map[:,1], 1] * a31   -
             nodpos[:, elem_node_map[:,2], 1] * a8127 -
             nodpos[:, elem_node_map[:,7], 1] * a5731 +
             nodpos[:, elem_node_map[:,4], 1] * a81   +
             nodpos[:, elem_node_map[:,6], 1] * a38)

      px5 = (-nodpos[:, elem_node_map[:,7], 1] * a7416 +
              nodpos[:, elem_node_map[:,6], 1] * a86   +
              nodpos[:, elem_node_map[:,5], 1] * a8127 -
              nodpos[:, elem_node_map[:,0], 1] * a8624 -
              nodpos[:, elem_node_map[:,3], 1] * a81   -
              nodpos[:, elem_node_map[:,1], 1] * a16)

      px6 = (-nodpos[:, elem_node_map[:,4], 1] * a8127 +
              nodpos[:, elem_node_map[:,7], 1] * a57   +
              nodpos[:, elem_node_map[:,6], 1] * a5238 -
              nodpos[:, elem_node_map[:,1], 1] * a5731 -
              nodpos[:, elem_node_map[:,0], 1] * a52   -
              nodpos[:, elem_node_map[:,2], 1] * a27)

      px7 = (-nodpos[:, elem_node_map[:,5], 1] * a5238 -
              nodpos[:, elem_node_map[:,4], 1] * a86   +
              nodpos[:, elem_node_map[:,7], 1] * a6345 +
              nodpos[:, elem_node_map[:,2], 1] * a8624 -
              nodpos[:, elem_node_map[:,1], 1] * a63   -
              nodpos[:, elem_node_map[:,3], 1] * a38)

      px8 = (-nodpos[:, elem_node_map[:,6], 1] * a6345 -
              nodpos[:, elem_node_map[:,5], 1] * a57   +
              nodpos[:, elem_node_map[:,4], 1] * a7416 +
              nodpos[:, elem_node_map[:,3], 1] * a5731 -
              nodpos[:, elem_node_map[:,2], 1] * a74   -
              nodpos[:, elem_node_map[:,0], 1] * a45)

      vol = (px1 * nodpos[:, elem_node_map[:,0], 0] +
             px2 * nodpos[:, elem_node_map[:,1], 0] +
             px3 * nodpos[:, elem_node_map[:,2], 0] +
             px4 * nodpos[:, elem_node_map[:,3], 0] +
             px5 * nodpos[:, elem_node_map[:,4], 0] +
             px6 * nodpos[:, elem_node_map[:,5], 0] +
             px7 * nodpos[:, elem_node_map[:,6], 0] +
             px8 * nodpos[:, elem_node_map[:,7], 0])
      vol = vol / 12.0

    elif class_def.sclass == Superclass.M_TET:
      u = nodpos[:,elem_node_map[:,1],:] - nodpos[:,elem_node_map[:,0],:]
      v = nodpos[:,elem_node_map[:,2],:] - nodpos[:,elem_node_map[:,0],:]
      w = nodpos[:,elem_node_map[:,3],:] - nodpos[:,elem_node_map[:,0],:]

      # vol = w dot (u cross v)
      vol = np.sum( w[:,:,:] * np.cross(u, v, axis=2), axis=2 )
      vol = vol / 6.0

    vol = np.reshape( vol, vol.shape + (1,) )
    derived_result["element_volume"]["data"] = vol
    return derived_result

  def __compute_quad_area(self,
                          result_name: str,
                          primal_data: Dict[str,QueryDict],
                          query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Compute the area of a QUAD element."""
    labels = query_args['labels']
    states = query_args['states']
    times = self.db.times(states)
    class_name = query_args['result_class_name']

    # Manually construct derived result dictionary since result element class
    # does not match primal data element class.
    derived_result = {
      result_name: QueryDict(
        class_name=class_name,
        source='derived',
        title=self.__derived_expressions[result_name]["title"],
        data=np.empty( (len(states), len(labels), 1), dtype=np.float32 ),
        layout=QueryLayout(
          states=states,
          labels=labels,
          components=[result_name],
          times=times,
        )
      )
    }

    elem_node_map = query_args["elem_node_map"]
    nodpos = primal_data["nodpos"]["data"]

    fs1  = -nodpos[:, elem_node_map[:,0], 0] + nodpos[:, elem_node_map[:,1], 0] + nodpos[:, elem_node_map[:,2], 0] - nodpos[:, elem_node_map[:,3], 0]
    fs2  = -nodpos[:, elem_node_map[:,0], 1] + nodpos[:, elem_node_map[:,1], 1] + nodpos[:, elem_node_map[:,2], 1] - nodpos[:, elem_node_map[:,3], 1]
    fs3  = -nodpos[:, elem_node_map[:,0], 2] + nodpos[:, elem_node_map[:,1], 2] + nodpos[:, elem_node_map[:,2], 2] - nodpos[:, elem_node_map[:,3], 2]
    ft1  = -nodpos[:, elem_node_map[:,0], 0] - nodpos[:, elem_node_map[:,1], 0] + nodpos[:, elem_node_map[:,2], 0] + nodpos[:, elem_node_map[:,3], 0]
    ft2  = -nodpos[:, elem_node_map[:,0], 1] - nodpos[:, elem_node_map[:,1], 1] + nodpos[:, elem_node_map[:,2], 1] + nodpos[:, elem_node_map[:,3], 1]
    ft3  = -nodpos[:, elem_node_map[:,0], 2] - nodpos[:, elem_node_map[:,1], 2] + nodpos[:, elem_node_map[:,2], 2] + nodpos[:, elem_node_map[:,3], 2]
    e    = fs1 * fs1 + fs2 * fs2 + fs3 * fs3
    f    = fs1 * ft1 + fs2 * ft2 + fs3 * ft3
    g    = ft1 * ft1 + ft2 * ft2 + ft3 * ft3
    area = np.sqrt( ((e * g - f * f) / 16.0) )

    area = np.reshape( area, area.shape + (1,) )
    derived_result[result_name]["data"] = area
    return derived_result

  def __compute_centroid(self,
                          result_name: str,
                          primal_data: Dict[str,QueryDict],
                          query_args: QueryArgs) -> Dict[str,QueryDict]:
    """Compute the Centroid of an element."""
    labels = query_args['labels']
    states = query_args['states']
    times = self.db.times(states)
    class_name = query_args['result_class_name']

    # Manually construct derived result dictionary since result element class
    # does not match primal data element class.
    derived_result = {
      result_name: QueryDict(
        class_name=class_name,
        source='derived',
        title=self.__derived_expressions[result_name]["title"],
        data=np.empty( (len(states), len(labels), 1), dtype=np.float32 ),
        layout=QueryLayout(
          states=states,
          labels=labels,
          components=primal_data['nodpos']['layout']['components'],
          times=times,
        )
      )
    }

    elem_node_map = query_args["elem_node_map"]
    nodpos = primal_data["nodpos"]["data"]

    # Need to differentiate between the connectivities of superclasses.
    class_def = self.db.mesh_object_classes().get(class_name)
    if class_def is None:
      raise ValueError(f"Failed to find class data for {class_name}")
    qty_conns = class_def.sclass.node_count()
    if class_def.sclass == Superclass.M_BEAM:
      qty_conns -= 1  # The third node in BEAM connectivity should be ignored.
    if qty_conns == 0 and class_def.sclass != Superclass.M_NODE:
      raise ValueError(f"Cannot calculate 'centroid' for class {class_name} with no connectivity")

    if class_def.sclass == Superclass.M_NODE:
      centroid = nodpos
    else:
      centroid = np.sum( nodpos[:, elem_node_map[:,:qty_conns], :], axis=2) / qty_conns

    derived_result[result_name]["data"] = centroid
    return derived_result

  def __compute_surface_strain(self,
                               result_name: str,
                               primal_data: Dict[str,QueryDict],
                               query_args: QueryArgs,
                               reference_state: int = 0,
                               face: int = -1) -> Dict[str,QueryDict]:
    """Compute the Surface Strain for each face of a Hex element."""
    labels = query_args['labels']
    states = query_args['states']
    num_labels = len(labels)
    num_states = len(states)
    times = self.db.times(states)
    class_name = query_args['result_class_name']

    # Validate the specified face number.
    if face == -1:
      raise ValueError("A valid face number (1-6) must be specified. Use the keyword argument 'face'.")

    face_to_nodes = {
      1 : [1, 2, 6, 5],
      2 : [2, 3, 7, 6],
      3 : [0, 4, 7, 3],
      4 : [1, 5, 4, 0],
      5 : [4, 5, 6, 7],
      6 : [0, 3, 2, 1],
    }
    face_nodes = face_to_nodes.get(face, None)
    if face_nodes is None:
      raise ValueError(f"The provided face ({face}) is invalid. Valid face numbers include 1-6")

    # Manually construct derived result dictionary since result element class
    # does not match primal data element class.
    derived_result = {
      result_name: QueryDict(
        class_name=class_name,
        source='derived',
        title=self.__derived_expressions[result_name]["title"],
        data=np.empty( (num_states, num_labels, 1), dtype=np.float32 ),
        layout=QueryLayout(
          states=states,
          labels=labels,
          components=[result_name],
          times=times,
        )
      )
    }

    # Get reference nodal positions
    node_labels = primal_data["ux"]["layout"]["labels"]
    reference_data = self.__get_nodal_reference_positions( ["ux", "uy", "uz"], reference_state, node_labels )

    elem_node_map = query_args["elem_node_map"]
    node_ux = primal_data["ux"]["data"]
    node_uy = primal_data["uy"]["data"]
    node_uz = primal_data["uz"]["data"]

    ux = node_ux[:, elem_node_map[:,face_nodes], :]
    uy = node_uy[:, elem_node_map[:,face_nodes], :]
    uz = node_uz[:, elem_node_map[:,face_nodes], :]

    ref_ux = reference_data[elem_node_map[:,face_nodes], 0]
    ref_uy = reference_data[elem_node_map[:,face_nodes], 1]
    ref_uz = reference_data[elem_node_map[:,face_nodes], 2]
    ref_ux = np.reshape( ref_ux, ref_ux.shape + (1,) )
    ref_uy = np.reshape( ref_uy, ref_uy.shape + (1,) )
    ref_uz = np.reshape( ref_uz, ref_uz.shape + (1,) )

    # Calculate nodal displacements
    disp_x = ux - ref_ux
    disp_y = uy - ref_uy
    disp_z = uz - ref_uz

    # Input Local Direction 1
    dt1 = np.empty( (num_states,num_labels,3), dtype=np.float32)
    dt1[:,:,0] = ux[:,:,1,0] - ux[:,:,0,0]
    dt1[:,:,1] = uy[:,:,1,0] - uy[:,:,0,0]
    dt1[:,:,2] = uz[:,:,1,0] - uz[:,:,0,0]

    # Tangent to the plane at the centroid in the reference coordinate sytem
    t1O = np.empty( (num_states,num_labels,3), dtype=np.float32)
    t2O = np.empty( (num_states,num_labels,3), dtype=np.float32)
    t1O[:,:,0] = 0.25 * (ux[:,:,1,0] + ux[:,:,2,0] - ux[:,:,0,0] - ux[:,:,3,0])
    t1O[:,:,1] = 0.25 * (uy[:,:,1,0] + uy[:,:,2,0] - uy[:,:,0,0] - uy[:,:,3,0])
    t1O[:,:,2] = 0.25 * (uz[:,:,1,0] + uz[:,:,2,0] - uz[:,:,0,0] - uz[:,:,3,0])
    t2O[:,:,0] = 0.25 * (ux[:,:,3,0] + ux[:,:,2,0] - ux[:,:,0,0] - ux[:,:,1,0])
    t2O[:,:,1] = 0.25 * (uy[:,:,3,0] + uy[:,:,2,0] - uy[:,:,0,0] - uy[:,:,1,0])
    t2O[:,:,2] = 0.25 * (uz[:,:,3,0] + uz[:,:,2,0] - uz[:,:,0,0] - uz[:,:,1,0])

    # Normal to plane at the centroid in reference coordinate system
    nnO = np.empty( (num_states,num_labels,3), dtype=np.float32)
    nnO[:,:,0] = t1O[:,:,1] * t2O[:,:,2] - t1O[:,:,2] * t2O[:,:,1]
    nnO[:,:,1] = t1O[:,:,2] * t2O[:,:,0] - t1O[:,:,0] * t2O[:,:,2]
    nnO[:,:,2] = t1O[:,:,0] * t2O[:,:,1] - t1O[:,:,1] * t2O[:,:,0]

    temp = np.empty( (num_states,num_labels,1), dtype=np.float32)
    temp[:,:,0] = 1.0 / np.sqrt((nnO[:,:,0] * nnO[:,:,0]) + (nnO[:,:,1] * nnO[:,:,1]) + (nnO[:,:,2] * nnO[:,:,2]))
    nnO = temp * nnO  # type: ignore

    # Normalized and Orthogonal
    t1Oh = np.empty( (num_states,num_labels,3), dtype=np.float32)
    t2Oh = np.empty( (num_states,num_labels,3), dtype=np.float32)
    temp[:,:,0] = 1.0 / np.sqrt((t1O[:,:,0] * t1O[:,:,0]) + (t1O[:,:,1] * t1O[:,:,1]) + (t1O[:,:,2] * t1O[:,:,2]))
    t1Oh = temp * t1O  # type: ignore

    t2Oh[:,:,0] = nnO[:,:,1] * t1Oh[:,:,2] - nnO[:,:,2] * t1Oh[:,:,1]
    t2Oh[:,:,1] = nnO[:,:,2] * t1Oh[:,:,0] - nnO[:,:,0] * t1Oh[:,:,2]
    t2Oh[:,:,2] = nnO[:,:,0] * t1Oh[:,:,1] - nnO[:,:,1] * t1Oh[:,:,0]

    # Now we have been given a defined direction 1 (it is not normalized)
    # and it is not necesarily orthogonal to nnO, so make it orthogonal to nnO
    dt1h = np.empty( (num_states,num_labels,3), dtype=np.float32)
    dt2h = np.empty( (num_states,num_labels,3), dtype=np.float32)

    temp[:,:,0] = dt1[:,:,0] * nnO[:,:,0] + dt1[:,:,1] * nnO[:,:,1] + dt1[:,:,2] * nnO[:,:,2]
    dt1h = dt1 - nnO * temp
    temp[:,:,0] = 1.0 / np.sqrt((dt1h[:,:,0] * dt1h[:,:,0]) + (dt1h[:,:,1] * dt1h[:,:,1]) + (dt1h[:,:,2] * dt1h[:,:,2]))
    dt1h = temp * dt1h  # type: ignore

    # Now we need to calculate dt2h as orthogonal to both
    dt2h[:,:,0] = nnO[:,:,1] * dt1h[:,:,2] - nnO[:,:,2] * dt1h[:,:,1]
    dt2h[:,:,1] = nnO[:,:,2] * dt1h[:,:,0] - nnO[:,:,0] * dt1h[:,:,2]
    dt2h[:,:,2] = nnO[:,:,0] * dt1h[:,:,1] - nnO[:,:,1] * dt1h[:,:,0]

    # Calculate the coordinates in the local coordinate system of the reference coordinate system
    dispOL = np.zeros( (num_states,num_labels,4,3), dtype=np.float32 )
    for j in range(0,4):
      dispOL[:,:,j,0] = dispOL[:,:,j,0] + dt1h[:,:,0] * disp_x[:,:,j,0]
      dispOL[:,:,j,0] = dispOL[:,:,j,0] + dt1h[:,:,1] * disp_y[:,:,j,0]
      dispOL[:,:,j,0] = dispOL[:,:,j,0] + dt1h[:,:,2] * disp_z[:,:,j,0]

      dispOL[:,:,j,1] = dispOL[:,:,j,1] + dt2h[:,:,0] * disp_x[:,:,j,0]
      dispOL[:,:,j,1] = dispOL[:,:,j,1] + dt2h[:,:,1] * disp_y[:,:,j,0]
      dispOL[:,:,j,1] = dispOL[:,:,j,1] + dt2h[:,:,2] * disp_z[:,:,j,0]

      dispOL[:,:,j,2] = dispOL[:,:,j,2] + nnO[:,:,0] * disp_x[:,:,j,0]
      dispOL[:,:,j,2] = dispOL[:,:,j,2] + nnO[:,:,1] * disp_y[:,:,j,0]
      dispOL[:,:,j,2] = dispOL[:,:,j,2] + nnO[:,:,2] * disp_z[:,:,j,0]

    # Calculate the shape function derivative at the centroid
    Nd = np.empty((4,2), dtype=np.float32)
    Nd[0][0] = -0.25
    Nd[0][1] = -0.25

    Nd[1][0] = 0.25
    Nd[1][1] = -0.25

    Nd[2][0] = 0.25
    Nd[2][1] = 0.25

    Nd[3][0] = -0.25
    Nd[3][1] = 0.25

    # Calculate the jacobian of transformation in the reference configuration at the centroid
    # in the local "hat" coordinate system.
    dxdxi = np.empty((num_states,num_labels,3,3), dtype=np.float32)
    dxdxi[:,:,0,0] = dt1h[:,:,0] * t1O[:,:,0] + dt1h[:,:,1] * t1O[:,:,1] + dt1h[:,:,2] * t1O[:,:,2]
    dxdxi[:,:,0,1] = dt1h[:,:,0] * t2O[:,:,0] + dt1h[:,:,1] * t2O[:,:,1] + dt1h[:,:,2] * t2O[:,:,2]
    dxdxi[:,:,0,2] = 0.0
    dxdxi[:,:,1,0] = dt2h[:,:,0] * t1O[:,:,0] + dt2h[:,:,1] * t1O[:,:,1] + dt2h[:,:,2] * t1O[:,:,2]
    dxdxi[:,:,1,1] = dt2h[:,:,0] * t2O[:,:,0] + dt2h[:,:,1] * t2O[:,:,1] + dt2h[:,:,2] * t2O[:,:,2]
    dxdxi[:,:,1,2] = 0.0
    dxdxi[:,:,2,0] = 0.0
    dxdxi[:,:,2,1] = 0.0
    dxdxi[:,:,2,2] = 1.0

    # Calculate the inverse
    dxidx = np.empty((num_states,num_labels,3,3), dtype=np.float32)
    onedet = np.empty((num_states,num_labels,1), dtype=np.float32)
    onedet[:,:,0] = 1.0 / (dxdxi[:,:,0,0] * dxdxi[:,:,1,1] - dxdxi[:,:,0,1] * dxdxi[:,:,1,0])
    dxidx[:,:,0,0] = onedet[:,:,0] * dxdxi[:,:,1,1]
    dxidx[:,:,0,1] = -onedet[:,:,0] * dxdxi[:,:,0,1]
    dxidx[:,:,0,2] = 0.0
    dxidx[:,:,1,0] = -onedet[:,:,0] * dxdxi[:,:,1,0]
    dxidx[:,:,1,1] = onedet[:,:,0] * dxdxi[:,:,0,0]
    dxidx[:,:,1,2] = 0.0
    dxidx[:,:,2,0] = 0.0
    dxidx[:,:,2,1] = 0.0
    dxidx[:,:,2,2] = 1.0

    # Now calculate the displacement gradient du/dxi
    dudxi = np.zeros((num_states,num_labels,3,2), dtype=np.float32)
    dudxi[:,:,:,0] = dudxi[:,:,:,0] + dispOL[:,:,0,:] * Nd[0,0]
    dudxi[:,:,:,0] = dudxi[:,:,:,0] + dispOL[:,:,1,:] * Nd[1,0]
    dudxi[:,:,:,0] = dudxi[:,:,:,0] + dispOL[:,:,2,:] * Nd[2,0]
    dudxi[:,:,:,0] = dudxi[:,:,:,0] + dispOL[:,:,3,:] * Nd[3,0]
    dudxi[:,:,:,1] = dudxi[:,:,:,1] + dispOL[:,:,0,:] * Nd[0,1]
    dudxi[:,:,:,1] = dudxi[:,:,:,1] + dispOL[:,:,1,:] * Nd[1,1]
    dudxi[:,:,:,1] = dudxi[:,:,:,1] + dispOL[:,:,2,:] * Nd[2,1]
    dudxi[:,:,:,1] = dudxi[:,:,:,1] + dispOL[:,:,3,:] * Nd[3,1]

    # Now calculate the displacement gradient du/dx
    dudx = np.zeros((num_states,num_labels,3,3), dtype=np.float32)
    dudx[:,:,:,0] = dudx[:,:,:,0] + dudxi[:,:,:,0] * dxidx[:,:,0,0:1]
    dudx[:,:,:,0] = dudx[:,:,:,0] + dudxi[:,:,:,1] * dxidx[:,:,1,0:1]
    dudx[:,:,:,1] = dudx[:,:,:,1] + dudxi[:,:,:,0] * dxidx[:,:,0,1:2]
    dudx[:,:,:,1] = dudx[:,:,:,1] + dudxi[:,:,:,1] * dxidx[:,:,1,1:2]
    dudx[:,:,:,2] = dudx[:,:,:,2] + dudxi[:,:,:,0] * dxidx[:,:,0,2:3]
    dudx[:,:,:,2] = dudx[:,:,:,2] + dudxi[:,:,:,1] * dxidx[:,:,1,2:3]

    # Construct 3d strain matrix in the local system t1h,t2h,nn
    e3dl = np.empty((num_states,num_labels,3,3), dtype=np.float32)
    e3dl[:,:,0,0] = dudx[:,:,0,0]
    e3dl[:,:,0,1] = 0.5 * (dudx[:,:,0,1] + dudx[:,:,1,0])
    e3dl[:,:,0,2] = 0.0
    e3dl[:,:,1,0] = 0.5 * (dudx[:,:,0,1] + dudx[:,:,1,0])
    e3dl[:,:,1,1] = dudx[:,:,1,1]
    e3dl[:,:,1,2] = 0.0
    e3dl[:,:,2,0] = 0.0
    e3dl[:,:,2,1] = 0.0
    e3dl[:,:,2,2] = 0.0

    # Construct the 3d rotation matrix r3d
    r3d = np.empty((num_states,num_labels,3,3), dtype=np.float32)
    r3d[:,:,0,0] = dt1h[:,:,0]
    r3d[:,:,0,1] = dt2h[:,:,0]
    r3d[:,:,0,2] = nnO[:,:,0]
    r3d[:,:,1,0] = dt1h[:,:,1]
    r3d[:,:,1,1] = dt2h[:,:,1]
    r3d[:,:,1,2] = nnO[:,:,1]
    r3d[:,:,2,0] = dt1h[:,:,2]
    r3d[:,:,2,1] = dt2h[:,:,2]
    r3d[:,:,2,2] = nnO[:,:,2]

    # Rotate the strain tensor from the local to the current system
    e3dg = np.zeros((num_states,num_labels,3,3), dtype=np.float32)
    for j in range(0,3):
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,0:1] * r3d[:,:,:,0] * e3dl[:,:,0,0:1]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,0:1] * r3d[:,:,:,1] * e3dl[:,:,0,1:2]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,0:1] * r3d[:,:,:,2] * e3dl[:,:,0,2:3]

      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,1:2] * r3d[:,:,:,0] * e3dl[:,:,1,0:1]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,1:2] * r3d[:,:,:,1] * e3dl[:,:,1,1:2]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,1:2] * r3d[:,:,:,2] * e3dl[:,:,1,2:3]

      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,2:3] * r3d[:,:,:,0] * e3dl[:,:,2,0:1]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,2:3] * r3d[:,:,:,1] * e3dl[:,:,2,1:2]
      e3dg[:,:,j,:] = e3dg[:,:,j,:] + r3d[:,:,j,2:3] * r3d[:,:,:,2] * e3dl[:,:,2,2:3]

    # Extract desired component
    if result_name == "surfstrainx":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,0,0]
    elif result_name == "surfstrainy":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,1,1]
    elif result_name == "surfstrainz":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,2,2]
    elif result_name == "surfstrainxy":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,1,0]
    elif result_name == "surfstrainyz":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,2,1]
    elif result_name == "surfstrainzx":
      derived_result[result_name]["data"][:,:,0] = e3dg[:,:,2,0]
    else:
      # This should never happen, but just in case.
      raise ValueError(f"Invalid value for result_name ({result_name})")

    return derived_result