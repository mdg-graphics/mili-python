==================
MDG Definitions
==================

Mili-python contains several Enums that define common Entity types and State Variables that are output by MDG's simulation codes. They are defined below.

**NOTE**: These enumerations are provided as a convenience for user's for accessing common values written to MDG's mili databases. We cover the known
default naming as best we can, but these values are not guaranteed to always be written the same way by the simulation codes. These values have
periodically changed throughout the history of the MDG codes and they may change again at any time. We will do our best to keep these values up to
date.


Entity Types
===============
.. autoclass:: mili.mdg_defines.EntityType
    :members:
    :undoc-members:
    :member-order: bysource

State Variables
=================
.. autoclass:: mili.mdg_defines.StateVariableName
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.GlobalStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.MaterialStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.NodalStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.LoadCurveStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.BeamStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.DiscreteStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.ShellStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.ContactSegmentStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.PressureSegmentStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.StressStrainStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.SharedStateVariables
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: mili.mdg_defines.DerivedVariables
    :members:
    :undoc-members:
    :member-order: bysource