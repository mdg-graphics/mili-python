"""
Copyright (c) 2016-2022, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by
 William Tobin (tobin6@llnl.hov), Kevin Durrenberger (durrenberger1@llnl.gov),
 and Ryan Hathaway (hathaway6@llnl.gov).
 CODE-OCEC-16-056.
 All rights reserved.

 This file is part of Mili. For details, see
 https://rzlc.llnl.gov/gitlab/mdg/mili/mili-python/. For read access to this repo
 please contact the authors listed above.

 Our Notice and GNU Lesser General Public License.

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License (as published by
 the Free Software Foundation) version 2.1 dated February 1999.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 and conditions of the GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software Foundation,
 Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
"""

# defer evaluation of type-annotations until after the module is processed, allowing class members to refer to the class
from __future__ import annotations
import os
from typing import *
from dataclasses import dataclass
from itertools import chain
from mili.datatypes import *
from mili.reader import *
import numpy.typing as npt


def open_griz_interface( base_filename : os.PathLike, procs = [], suppress_parallel = False, experimental = False ):
  """Open a Mili database and create GrizInterface object.

  Args:
   base_file (os.PathLike): the base filename of the mili database (e.g. for 'pltA', just 'plt', for parallel
                            databases like 'dblplt00A', also exclude the rank-digits, giving 'dblplt')
   procs (Optional[List[int]]) : optionally a list of process-files to open in parallel, default is all
   suppress_parallel (Optional[Bool]) : optionally return a serial database reader object if possible (for serial databases).
                                        Note: if the database is parallel, suppress_parallel==True will return a reader that will
                                        query each processes database files in series.
   experimental (Optional[Bool]) : optional developer-only argument to try experimental parallel features
  """
  return GrizInterface(open_database(base_filename, procs, suppress_parallel, experimental))


@dataclass
class GrizDataContainer:
  """Data class for all data that needs to be returned by each processor."""
  params: dict
  smaps: List[StateMap]
  nodes: list
  conns: dict
  labels: dict
  mo_classes: list
  materials: dict
  parts: dict
  element_sets: dict
  subrecords: list
  classes_with_sand: List[str]


@dataclass
class FreeNodeData:
  """Data class for storing free node mass/volume in Griz Interface."""
  mass: List[List[float]]
  vol: List[List[float]]


class GrizInterface:
  """Wrapper class for MiliDatabase object that simplifies use of Mili reader in Griz.

  Gathers most of the required data for Griz on instantiation so that Griz doesn't need
  to waste time doing all the lookups and queries.

  Args:
    db (MiliDabase): The Milidatabase object to wrap
  """
  def __init__(self, db: MiliDatabase):

    # Store the Mili Database
    self.db = db

    # Gather data from each processor
    self.griz_data = db.get_griz_data()
    self.processor_count = len(self.griz_data)
    self.dimensions = db.mesh_dimensions()[0]
    self.srec_fmt_qty = db.srec_fmt_qty()[0]

    # Load in Free Node mass/volume if it exists
    self.free_node_data: FreeNodeData = None
    self.load_free_node_data()

    # Each processor has its own parameter database. merge them together
    # NOTE: Some values may not be merged together correctly.
    self.params = {}
    self.merge_parameters()

    # Statemaps are the same on all processors so just get first processor.
    self.smaps = self.griz_data[0].smaps

    # Get the names of all element classes with sand flags.
    self.sand_class_names = set(chain.from_iterable([data.classes_with_sand for data in self.griz_data]))

    # Reflection for mili reader functions we need to access
    for func in ["query_all_classes", "query"]:
      if func in dir(self.db):
        call_lambda = lambda *pargs, **kwargs : getattr(self.db, func)(*pargs, **kwargs)
        setattr( self, func, call_lambda )

  def reload(self) -> None:
    """Reload the state maps for the plot file."""
    self.db.reload_state_maps()
    self.smaps = self.db.state_maps()[0]


  def load_free_node_data(self) -> None:
    """Get Free Node Mass/Volume if it exists."""
    fn_mass = []
    fn_vol = []
    for proc_params in self.db.parameters():
      # Get Nodal Mass
      if "Nodal Mass" in proc_params:
        fn_mass.append(proc_params["Nodal Mass"])
      else:
        fn_mass.append(None)
      # Get Nodal Volume
      if "Nodal Volume" in proc_params:
        fn_vol.append(proc_params["Nodal Volume"])
      else:
        fn_vol.append(None)

    self.free_node_data = FreeNodeData( fn_mass, fn_vol )


  def merge_parameters(self) -> None:
    """Merge the parameter dictionaries for each processor into a single dict object."""
    for proc_params in self.db.parameters():
      for proc_key, proc_value in proc_params.items():
        if proc_key not in self.params:
          self.params[proc_key] = proc_value


  def parameters(self) -> dict:
    """Getter for params dictionary."""
    return self.params

  def state_maps(self) -> List:
    """Getter for state maps list."""
    return self.smaps

  def parameter_wildcard_search(self, key: str) -> List[str]:
    """Wildcard search of Mili parameters dictionary.

    Args:
      key (str): The key to search for.

    Returns:
      List of keys that match or startwith the key searched for.
    """
    matches = set()
    for param_key in self.params:
      if param_key == key or param_key.startswith(key):
        matches.add(param_key)
    return list(matches)


  @dataclass
  class GrizGeomCallData:
    """Dataclass containing all data needed by Griz get_geom call."""
    nodes: List[Dict]
    connectivity: List[Dict]
    labels: List[Dict]
    mo_classes: List[Dict]
    materials: List[Dict]
    parts: List[Dict]


  def griz_get_geom_call(self):
    """Populate GrizGeomCallData object for each processor."""
    return [
      GrizInterface.GrizGeomCallData(
        data.nodes, data.conns,
        data.labels, data.mo_classes,
        data.materials, data.parts
      ) for data in self.griz_data
    ]

  @dataclass
  class GrizStDescriptorsCallData:
    """Dataclass containing all data needed by Griz get_st_descriptors call."""
    element_sets: List[dict]
    subrecords: List[List[Subrecord]]

  def griz_get_st_descriptors_call(self):
    """Populate GrizStDescriptorsCallData object for each processor."""
    return [
      GrizInterface.GrizStDescriptorsCallData(
        data.element_sets,
        data.subrecords
      ) for data in self.griz_data
    ]

  @dataclass
  class GrizGetStateCallData:
    """Dataclass containing all data needed by Griz get_state call."""
    nodpos: dict
    sand: dict

  def griz_get_state_call(self):
    """Populate GrizGetStateCallData object for each processor."""
    node_data = self.db.query("nodpos", "node")
    if len(self.sand_class_names) == 0:
      sand_data = {}
    else:
      sand_data = self.db.query_all_classes( "sand" )
    return GrizInterface.GrizGetStateCallData(
      node_data,
      sand_data,
    )