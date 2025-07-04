"""Plotting module.

Various Classes and Functions related to plotting Mili results.

SPDX-License-Identifier: (MIT)
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Union, Optional
from typing import overload, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mili.reductions import combine
from mili.datatypes import QueryDict

@dataclass
class PlotObject:
  """A Plot Object storing all the data to plot a single line."""
  name: str = ""
  title: str = ""
  class_name: str = ""
  label: Union[int,np.integer] = -1
  x_data : NDArray[np.floating] = field(default_factory=lambda: np.empty([0],dtype = np.float64))
  y_data : NDArray[np.floating] = field(default_factory=lambda: np.empty([0],dtype = np.float64))


class Plotter(ABC):
  """Abstract Plotter Class."""
  def __init__(self) -> None:
    pass

  def _query_data_to_plot_data(self, query_data: Union[Dict[str,QueryDict],List[Dict[str,QueryDict]]]) -> List[PlotObject]:
    """Convert Mili-python query data to plot data."""
    plot_objects = []
    if isinstance(query_data, list):
      query_data = combine(query_data)
    for _, result_data in query_data.items():
      if isinstance( result_data, pd.DataFrame ):
        raise ValueError("Plotting is not supported for Dataframe results. Please set on_dataframe to False.")
      class_name = result_data["class_name"]
      svar_title = result_data["title"]
      data = result_data["data"]
      labels = result_data["layout"]["labels"]
      x_data = result_data["layout"]["times"]
      components = result_data["layout"]["components"]

      # Group data by line and generate label
      for idx, label in enumerate(labels):
        for comp_idx, component in enumerate(components):
          y_data = data[:,idx,comp_idx]
          plot_obj = PlotObject(component, svar_title, class_name, label, x_data, y_data)
          plot_objects.append( plot_obj )
    return plot_objects

  @abstractmethod
  def initialize_plot(self, *_: Any, **__: Any) -> Any: ...

  @abstractmethod
  def update_plot(self, query_data: Dict[str,QueryDict], *_: Any, **__: Any) -> Any: ...


class MatPlotLibPlotter(Plotter):
  """Plotter for Matplotlib."""
  def __init__(self) -> None:
    super(MatPlotLibPlotter, self).__init__()

  @overload
  def initialize_plot(self) -> Tuple[Figure,Axes]: ...

  @overload
  def initialize_plot(self, nrows: int, ncols: int) -> Tuple[Figure,List[Axes]]: ...

  @overload
  def initialize_plot(self, nrows: int = 1, ncols: int = 1) -> Tuple[Figure,Union[Axes,List[Axes]]]: ...

  def initialize_plot(self, nrows: int = 1, ncols: int = 1) -> Tuple[Figure,Union[Axes,List[Axes]]]:
    """Initialize a Figure and one or more Axes object to plot results on.

    Args:
      nrows (int, default=1): The number of rows of plots.
      ncols (int, default=1): The number of colums of plots.

    Returns: Tuple[Figure,Union[Axes,List[Axes]]]
      A Matplotlib Figure object and either a single Axes object or an array of Axes objects
      if more that one plots was generated using nrows and ncols.
    """
    fig, ax = plt.subplots(nrows, ncols, layout="constrained")
    return fig, ax

  def update_plot(self, query_data: Dict[str,QueryDict], ax: Axes) -> None:
    """Add plot data to the specified Axes object.

    Args:
      query_data (Dict): The query data
    """
    plot_objects = self._query_data_to_plot_data( query_data )
    for plot_obj in plot_objects:
      line_label = f"{plot_obj.name} {plot_obj.class_name} {plot_obj.label}"
      ax.plot(plot_obj.x_data, plot_obj.y_data, label=line_label)
    ax.legend()