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

import multiprocessing
import dill
import pathos.multiprocessing as mp
from functools import partial
from multiprocessing import Process
import numpy as np
import atexit
import psutil

from typing import *

# TODO: probably just create a wrapper/dispatch superclass, and implement loop/pool/client-server versions instead of loop/pool in one and client/server in another

class LoopWrapper:
  def __init__( self,
                cls_obj : Type,
                proc_pargs : List[List[Any]] = [],
                proc_kwargs : List[Mapping[Any,Any]] = [] ):
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate object list.')

    objs = [ cls_obj( *pargs, **kwargs ) for pargs, kwargs in zip(proc_pargs, proc_kwargs) ]

    # ensure all contained objects are the same exact type (no instances, subclasses are not valid)
    obj_type = type(objs[0])
    assert( all( type(obj) == obj_type for obj in objs ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    self._objs = objs

    # Add member functions to this object based on the member functions of the wrapped objects
    call_lambda = lambda _attr, _cls_obj, _objs : lambda *pargs, **kwargs: self.loop_caller(_attr,_cls_obj,_objs,*pargs,**kwargs)
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') and callable(getattr(cls_obj,func)) ):
      setattr( self, func, call_lambda(func,cls_obj,objs) )

  def loop_caller(self,attr,cls_obj,objs,*pargs,**kwargs):
    return [ getattr(cls_obj,attr)( obj, *pargs, **kwargs ) for obj in objs ]

class PoolWrapper:
  def __init__( self,
                cls_obj : Type,
                proc_pargs : List[List[Any]] = [],
                proc_kwargs : List[Mapping[Any,Any]] = [] ):
    # validate parameters
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate worker processes')

    with mp.ProcessingPool(len(proc_pargs)) as pool:
      objs = pool.map( lambda args: cls_obj( *args[0], **args[1] ), list( zip(proc_pargs, proc_kwargs) ) )

    # ensure all contained objects are the same exact type (no instances, subclasses are not valid)
    obj_type = type(objs[0])
    assert( all( type(obj) == obj_type for obj in objs ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    self._objs = objs

    # Add member functions to this object based on the member functions of the wrapped objects
    call_lambda = lambda _attr, _cls_obj, _objs : lambda *pargs, **kwargs: self.pool_caller(_attr,_cls_obj,_objs,*pargs,**kwargs)
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') and callable(getattr(cls_obj,func)) ):
      setattr( self, func, call_lambda(func,cls_obj,objs) )

  def pool_caller(self,attr,cls_obj,objs,*pargs,**kwargs):
    # we make the list ahead of time to bind the pargs and kwargs identically across the pool, instead of having to pass in or make arrays for them
    to_invoke = [ partial( getattr(cls_obj,attr), obj, *pargs, **kwargs ) for obj in objs ]
    with mp.ProcessingPool(len(objs)) as pool:
      res = pool.map(lambda f: f(), to_invoke) # can't partial the pargs or the first will bind to self
    return res


# check asyncio module

class ServerWrapper:
  """Multiprocessing client/server wrapper for reading Mili databases in parallel.

  When more plot files exist than processors, each processor handles multiple file
  using the same concept as the loop wrapper.
  """
  class ClientWrapper(Process):
    def __init__( self,
                  conn, 
                  cls_obj: Type,
                  proc_pargs: List[List[Any]] = [],
                  proc_kwargs: List[Mapping[Any,Any]] = [] ):
      super(Process,self).__init__()

      num_pargs = len(proc_pargs)
      num_kwargs = len(proc_kwargs)
      if num_pargs > 0 and num_kwargs == 0:
        proc_kwargs = [ {} ] * num_pargs
      elif num_kwargs > 0 and num_pargs == 0:
        proc_pargs = [] * num_kwargs
      elif num_kwargs != num_pargs:
        raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate object list.')

      self.__conn = conn
      self.__cls_obj = cls_obj
      self.__proc_pargs = proc_pargs
      self.__proc_kwargs = proc_kwargs

    def run( self ):
      self.__wrapped = [ self.__cls_obj(*pargs, **kwargs) for pargs, kwargs in zip(self.__proc_pargs, self.__proc_kwargs) ]

      # ensure all contained objects are the same exact type (no instances, subclasses are not valid)
      obj_iter = iter(self.__wrapped)
      obj_type = type(next(obj_iter))
      assert( all( type(obj) == obj_type for obj in obj_iter ) )

      while True:
        if self.__conn.poll():
          cmd, pargs, kwargs = dill.loads( self.__conn.recv() )
          if cmd == '__exit':
            break
          else:
            result = [ getattr( self.__cls_obj, cmd )( wrapped, *pargs, **kwargs) for wrapped in self.__wrapped ]
            self.__conn.send_bytes( dill.dumps(result) )
      return
  
  def __divide_into_sequential_groups( self, values, n_groups ):
    if n_groups > len(values):
      n_groups = len(values)
    d, r = divmod( len(values), n_groups )
    for i in range( n_groups ):
      si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
      yield values[si:si+(d+1 if i < r else d)]
  
  def __init__( self, cls_obj : Type, proc_pargs : List[List[Any]] = [], proc_kwargs : List[Mapping[Any,Any]] = [] ):
    # validate parameters
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate worker processes')

    # need to have a lambda that returns new lambdas for each func to avoid only have a single lambda bound for all funcs
    mem_maker = lambda attr : lambda *pargs, **kwargs : self.__worker_call( attr, *pargs, **kwargs )
    # generate member functions mimicking those in the wrapped class, excluding private and class functions
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') and callable(getattr(cls_obj,func)) ):
      setattr( self, func, mem_maker(func) )
    
    # Get the number of processors that are available
    n_cores = int( psutil.cpu_count(logical=False) )

    # Break out into groups for each processor available
    proc_pargs = self.__divide_into_sequential_groups( proc_pargs, n_cores )
    proc_kwargs = self.__divide_into_sequential_groups( proc_kwargs, n_cores )

    # create a pool of worker processes and supply the constructor arguments to each one
    self.__pool = []
    self.__conns = []
    for pargs, kwargs in zip( proc_pargs, proc_kwargs ):
      c0, c1 = multiprocessing.Pipe()
      self.__pool.append( ServerWrapper.ClientWrapper( c1, cls_obj, pargs, kwargs ) )
      self.__conns.append( c0 )

    atexit.register(self.__cleanup_processes)

    # spawn the worker processes
    for proc in self.__pool:
      proc.start()
  
  def close(self):
    """Close the database and end all subprocesses."""
    atexit.unregister(self.__cleanup_processes)
    self.__cleanup_processes()
    
  # Need this so processes don't hang and close properly
  def __cleanup_processes(self):
    if hasattr( self, '_ServerWrapper__conns' ):
      for conn in self.__conns:
        conn.send( dill.dumps( ( '__exit', [], {} ) ) )
    self.__conns = []
    if hasattr( self, '_ServerWrapper__pool'):
      for proc in self.__pool:
        proc.join( )
    self.__procs = []

  def __flatten( self, alist: List[Any] ):
    return [item for sublist in alist for item in sublist]

  # called by the generated member functions to send the command (attr name) and function arguments to
  #  each of the processes managed by this server. All arguments are sent to all processes, sending specific
  #  arguments to individual processes is not supported currently.
  def __worker_call(self,attr,*pargs,**kwargs):
    results = [None] * len(self.__conns)
    data = ( attr, pargs, kwargs )

    for conn in self.__conns:
      conn.send( dill.dumps( data ) )

    for idx, conn in enumerate(self.__conns):
      results[idx] = dill.loads( conn.recv_bytes() )

    return self.__flatten(results)


def get_wrapper( suppress_parallel = False, experimental = False ):
  Wrapper = None
  if suppress_parallel:
    Wrapper = LoopWrapper
  else:
    if experimental:
      Wrapper = ServerWrapper
    else:
      Wrapper = PoolWrapper
  return Wrapper
