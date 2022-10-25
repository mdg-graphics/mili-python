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
    obj_iter = iter(objs)
    obj_type = type(next(obj_iter))
    assert( all( type(obj) == obj_type for obj in obj_iter ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    # TODO: this will eventually be the live pool
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
    obj_iter = iter(objs)
    obj_type = type(next(obj_iter))
    assert( all( type(obj) == obj_type for obj in obj_iter ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    # TODO: this will eventually be the live pool
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

# generic multiprocessing client/server wrapper
class ServerWrapper:
  class ClientWrapper(Process):
    def __init__( self, conn, cls_obj : Type, *pargs, **kwargs ):
      super(Process,self).__init__()
      self.__conn = conn
      self.__cls_obj = cls_obj
      self.__pargs = pargs
      self.__kwargs = kwargs

    def run( self ):
      self.__last_result = None
      self.__wrapped = self.__cls_obj( *self.__pargs, **self.__kwargs )
      while True:
        if self.__conn.poll():
          cmd, pargs, kwargs = dill.loads( self.__conn.recv() )
          if cmd == '__exit':
            break
          elif cmd == '__return':
            self.__conn.send( dill.dumps( self.__last_result ) )
          else:
            self.__last_result = getattr( self.__wrapped, cmd )( *pargs, **kwargs )
      return

  def __init__( self, cls_obj : Type, proc_pargs : List[List[Any]] = [], proc_kwargs : List[Mapping[Any,Any]] = [], immediate_mode : bool = True ):
    self.__immediate_mode = immediate_mode

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

    # create a pool of worker processes and supply the constructor arguments to each one
    self.__pool = []
    self.__conns = []
    for pargs, kwargs in zip( proc_pargs, proc_kwargs ):
      c0, c1 = multiprocessing.Pipe()
      self.__pool.append( ServerWrapper.ClientWrapper( c1, cls_obj, *pargs, **kwargs ) )
      self.__conns.append( c0 )

    # spawn the worker processes
    for proc in self.__pool:
      proc.start()

  # TODO : this *really* doesn't seem to be working
  def __del__( self ):
    # need to check for attr existence in __del__ to handle partially-constructed failure modes
    print('Sending exit command to worker processes...')
    if hasattr( self, '__conns' ):
      for conn in self.__conns:
        conn.send( dill.dumps( ( '__exit', [], {} ) ) )
    print('Waiting for worker processes to exit...')
    if hasattr( self, '__procs'):
      for proc in self.__procs:
        proc.join( )
    print('All worker processes terminated.')

  # called by the generated member functions to send the command (attr name) and function arguments to
  #  each of the processes managed by this server. All arguments are sent to all processes, sending specific
  #  arguments to individual processes is not supported currently.
  def __worker_call(self,attr,*pargs,**kwargs):
    data = ( attr, pargs, kwargs )
    for conn in self.__conns:
      conn.send( dill.dumps( data ) )

    if self.__immediate_mode:
      return self.result()
    return None

  # collect the last result from each worker process
  def result(self, procs : List[int] = None ):
    result = []
    if procs == None:
      for conn in self.__conns:
        conn.send( dill.dumps( ('__return', [], {}) ) )
      for conn in self.__conns:
        result.append( conn.recv() )
    else:
      for pid in procs:
        self.__conns[pid].send( dill.dumps( ('__return', [], {} ) ) )
      for pid in procs:
        result.append( self.__conns[pid].recv() )
    return result, procs
