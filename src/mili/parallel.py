"""
SPDX-License-Identifier: (MIT)
"""
import multiprocessing
import dill
import pathos.multiprocessing as mp
from functools import partial
from multiprocessing import Process, shared_memory
import numpy as np
import atexit
import psutil
import uuid
from enum import Enum
from typing import *
from functools import reduce

# TODO: probably just create a wrapper/dispatch superclass, and implement loop/pool/client-server versions instead of loop/pool in one and client/server in another

class ReturnCode(Enum):
  OK = 0
  ERROR = 1
  CRITICAL = 2

  def str_repr(self):
    return ["Success", "Error", "Critical"][self.value]

def parse_return_codes(return_codes):
  if not np.all([rcode_tup[0] == ReturnCode.OK for rcode_tup in return_codes]):
    # An error has occurred. Need to determine severity.
    num_ret_codes = len(return_codes)
    error_types = np.array(return_codes)[:,0]
    errors = np.where(np.isin(error_types, ReturnCode.ERROR))[0]
    critical = np.where(np.isin(error_types, ReturnCode.CRITICAL))[0]
    if len(critical) > 0 or len(errors) == num_ret_codes:
      all_errors = np.concatenate((return_codes[errors], return_codes[critical]))
      error_msgs = list(set([f"{retcode[0].str_repr()}: {retcode[1]}" for retcode in all_errors]))
      raise ValueError(", ".join(error_msgs))

class Wrapper:
  """Wrapper superclass."""
  def __init__(self, cls_obj):
    self.cls_obj = cls_obj
    self.supports_returncode = False
    if hasattr(cls_obj, "returncode"):
      self.supports_returncode = True

class LoopWrapper(Wrapper):
  def __init__( self,
                cls_obj : Type,
                proc_pargs : List[List[Any]] = [],
                proc_kwargs : List[Mapping[Any,Any]] = [],
                objects : List[Any] = None ):
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate object list.')

    # Call super class constructor to set up return code if available
    super(LoopWrapper, self).__init__(cls_obj)

    if objects:
      objs = objects
    else:
      objs = [ cls_obj( *pargs, **kwargs ) for pargs, kwargs in zip(proc_pargs, proc_kwargs) ]

    # ensure all contained objects are the same exact type (no instances, subclasses are not valid)
    obj_type = type(objs[0])
    assert( all( type(obj) == obj_type for obj in objs ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    self._objs = objs

    # Add member functions to this object based on the member functions of the wrapped objects
    call_lambda = lambda _attr, _cls_obj, _objs : lambda *pargs, **kwargs: self.loop_caller(_attr,_cls_obj,_objs,*pargs,**kwargs)
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') ):
      # Class Methods
      if callable(getattr(cls_obj,func)):
        setattr( self, func, call_lambda(func,cls_obj,objs) )
      # Support for properties
      elif isinstance(getattr(cls_obj, func), property):
        prop_objs = [ getattr(obj, func) for obj in objs ]
        prop_objs_type = type(prop_objs[0])
        setattr( self, func, LoopWrapper(prop_objs_type, objects=prop_objs))

  def loop_caller(self,attr,cls_obj,objs,*pargs,**kwargs):
    if callable(getattr(cls_obj, attr)):
      result = [ getattr(cls_obj,attr)( obj, *pargs, **kwargs ) for obj in objs ]
    else:
      result = [ getattr(obj,attr) for obj in objs ]

    if self.supports_returncode:
      return_codes = np.array([ obj.returncode() for obj in objs ])
      parse_return_codes( return_codes )
    return result

class PoolWrapper(Wrapper):
  def __init__( self,
                cls_obj : Type,
                proc_pargs : List[List[Any]] = [],
                proc_kwargs : List[Mapping[Any,Any]] = [],
                objects: List[Any] = None ):
    # validate parameters
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate worker processes')

    # Call super class constructor to set up return code if available
    super(PoolWrapper, self).__init__(cls_obj)

    if objects:
      objs = objects
    else:
      with mp.ProcessingPool(len(proc_pargs)) as pool:
        objs = pool.map( lambda args: cls_obj( *args[0], **args[1] ), list( zip(proc_pargs, proc_kwargs) ) )

    # ensure all contained objects are the same exact type (no instances, subclasses are not valid)
    obj_type = type(objs[0])
    assert( all( type(obj) == obj_type for obj in objs ) )

    # right now this is just to retain the objs for debuging, but they are all captured in lambdas so this isn't explicitly required
    self._objs = objs

    # Add member functions to this object based on the member functions of the wrapped objects
    call_lambda = lambda _attr, _cls_obj, _objs : lambda *pargs, **kwargs: self.pool_caller(_attr,_cls_obj,_objs,*pargs,**kwargs)
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') ):
      # Class Methods
      if callable(getattr(cls_obj,func)):
        setattr( self, func, call_lambda(func,cls_obj,objs) )
      # Support for properties
      elif isinstance(getattr(cls_obj, func), property):
        prop_objs = [ getattr(obj, func) for obj in objs ]
        prop_objs_type = type(prop_objs[0])
        setattr( self, func, LoopWrapper(prop_objs_type, objects=prop_objs))

  def returncode_wrapper(func):
    """Wrapper to run function and get returncode in single call.

    NOTE: This is needed because the state of each object is not maintained across calls
          to pool.map, so all data needs to be retrieved in one call.
    """
    def wrapper(self, *pargs, **kwargs):
      res = func(self, *pargs, **kwargs)
      return_codes = self.returncode()
      return res, return_codes
    return wrapper

  def pool_caller(self,attr,cls_obj,objs,*pargs,**kwargs):
    # we make the list ahead of time to bind the pargs and kwargs identically across the pool, instead of having to pass in or make arrays for them
    if self.supports_returncode:
      to_invoke = [ partial( PoolWrapper.returncode_wrapper(getattr(cls_obj,attr)), obj, *pargs, **kwargs ) for obj in objs ]
      with mp.ProcessingPool(len(objs)) as pool:
        result = pool.map(lambda f: f(), to_invoke) # can't partial the pargs or the first will bind to self
        res, return_codes = zip(*result)
        parse_return_codes( np.array(return_codes) )
    else:
      to_invoke = [ partial( getattr(cls_obj,attr), obj, *pargs, **kwargs ) for obj in objs ]
      with mp.ProcessingPool(len(objs)) as pool:
        res = pool.map(lambda f: f(), to_invoke) # can't partial the pargs or the first will bind to self
    return res

class SharedMemKey(dict):
  """This is just here so we can compare using ininstance. Not sure if there is a better way to do this."""
  pass

class Shmallocate:
  def __init__( self ):
    self.__shmem = {}
    self.__metadata = {}
    self.__all_shmem = {}

  def active_keys( self ):
    return list( self.__shmem.keys() )

  def metadata( self, ukey ):
    return self.__metadata.get(ukey,{})

  def alloc( self, name : str, create : bool, **kwargs ):
    ukey = name
    if create:
      alloc_uuid = uuid.uuid4()
      # if alloc_uuid.is_safe != uuid.SafeUUID.safe:
        # raise SystemError("Cannot generate a multiprocessing-safe UUID for shared memory usage! Please revert to a different parallel operation mode.")
      ukey =  f"{name}-{alloc_uuid.hex}"
    shmem = shared_memory.SharedMemory( name = ukey, create = create, **kwargs )
    self.__shmem[ ukey ] = shmem
    self.__all_shmem[ ukey ] = shmem
    self.__metadata[ ukey ] = { }
    return shmem, ukey

  def ndarray( self, name : str, create : bool, shape : Any, dtype : np.dtype, **kwargs ):
    if 'buffer' in kwargs.keys():
      raise ValueError("Do not attempt to specify a buffer for shared-memory ndarray creation.")
    shmem, ukey = self.alloc( name = name, create = create, size = reduce( lambda x, y : x * y, shape ) * np.dtype(dtype).itemsize )
    self.__metadata[ ukey ][ 'shape' ] = shape
    self.__metadata[ ukey ][ 'dtype' ] = dtype
    return np.ndarray( shape = shape, dtype = dtype, buffer = shmem.buf, **kwargs ), ukey

  def close( self ):
    for shmem in self.__shmem.values():
      shmem.close()
    self.__metadata = {}
    self.__shmem = {}

  # this removes responsiblity for managing the allocated shared memory
  #  from this shmallocate object and places the responsibility on the user
  def relinquish( self, ukey ):
    shmem = self.__all_shmem.pop( ukey, None )
    return shmem

  def unlink( self ):
    for shmem in self.__all_shmem.values():
      shmem.close()
      shmem.unlink()
    self.__all_shmem = {}


class ServerWrapper(Wrapper):
  """Multiprocessing client/server wrapper for reading Mili databases in parallel.

  When more plot files exist than processors, each processor handles multiple file
  using the same concept as the loop wrapper.
  """
  class ClientWrapper(Process):
    def __init__( self,
                  conn,
                  cls_obj: Type,
                  use_shared_memory: bool,
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
      self.__use_shared_memory = use_shared_memory
      self.__shmalloc = Shmallocate()

    def __from_shared_mem(self, item):
      if not self.__use_shared_memory:
        return item
      elif isinstance(item, SharedMemKey):
        shared_mem, _  = self.__shmalloc.ndarray( **item, create=False )
        return shared_mem
      elif isinstance(item, list):
        return [self.__from_shared_mem(i) for i in item]
      elif isinstance(item, dict):
        return { k:self.__from_shared_mem(v) for k,v in item.items() }
      else:
        return item

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
            pargs = self.__from_shared_mem( pargs )
            kwargs = self.__from_shared_mem( kwargs )
            try:
              if callable(getattr(obj_type, cmd)):
                result = [ getattr( self.__cls_obj, cmd )( wrapped, *pargs, **kwargs) for wrapped in self.__wrapped ]
              elif isinstance(getattr(obj_type, cmd), property):
                result = [ getattr( wrapped, cmd ) for wrapped in self.__wrapped ]
            except:
              result = []
            self.__conn.send_bytes( dill.dumps(result) )
            # Call close() since we don't need shared_mem pargs/kwargs anymore on this subprocess
            self.__shmalloc.close()

      return

  def __to_shared_mem(self, item):
    if not self.__use_shared_memory:
      return item
    elif isinstance(item, np.ndarray):
      if item.size == 0:
        return item
      array, key = self.__shmalloc.ndarray( name = "shm", create=True, shape=item.shape, dtype=item.dtype )
      array[:] = item[:]
      return SharedMemKey({'name': key, 'shape': item.shape, 'dtype': item.dtype})
    elif isinstance(item, list):
      # Try to convert lists to numpy arrays so we can use shared memory
      if len(item) > 0 and isinstance(item[0], (int,np.integer,float,np.floating)):
        return self.__to_shared_mem( np.array(item) )
      else:
        return [ self.__to_shared_mem(i) for i in item]
    elif isinstance(item, dict):
      return { k:self.__to_shared_mem(v) for k,v in item.items() }
    else:
      return item

  def __divide_into_sequential_groups( self, values, n_groups ):
    if n_groups > len(values):
      n_groups = len(values)
    d, r = divmod( len(values), n_groups )
    for i in range( n_groups ):
      si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
      yield values[si:si+(d+1 if i < r else d)]

  def __init__( self,
               cls_obj : Type,
               proc_pargs : List[List[Any]] = [],
               proc_kwargs : List[Mapping[Any,Any]] = [],
               use_shared_memory: Optional[bool] = True ):
    # validate parameters
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate worker processes')

    # Call super class constructor to set up return code if available
    super(ServerWrapper, self).__init__(cls_obj)

    # need to have a lambda that returns new lambdas for each func to avoid only have a single lambda bound for all funcs
    mem_maker = lambda attr : lambda *pargs, **kwargs : self.__worker_call( attr, *pargs, **kwargs )
    # generate member functions mimicking those in the wrapped class, excluding private and class functions
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') ):
      # Class Methods and Properties
      if callable(getattr(cls_obj,func)) or isinstance(getattr(cls_obj, func), property):
        setattr( self, func, mem_maker(func) )

    # Get the number of processors that are available
    n_cores = int( psutil.cpu_count(logical=False) )

    self.__use_shared_memory = use_shared_memory
    self.__shmalloc = Shmallocate()

    # WARNING: Do not remove this.
    # The multiprocessing resource_tracker is instantiated the first time a SharedMemory object
    # is created. We need to create some memory here and then delete it so that the resource_tracker
    # is created now and gets inherited by all the subprocesses when we fork().
    # This prevents erroneous warnings that shared memory is being leaked because now all processes are
    # using the same resource tracker.
    if self.__use_shared_memory:
      shmem = shared_memory.SharedMemory( create = True, size = 4 )
      shmem.close()
      shmem.unlink()

    # Break out into groups for each processor available
    proc_pargs = self.__divide_into_sequential_groups( proc_pargs, n_cores )
    proc_kwargs = self.__divide_into_sequential_groups( proc_kwargs, n_cores )

    # create a pool of worker processes and supply the constructor arguments to each one
    self.__pool = []
    self.__conns = []
    for pargs, kwargs in zip( proc_pargs, proc_kwargs ):
      c0, c1 = multiprocessing.Pipe()
      self.__pool.append( ServerWrapper.ClientWrapper( c1, cls_obj, self.__use_shared_memory, pargs, kwargs ) )
      self.__conns.append( c0 )

    atexit.register(self.__cleanup_processes)

    # spawn the worker processes
    for proc in self.__pool:
      proc.start()

    for func in ( func for func in dir(cls_obj) if not func.startswith('__') and not func.startswith(f'_{cls_obj.__name__}') ):
      # Rewrap properties with PoolWrapper
      if isinstance(getattr(cls_obj, func), property):
        prop_objs = getattr(self, func)([],{})
        prop_objs_type = type(prop_objs[0])
        setattr( self, func, LoopWrapper(prop_objs_type, objects=prop_objs))

  def close(self):
    """Close the database and end all subprocesses."""
    atexit.unregister(self.__cleanup_processes)
    self.__cleanup_processes()

  # Need this so processes don't hang and close properly
  def __cleanup_processes(self):
    if hasattr( self, f'_{__class__.__name__}__conns' ):
      for conn in self.__conns:
        conn.send( dill.dumps( ('__exit', [], {} ) ) )
    self.__conns = []
    if hasattr( self, f'_{__class__.__name__}__pool'):
      for proc in self.__pool:
        proc.join( )
    self.__pool = []

  def __flatten( self, alist: List[Any] ):
    return [item for sublist in alist for item in sublist]

  # called by the generated member functions to send the command (attr name) and function arguments to
  #  each of the processes managed by this server. All arguments are sent to all processes, sending specific
  #  arguments to individual processes is not supported currently.
  def __worker_call(self,attr,*pargs,**kwargs):
    results = [None] * len(self.__conns)

    pargs = self.__to_shared_mem( pargs )
    kwargs = self.__to_shared_mem( kwargs )
    data = (attr, pargs, kwargs)

    # Send command information to subprocesses
    for conn in self.__conns:
      conn.send( dill.dumps( data ) )

    for idx, conn in enumerate(self.__conns):
      results[idx] = dill.loads( conn.recv_bytes() )

    self.__shmalloc.unlink()

    if self.supports_returncode:
      data = ("returncode", [], {})
      for conn in self.__conns:
        conn.send( dill.dumps( data ) )

      return_codes = []
      for idx, conn in enumerate(self.__conns):
        return_codes.append( dill.loads( conn.recv_bytes() ) )
      return_codes = np.array(self.__flatten(return_codes))
      parse_return_codes(return_codes)

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
