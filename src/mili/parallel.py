"""
SPDX-License-Identifier: (MIT)
"""
import multiprocessing
import dill  # type: ignore
import numpy as np
import atexit
import psutil
import uuid

from multiprocessing import Process, shared_memory
from multiprocessing.connection import Connection
from typing import List, Dict, Tuple, Optional, Any, Type, Union
from numpy.typing import NDArray
from functools import reduce

# TODO: probably just create a wrapper/dispatch superclass, and implement loop/pool/client-server versions instead of loop/pool in one and client/server in another

class LoopWrapper:
  def __init__(self,
               cls_obj: Type[Any],
               proc_pargs: List[List[Any]] = [],
               proc_kwargs: List[Dict[str,Any]] = [],
               objects: Optional[List[Any]] = None ) -> None:
    num_pargs = len(proc_pargs)
    num_kwargs = len(proc_kwargs)
    if num_pargs > 0 and num_kwargs == 0:
      proc_kwargs = [ {} ] * num_pargs
    elif num_kwargs > 0 and num_pargs == 0:
      proc_pargs = [] * num_kwargs
    elif num_kwargs != num_pargs:
      raise ValueError(f'Must supply the same number of pargs ({num_pargs}) and kwargs ({num_kwargs}) to instantiate object list.')

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
    call_lambda = lambda _attr, _cls_obj, _objs : lambda *pargs, **kwargs: self.__loop_caller(_attr,_cls_obj,_objs,*pargs,**kwargs)
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') ):
      # Class Methods
      if callable(getattr(cls_obj,func)):
        setattr( self, func, call_lambda(func,cls_obj,objs) )  # type: ignore  # Error because we don't type call_lambda
      # Support for properties
      elif isinstance(getattr(cls_obj, func), property):
        prop_objs = [ getattr(obj, func) for obj in objs ]
        prop_objs_type = type(prop_objs[0])
        setattr( self, func, LoopWrapper(prop_objs_type, objects=prop_objs))

  def __loop_caller(self, attr: str, cls_obj: Type[Any], objs: List[Any], *pargs: Any, **kwargs: Any) -> Any:
    """Helper function to call a specified method for all wrapped objects."""
    if callable(getattr(cls_obj, attr)):
      try:
        result = [ getattr(cls_obj,attr)( obj, *pargs, **kwargs ) for obj in objs ]
      except Exception as e:
        result = [e]
    else:
      try:
        result = [ getattr(obj,attr) for obj in objs ]
      except Exception as e:
        result = [e]
    return result

class SharedMemKey(dict):  # type: ignore  # mypy expects a type for dict
  """This is just here so we can compare using ininstance. Not sure if there is a better way to do this."""
  pass

class Shmallocate:
  def __init__(self) -> None:
    self.__shmem: Dict[str,shared_memory.SharedMemory] = {}
    self.__all_shmem: Dict[str,shared_memory.SharedMemory] = {}
    self.__metadata: Dict[str,Dict[str,Any]] = {}

  def active_keys(self) -> List[str]:
    """Get list of keys to active shared memory."""
    return list( self.__shmem.keys() )

  def metadata(self, ukey: str) -> Dict[str,Any]:
    """Get metadata for a specified piece of shared memory."""
    return self.__metadata.get(ukey,{})

  def alloc(self, name: str, create: bool, size: int) -> Tuple[shared_memory.SharedMemory,str]:
    """Allocate a new piece of shared memory."""
    ukey = name
    if create:
      alloc_uuid = uuid.uuid4()
      # if alloc_uuid.is_safe != uuid.SafeUUID.safe:
        # raise SystemError("Cannot generate a multiprocessing-safe UUID for shared memory usage! Please revert to a different parallel operation mode.")
      ukey =  f"{name}-{alloc_uuid.hex}"
    shmem = shared_memory.SharedMemory( name = ukey, create = create, size = size )
    self.__shmem[ ukey ] = shmem
    self.__all_shmem[ ukey ] = shmem
    self.__metadata[ ukey ] = { }
    return shmem, ukey

  def ndarray(self, name: str, create: bool, shape: Any, dtype: Any) -> Tuple[NDArray[Any],str]:
    """Allocate a shared memory Numpy array."""
    shmem, ukey = self.alloc( name = name, create = create, size = reduce( lambda x, y : x * y, shape ) * np.dtype(dtype).itemsize )
    self.__metadata[ ukey ][ 'shape' ] = shape
    self.__metadata[ ukey ][ 'dtype' ] = dtype
    return np.ndarray( shape = shape, dtype = dtype, buffer = shmem.buf ), ukey

  def close(self) -> None:
    """Close all active shared memory."""
    for shmem in self.__shmem.values():
      shmem.close()
    self.__metadata = {}
    self.__shmem = {}

  def relinquish(self, ukey: str) -> Optional[shared_memory.SharedMemory]:
    """Reliquish a specified piece of shared memory.

    NOTE: This removes responsiblity for managing the allocated shared memory
          from this shmallocate object and places the responsibility on the user.
    """
    shmem = self.__all_shmem.pop( ukey, None )
    return shmem

  def unlink(self) -> None:
    """Close and Unlink all active shared memory."""
    for shmem in self.__all_shmem.values():
      shmem.close()
      shmem.unlink()
    self.__all_shmem = {}


class ServerWrapper:
  """Multiprocessing client/server wrapper for reading Mili databases in parallel.

  When more plot files exist than processors, each processor handles multiple file
  using the same concept as the loop wrapper.
  """
  class _ClientWrapper(Process):
    def __init__( self,
                  conn: Connection,
                  cls_obj: Type[Any],
                  use_shared_memory: bool,
                  proc_pargs: List[List[Any]] = [],
                  proc_kwargs: List[Dict[Any,Any]] = [] ):
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

    def __from_shared_mem(self, item: Any) -> Any:
      """Convert item back from shared memory."""
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

    def run(self) -> None:
      """TODO: document"""
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
            except Exception as e:
              result = [e]
            self.__conn.send_bytes( dill.dumps(result) )
            # Call close() since we don't need shared_mem pargs/kwargs anymore on this subprocess
            self.__shmalloc.close()

      return

  def __to_shared_mem(self, item: Any) -> Any:
    """Convert item to shared memory."""
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

  def __divide_into_sequential_groups(self, values: List[Any], n_groups: int) -> Any:
    """Divide a list into sublists."""
    if n_groups > len(values):
      n_groups = len(values)
    d, r = divmod( len(values), n_groups )
    for i in range( n_groups ):
      si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
      yield values[si:si+(d+1 if i < r else d)]

  def __init__( self,
               cls_obj: Type[Any],
               proc_pargs: List[List[Any]] = [],
               proc_kwargs: List[Dict[Any,Any]] = [],
               use_shared_memory: bool = True ):
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
    for func in ( func for func in dir(cls_obj) if not func.startswith('__') ):
      # Class Methods and Properties
      if callable(getattr(cls_obj,func)) or isinstance(getattr(cls_obj, func), property):
        setattr( self, func, mem_maker(func) )  # type: ignore   # Ignore error that mem_maker is untyped.

    # Get the number of processors that are available
    n_cores = psutil.cpu_count(logical=False)
    if n_cores is None:
      raise ValueError(f"psutil.cpu_count returned None")

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
    grouped_pargs: List[List[List[Any]]] = self.__divide_into_sequential_groups( proc_pargs, n_cores )
    grouped_kwargs: List[List[Dict[Any,Any]]] = self.__divide_into_sequential_groups( proc_kwargs, n_cores )

    # create a pool of worker processes and supply the constructor arguments to each one
    self.__pool = []
    self.__conns = []
    for pargs, kwargs in zip( grouped_pargs, grouped_kwargs ):
      c0, c1 = multiprocessing.Pipe()
      self.__pool.append( ServerWrapper._ClientWrapper( c1, cls_obj, self.__use_shared_memory, pargs, kwargs ) )
      self.__conns.append( c0 )

    atexit.register(self.__cleanup_processes)

    # spawn the worker processes
    for proc in self.__pool:
      proc.start()

    for func in ( func for func in dir(cls_obj) if not func.startswith('__') ):
      # Rewrap properties with LoopWrapper
      if isinstance(getattr(cls_obj, func), property):
        prop_objs = getattr(self, func)([],{})
        prop_objs_type = type(prop_objs[0])
        setattr( self, func, LoopWrapper(prop_objs_type, objects=prop_objs))

  def close(self) -> None:
    """Close the database and end all subprocesses."""
    atexit.unregister(self.__cleanup_processes)
    self.__cleanup_processes()

  def __cleanup_processes(self) -> None:
    """Needed so processes don't hand and close properly."""
    if hasattr( self, f'_{__class__.__name__}__conns' ):  # type: ignore  # Ignore error that __class__ is undefined
      for conn in self.__conns:
        conn.send( dill.dumps( ('__exit', [], {} ) ) )
    self.__conns = []
    if hasattr( self, f'_{__class__.__name__}__pool'):  # type: ignore  # Ignore error that __class__ is undefined
      for proc in self.__pool:
        proc.join( )
    self.__pool = []

  def __flatten( self, alist: List[List[Any]] ) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in alist for item in sublist]

  def __worker_call(self, attr: str, *pargs: Any, **kwargs: Any) -> List[Any]:
    """Called by the generated member functions to send the command (attr name) and function arguments to
    each of the processes managed by this server. All arguments are sent to all processes, sending specific
    arguments to individual processes is not supported currently.
    """
    results: List[Any] = [None] * len(self.__conns)

    pargs = self.__to_shared_mem( pargs )
    kwargs = self.__to_shared_mem( kwargs )
    data = (attr, pargs, kwargs)

    # Send command information to subprocesses
    for conn in self.__conns:
      conn.send( dill.dumps( data ) )

    for idx, conn in enumerate(self.__conns):
      results[idx] = dill.loads( conn.recv_bytes() )

    self.__shmalloc.unlink()

    return self.__flatten(results)

def get_wrapper( suppress_parallel: bool = False, experimental: bool = False ) -> Union[Type[LoopWrapper],Type[ServerWrapper]]:
  Wrapper: Union[Type[LoopWrapper],Type[ServerWrapper]]
  if suppress_parallel:
    Wrapper = LoopWrapper
  else:
    Wrapper = ServerWrapper
  return Wrapper
