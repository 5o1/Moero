import importlib
from types import ModuleType
import inspect
from typing import Optional

import numpy

class LazyModule(ModuleType):
    """
    Also see: https://github.com/5o1/pyLazyModule
    """
    __redirect_attrs = [
        # Basic attributes
        "__class__", "__doc__", "__name__", "__dict__", "__annotations__",
        "__file__", "__package__", "__path__", "__spec__", "__loader__", "__builtins__",
        
        # Type and identity related
        "__module__", "__qualname__", "__bases__", "__mro__", "__subclasses__", "__slots__",
        
        # Operator overloading methods
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",  # Comparison operators
        "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__",  # Arithmetic operators
        "__and__", "__or__", "__xor__", "__lshift__", "__rshift__",  # Bitwise operators
        "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__",  # Reverse operators
        "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__",  # In-place operators
        "__neg__", "__pos__", "__abs__", "__invert__",  # Unary operators
        "__round__", "__floor__", "__ceil__", "__trunc__",  # Mathematical operations

        # Callable and iterable related
        "__call__", "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__",
        "__len__", "__contains__", "__reversed__",

        # Attribute access and management
        "__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__dir__",
        "__slots__", "__dict__", "__weakref__",
        
        # Memory management related
        "__new__", "__init__", "__del__", "__copy__", "__deepcopy__", "__sizeof__",

        # Descriptor related
        "__get__", "__set__", "__delete__",

        # String representation and debugging
        "__str__", "__repr__", "__format__", "__hash__",
        
        # Exception and context management
        "__enter__", "__exit__", "__cause__", "__context__",

        # Metaclass related
        "__metaclass__", "__prepare__",

        # Extended protocols
        "__fspath__",  # Filesystem path protocol (Python 3.6+)

        # Custom attributes (placeholder, add based on the target class)
        "__custom_attr__",  # Example
    ]

    def __init__(self, module_path: str):
        assert module_path
        self.__module_path = module_path

        stack = inspect.stack(1)
        caller_frame = stack[1] 
        caller_module = inspect.getmodule(caller_frame[0])
        self.__package = caller_module.__package__ if caller_module else None

        self.__module = None

        self.cache = numpy.ones((100,100,100,100))

    def _resolve_import_path(self, path: str, package: Optional[str]):
        prefix_dots = len(path) - len(path.lstrip("."))
        parts = path.lstrip(".").split(".")
        parts[0] = "." * prefix_dots + parts[0]
        
        module = None
        attr_chain = []
        e = None

        for i in range(len(parts) - 1, -1, -1):
            try:
                module = importlib.import_module(".".join(parts[:i + 1]), package)
                attr_chain = parts[i+1:]
            except ModuleNotFoundError as _e:
                e = _e
                continue

        if module is None:
            raise e

        for attr in attr_chain:
            module = getattr(module, attr)
        return module

    def _load_module(self):
        if self.__module is None:
            self.__module = self._resolve_import_path(f"{self.__module_path}", self.__package)
        return self.__module
    
    def __getattr__(self, attr):
        return getattr(self._load_module(), attr)
    
    def __getattribute__(self, attr):
        if str(attr) in LazyModule.__redirect_attrs:
            module = super().__getattribute__('_load_module')()
            return getattr(module, attr)
        return super().__getattribute__(attr)
    
    def __call__(self, *args, **kwargs):
        module = self._load_module()
        if callable(module):
            return module(*args, **kwargs)
        raise TypeError(f"'{self.__module.__path__}' is not callable")