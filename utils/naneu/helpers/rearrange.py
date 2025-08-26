from abc import abstractmethod
from einops import rearrange
from functools import wraps
import inspect
import torch
import re
from typing import List, Union, Literal, Callable, Iterable, Tuple, Type, Any

Size = Any

def is_unique(iterable):
    return len(iterable) == len(set(iterable))

# def func_argnames_parser(func: callable, white_list: List[Union[str, int]] | Literal["all"] | Literal["firstonly"]) -> List[str]:
#     signature = inspect.signature(func)
#     parameters = signature.parameters
    
#     # params_offset = 1 if inspect.ismethod(forward) else 0
#     params_offset = 0
#     params_mapper = {index: name for index, (name, param) in enumerate(parameters.items())}

#     if white_list == "firstonly":
#         accept_list = [params_mapper[0 + params_offset]]
#     elif white_list == "all":
#         accept_list = [params_mapper[i] for i in range(len(parameters))]
#     else:
#         accept_list = []
#         for key in white_list:
#             if isinstance(key, int):
#                 if key > len(parameters) - params_offset:
#                     raise ValueError(f"Index `{key}` out of range for parameters.")
#                 key = params_mapper.get(key + params_offset)
#             else:
#                 if key not in parameters:
#                     raise ValueError(f"Parameter `{key}` not found in the function signature.")
#             accept_list.append(key)

#         if len(accept_list) == 0:
#             raise ValueError("No valid parameters provided in `for_in`.")
#     return accept_list

# def process_args_kwargs(parameters, *args, **kwargs):
#     bound_arguments = {}
#     for i, (name, param) in enumerate(parameters.items()):
#         if i < len(args):
#             bound_arguments[name] = args[i]
#         elif name in kwargs:
#             bound_arguments[name] = kwargs[name]
#         elif param.default != inspect.Parameter.empty:
#             bound_arguments[name] = param.default
#         else:
#             raise ValueError(f"Missing required argument: `{name}`")
    
#     return bound_arguments

# def quxarrange_impl(obj: torch.nn.Module, input: str, hidden: str = "$ c h w", output: str = None, qux: str = "$", 
#                       for_in: Union[List[Union[str, int]], Literal["all"], Literal["firstonly"]] = "firstonly",
#                       for_out: Union[List[int], Literal["all"]] = "all") -> torch.nn.Module:
#     """
#     A wrapper function to modify the `forward` method of a PyTorch module by dynamically rearranging tensor dimensions
#     during the forward pass. The function allows flexible input-output dimension mappings, ensuring that tensors
#     conform to specified patterns before and after being passed through the module.

#     In most cases because modifying the tensor's view does not extra memory.

#     ### Parameters:
#     - **obj** (`torch.nn.Module`):  
#       The PyTorch module whose `forward` method will be wrapped and modified.
    
#     - **input** (`str`):  
#       A space-separated string representing the dimension names of the input tensor (e.g., `"b c h w"`).
    
#     - **hidden** (`str`, optional):  
#       A space-separated string representing the intermediate "hidden" dimension names used during the forward pass.  
#       Default is `"$ c h w"`.
      
#     - **output** (`str`, optional):  
#       A space-separated string representing the dimension names of the output tensor. If not provided, it defaults to the same as `input`.
    
#     - **qux** (`str`, optional):  
#       The dimension name in `hidden` that will combine dimensions not explicitly listed in `hidden`.  
#       Default is `"$"`.
    
#     - **for_in** (`Union[List[Union[str, int]], Literal["all"], Literal["firstonly"]]`, optional):  
#       Specifies which input arguments to the `forward` method should be rearranged:  
#       - `"firstonly"`: Rearrange only the first input argument (default).  
#       - `"all"`: Rearrange all input arguments.  
#       - `List[str | int]`: Explicit list of argument names or indices to rearrange.
    
#     - **for_out** (`Union[List[int], Literal["all"]]`, optional):  
#       Specifies which outputs from the `forward` method should be rearranged:  
#       - `"all"`: Rearrange all outputs (default).  
#       - `List[int]`: Explicit list of output indices to rearrange.

#     ### Example Usage:
#     ```python
#     from torch import nn
#     from einops import rearrange

#     class SimpleModel(nn.Module):
#         def forward(self, x):
#             # Example forward pass
#             return x.mean(dim=1, keepdim=True)

#     model = SimpleModel()

#     # Wrap the model with rearrange_wrapper
#     model = rearrange_wrapper(
#         obj=model,
#         input="b c h w",
#         output="b h w",
#         hidden="$ c h w",
#         qux="$",
#         for_in="firstonly",
#         for_out="all"
#     )

#     # Input tensor
#     x = torch.randn(2, 3, 4, 4)

#     # Forward pass with automatic rearrangement
#     y = model(x)  # Input is rearranged from "b c h w" -> "$ c h w", and output is rearranged back to "b h w"
#     print(y.shape)  # Output shape: (2, 4, 4)
#     ```
#     """

#     input = input.strip().split()
#     output = output.strip().split() if output is not None else input
#     hidden = hidden.strip().split()
#     qux = qux.strip()

#     # Args checking
#     def is_unique(lst):
#          return len(lst) == len(set(lst))
    
#     for name, lst in zip(["input", "output", "hidden"], [input, output, hidden]):
#         if not is_unique(lst):
#             raise ValueError(f"`{name}` must be unique, but got {lst}.")

#     if qux not in hidden:
#         raise ValueError(f"`combine_to` {qux} must in {hidden}.")

#     for dim in hidden:
#         if dim == qux:
#             continue
#         if dim not in input:
#             raise ValueError(f"Expected `{dim}` in input {input}")
#         if dim not in output:
#             raise ValueError(f"Expected `{dim}` in output {output}")
    
#     idx_qux = hidden.index(qux)
#     combine = [dim for dim in input if dim not in hidden]
#     hidden[idx_qux] = f"({' '.join(combine)})"

#     in_pattern = " ".join(input)
#     out_pattern = " ".join(output)
#     hidden_pattern = " ".join(hidden)

#     forward = obj.forward
#     parameters = func_argnames_parser(forward, white_list = for_in)

#     def output_filter(output, origin_shape):
#         if not isinstance(output, (list, tuple)):
#             output = rearrange(output, f"{hidden_pattern} -> {out_pattern}", **{dim: origin_shape[dim] for dim in combine})
#         else:
#             output_ = []
#             for i, output_item in enumerate(output):
#                 if for_out == "all" or i in for_out:
#                     output_item = rearrange(output_item, f"{hidden_pattern} -> {out_pattern}", **{dim: origin_shape[dim] for dim in combine})
#                 output_.append(output_item)
#             if isinstance(output, tuple):
#                 output_ = tuple(output_)
#             output = output_
#         return output
    
#     @wraps(forward)
#     def rearrange_aspect_variadic(*args, **kwargs):
#         kwargs = process_args_kwargs(parameters, *args, **kwargs)
#         origin_shape = None
#         for argi, argv in kwargs.items():
#             if argi in for_in:
#                 if not isinstance(argv, torch.Tensor):
#                     raise ValueError(f"Parameter `{argi}` must be a `torch.Tensor`, but got {type(argv)}.")
#                 origin_shape_ = dict(zip(input, argv.shape))
#                 if origin_shape is None:
#                     origin_shape = origin_shape_
#                 argv = rearrange(argv, f"{in_pattern} -> {hidden_pattern}")
#                 kwargs[argi] = argv
#         output = forward(**kwargs)
#         output = output_filter(output, origin_shape)
#         return output

#     obj.forward = rearrange_aspect_variadic
#     return obj

class TorchModuleForwardInputCallbacksManager:
    def __init__(self, forward: Callable):
        self.forward = forward # Must be a bound method
        if not inspect.ismethod(self.forward):
            raise ValueError("Expected `module.forward` to be a bound method, but got a function.")
        self.callbacks: List[Tuple[List[str], Callable[[torch.Tensor], torch.Tensor]]] = list()

        self.args_idx2name = list(inspect.signature(self.forward).parameters)

    def register(self, argnames: List[str| int] | Literal["all"] | Literal["firstonly"], callback: Callable[[torch.Tensor], torch.Tensor]):
        if not callable(callback):
            raise ValueError("Callback must be a callable function.")
        
        solved_argname = []
        if isinstance(argnames, str):
            if argnames == "all":
                solved_argname = self.args_idx2name
            elif argnames == "firstonly":
                solved_argname = [self.args_idx2name[0]]
            else:
                raise ValueError("`argnames` must be 'all', 'firstonly', or a list of argument names.")
        elif isinstance(argnames, Iterable):
            for name in argnames:
                if isinstance(name, int):
                    if name < 0 or name >= len(self.args_idx2name):
                        raise ValueError(f"Index `{name}` out of range for parameters.")
                    solved_argname.append(self.args_idx2name[name])
                elif isinstance(name, str):
                    solved_argname.append(name)
                else:
                    raise ValueError(f"Expected `argnames` to be a list of strings or integers, but got {type(name)}.")
        else:
            raise ValueError(f"Expected `argnames` to be a list of strings or integers, but got {type(argnames)}.")

        if not is_unique(solved_argname):
                raise ValueError(f"Expected `argnames` to be unique, but got {argnames}.")

        self.callbacks.append((solved_argname, callback))

    def __call__(self, *args, **kwargs):
        bound_arguments = inspect.signature(self.forward).bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        for argname, value in bound_arguments.arguments.items():
            for callback_argnames, callback in self.callbacks:
                if argname in callback_argnames:
                    value = callback(value)
            bound_arguments.arguments[argname] = value

        return bound_arguments.args, bound_arguments.kwargs

class TorchModuleForwardOutputCallbacksManager:
    def __init__(self, forward: Callable):
        self.forward = forward # Must be a bound method
        if not inspect.ismethod(self.forward):
            raise ValueError("Expected `module.forward` to be a bound method, but got a function.")
        self.callbacks: List[Tuple[List[int] | str, Callable[[torch.Tensor], torch.Tensor]]]  = list()

    def register(self, output_indices: List[int] | Literal["all"], callback: Callable[[torch.Tensor], torch.Tensor]):
        if not callable(callback):
            raise ValueError("Callback must be a callable function.")
        
        solved_output_indices = []
        if isinstance(output_indices, str):
            if output_indices == "all":
                solved_output_indices = "all"
            else:
                raise ValueError("`output_indices` must be 'all' or a list of output indices.")
        elif isinstance(output_indices, Iterable):
            for idx in output_indices:
                if isinstance(idx, int):
                    solved_output_indices.append(idx)
                else:
                    raise ValueError(f"Expected `output_indices` to be a list of integers, but got {type(idx)}.")
            if not is_unique(solved_output_indices):
                raise ValueError(f"Expected `output_indices` to be unique, but got {solved_output_indices}.")
        else:
            raise ValueError(f"Expected `output_indices` to be a list of integers, but got {type(output_indices)}.")
        self.callbacks.append((solved_output_indices, callback))
        
    def __call__(self, output):
        if output is None:
            return output
        
        if isinstance(output, tuple):
            output = list(output)
        else:
            output = [output]

        for i, out in enumerate(output):
            for output_indices, callback in reversed(self.callbacks):
                if output_indices == "all" or i in output_indices:
                    out = callback(out)
            output[i] = out

        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)

        return output

class TorchModuleForwardHook(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise ValueError("Expected `module` to be an instance of `torch.nn.Module`.")
        if not hasattr(module, "forward"):
            raise ValueError("Module must have a `forward` method.")
        if isinstance(module, TorchModuleForwardHook):
            raise ValueError("Module is already wrapped with `TorchModuleForwardHook`.")
        
        self.module = module
        self.input_callbacks = TorchModuleForwardInputCallbacksManager(self.module.forward)
        self.output_callbacks = TorchModuleForwardOutputCallbacksManager(self.module.forward)

    def __getattr__(self, name):
        if name == "module":
            return super().__getattr__(name)
        try:
            return getattr(self.module, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        args, kwargs = self.input_callbacks(*args, **kwargs)
        output = self.module.forward(*args, **kwargs)
        output = self.output_callbacks(output)
        return output
            

class TorchModuleForwardCallback:
    def __init__(
        self,
        hook: TorchModuleForwardHook,
        for_input: List[str| int] | Literal["all"] | Literal["firstonly"] = None,
        for_output: List[int] | Literal["all"] = None,
        ):
        if for_input is None and for_output is None:
            raise ValueError("At least one of `for_input` or `for_output` must be provided.")
        self.hook = hook
        self.hook.input_callbacks.register(for_input, self.input_callback)
        self.hook.output_callbacks.register(for_output, self.output_callback)

    @classmethod
    def bind(cls, method_name: str):
        if hasattr(torch.nn.Module, method_name):
            return
        
        def register(self, *args, **kwargs):
            if not isinstance(self, TorchModuleForwardHook):
                self = TorchModuleForwardHook(self)
            cls(self, *args, **kwargs)
            return self
        setattr(torch.nn.Module, method_name, register)

    @abstractmethod
    def input_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Callback function to process the input tensor before passing it to the module's forward method.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `input_callback` method.")
    
    @abstractmethod
    def output_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Callback function to process the output tensor after the module's forward method has been called.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement `output_callback` method.")

class ViewAsReal(TorchModuleForwardCallback):
    def input_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return torch.view_as_real(tensor).contiguous()
    
    def output_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return torch.view_as_complex(tensor.contiguous())
    
    def __init__(self, module: torch.nn.Module, for_input: List[str| int] | Literal["all"] | Literal["firstonly"] = "all", for_output: List[int] | Literal["all"] = "all"):
        super().__init__(module, for_input=for_input, for_output=for_output)

class ViewAsReal2Chan(TorchModuleForwardCallback):
    def input_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        original_shape = list(tensor.shape)
        real_imag = torch.view_as_real(tensor)

        n = self.dim
        ndim = len(original_shape)
        if n < 0:
            n = ndim + n
        permute_order = list(range(ndim)) + [ndim]
        permute_order.insert(n + 1, permute_order.pop(-1))
        real_imag = real_imag.permute(permute_order)
        new_shape = real_imag.shape[:n] + (-1,) + real_imag.shape[n+2:]
        return real_imag.reshape(new_shape).contiguous()

    def output_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        tensor = tensor.contiguous()

        original_shape = list(tensor.shape)
        ndim = len(original_shape)
        
        if n < 0:
            n = ndim + n

        if original_shape[n] % 2 != 0:
            raise ValueError("The size of the channel dimension must be even to represent real and imaginary parts.")

        new_shape = original_shape[:n] + [original_shape[n] // 2, 2] + original_shape[n+1:]
        real_imag = tensor.reshape(new_shape)

        permute_order = list(range(ndim + 1))
        permute_order.append(permute_order.pop(n + 1))
        real_imag = real_imag.permute(permute_order)

        return torch.view_as_complex(real_imag)

    def __init__(self, module: torch.nn.Module, dim:int = -3, for_input: List[str| int] | Literal["all"] | Literal["firstonly"] = "all", for_output: List[int] | Literal["all"] = "all"):
        super().__init__(module, for_input=for_input, for_output=for_output)
        self.dim = dim

class Rearrange(TorchModuleForwardCallback):
    def split(self, pattern: str):
        return re.findall(r'\(.*?\)|\S+', pattern)
    
    def input_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        pattern = self.pattern_outer + "->" + self.pattern_inner
        axes_lengths = self.axes_lengths_outer.copy()

        if "(" in self.pattern_inner and ")" in self.pattern_inner:
            groups_outer = self.split(self.pattern_outer)
            groups_inner = self.split(self.pattern_inner)
            self.axes_lengths_inner = dict()
            for group in groups_inner:
                if "(" in group and ")" in group:
                    subpattern = re.sub(r'[()]', '', group)
                    for subgroup in subpattern.split():
                        if subgroup not in groups_outer:
                            raise ValueError(f"Expected `{subgroup}` in outer pattern `{self.pattern_outer}`.")
                        self.axes_lengths_inner[subgroup] = tensor.size(groups_outer.index(subgroup))
            axes_lengths.update(self.axes_lengths_inner)
        else:
            self.axes_lengths_inner = None

        return rearrange(tensor, pattern, **axes_lengths).contiguous()

    def output_callback(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            return tensor
        pattern = self.pattern_inner + "->" + self.pattern_outer
        if self.axes_lengths_inner is None:
            axes_lengths = self.axes_lengths_outer.copy()
        else:
            axes_lengths = self.axes_lengths_outer.copy()
            axes_lengths.update(self.axes_lengths_inner)

        return rearrange(tensor, pattern, **axes_lengths)

    def __init__(self, module: torch.nn.Module, pattern: str, for_input: List[str| int] | Literal["all"] | Literal["firstonly"] = "all", for_output: List[int] | Literal["all"] = "all", **axes_lengths: Size):
        super().__init__(module, for_input=for_input, for_output=for_output)
        self.pattern_outer, self.pattern_inner = pattern.split("->")
        self.axes_lengths_outer, self.axes_lengths_inner = axes_lengths, None

# Monkey patching
ViewAsReal.bind("view_as_real")
ViewAsReal2Chan.bind("view_as_real_to_chan")
Rearrange.bind("rearrange")
