import torch
import functools
from torch.nn import functional as F

@functools.wraps(F.interpolate)
def interpolate(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Extension of F.interpolate to support complex tensors.

    If the input tensor is complex, the real and imaginary parts are interpolated
    separately using F.interpolate and then combined back into a complex tensor.

    If the input tensor is real, it behaves exactly like F.interpolate.
    """
    if torch.is_complex(tensor):
        real = F.interpolate(tensor.real, *args, **kwargs)
        imag = F.interpolate(tensor.imag, *args, **kwargs)
        return real + 1j * imag
    else:
        return F.interpolate(tensor, *args, **kwargs)
