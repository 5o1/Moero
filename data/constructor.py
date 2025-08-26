import torch
import warnings
from typing import Type, Callable, List

class GridConstructor(torch.nn.Module):
    grid_flags: torch.Tensor # Tensor[bool]
    grid_elements: dict[tuple, torch.Tensor]  # dict[idx: Tensor]

    def __init__(self, grid_shape: tuple, src_class: Type[torch.Tensor], dst_class: Type[torch.Tensor], **kwargs_for_build):
        super().__init__()
        self.grid_shape = grid_shape
        self.src_class = src_class
        self.dst_class = dst_class
        self.kwargs_for_build = kwargs_for_build

        self.grid_elements = dict() # idx: Tensor
        self.grid_flags = torch.zeros(grid_shape, dtype=torch.bool)
        self._result = None

    def is_completed(self) -> bool:
        """
        Check if the grid is full.
        A grid is considered full if all positions are occupied.
        """
        return self.grid_flags.all()

    def construct(self) -> torch.Tensor:
        """
        Construct the grid from the elements.
        If the grid is not full, raise an error.
        """

        if not self.is_completed():
            raise ValueError("Grid is not full. Cannot construct.")

        # Convert grid_elements to a tensor
        first_element = next(iter(self.grid_elements.values()))
        dst = torch.empty([*self.grid_shape, *first_element.shape], device=first_element.device, dtype=first_element.dtype)
        for idx, element in self.grid_elements.items():
            dst[idx] = element
        return self.dst_class(dst, **self.kwargs_for_build)

    def update(self, other: "GridConstructor") -> None:
        """
        Update the grid_elements and grid_flags from another GridConstructor object.

        Args:
            other (GridConstructor): Another GridConstructor object to update from.

        Raises:
            ValueError: If the grid_shape or src_class of the other object does not match.
        """
        if self.grid_shape != other.grid_shape:
            raise ValueError(f"Grid shapes do not match: {self.grid_shape} != {other.grid_shape}")
        if self.src_class != other.src_class:
            raise ValueError(f"Source classes do not match: {self.src_class} != {other.src_class}")

        self.grid_elements.update(other.grid_elements)
        self.grid_flags = self.grid_flags | other.grid_flags

    def forward(self, x: torch.Tensor, idx) -> None | torch.Tensor:
        """
        Add a new sample to the grid.
        If the grid is full, return dst_class with the current grid.
        """
        if not isinstance(x, self.src_class):
            raise TypeError(f"Expected input of type {self.src_class}, got {type(x)}")

        x = x.detach().cpu()

        self.grid_elements[idx] = x
        self.grid_flags[idx] = True

    def __repr__(self):
        return f"GridConstructor(grid_shape={self.grid_shape}, src_class={self.src_class.__name__}, dst_class={self.dst_class.__name__})"