import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexGaussianBlur(nn.Module):
    kernel: torch.Tensor
    def __init__(self, kernel_size: int = 7, sigma: float = 1.0, is_magnitude: bool = False, is_phase: bool = False):
        """
        Initializes the ComplexGaussianBlur module.

        Args:
            kernel_size (int): Size of the Gaussian kernel. Must be odd.
            sigma (float): Standard deviation for the Gaussian kernel.
            is_phase (bool): If True, applies the blur to the phase component.
            is_magnitude (bool): If True, applies the blur to the magnitude component.
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        if not (is_magnitude or is_phase):
            raise ValueError("At least one of is_magnitude or is_phase must be True.")
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.is_magnitude = is_magnitude
        self.is_phase = is_phase

        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        coords = coords.float()

        gaussian_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gaussian_1d /= gaussian_1d.sum()

        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
        gaussian_2d /= gaussian_2d.sum()

        return gaussian_2d.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, kernel_size, kernel_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        
        raw_shape = tensor.shape
        tensor = tensor.view(-1, *raw_shape[-3:])  # Reshape to (batch_size, channels, height, width)

        magnitude = tensor.abs()
        phase = tensor.angle()
        kernel = self.kernel.expand(phase.size(-3), -1, self.kernel_size, self.kernel_size)

        if self.is_magnitude:
            magnitude = F.conv2d(
                magnitude,
                kernel,
                padding=self.kernel_size // 2,
                groups=magnitude.size(-3)
            )

        if self.is_phase:
            phase = F.conv2d(
                phase,
                kernel,
                padding=self.kernel_size // 2,
                groups=phase.size(-3)
            )

        result = magnitude * torch.exp(1j * phase)
        result = result.view(*raw_shape)

        return result