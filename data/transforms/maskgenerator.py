"""
**Caution**: The implementations of mask generators differs from the sample code provided by cmrxrecon25, which may lead to errors.
"""
import torch
import math
from abc import abstractmethod
from typing import Sequence, List

class MaskGenerator(torch.nn.Module):
    def __init__(
            self,
            accel_factors: List[int] = [8, 16, 24],
            accel_weights:List[float] = None,
            ncalibs: List[int] = [16, 20],
            calib_weights: List[float] = None,
            rng: torch.Generator = None,
            dimidx_phase: int = -1,
            ):
        super().__init__()
        self.accel_factors = torch.as_tensor(accel_factors, dtype = torch.int32)
        self.accel_weights = torch.as_tensor(accel_weights if accel_weights is not None else [1] * len(accel_factors), dtype=torch.float32)
        self.ncalibs = torch.as_tensor(ncalibs, dtype = torch.int32)
        self.calib_weights = torch.as_tensor(calib_weights if calib_weights is not None else [1] * len(ncalibs), dtype=torch.float32)
        self.dimidx_phase = dimidx_phase

        self.accel_weights = self.accel_weights / self.accel_weights.sum()
        self.calib_weights = self.calib_weights / self.calib_weights.sum()

        if rng is None:
            self.rng = torch.Generator()
            self.rng.manual_seed(torch.initial_seed())
        else:
            self.rng = rng

    @abstractmethod
    def make_mask(self, size: Sequence[int], accel_factor: float, ncalib: int) -> torch.Tensor:
        """
        Generate a mask for the given size.
        
        Args:
            size (Sequence[int]): The size of the mask to generate.
        
        Returns:
            torch.Tensor[float] shape[batch phase readout] : A tensor representing the mask.
        """
        pass

    def forward(self, size: Sequence[int]) -> torch.Tensor:
        """
        Generate a mask for the given size.
        
        Args:
            size (Sequence[int]): The size of the mask to generate.
        
        Returns:
            torch.Tensor[float] shape[batch phase readout] : A tensor representing the mask.
        """
        acc_ratio = self.accel_factors[torch.multinomial(self.accel_weights, num_samples=1, generator = self.rng).item()].item()
        acs_size = self.ncalibs[torch.multinomial(self.calib_weights, num_samples=1, generator = self.rng).item()].item()
        return self.make_mask(size, acc_ratio, acs_size), acc_ratio, acs_size

class UniformMaskGenerator(MaskGenerator):
    def make_mask(self, size: Sequence[int], accel_factor: float, ncalib: int) -> torch.Tensor:
        nshot = size[self.dimidx_phase]

        ncalib = ncalib + ((nshot % 2) != (ncalib % 2))

        acs_start = math.floor(nshot / 2) + math.ceil(-ncalib / 2)
        acs_end = math.floor(nshot / 2) + math.ceil(ncalib / 2)

        indices = list(range(0, nshot, accel_factor)) + list(range(acs_start, acs_end))

        # Create masks using the sampled indices
        mask = torch.zeros(size, dtype=torch.float32, device = self.accel_factors.device)
        mask.index_fill_(self.dimidx_phase, torch.tensor(indices, dtype=torch.long), 1.0)
        return mask


class KtGaussianMaskGenerator(MaskGenerator):
    mu: torch.Tensor
    sigma: torch.Tensor
    def __init__(self, mu: float = 0.0, sigma: float = 0.28, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sigma", torch.as_tensor(sigma))

    def gau(self, x: float | torch.Tensor, mean: float | torch.Tensor, std: float | torch.Tensor):
        return (1 / (torch.sqrt(torch.as_tensor(2) * torch.pi) * std)) * torch.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def make_mask(self, size: Sequence[int], accel_factor: float, ncalib: int) -> torch.Tensor:
        batch_size, nshot = size[0], size[self.dimidx_phase]
        ncalib = ncalib + ((nshot % 2) != (ncalib % 2))
        nacq = nshot // accel_factor

        acs_start = math.floor(nshot / 2) + math.ceil(-ncalib / 2)
        acs_end = math.floor(nshot / 2) + math.ceil(ncalib / 2)

        acs_indices = list(range(acs_start, acs_end))

        # Create a tensor for the indices
        mask = torch.zeros(size, dtype=torch.float32, device=self.mu.device)
        mask.index_fill_(self.dimidx_phase, torch.tensor(acs_indices, dtype=torch.long, device=self.mu.device), 1.0)

        # Precompute the Gaussian PDF for all indices
        xs = torch.arange(nshot, dtype=torch.int32, device=self.mu.device)
        pdf = self.gau((xs / nshot - 0.5), self.mu, self.sigma)
        pdf[acs_start:acs_end] = 0  # Exclude ACS region from Gaussian sampling

        # ktdup
        recorded_indices = torch.zeros(pdf.shape, dtype=torch.bool, device=pdf.device)
        for b in range(batch_size):
            # Sample Gaussian indices
            gau_indices = torch.multinomial(pdf, num_samples=nacq, replacement=False, generator=self.rng)
            already_sampled = recorded_indices[gau_indices]
            for iretry in range(nacq):
                dup_indices = gau_indices[already_sampled]
                available_indices = (~recorded_indices).nonzero(as_tuple=False).squeeze(-1)
                dup_indices_expanded = dup_indices.unsqueeze(1)
                distances = torch.abs(available_indices - dup_indices_expanded)
                nearest_indices = distances.argmin(dim=1)
                nearest_available = available_indices[nearest_indices]
                gau_indices[already_sampled] = nearest_available
                already_sampled = recorded_indices[gau_indices]
            recorded_indices[gau_indices] = True
            mask[b:b+1].index_fill_(self.dimidx_phase, gau_indices, 1.0)

        return mask

class KtRadialMaskGenerator(MaskGenerator):
    def __init__(self, rotate_angle: float = 137.5, max_frames = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotate_angle = rotate_angle
        self.max_frames = max_frames

    def acc_factor_to_nlines(self, accel_factor: float) -> int:
        return math.floor(1/(accel_factor * 0.6) * 180) * 2

    def draw_radial(self, tensor: torch.Tensor, angles_target: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """
        Batch version of draw_radial. `angles_target` is now [batch, n_angles].
        
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch, height, width].
            angles_target (torch.Tensor): Angles tensor of shape [batch, n_angles].
            inplace (bool): Whether to modify the input tensor in-place.

        Returns:
            torch.Tensor: Modified tensor after drawing radial lines.
        """
        if not inplace:
            tensor = tensor.clone()

        batch_size, height, width = tensor.size()
        cx, cy = height / 2, width / 2
        r = max(height, width) / 2

        # Create meshgrid for coordinates
        x = torch.arange(height, device=tensor.device)
        y = torch.arange(width, device=tensor.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        dx = X - cx
        dy = Y - cy
        distance = torch.sqrt(dx**2 + dy**2)
        distance_weights = torch.exp((distance - 5) * (4 / r))
        distance_weights = (distance_weights - distance_weights.min()).clamp(0, 1)

        # Compute corner areas of the grid
        corner_bias_min, corner_bias_max = 0.45, 0.5
        corner_bias = distance_weights * (corner_bias_max - corner_bias_min) + corner_bias_min
        corners = torch.empty(X.size() + (4, 2), device=X.device)  # [height, width, 4, 2]
        corners[..., 0, 0], corners[..., 0, 1] = X - corner_bias, Y - corner_bias  # top_left
        corners[..., 1, 0], corners[..., 1, 1] = X - corner_bias, Y + corner_bias  # top_right
        corners[..., 2, 0], corners[..., 2, 1] = X + corner_bias, Y - corner_bias  # bottom_left
        corners[..., 3, 0], corners[..., 3, 1] = X + corner_bias, Y + corner_bias  # bottom_right

        # Calculate radial lines
        dx_corner = corners[..., 0] - cx  # [height, width, 4]
        dy_corner = corners[..., 1] - cy  # [height, width, 4]
        corner_angles = torch.atan2(dy_corner, dx_corner)  # [height, width, 4]

        min_angle = torch.min(corner_angles, dim=-1).values  # [height, width]
        max_angle = torch.max(corner_angles, dim=-1).values  # [height, width]

        # Normalize angles to [0, 2*pi]
        angles_target = (angles_target + 2 * torch.pi) % ( 2 * torch.pi)  # [batch, n_angles]

        # Expand to wrap bound
        bound = (max_angle - min_angle) > (2 * torch.pi * (350 / 360))
        min_angle[bound], max_angle[bound] = max_angle[bound] - (torch.pi * 2), min_angle[bound]
        angles_target = torch.cat([
            angles_target,
            angles_target + torch.pi * 2,
            angles_target - torch.pi * 2
        ], dim=1)

        # Expand min_angle and max_angle for batch processing
        min_angle = min_angle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height, width]
        max_angle = max_angle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height, width]

        # Check if angles_target are within the radial bounds
        contains = (angles_target[:, :, None, None] >= min_angle[:, None, :, :]) & \
                (angles_target[:, :, None, None] <= max_angle[:, None, :, :])  # [batch, n_angles, height, width]

        # Combine across all angles in the batch
        contains = contains.any(dim=1)  # [batch, height, width]

        # Crop corner
        circle = distance < r  # [height, width]
        circle = circle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height, width]
        tensor[(contains > 0) & circle] = 1

        return tensor

    def make_mask(self, size: Sequence[int], accel_factor: float, ncalib: int) -> torch.Tensor:
        batch_size, nx, ny = size

        ncalib_x = ncalib + ((nx % 2) != (ncalib % 2))
        ncalib_y = ncalib + ((ny % 2) != (ncalib % 2))

        acs_x_start = math.floor(nx / 2) + math.ceil(-ncalib_x / 2)
        acs_x_end = math.floor(nx / 2) + math.ceil(ncalib_x / 2)

        acs_y_start = math.floor(ny / 2) + math.ceil(-ncalib_y / 2)
        acs_y_end = math.floor(ny / 2) + math.ceil(ncalib_y / 2)

        mask = torch.zeros(size, dtype=torch.float32, device = self.accel_factors.device)
        mask[:, acs_x_start:acs_x_end, acs_y_start:acs_y_end] = 1.0

        # Draw Lines
        nlines = self.acc_factor_to_nlines(accel_factor)
        angles = torch.linspace(0, 2 * torch.pi, nlines + 1, device=mask.device)[:-1]
        
        # Rotate angles by the golden angle
        iframes = (torch.arange(batch_size, device=mask.device) + torch.randint(0, self.max_frames, (1,), device=mask.device, generator=self.rng))
        angles = (angles + iframes[:, None] * self.rotate_angle * (torch.pi / 180)) % (2 * torch.pi)
        mask = self.draw_radial(mask, angles, inplace=True)
        return mask