import torch
from torch import nn, SymInt
from torchvision.transforms import GaussianBlur
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Literal, Callable, Sequence, Union
from torchvision.models import vgg16
from .conv import AverageConvnd, GaussianConvnd

class SSIMLoss(nn.Module):
    r"""
    Creates a criterion that measures the SSIM (structural similarity index measure) between
    each element in the input and target.

    Supports 1d, 2d, 3d input.

    Attributes:
        kernel_size: The size of the sliding window. Must be an int, or a shape with 1, 2 or 3 dimensions.
        *,
        kernel_type: Type of kernel ("avg" or "gauss") or a Custom callable object.
        reduction: Reduction method ("mean", "sum", or "none").
        data_range: Dynamic range of input tensors.
        k1,k2: Stabilization constants for SSIM calculation.
    """

    conv: Callable[..., torch.Tensor]
    data_range: torch.Tensor

    def __init__(
            self,
            kernel_size: int | Sequence[Union[int, SymInt]],
            *,
            kernel: Literal["avg", "gauss"] | Callable[[torch.Tensor], torch.Tensor] = "avg",
            reduction: Literal["mean", "sum", "none"] = "mean",
            data_range: float = 1.0,
            k1: float = 0.01, k2: float = 0.03
        ):
        super().__init__()
        self.kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
        if self.kernel_size[0] <= 5:
            self.sigma = 0.7
        elif self.kernel_size[0] == 7:
            self.sigma = 1.0
        else:
            self.sigma = 1.5
        self.kernel = kernel
        self.reduction = reduction
        self.register_buffer("data_range", torch.as_tensor(data_range))
        self.k1, self.k2 = k1, k2

        # unbiased estimate
        npts = torch.prod(torch.as_tensor(self.kernel_size)).item()
        self.cov_norm = npts / (npts - 1)

        ndim = len(self.kernel_size)
        if self.kernel == "avg":
            self.conv = AverageConvnd(self.kernel_size)
        elif self.kernel == "gauss":
            if ndim == 2:
                self.conv = GaussianBlur(self.kernel_size, sigma = self.sigma)
            else:
                self.conv = GaussianConvnd(self.kernel_size, sigma = self.sigma)
        elif callable(self.kernel):
            self.conv = self.kernel
        else:
            raise ValueError("`kernel` only supports 'avg', 'gauss' or Callable object.")

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            batch_data_range: torch.Tensor = None,
        ):
        if batch_data_range is None:
            data_range = self.data_range
        else:
            data_range = batch_data_range

        if data_range.ndim > 0 and input.ndim >= 3:
            if input.size(0) != data_range.size(0):
                raise ValueError(f"`input` and `batch_data_range` must have the same batchsize. Got shape `input` {input.shape}, `data_range` {data_range.shape}")
            data_range = data_range.view(input.size(0),*([1] * (input.ndim - 1)))

        device = input.device
        run_device = self.data_range.device
        input = input.to(run_device)
        target = target.to(run_device)
        data_range = data_range.to(run_device)

        if input.ndim < 4:
            input = input.view(
                *(1,) * (4 - input.ndim), *input.shape,
            )
        if target.ndim < 4:
            target = target.view(
                *(1,) * (4 - target.ndim), *target.shape,
            )
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = self.conv(input)
        uy = self.conv(target)
        uxx = self.conv(input**2)
        uyy = self.conv(target**2)
        uxy = self.conv(input * target)

        vx = self.cov_norm * (uxx - ux**2)
        vy = self.cov_norm * (uyy - uy**2)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2
        
        ssim_map = (A1 * A2) / (B1 * B2)
        
        if self.reduction == "mean":
            score =  1 - ssim_map.mean()
        elif self.reduction == "sum":
            score =  ssim_map.size(0) - ssim_map.mean(tuple(range(1, ssim_map.ndim))).sum()
        else:
            score =  1 - ssim_map
        
        return score.to(device)


class VGGLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu3_3', 'relu5_3']):
        super().__init__()
        self.layers = layers
        self.layers_mapping = {
            'relu1_1': 'features.1', 'relu1_2': 'features.3', 
            'relu2_1': 'features.6', 'relu2_2': 'features.8',
            'relu3_1': 'features.11', 'relu3_2': 'features.13', 'relu3_3': 'features.15',
            'relu4_1': 'features.18', 'relu4_2': 'features.20', 'relu4_3': 'features.22',
            'relu5_1': 'features.25', 'relu5_2': 'features.27', 'relu5_3': 'features.29',
        }

        if not all(layer in self.layers_mapping for layer in layers):
            raise ValueError(f"Invalid layers specified. Available layers: {list(self.layers_mapping.keys())}")

        vgg = vgg16(pretrained=True).eval().requires_grad_(False)

        self.feat_extractor = create_feature_extractor(
            vgg,
            return_nodes={
                self.layers_mapping[key]: key for key in layers
            }
        )
        
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(f"Input and target must have 4 dims. Got {pred.ndim} and {target.ndim}.")
        
        if pred.size(-3) != target.size(-3):
            raise ValueError(f"Input and target must have the same batch size. Got {pred.size(-3)} and {target.size(-3)}.")
        
        target_mean = target.mean()
        target_std = target.std()

        pred = (pred - target_mean) / target_std
        target = (target - target_mean) / target_std

        if target.size(-3) == 1:
            target = target.expand(-1, 3, -1, -1)
            pred = pred.expand(-1, 3, -1, -1)
        elif target.size(-3) != 3:
            raise ValueError(
                f"Expected the target to have 1 or 3 channels (size[-3]), but got {target.size(-3)}. "
                "Ensure the target tensor has a channel dimension of size 1 or 3."
            )

        features_pred = self.feat_extractor(pred)
        features_target = self.feat_extractor(target)

        loss = torch.stack(
            [self.criterion(features_pred[layer], features_target[layer]) for layer in self.layers],
            dim=0
        ).mean()

        return loss
        