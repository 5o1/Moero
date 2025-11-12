import torch
from typing import List, Sequence
from data.cmrsample import CmrSample
from einops import rearrange
from data.transforms.maskgenerator import UniformMaskGenerator, KtGaussianMaskGenerator, KtRadialMaskGenerator


class MixedRandomMaskGenerator(torch.nn.Module):
    def __init__(
            self,
            acc_factors: List[int] = [2, 4, 8, 12, 16, 20, 24],
            n_calibs: List[int] = [16, 20],
            mask_weights: dict = {
                "Uniform": 1,
                "ktGaussian": 1,
                "ktRadial": 1,
            }
    ):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(torch.initial_seed())
        self.maskgen_pool = [
            UniformMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
            KtGaussianMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
            KtRadialMaskGenerator(accel_factors=acc_factors, ncalibs = n_calibs, rng = self.rng),
        ]
        self.masktype_pool = [
            "Uniform",
            "ktGaussian",
            "ktRadial"
        ]

        self.maskgen_weights = torch.as_tensor([mask_weights[masktype] for masktype in self.masktype_pool], dtype=torch.float32)

    def set_seed(self, seed: int):
        self.rng.manual_seed(seed)

    def forward(self, size: Sequence[int]) -> torch.Tensor:
        idx = torch.multinomial(self.maskgen_weights, num_samples=1, generator = self.rng).item()
        maskgen = self.maskgen_pool[idx]
        masktype = self.masktype_pool[idx]

        mask, accel_factor, ncalib = maskgen(size)
        masktype = self.masktype_pool[idx] + str(accel_factor)
        return mask, masktype


class Cmr25TrainingTransform(torch.nn.Module):
    def __init__(
            self,
            acc_factors: List[int] = [2, 4, 8, 12, 16, 20, 24],
            n_calibs: List[int] = [16, 20],
            mask_weights: dict = {
                "Uniform": 1,
                "ktGaussian": 1,
                "ktRadial": 1,
            }
            ):
        super().__init__()
        self.maskgen = MixedRandomMaskGenerator(acc_factors, n_calibs, mask_weights)

    def set_seed(self, seed: int):
        self.maskgen.set_seed(seed)
    
    def forward(self, sample: CmrSample) -> CmrSample:
        size = (sample.masked_kspace.size(0), sample.masked_kspace.size(-2), sample.masked_kspace.size(-1))
        mask, masktype = self.maskgen(size) # (batch, phase, readout)

        mask = rearrange(mask, "t readout phase -> t 1 1 readout phase")  # Add channel dimension

        sample.masked_kspace = sample.masked_kspace * mask # Apply mask to k-space data
        sample.mask = mask
        sample.mask_type = masktype
        return sample
