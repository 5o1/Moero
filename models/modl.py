import torch
import torch.nn as nn
from utils.naneu import fft
import torch
from torch import nn
from typing import List, Tuple
from utils.naneu import fft
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from utils.naneu.helpers.context import register_extra_output, register_extra_metric
from utils.algos.acs import find_max_square
from utils.complex import interpolate
from .modules.format import Format4Unet2d
from data.transforms.crop import center_crop, make_center_mask


def fft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def ifft2(data):
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def A(data, csm, mask):
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    return data


def At(data, csm, mask):
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data


def AtA(data, csm, mask):
    data = data[:, None, ...] * csm
    data = fft2(data)
    data = data * mask[:, None, ...]
    data = ifft2(data)
    data = torch.sum(data * torch.conj(csm), dim=1)
    return data

class Dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dw, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        return x + self.layers(x)


class ConjugatedGrad(nn.Module):
    def __init__(self):
        super(ConjugatedGrad, self).__init__()

    def forward(self, rhs, csm, mask, lam):
        rhs = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())
        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
        num_iter, epsilon = 10, 1e-10
        for i in range(num_iter):
            Ap = AtA(p, csm, mask) + lam * p
            alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=(-2, -1)).real
            x = x + alpha[:, None, None] * p
            r = r - alpha[:, None, None] * Ap
            rTrNew = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real
            if rTrNew.max() < epsilon:
                break
            beta = rTrNew / rTr
            rTr = rTrNew
            p = r + beta[:, None, None] * p
        return x


class CsmBlock(nn.Module):
    """
    Search the ACS regions and then generate coil sensitivity maps based on ACS signals.

    Args:
        model: The model to generate coil sensitivity maps.
        cropsize_max: The maximum size of the ACS region to search for.
        cropsize_min: The minimum size of the ACS region to search for.
        ncalib_mincheck: The minimum number of calibration points to check for ACS region.
        It will raise an error if the ACS region is smaller than this value.

    Returns:
        The coil sensitivity maps.
    
    Note:
        The input masked_kspace should be a complex tensor of shape (b, ref, adj, coils, h, w).
        The mask should be a float tensor of shape (b, ref, adj, 1, h, w).
    """
    def __init__(self, model: nn.Module, cropsize_max = 128, cropsize_min = 48, ncalib_mincheck = 8, crop: bool = True):
        super().__init__()
        self.cropsize_max = cropsize_max
        self.cropsize_min = cropsize_min
        self.ncalib_mincheck = ncalib_mincheck
        self.is_crop = crop

        self.model = model.view_as_real(for_input = [0], for_output = [0]).rearrange("b ref adj coil h w two-> (b coil) ref (adj two) h w", for_input = [0], for_output = [0])
        self.norm:Format4Unet2d = Format4Unet2d(ndownsample=self.model.depth, is_resize=False)

        if crop:
            self.to_out:nn.Conv2d = nn.Conv2d(2, 2, kernel_size=7, padding="same").view_as_real().rearrange("b ref adj coil h w two-> (b ref adj coil) two h w")
    
    def acs_crop(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dimlength_min = min(mask.size(-1), mask.size(-2))

            if dimlength_min < self.cropsize_min:
                raise ValueError(f"Minimum ACS size {dimlength_min} is smaller than cropsize_min {self.cropsize_min}.")

            # Search for the largest square ACS region
            search_size = min(dimlength_min, self.cropsize_max)
            mask_subregion = center_crop(mask, (search_size, search_size))
            ncalib_x, ncalib_y = find_max_square(mask_subregion, threshold=1e-13)

            # Check if the ACS size is too small, which may be an error
            if ncalib_x < self.ncalib_mincheck or ncalib_y < self.ncalib_mincheck:
                raise ValueError(f"ACS size {(ncalib_x, ncalib_y)} is too small. Minimum ACS size is {self.ncalib_mincheck}.")
            
        if self.is_crop:
            # Clamp to the range [cropsize_min, cropsize_max]
            cropsize_x = max(self.cropsize_min, min(ncalib_x, self.cropsize_max)) 
            cropsize_y = max(self.cropsize_min, min(ncalib_y, self.cropsize_max))

            # Crop the masked k-space and mask to the cropsize
            masked_kspace = center_crop(masked_kspace, (cropsize_x, cropsize_y))
            mask = center_crop(mask, (cropsize_x, cropsize_y))

        with torch.no_grad():
            # Apply the ACS mask to the cropped k-space
            acs_mask = make_center_mask(masked_kspace, (ncalib_x, ncalib_y))

        masked_kspace = masked_kspace * acs_mask
        mask = mask * acs_mask
        return masked_kspace, mask


    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        masked_kspace: (b, ref, adj, c, h, w)
        mask: (b, ref, adj, 1, h, w)
        """
        masked_kspace_cropped, mask_cropped = self.acs_crop(masked_kspace, mask) # Crop to use only low frequencies
        masked_image = fft.ktoi(masked_kspace_cropped)

        masked_image = self.norm(masked_image)
        csm = self.model(masked_image)
        csm = csm[0] if isinstance(csm, tuple) else csm
        csm = self.norm.pad_adjoint(csm)
        if self.is_crop:
            csm = interpolate(csm.view(-1, *csm.shape[-3:]), size=masked_kspace.shape[-2:], mode='bilinear', align_corners=False).view(*csm.shape[:-3], -1, *masked_kspace.shape[-2:])
            csm = self.to_out(csm)
            
        csm = self.norm.norm_adjoint(csm)

        csm = csm / ((csm.abs()**2).sum(dim=-3, keepdim=True).sqrt()+1e-13) # Normalize
        return csm # (b, ref, adj, c, h, w) complex tensor


class MoDL(nn.Module):
    def __init__(
            self, 
            csm_model: nn.Module,
            in_channels, 
            out_channels, 
            num_layers, 
            csmblock_kwargs: dict = {},
            ):
        super(MoDL, self).__init__()
        self.csm_model = CsmBlock(csm_model, **csmblock_kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layers = Dw(self.in_channels, self.out_channels)
        self.lam = nn.Parameter(torch.FloatTensor([0.05]), requires_grad=True)
        self.CG = ConjugatedGrad()

        self.norm:Format4Unet2d = Format4Unet2d(ndownsample=0, is_resize=False, is_pad=False)

    def sens_reduce(self, kspace: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        return (fft.ktoi(kspace) * csm.conj()).sum(dim=-3, keepdim=True)

    def rss(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (b, h, w) complex tensor
        """
        img = (img.abs() ** 2).sum(dim=-3, keepdim=True).sqrt()  # (b, 1, h, w)
        return img

    def forward(self, masked_kspace, mask):
        csm = self.csm_model(masked_kspace, mask)
        if csm.isnan().any():
            raise ValueError("Coil sensitivity maps contains NaN values.")
        img_zf = self.sens_reduce(masked_kspace, csm)

        mask = mask.squeeze(1).squeeze(1)
        img_zf = img_zf.squeeze(1).squeeze(1)
        csm = csm.squeeze(1).squeeze(1)

        img_zf = self.norm(img_zf)
        x = img_zf.clone()

        x = x.squeeze(1)
        mask = mask.squeeze(1)
        img_zf = img_zf.squeeze(1)


        x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        img_zf = torch.view_as_real(img_zf).permute(0, 3, 1, 2).contiguous()

        for _ in range(self.num_layers):
            x = self.layers(x)
            x = img_zf + self.lam * x
            x = self.CG(x, csm, mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
        x_final = x
        x_final = torch.view_as_complex(x_final.permute(0, 2, 3, 1).contiguous())
        x_final = x_final[:, None, ...]  # b 1 h w
        img_zf = torch.view_as_complex(img_zf.permute(0, 2, 3, 1).contiguous())
        img_zf = img_zf[:, None, ...]  # b 1 h w


        x_final = self.norm.adjoint(x_final)
        img_zf = self.norm.adjoint(img_zf)


        return {
                'img_pred': x_final.abs(), # b 1 h w
                'img_zf': img_zf.abs(),   # b 1 h w
                'csm': csm.abs() # b c h w
            }