import torch
from torch import nn
from typing import List, Tuple
from utils.naneu import fft
from utils.naneu.helpers.rearrange import TorchModuleForwardHook # Don't touch this import
from utils.naneu.helpers.context import register_extra_output, register_extra_metric
from utils.algos.acs import find_max_square
from utils.complex import interpolate
from .modules.format import Format4Unet2d
from .modules.complex import ComplexGaussianBlur
from data.transforms.crop import center_crop, make_center_mask
from einops import rearrange
from copy import deepcopy
from warnings import warn


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
    def __init__(self, model: nn.Module, cropsize_max = 128, cropsize_min = 48, ncalib_mincheck = 8,crop: bool = True):
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
        csm, _, wordfreq = self.model(masked_image)
        csm = self.norm.pad_adjoint(csm)

        wordfreq = rearrange(wordfreq, "(b c) words -> b c words", b = masked_kspace.size(0), c = masked_kspace.size(-3)).mean(dim = 1)

        if self.is_crop:
            csm = interpolate(csm.view(-1, *csm.shape[-3:]), size=masked_kspace.shape[-2:], mode='bilinear', align_corners=False).view(*csm.shape[:-3], -1, *masked_kspace.shape[-2:])
            csm = self.to_out(csm)
            
        csm = self.norm.norm_adjoint(csm)
        csm = csm / ((csm.abs()**2).sum(dim=-3, keepdim=True).sqrt() + 1e-13)  # Normalize
        return csm, wordfreq # (b, t, s, c, h, w), [b c]

class SenseBlock(nn.Module):
    """
    Torch Implementation of Forward and Backward Processes of Sense Operator in MRI Compressed Sensing Reconstruction.  
    Modified from PromptMR[1].

    Args:
        model: The model to generate the sense operator.
    
    Returns:
        The predicted image, latent vector, and history features.

    References:  
    [1] https://github.com/hellopipu/PromptMR/blob/main/models/promptmr.py
    """
    def __init__(
            self,
            model: nn.Module,
            use_noise: bool = False
            ):
        super().__init__()
        self.model = model.view_as_real(for_input = [0], for_output = [0]).rearrange("b ref adj c h w two-> b ref (adj c two) h w", for_input = [0], for_output = [0])
        self.use_noise = use_noise

        self.norm: Format4Unet2d = Format4Unet2d(ndownsample=self.model.depth, is_resize=False)
        self.dc_weight = nn.Parameter(torch.tensor(1.0, dtype = torch.float32))  # DC weight for the model

        if self.use_noise:
            # self.noise_filter = ComplexGaussianBlur(7, 0.5, is_magnitude=True, is_phase=True)
            pass

    def sense_expand(self, img: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        return fft.itok(img * csm)

    def sense_reduce(self, kspace: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        return (fft.ktoi(kspace) * csm.conj()).sum(dim=-3, keepdim=True)

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        csm: torch.Tensor,
        history_feat: Tuple[torch.Tensor, ...] | None = None,
    ):
        """
        complex
        b ref adj c h w
        """
        ffx = self.sense_reduce(self.sense_expand(current_img, csm) * mask, csm)
        # buffer: A^H(A(x)), s_i, x0
        # Note: `ffx - img_zf` greatly enhances noise and is therefore discarded
        current_img_with_buffer = [current_img, ffx, latent, img_zf]
        raw_nchannels = [x.size(-3) for x in current_img_with_buffer]
        current_img_with_buffer = torch.cat(current_img_with_buffer, dim=-3)

        if self.use_noise:
            noise = ffx - img_zf
            # noise = self.noise_filter(noise)

        # Normalize
        current_img_with_buffer = self.norm(current_img_with_buffer)
        if self.use_noise:
            noise = self.norm(noise, is_fit = False)
            raw_nchannels = raw_nchannels + [noise.size(-3)]
            current_img_with_buffer = torch.cat([current_img_with_buffer, noise], dim=-3)

        # Model forward pass
        model_term_with_buffer, feat_cached, wordfreq = self.model(current_img_with_buffer, history_feat)

        # Restore from normalization
        model_term_with_buffer = self.norm.adjoint(model_term_with_buffer)

        # Split
        model_term_with_buffer = torch.split(model_term_with_buffer, raw_nchannels, dim=-3)
        model_term, latent = model_term_with_buffer[0], model_term_with_buffer[2]

        # DC
        dc_weight = self.dc_weight
        current_img = current_img - (ffx - img_zf) * dc_weight - model_term

        with torch.no_grad():
            register_extra_metric(self, f"dc_weight_max", dc_weight.detach(), op ="max")
            register_extra_metric(self, f"dc_weight_min", dc_weight.detach(), op ="min")

        return current_img, latent, feat_cached, wordfreq

    def get_model(self) -> torch.nn.Module:
        return self.model


class Moero(nn.Module):
    """
    Modular Cascaded Reconstruction Network

    Args:
        csm_model: The model to generate coil sensitivity maps.
        cascades: A list of models for each cascade in the network.

    Returns:
        The predicted image, zero-filled image, and coil sensitivity maps.

    """
    def __init__(
            self,
            csm_model: nn.Module,
            cascades: List[nn.Module],
            branchnav: nn.Module,
            nbranch: int = 2,
            csmblock_kwargs: dict = {},
            senseblock_kwargs: dict = {}
    ):
        super().__init__()
        self.csm_model = CsmBlock(csm_model, **csmblock_kwargs)

        self.branchgrids = nn.ModuleList([
            nn.ModuleList([
                SenseBlock(deepcopy(cascade), **senseblock_kwargs) for _ in range(nbranch)
            ])
            for cascade in cascades
        ]) # (n_cascades, n_branch, SenseBlock)

        self.sync_vq_module_share_state()

        self.branchnavs = nn.ModuleList([
            deepcopy(branchnav) for _ in range(len(cascades))
        ]) # (n_cascades, BranchNav)

        for i in range(len(self.branchnavs)):
            self.branchnavs[i].idx=i
    
    def sync_vq_module_share_state(self):
        """
        Share the VQ module across all cascades.
        """
        results = []
        for i, cascade_group in enumerate(self.branchgrids):
            vq_module = cascade_group[0].get_model().get_vq_module()
            for j, unit in enumerate(cascade_group[1:]):
                if unit.get_model().get_vq_module() is not vq_module:
                    unit.get_model().set_vq_module(vq_module)
                    results.append((i, j+1))
        if results:
            print(f"Perform object sharing in BranchGrids: {results}")


    def load_state_dict(self, state_dict, strict = True, assign = False):
        result = super().load_state_dict(state_dict, strict, assign)
        self.sync_vq_module_share_state()  # Ensure VQ module is shared after loading state dict
        return result

    def sens_reduce(self, kspace: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        return (fft.ktoi(kspace) * csm.conj()).sum(dim=-3, keepdim=True)

    def rss(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (b, h, w) complex tensor
        """
        img = (img.abs() ** 2).sum(dim=-3, keepdim=True).sqrt()  # (b, 1, h, w)
        return img

    def phase_preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the phase of the tensor by applying a Gaussian blur.
        tensor: (b, ref, adj, c, h, w) complex tensor
        """
        tensor = fft.ktoi(tensor)
        tensor = self.phasefilter(tensor)
        tensor = fft.itok(tensor)
        return tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Args:
            masked_kspace(complex): (b, ref, adj, c, h, w) complex input k-space data
            mask(float): (b, ref, adj, 1, h, w) or (b, 1, 1, 1, h, w) mask
            num_low_frequencies(int): (b) number of low frequencies
        '''
        if masked_kspace.ndim != 6:
            raise ValueError(f"Input masked_kspace must be 6D tensor (b, ref, adj, c, h, w). But got {masked_kspace.ndim}D tensor.")
        if mask.ndim != 6:
            raise ValueError(f"Input mask must be 6D tensor (b, ref, adj, c, h, w). But got {mask.ndim}D tensor.")
        if mask.shape[-2:] != masked_kspace.shape[-2:]:
            raise ValueError(f"Mask shape {mask.shape[-2:]} does not match masked_kspace shape {masked_kspace.shape[-2:]}.")
        if not mask.dtype.is_floating_point:
            raise ValueError(f"Unexpected dtype for mask: {mask.dtype}. Expected a floating-point dtype.")
        if masked_kspace.size(1) % 2 != 1 or masked_kspace.size(2) % 2 != 1:
            raise ValueError(f"Input masked_kspace must have odd number of frames and slices. But got {masked_kspace.size(1)} frames and {masked_kspace.size(2)} slices.")

        # TODO: Because most of the data is dirty and noisy, a phase unwrapping module is needed.

        # register_extra_output(self, "masked_kspace", masked_kspace) # Debug print

        csm, wordfreq = self.csm_model(masked_kspace, mask)

        if csm.isnan().any():
            if self.training:
                raise ValueError("Coil sensitivity maps contains NaN values.")
            else:
                warn(f"Coil sensitivity maps contains NaN values. This may cause issues in inference. Consider checking the input data or the model parameters.")
                csm = torch.nan_to_num(csm, nan=0.0, posinf=0, neginf=0)

        img_zf = self.sens_reduce(masked_kspace, csm)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        feat_history = [[] for _ in range(3)]

        for cascade_idx, cascade_group in enumerate(self.branchgrids):
            try:
                # Branch Model Learning
                with torch.no_grad():
                    route: torch.Tensor = self.branchnavs[cascade_idx](wordfreq)
                    if len(route) != len(cascade_group):
                        raise ValueError(f"Route length {len(route)} does not match the number of cascades branchs {len(cascade_group)}.")
                    
                    indices_available: List[int] = route.nonzero().flatten().tolist()
                    indices_not_available: List[int] = [i for i in range(len(route)) if i not in indices_available]

                # Sense forward
                img_pred_branches: List[torch.Tensor] = []
                latent_branches: List[torch.Tensor] = []
                feat_cached_branches: List[List[torch.Tensor]] = []
                wordfreq_branches: List[torch.Tensor] = []

                for unit in [cascade_group[i] for i in indices_available]:
                    img_pred_unit, latent_unit, feat_cached_unit, word_freq_unit = unit(img_pred, img_zf, latent, mask, csm, feat_history)

                    for l, v in zip([img_pred_branches, latent_branches, feat_cached_branches, wordfreq_branches], [img_pred_unit, latent_unit, feat_cached_unit, word_freq_unit]):
                        l.append(v)

                if len(img_pred_branches) == 0:
                    raise ValueError(f"All cascades in cascade group {cascade_idx} are not available. Please check the route {route}.")
                
                # Merge Branch Results
                img_pred: torch.Tensor = torch.stack(img_pred_branches, dim=0).mean(0)
                latent: torch.Tensor = torch.stack(latent_branches, dim=0).mean(0)
                feat_cached: List[torch.Tensor] = [
                    torch.stack([
                        feat_cached_branches[branch_idx][feat_idx]
                        for branch_idx in range(len(feat_cached_branches))
                        ], dim=0).mean(0)
                        for feat_idx in range(len(feat_cached_branches[0]))
                        ]
                wordfreq: torch.Tensor = torch.stack(wordfreq_branches, dim=0).mean(0)

                # Clean cache
                img_pred_branches = None
                latent_branches = None
                feat_cached_branches = None
                wordfreq_branches = None

            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(
                    f"Out of memory in cascade {cascade_idx}. Consider reducing cascade size."
                ) from e
            
            for ilevel, level_history in enumerate(feat_history):
                level_history.append(feat_cached[ilevel])

        # Get reduced central slice as final output
        img_pred = img_pred[:, img_pred.size(1) // 2, img_pred.size(2) // 2, ...]
        csm = csm[:, csm.size(1) // 2, csm.size(2) // 2, ...]
        masked_kspace = masked_kspace[:, masked_kspace.size(1) // 2, masked_kspace.size(2) // 2, ...]
        mask = mask[:, mask.size(1) // 2, mask.size(2) // 2, ...]

        img_pred = self.rss(img_pred * csm)  # (b, 1, h, w)

        if not self.training:
            img_zf = fft.ktoi(masked_kspace)  # (b, c, h, w)
            img_zf = self.rss(img_zf)  # (b, 1, h, w)
            return {
                'img_pred': img_pred.abs(), # b 1 h w
                'img_zf': img_zf.abs(), # b 1 h w
                'csm': csm.abs() # b c h w
            }
        else:
            return {
                'img_pred': img_pred.abs(), # b 1 h w
                'csm': csm.abs() # b c h w
            }