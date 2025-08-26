import torch
from torch import nn
import math
import tinycudann as tcnn
from typing import Sequence

class TinyInr(nn.Module):

    def __init__(self, in_channels = 2, out_channels = 3,nlayers_mlp = 8,  is_continue=False, image_size: Sequence[int] = [256,256], dynamic_range: int = 256, pos_range: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_continue:
            self.pe_config = {
                "otype": "Frequency",
                "n_frequencies": math.ceil(math.log2(max(image_size) / (pos_range * 2) + 1)) # Nquist sampling theorem
            }
        else:
            self.pe_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": math.ceil(math.log2(math.prod(image_size) * math.log2(dynamic_range))),
                "base_resolution": 16,
                "per_level_scale": 2.0,
                "interpolation": "Linear"
            }

        self.encoding = tcnn.Encoding(
            n_input_dims=in_channels,
            encoding_config=self.pe_config)
        
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=out_channels,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": nlayers_mlp
            }
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = self.encoding(coords)
        x = self.network(coords) # pts channels
        return x