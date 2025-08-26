import torch
import math
from typing import List

class SinusoidalPositionEncoding1d(torch.nn.Module):
    """Sinusoidal Position Encoding for 1-dimensions"""
    def __init__(self, n_feat, pos_scale = 1.0):
        super().__init__()
        self.n_feat = n_feat
        self.pos_scale = pos_scale
        self.register_buffer('_div_term', torch.exp(torch.arange(0, n_feat, 2).float() * -(math.log(10000.0) / n_feat)))

    def forward(self, input = None, position = None) -> torch.Tensor:
        """tensor:[batch_size, n_token, n_feat]  
        position:[batch_size, n_token, 1]  
        """
        assert input is not None or position is not None, "Either tensor or position must be provided"

        if position is None:
            position = torch.arange(input.size(1), device=self._div_term.device).view(1,input.size(1),1).float()
        position *= self.pos_scale

        assert len(position.size()) == 3, "position must be [batch_size, n_token, 1]"

        pe = torch.zeros(position.size(0), position.size(1), self.n_feat, device=self._div_term.device)
        pe[:, :, 0::2] = torch.sin(position * self._div_term)
        pe[:, :, 1::2] = torch.cos(position * self._div_term)

        if input is None:
            return pe        
        return input + pe
    

class SinusoidalPositionEncodingmd(torch.nn.Module):
    """Multi-dimensional Sinusoidal Position Encoding"""
    def __init__(self, n_feat, pos_scale: float | List[float] = 1.0, n_dim=2):
        super(SinusoidalPositionEncodingmd, self).__init__()
        self.n_dim = n_dim
        self.n_feat = n_feat
        self.register_buffer('pos_scale', torch.tensor(pos_scale if isinstance(pos_scale, list) else [pos_scale] * n_dim))

        assert n_feat % n_dim == 0, "n_feat must be divisible by n_dim"
        self.register_buffer('_div_term', torch.exp(torch.arange(0, self.n_feat // self.n_dim, 2).float() * -(math.log(10000.0) / self.n_feat // self.n_dim)))

    
    def forward(self, input, position = None):
        """tensor:[batch_size, n_token, n_feat]  
        position:[batch_size, n_token, n_dim]   
        """
        assert input is not None or position is not None, "Either tensor or position must be provided"

        if position is None:
            position = torch.cat([torch.arange(input.size(1), device=self._div_term.device).view(1,input.size(1),1).float() for _ in range(self.n_dim)], dim=-1)
        position *= self.pos_scale

        assert len(position.size()) == 3, "position must be [batch_size, n_token, n_dim]"

        pe = torch.zeros(position.size(0), position.size(1), self.n_feat, device=self._div_term.device)
        for i in range(self.n_dim): # [aaabbbccc]
            pe[:, :, i * self.n_feat // self.n_dim:(i+1) * self.n_feat // self.n_dim:2] = torch.sin(position[:, :, i:i+1] * self._div_term)
            pe[:, :, i * self.n_feat // self.n_dim + 1:(i+1) * self.n_feat // self.n_dim:2] = torch.cos(position[:, :, i:i+1] * self._div_term)

        if input is None:
            return pe
        return input + pe
    

class LearnablePositionEncoding(torch.nn.Module):
    """Learnable Position Encoding"""
    def __init__(self, n_feat, max_n_token=512):
        super(LearnablePositionEncoding, self).__init__()
        self.n_feat = n_feat
        self.max_n_token = max_n_token
        self._pe = torch.nn.Parameter(torch.randn(max_n_token, n_feat))

    def forward(self, input) -> torch.Tensor:
        """
        tensor:[batch_size, n_token, n_feat]  
        """
        return input + self._pe[:input.size(1), :]