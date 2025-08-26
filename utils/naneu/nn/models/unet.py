# import torch
# from torch import nn
# from typing import List, Optional

# from .modules.conv import *
# from .modules.prompt import *
    
# class Down(nn.Module):
#     def __init__(self, n_in_feat: int, n_out_feat: int, n_cab: int, kernel_size, reduction, bias, act, first_act=False):
#         super().__init__()
#         if first_act:
#             self.encoder = [CAB2d(n_in_feat, kernel_size, reduction,bias=bias, act=nn.PReLU())]
#             self.encoder = nn.Sequential(
#                     *(self.encoder+[CAB2d(n_in_feat, kernel_size, reduction, bias=bias, act=act) 
#                                     for _ in range(n_cab-1)]))
#         else:
#             self.encoder = nn.Sequential(
#                 *[CAB2d(n_in_feat, kernel_size, reduction, bias=bias, act=act) 
#                   for _ in range(n_cab)])
#         self.down = nn.Conv2d(n_in_feat, n_out_feat,kernel_size=3, stride=2, padding=1, bias=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         res = self.encoder(x)
#         x = self.down(res)
#         return x, res

# class Up(nn.Module):
#     def __init__(self, n_in_feat, n_out_feat, n_cab, kernel_size, reduction, bias, act):
#         super().__init__()
#         self.conv = CABChain(n_in_feat, n_in_feat, n_cab, kernel_size, reduction, bias, act)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.proj = conv(n_in_feat, n_out_feat, 1, bias=False)
#         self.ca = CAB2d(n_out_feat, kernel_size, reduction, bias=bias, act=act)

#     def forward(self, x, res):
#         x = self.conv(x)
#         x = self.up(x)
#         x = self.proj(x)
#         x += res
#         x = self.ca(x)
#         return x


# class PromptUp(nn.Module):
#     def __init__(self, n_in_feat, n_out_feat, n_prompt_feat, n_cab, kernel_size, reduction, bias, act):
#         super().__init__()
#         self.conv = CABChain(n_in_feat + n_prompt_feat, n_in_feat, n_cab, kernel_size, reduction, bias, act)
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.proj = conv(n_in_feat, n_out_feat, 1, bias=False)
#         self.ca = CAB2d(n_out_feat, kernel_size, reduction, bias=bias, act=act)

#     def forward(self, x, prompt, res):
#         x = torch.cat([x, prompt], dim=1)
#         x = self.conv(x)
#         x = self.up(x)
#         x = self.proj(x)
#         x += res
#         x = self.ca(x)
#         return x

# class PromptUnet(nn.Module):
#     def __init__(self,
#                 n_in_feat: int,
#                 n_out_feat: int,
#                 n_feat0: int, # Feature extraction at the beginning
#                 n_feat: List[int], # Along levels
#                 n_prompt_feat: List[int],
#                 n_prompt_token: List[int],
#                 prompt_size: List[int],
#                 n_enc_cab: List[int],
#                 n_dec_cab: List[int],
#                 n_skip_cab: List[int],
#                 n_bottleneck_cab: int,
#                 kernel_size=3,
#                 reduction=4,
#                 act=nn.PReLU(),
#                 bias=False,
#                 no_use_ca=False,
#                 learnable_prompt=False,
#                 adaptive_input=False,
#                 n_buffer=0,
#                 n_history=0,
#                  ):
#         super().__init__()
#         self.n_feat = n_feat
#         self.n_history = n_history
#         self.n_buffer = n_buffer if adaptive_input else 0

#         n_in_feat = n_in_feat * (1+self.n_buffer) if adaptive_input else n_in_feat
#         n_out_feat = n_out_feat * (1+self.n_buffer) if adaptive_input else n_in_feat

#         # Feature extraction
#         self.feat_extract = conv(n_in_feat, n_feat0, kernel_size, bias=bias)

#         # Encoder - 3 DownBlocks
#         self.enc_level1 = Down(n_feat0, n_feat[0], n_enc_cab[0], kernel_size, reduction, bias, act, first_act=True)
#         self.enc_level2 = Down(n_feat[0], n_feat[1], n_enc_cab[1], kernel_size, reduction, bias, act)
#         self.enc_level3 = Down(n_feat[1], n_feat[2], n_enc_cab[2], kernel_size, reduction, bias, act)

#         # Skip Connections - 3 SkipBlocks
#         self.skip_attn1 = CABChain(n_feat0, n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act)
#         self.skip_attn2 = CABChain(n_feat[0], n_feat[0], n_skip_cab[1], kernel_size, reduction, bias, act)
#         self.skip_attn3 = CABChain(n_feat[1], n_feat[1], n_skip_cab[2], kernel_size, reduction, bias, act)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(*[CAB2d(n_feat[2], kernel_size, reduction, bias, act, no_use_ca)
#                                           for _ in range(n_bottleneck_cab)])
#         # Decoder - 3 UpBlocks
#         self.momentum_level3 = MomentumConv2d(n_feat[2], kernel_size, reduction, bias, act, n_history)
#         self.prompt_level3 = EncodedPromptWithFallback2d(n_prompt_token[2], n_prompt_feat[2], prompt_size[2], n_feat[2], learnable_prompt)
#         self.dec_level3 = Up(n_feat[2], n_feat[1], n_prompt_feat[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history)

#         self.momentum_level2 = MomentumConv2d(n_feat[1], kernel_size, reduction, bias, act, n_history)
#         self.prompt_level2 = EncodedPromptWithFallback2d(n_prompt_token[1], n_prompt_feat[1],  prompt_size[1], n_feat[1], learnable_prompt)
#         self.dec_level2 = Up(n_feat[1], n_feat[0], n_prompt_feat[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history)

#         self.momentum_level1 = MomentumConv2d(n_feat[0], kernel_size, reduction, bias, act, n_history)
#         self.prompt_level1 = EncodedPromptWithFallback2d(n_prompt_token[0], n_prompt_feat[0],  prompt_size[0], n_feat[0], learnable_prompt)
#         self.dec_level1 = Up(n_feat[0], n_feat0, n_prompt_feat[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history)

#         # OutConv
#         self.conv_last = conv(n_feat0, n_out_feat, 5, bias=bias)

#     def forward(self, x, history_feat: Optional[List[torch.Tensor]] = None):
#         if history_feat is None:
#             history_feat = [None, None, None]

#         history_feat3, history_feat2, history_feat1 = history_feat
#         current_feat = []

#         # 0. featue extraction
#         x = self.feat_extract(x)

#         # 1. encoder
#         x, enc1 = self.enc_level1(x)
#         x, enc2 = self.enc_level2(x)
#         x, enc3 = self.enc_level3(x)

#         # 2. bottleneck
#         x = self.bottleneck(x)

#         # 3. decoder
#         current_feat.append(x.clone())
#         dec_prompt3 = self.prompt_level3(x)
#         x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

#         current_feat.append(x.clone())
#         dec_prompt2 = self.prompt_level2(x)
#         x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

#         current_feat.append(x.clone())
#         dec_prompt1 = self.prompt_level1(x)
#         x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

#         # 4. last conv
#         if self.n_history > 0:
#             for i, history_feat_i in enumerate(history_feat):
#                 if history_feat_i is None:  # for the first cascade, repeat the current feature
#                     history_feat[i] = torch.cat([torch.tile(current_feat[i], (1, self.n_history, 1, 1))], dim=1)
#                 else:  # for the rest cascades: pop the oldest feature and append the current feature
#                     history_feat[i] = torch.cat([current_feat[i], history_feat[i][:, :-self.n_feat[2-i]]], dim=1)
#         return self.conv_last(x), history_feat