# from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock, _log_api_usage_once, partial, Optional, Callable, Permute
# from torch import nn



# class SwinTransformer(nn.Module):
#     """
#     Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
#     Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
#     Args:
#         patch_size (List[int]): Patch size.
#         embed_dim (int): Patch embedding dimension.
#         depths (List(int)): Depth of each Swin Transformer layer.
#         num_heads (List(int)): Number of attention heads in different layers.
#         window_size (List[int]): Window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
#         dropout (float): Dropout rate. Default: 0.0.
#         attention_dropout (float): Attention dropout rate. Default: 0.0.
#         stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
#         num_classes (int): Number of classes for classification head. Default: 1000.
#         block (nn.Module, optional): SwinTransformer Block. Default: None.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None.
#         downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         embed_channels: int,
#         patch_size: list[int],
#         depths: list[int],
#         num_heads: list[int],
#         window_size: list[int],
#         mlp_ratio: float = 4.0,
#         *,
#         bias: bool = True,
#         norm: bool = False
#     ):
#         super().__init__()

#         if norm:
#             norm_layer = partial(nn.LayerNorm, eps=1e-13)
#         else:
#             norm_layer = nn.Identity

#         self.to_input = nn.Sequential(
#             nn.Conv2d(in_channels, embed_channels, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])),
#             Permute([0, 2, 3, 1]),
#             norm_layer(embed_channels)
#         )

#         self.enc = nn.ModuleList([
#             *[
#                 SwinTransformerBlock(
#                     dim=embed_channels * 2**i_layer,
#                     input_resolution=(64 // (2**i_layer), 64 // (2**i_layer)),
#                     num_heads=num_heads[i_layer],
#                     window_size=window_size,
#                     shift_size=[0 if (i_layer % 2 == 0) else w // 2 for w in window_size],
#                     mlp_ratio=mlp_ratio,
#                     norm_layer=norm_layer,
#                 )
#                 for i_layer in range(len(depths))
#             ],
#             PatchMerging(embed_channels * 2**(len(depths) - 1), norm_layer),
#         ])

#         self.dec,

#         self.to_output

#         # build SwinTransformer blocks
#         for i_stage in range(len(depths)):
#             stage: list[nn.Module] = []
#             dim = embed_channels * 2**i_stage
#             for i_layer in range(depths[i_stage]):
#                 # adjust stochastic depth probability based on the depth of the stage block
#                 stage.append(
#                     block(
#                         dim,
#                         num_heads[i_stage],
#                         window_size=window_size,
#                         shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
#                         mlp_ratio=mlp_ratio,
#                     )
#                 )
#             layers.append(nn.Sequential(*stage))
#             # add patch merging layer
#             if i_stage < (len(depths) - 1):
#                 layers.append(PatchMerging(dim, norm_layer))
#         self.features = nn.Sequential(*layers)

#         num_features = embed_channels * 2 ** (len(depths) - 1)
#         self.norm = norm_layer(num_features)
#         self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.norm(x)
#         x = self.permute(x)
#         return x