'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-05-26 16:01:54
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-05-26 18:26:48
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''

"""
transformer 的各种位置编码
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    正余弦位置编码
    这是一个更为标准的位置嵌入版本，与“Attention is All You Need”论文中所使用的非常相似，只是它被推广应用到图像处理领域。
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 位置编码维度（单方向）
        self.temperature = temperature      # 调节频率分布的参数
        self.normalize = normalize          # 是否归一化坐标到 [0, scale]
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi             # 归一化范围，默认是 [0, 2*pi]
        self.scale = scale                  # 坐标范围

    def forward(self, tensor_list: NestedTensor):
        """
        位置编码公式
        PE(pos, 2i)   = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        其中 pos 是位置，d_model 是模型的维度（这里是 num_pos_feats），i 是位置编码的维度（这里是 num_pos_feats // 2）
        """
        x = tensor_list.tensors # [B, C, H, W]
        mask = tensor_list.mask # [B, H, W]
        assert mask is not None
        not_mask = ~mask # 非mask区域
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # 行(高度)方向的累积, 也就是每个行的位置 # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # 列(宽度)方向的累积, 也就是每个列的位置 # [B, H, W]
        # 归一化坐标到 [0, scale]
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed / (y_embed[:, -1:, :] + eps)) * self.scale # [B, H, W]
            x_embed = (x_embed / (x_embed[:, :, -1:] + eps)) * self.scale # [B, H, W]
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [num_pos_feats]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 调节频率分布

        pos_x = x_embed[:, :, :, None] / dim_t # [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # [B, H, W, num_pos_feats]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B, 2*num_pos_feats, H, W]
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的绝对位置编码
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats) # 输入的图像尺寸最大为 [50, 50], 也就是 token 数量最大为 50*50
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, tensor_list: NestedTensor):
        """
        
        """
        x = tensor_list.tensors # [B, C, H, W]
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1), # [H, W, num_pos_feats]
            y_emb.unsqueeze(1).repeat(1, w, 1), # [H, W, num_pos_feats]
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos  # [B, 2*num_pos_feats, H, W]


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2 # 将总维度分成水平和竖直两个部分，每部分的维度为 N_steps
    if args.position_embedding in ('v2', 'sine'):
        # 位置嵌入
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        # 可学习的绝对位置嵌入
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


