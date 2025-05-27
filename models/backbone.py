'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-05-26 11:57:35
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-05-26 15:21:58
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    BatchNorm2d, 其中批统计量和仿射参数是固定的。创建一个冻结的BatchNorm2d层

    在rqsrt之前，从添加了eps的torchvision.misc.ops中复制粘贴，
    否则除torchvision.models.resnet[18,34,50,101]之外的任何其他模型都会产生nans。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 使用 register_buffer 设定buffer变量，保存进模型的state_dict中
        self.register_buffer("weight", torch.ones(n)) 
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("runnung_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        在加载模型参数时，删除 num_batches_tracked 这个键。
        因为在标准的BatchNorm2d中，这个参数用来跟踪训练时的batch数量，但在冻结BN层中，这个参数是没有意义的。
        """
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )
    
    def forward(self, x):
        # 计算BN层的输出
        w = self.weight.reshape(1, -1, 1, 1) # (B, C_origin, H, W) -> (1, C, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt() # 计算BN层的缩放因子, scale = w / sqrt(var + eps)
        bias = b - rm * scale # 计算BN层的偏移量, bias = b - mean * scale
        return x * scale + bias # 计算BN层的输出, y = x * scale + bias


class BackboneBase(nn.Module):
    """
    封装预训练的主干网络，支持冻结部分层参数和提取中间层特征，主要用于目标检测和分割任务的特征提取
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """
        Args:
            backbone: 预训练的主干网络
            train_backbone: 冻结参数层, False: 冻结所有层，True: 解冻 layer2 layer3 layer4 的参数
            num_channels: 输入图像的通道数
            return_interm_layers: 是否返回中间层特征，False: 只返回最后一层 layer4 特征, True: 返回所有中间层(layer1-layer4)特征
        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask # 原始掩码
            assert m is not None
            # 调整掩码尺寸至特征图大小
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask) # 封装成 NestedTensor
        return out


class Backbone(BackboneBase):
    """
    使用冻结的 BatchNorm 层的 ResNet 主干网络
    """
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)( # 动态加载模型
            replace_stride_with_dilation=[False, False, dilation], # False, False 表示 layer2 和 layer3 不使用空洞卷积。dilation 控制 layer4 是否将步长替换为空洞卷积（用于保持特征图分辨率）。
            pretrained=is_main_process(), # 仅在主进程下载权重（避免多进程冲突）。
            norm_layer=FrozenBatchNorm2d # 替换所有 BN 层为冻结版本。
        )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """
    将 backbone 和 position encoding 层连接起来
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list) # Backbone 前向传播
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype)) # 生成位置编码
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model



