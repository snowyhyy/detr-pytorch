'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-06-05 14:54:39
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-06-05 16:01:23
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''
"""
用于边界框和GIou操作
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """
    将边界框的中心点坐标和宽高转换为左上角和右下角坐标
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    将边界框的左上角和右下角坐标转换为中心点坐标和宽高
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)
    ]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    计算两个边界框的IoU
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    计算两个边界框的GIoU
    boxes 的数据格式为 [x0, y0, x1, y1]
    返回 [N, M] 维度的矩阵，其中 N 为 boxes1 的数量，M 为 boxes2 的数量
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2], 闭包矩形的宽高
    area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return iou - (area - union) / area
    

def masks_to_boxes(masks):
    """
    
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)  # 空掩码处理
    
    h, w = masks.shape[-2:]
    
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

