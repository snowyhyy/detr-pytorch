'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-05-27 10:28:35
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-05-27 11:55:24
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''

"""
用于计算匹配成本并解决相应线性求和分配问题（LSAP）的模块。匈牙利匹配
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
        This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    此类用于计算目标与网络预测的对应关系。

    出于效率因素，目标不包括 no_object 类别。因此，一般来说，预测值多于目标。
    在此情况下，我们为最佳预测做1:1匹配，而其他预测值是不匹配的（因此被视为非目标）
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost

        创建匹配器

        Args:
            cost_class: 匹配成本中类别损失的相对权重
            cost_bbox: 匹配成本中边界框损失的L1距离的相对权重
            cost_giou: 匹配成本中边界框的GIoU损失的相对权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        
        执行匹配

        Args:
            outputs: 包含至少以下条目的字典：
                 "pred_logits": 维度为 [batch_size, num_queries, num_classes] 的分类logits张量
                 "pred_boxes": 维度为 [batch_size, num_queries, 4] 的预测边界框坐标张量
            
            targets: 一个目标的列表（len(targets) = batch_size），其中每个目标是一个字典，包含：
                 "labels": 维度为 [num_target_boxes] 的类标签张量（其中 num_target_boxes 是目标中的目标对象数）
                 "boxes": 维度为 [num_target_boxes, 4] 的目标边界框坐标张量

        Returns:
            一个大小为 batch_size 的列表，其中包含元组 (index_i, index_j)：
                - index_i 是所选预测的索引（按顺序）
                - index_j 是相应的所选目标的索引（按顺序）
            对于每个批处理元素，它满足：
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 展平以计算一个批次的匹配成本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 同时，拼接目标标签和边界框
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本。与损失相反，不使用 NLL，
        # 而是将其近似为 1-proba[target class]
        # 这里的 1 是一个不改变匹配结果的常数，因此可以省略
        cost_class = -out_prob[:, tgt_ids]

        # 计算边界框成本。使用 L1 距离，
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算 GIoU 成本。使用 generalized_box_iou 函数
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 合并成本
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu() # [batch_size, num_queries, num_target_boxes]

        # 执行匈牙利匹配
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)



