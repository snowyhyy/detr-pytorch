'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-05-28 09:47:22
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-05-28 14:32:52
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''

"""
DETR 模型和标准类
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)


class DETR(nn.Module):
    """执行目标检测的DETR模块"""
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        Args:
            backbone: 将要使用的骨干网络的 torch 模块
            transformer: transformer 结构的 torch 模块
            num_classes: 类别数
            num_queries: 预测的查询数, 一幅图像的最大目标数量，论文推荐100个
            aux_loss: 是否使用辅助损失
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # 分类头
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # 边界框回归头
        self.query_embed = nn.Embedding(num_queries, hidden_dim) # 初始化查询向量
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
    
    def forward(self, samples: NestedTensor):
        """
        前向传播期望得到 NestedTensor 类型的输入，其中包含图像和目标的位置信息
            - samples.tensor: 批次的图像张量，形状为 [B, 3, H, W]
            - samples.mask: 批次的图像掩码张量，形状为 [B, H, W]，1为填充
        
        返回以下元素的 dict:
            - "pred_logits": 所有查询的分类概率（包含非对象目标）， 形状为 [B * num_queries * (num_classes + 1)]
            - "pred_boxes": 所有查询的边界框归一化坐标，(cx, cy, w, h)，归一化值在[0, 1]
            - "aux_outputs": 可选，辅助输出，如果 aux_loss 为 True，则返回辅助损失的结果
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose() # 取出最后一层特征图
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 不支持有非同种类值的dict，例如一个字典中不能同时存在 tensor 和 list
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """DETR的损失函数
    该过程分为两个步骤：
        1）我们计算地面真值框与模型输出之间的匈牙利算法匹配
        2）我们监督每一对匹配的地面真值/预测结果（包括类别和框的监督）
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Args:
            num_classes: 类别数, 这里忽略了 非目标 类
            matcher: 匹配器
            weight_dict: 权重字典
            eos_coef: 应用于 非目标 类别的相对分类权重
            losses: 损失函数列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight) # 注册缓冲区, 不更新参数

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        计算 NLL 损失
        目标字典必须包含键 labels ，该键对应的张量维度为[nb_target_boxes]
        Args:
            outputs: 模型输出，需包含 pred_logits（分类逻辑值）
            targets: 真实目标列表，每个元素需包含 "labels"（真实类别索引）
            indices: 匈牙利算法匹配结果，每个元素为 (预测索引, 真实框索引) 元组
            num_boxes: 当前批次的总真实框数（用于损失归一化）
            log: 是否记录分类错误率
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # 

        idx = self._get_src_permutation_idx(indices) # 将匈牙利匹配结果转换为二维索引
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(srfc_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o # 填充目标类别

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight) # 计算分类损失
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # 计算分类错误率
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        计算 cardinality 指标，不进行梯度回传，仅仅用来记录
        表示预测的非空框数量与真实目标数的差异
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # 获取每个样本的真实目标数量 [batch_size]
        # 计算不是 "无目标" 的预测数目
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) # 统计预测的非空框数量 [batch_size]
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float()) # 计算平均绝对误差（MAE）
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        计算与边界框、L1回归损失和GIoU损失相关的损失时，
        目标字典必须包含 "boxes" 键，该键包含维度为 [nb_target_boxes, 4] 的张量
        目标边界框的格式为 (cx, cy, w, h)，根据图像大小进行归一化。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        计算与分割损失相关的损失时（焦点损失和Dice损失），
        目标字典必须包含 "masks" 键，该键包含维度为 [nb_target_boxes, h, w] 的张量
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices) # 预测框匹配索引
        tgt_idx = self._get_tgt_permutation_idx(indices) # 真实框匹配索引
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # 上采样预测值到目标尺寸
        src_masks = interpolate(src_masks[:, None],
                                size=target_masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # 获取预测框匹配索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        # 获取真实框匹配索引
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets):
        """
        损失计算的实现
        Args:
            outputs: 张量字典，模型的输出
            targets: 字典列表，len(targets) == batch_size
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'} # 去除辅助输出

        # 检索最后一层的输出与目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets) # 匈牙利算法匹配

        # 计算所有节点上目标框的平均数量，以进行规范化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 计算辅助损失时，需要在每个中间层的输出上计算损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # 中间层的掩码损失计算成本太高，因此忽略
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 仅对最后一层进行日志记录
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


class PostProcess(nn.Module):
    """
    此模块将模型的输出转换为cocoapi所期望的格式
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Args:
            outputs: 模型的原始输出
            target_sizes: 图像的目标尺寸，形状为 [batch_size, 2]
                对于评估，这必须是原始图像大小
                对于可视化，这是数据增强后但填充前的图像尺寸
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1) # 排除最后一个维度（背景类）

        # 转换到 [x0, y0, x1, y1] 格式
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) # [cx, cy, w, h] -> [x0, y0, x1, y1]
        # 从相对坐标[0, 1]转换到[0, height]绝对坐标
        img_h, img_w = target_sizes.unbind(1) # 取出图像高度和宽度
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """
    多层感知机, 也就是 FFN
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # 这里的 num_classes 指的是 目标最大类别 + 1（背景类）
    # 比如在 coco 数据集中，目标检测类别数是 80 类，其中最大的类别是 90，所以 num_classes = 91
    # 在全景分割中，类别包含 物体+背景，为了预留足够的空间，num_classes = 250
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # 对于全景分割，只需要添加一个足够大 num_classes, 去容纳 max_obj_id + 1 个类别, 不需要确切的类别数
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

