
"""

"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from packaging import version
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# 由于PyTorch和TorchVision 0.5版本中存在的空张量漏洞而必不可少
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """
    用于跟踪和计算数值序列平滑指标的工具，特别适用于机器学习训练过程中监控损失和指标的变化
    跟踪一系列数值，并提供访问功能，以便获取窗口内的平滑数值或全局序列的平均值。
    """
    # 初始化：设置窗口大小和输出格式
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 滑动窗口存储
        self.total = 0.0    # 总数
        self.count = 0      # 计数
        self.fmt = fmt      # 输出格式
    # 更新数值
    def update(self, value, n=1):
        self.deque.append(value)    # 加入滑动窗口
        self.count += n             # 增加计数
        self.total += value * n     # 增加总数
    
    # 分布式训练同步
    def synchronize_between_processes(self):
        # 将 count/total 同步到所有进程
        # 注意：滑动窗口数据(deque)不同步，因各GPU窗口内容可能不同
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property # 只读属性，返回滑动窗口的平均值
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property # 只读属性，返回滑动平均值
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property # 只读属性，返回全局平均值
    def global_avg(self):
        return self.total / self.count
    
    @property # 只读属性，返回最大值
    def max(self):
        return max(self.deque)
    
    @property # 只读属性，返回最新值
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


def all_gather(data):
    """
    用于分布式训练的all_gather函数，将数据从所有进程收集到一个列表中
    Args:
        data: 任意可收集的数据
    Returns:
        一个列表，包含所有进程中的数据
    """
    # 检查是否是单机模式
    world_size = get_world_size()
    if world_sizze == 1:
        return [data]
    
    # 序列化数据
    buffer = pickle.dumps(data)  # 序列化为字节流
    storage = torch.ByteStorage.from_buffer(buffer) # 转为张量
    tensor = torch.ByteTensor(storage).to("cuda") # 转为GPU张量

    # 获取各进程的张量大小
    local_size = torch.tensor([torch.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 接收各进程的数据
    # 注意：由于各进程的张量大小可能不同，因此需要使用torch.empty()创建张量
    # 并使用torch.cuda.ByteTensor()创建GPU张量

