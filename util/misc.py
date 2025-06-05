'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-06-05 09:27:13
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-06-05 15:14:37
Description: 根据官方的实现，添加中文注释，源码：https://github.com/facebookresearch/detr.git
'''
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
    local_size = torch.tensor([torch.numel()], device="cuda") # 当前进程数据大小
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)] # 初始化大小列表
    dist.all_gather(size_list, local_size)  # 收集各进程数据大小
    size_list = [int(size.item()) for size in size_list]    # 转为整数列表
    max_size = max(size_list)    # 最大数据大小

    # 接收各进程的数据
    # 注意：由于各进程的张量大小可能不同，因此需要使用torch.empty()创建张量
    # 并使用torch.cuda.ByteTensor()创建GPU张量
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda")) # 创建接收缓冲区
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0) # 填充当前数据至最大尺寸
    dist.all_gather(tensor_list, tensor)  # 接收各进程数据

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]  # 截取有效字节流（去除填充）
        data_list.append(pickle.loads(buffer)) # 反序列化为原始数据
    
    return data_list


def reduce_dict(input_dict, average=True):
    """
    聚合多进程字典中的张量值，支持全局平均或求和
    Args:
        input_dict (dict): 所有值将被聚合的字典
        average (bool): 选择求平均（True）或求和（False）
    减少字典中所有进程的值，使所有进程都有平均结果。还原后返回一个与input_dict具有相同字段的dict。
    """
    world_size = get_world_size() # 获取进程数
    if world_size < 2:
        return input_dict
    with torch.no_grad():   # 禁用梯度计算
        names = []
        values = []
        # 对键进行排序，以便他们在各个进程中保持一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0) # 堆叠张量
        dist.all_reduce(values) # 所有进程的 values 张量相加
        if average:
            values /= world_size # 平均值
        reduce_dict = {k: v for k, v in zip(names, values)} # 还原字典
    return reduce_dict


class MetricLogger(object):
    """
    用于在训练过程中记录、同步和打印指标，支持以下功能：
    - 动态添加和更新指标
    - 自动将张量转换为标量
    - 分布式训练中的进程同步
    - 定期打印训练状态，包括进度、剩余时间、指标值、时间和显存使用情况
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)    # 存储指标的字典，默认使用SmoothedValue
        self.delimiter = delimiter    # 分隔符
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()    # 张量转标量
            assert isinstance(v, (float, int))    # 必须是浮点数或整数
            self.meters[k].update(v)    # 更新指标

    def __getattr__(self, attr):    # 用于处理对象的属性访问
        # 这个方法的设计使得可以通过点号.来访问MetricLogger对象的指标，例如logger.loss，其中loss是指标的名称。这样可以方便地获取和记录指标值，而不需要直接访问self.meters字典。
        if attr in self.meters: # 检查所请求的属性attr是否存在于self.meters字典中。如果存在，表示用户希望访问某个指标的值，于是返回该指标的SmoothedValue对象。
            return self.meters[attr]    # 直接访问指标对象
        if attr in self.__dict__:   # 如果attr不存在于self.meters字典中，继续检查是否存在于对象的__dict__属性中。这是因为除了指标之外，MetricLogger对象还可以有其他自定义属性。
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(    # 如果attr既不是指标也不是自定义属性，则抛出AttributeError异常，表示对象没有这个属性
            type(self).__name__, attr))

    def __str__(self):
        # 通过这个方法，你可以通过 print(logger) 来打印 MetricLogger 对象的状态，从而方便地查看所有指标的值。这有助于监控训练过程中的指标变化和性能
        loss_str = [] # 创建一个空列表 loss_str 用于存储每个指标的字符串表示。
        for name, meter in self.meters.items(): # 遍历 self.meters 字典中的每个指标（以指标名称为键，以 SmoothedValue 对象为值）。
            loss_str.append(
                "{}: {}".format(name, str(meter)) # 对每个指标，使用 str(meter) 来获取 SmoothedValue 对象的字符串表示，该字符串表示包括指标的平均值、中位数等统计信息。
            )
        return self.delimiter.join(loss_str)    # 按分隔符连接各指标的字符串表示

    def synchronize_between_processes(self):
        # 用于在多个进程之间同步指标数据
        for meter in self.meters.values(): # 遍历 self.meters 字典中的每个指标名称和相应的 SmoothedValue 对象。
            meter.synchronize_between_processes() # 对每个 SmoothedValue 对象调用其自身的 synchronize_between_processes 方法，以确保该指标在所有进程之间同步

    def add_meter(self, name, meter):
        # add_meter 方法用于向 MetricLogger 对象中添加新的指标
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        # 用于在迭代训练中定期记录和打印训练进程的信息
        i = 0 # 初始化迭代计数器 i
        if not header: # 如果没有提供 header，则将其设置为空字符串
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}') # 用于记录迭代时间的 SmoothedValue 对象
        data_time = SmoothedValue(fmt='{avg:.4f}') # 用于记录数据加载时间的 SmoothedValue 对象
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd' # 根据 iterable 的长度确定输出格式中迭代次数的显示宽度
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0 # 定义 MB 常量
        for obj in iterable: # 遍历 iterable
            data_time.update(time.time() - end) # 更新 data_time，记录从上一次迭代到当前迭代的时间
            yield obj # 使用 yield obj 从迭代器中获取下一个对象，并将其返回
            iter_time.update(time.time() - end) # 更新 iter_time，记录从上一次迭代到当前迭代的时间
            if i % print_freq == 0 or i == len(iterable) - 1: # 检查是否达到了指定的 print_freq 或是否已经遍历完了 iterable 中的所有对象
                eta_seconds = iter_time.global_avg * (len(iterable) - i) # 计算剩余时间
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)
                    ))
            i += 1
            end = time.time() # 更新 end，记录当前时间
        total_time = time.time() - start_time # 计算总时间
        total_time_str = str(datetime.timedelta(seconds=int(total_time))) # 格式化总时间为字符串
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)
        ))


def get_sha():
    # 用于获取当前代码库的 Git 信息，包括提交的 SHA（commit hash）、工作目录的状态以及当前的分支
    cwd = os.path.dirname(os.path.abspath(__file__)) # 获取当前工作目录

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git','rev-parse', 'HEAD']) # 这一行运行 Git 命令 git rev-parse HEAD，以获取当前代码库的最新提交的 SHA（commit hash）
        subprocess.check_output(['git', 'diff'], cwd=cwd) # 这一行运行 Git 命令 git diff，以检查工作目录中是否有未提交的更改
        diff = _run(['git', 'diff-index', 'HEAD']) # 这一行运行 Git 命令 git diff-index HEAD，以获取工作目录中未提交的更改
        diff = "has uncommited changes" if diff else "clean" # 如果 diff 不为空，说明有未提交的更改，否则说明工作目录是干净的
        branch = _run(['git','rev-parse', '--abbrev-ref', 'HEAD']) # 这一行运行 Git 命令 git rev-parse --abbrev-ref HEAD，以获取当前分支的名称
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    """
    将具有不同大小的图像和与之相关的信息组合成一个批次
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # 找到输入列表中的每个列的最大值
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """
    用于存储具有不同大小的图像和与之相关的信息的类
    """
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # 用于将 NestedTensor 对象转移到指定设备
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        # 将 NestedTensor 对象分解为一系列张量
        return self.tensors, self.mask
    
    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    将一系列张量转换为 NestedTensor 对象
    """
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing(): # 如果当前代码处于ONNX追踪模式，那么它会调用 _onnx_nested_tensor_from_tensor_list(tensor_list) 函数来创建 NestedTensor 对象
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        
        max_size = _max_by_axis([list(img.shape) for img in tensor_list]) # 找到张量列表中图像的最大尺寸    
        batch_shape = [len(tensor_list)] + max_size # 构造批次形状
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device) # 创建一个零张量，其形状为 batch_shape
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device) # 创建一个布尔掩码张量，其形状为 (b, h, w)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img) # 将图像填充到 tensor 中
            m[: img.shape[1], : img.shape[2]] = False # 将掩码填充到 mask 中
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # 处理张量填充，确保他们有相同的大小
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))
    
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    在分布式环境中设置打印行为，以便在非主进程中禁用打印
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    检查是否支持分布式计算，并且是否已经初始化了分布式环境
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取当前进程的进程数
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的进程号
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    检查当前进程是否是主进程
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    在主进程上保存模型和其他信息
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式计算环境
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) # 全局进程编号（0~world_size-1）
        args.world_size = int(os.environ['WORLD_SIZE']) # 总进程数，通常等于GPU卡数
        args.gpu = int(os.environ['LOCAL_RANK']) # 当前节点内GPU编号（单机多卡时为 RANK）
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID']) # SLURM分配的进程ID
        args.gpu = args.rank % torch.cuda.device_count() # 按GPU数量取模绑定设备
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True # 标记已启用分布式计算模式

    torch.cuda.set_device(args.gpu) # 将当前进程绑定到指定GPU
    args.dist_backend = 'nccl' # 设置通信后端为NCCL, NVIDIA专用
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank) # 初始化分布式进程组
    torch.distributed.barrier() # 等待所有进程同步
    setup_for_distributed(args.rank == 0) # 设置打印行为，以便在非主进程中禁用打印


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    计算了模型的预测输出 output 与真实标签 target 之间的精度（accuracy），并且支持不同的精度计算，即可以计算前k个预测的精度
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:K].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    图像插值函数，用于调整图像大小
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)
        
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)



