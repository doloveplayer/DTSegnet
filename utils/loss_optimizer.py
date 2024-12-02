import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


# ===================== 定义损失函数 =====================
def CE_Loss(inputs, target, cls_weights):
    cls_weights = np.array(cls_weights)
    weights = torch.from_numpy(cls_weights).to(inputs.device).float()
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=0)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, alpha=0.5, gamma=2):
    cls_weights = np.array(cls_weights)
    weights = torch.from_numpy(cls_weights).to(inputs.device).float()
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=weights, ignore_index=0, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """
    计算带有 beta 的 DICE 损失，适用于多类别分割任务

    :param inputs: 预测结果 (n, c+1, h, w)，其中 c+1 是类别数+1（包括背景）
    :param target: 真实标签 (n, h, w)，每个像素是一个整数（类别索引）
    :param beta: beta 值，用于调节精度和召回之间的权重，默认是 1，表示平衡
    :param smooth: 平滑因子，避免除零，默认是 1e-5
    :return: DICE 损失值
    """
    n, c, h, w = inputs.size()  # c_plus_1 是类别数 + 1（包括背景）

    # 对于多类别情况，计算每个类别的 DICE
    temp_inputs = F.softmax(inputs, dim=1)  # 将 logits 转换为概率分布

    # 将目标标签转换为 one-hot 编码，确保 torch.eye 在 target.device 上
    temp_target = torch.eye(c, device=target.device)[target.view(-1).long()].view(n, h, w, c)

    # 转置 temp_target，使其形状与 temp_inputs 匹配
    temp_target = temp_target.permute(0, 3, 1, 2)  # 变为 (n, c, h, w)

    # 计算每个类别的 TP、FP 和 FN
    tp = torch.sum(temp_target * temp_inputs, dim=[0, 2, 3])  # True positives
    fp = torch.sum(temp_inputs, dim=[0, 2, 3]) - tp  # False positives
    fn = torch.sum(temp_target, dim=[0, 2, 3]) - tp  # False negatives

    # 加权 DICE 系数的计算
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # 返回 DICE 损失
    dice_loss = 1 - torch.mean(score)
    return dice_loss


# ===================== 修改 get_loss_function 函数 =====================

def get_loss_function(name, **kwargs):
    """
    获取损失函数的工厂方法。

    参数:
    - name (str): 损失函数名称，如 'MSELoss', 'CrossEntropyLoss', 'BCEWithLogitsLoss', 'DiceLoss', 'TverskyLoss', 'FocalLoss' 等。
    - kwargs (dict): 传递给损失函数的其他参数。

    返回:
    - torch.nn.Module: 损失函数实例。
    """
    if name == 'MSELoss':
        return nn.MSELoss(**kwargs)
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'NLLLoss':
        return nn.NLLLoss(**kwargs)
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss(**kwargs)
    elif name == 'L1Loss':
        return nn.L1Loss(**kwargs)
    elif name == 'HuberLoss':
        return nn.HuberLoss(**kwargs)
    elif name == 'DiceLoss':
        return Dice_loss
    elif name == 'FocalLoss':
        return Focal_Loss
    elif name == 'CELoss':
        return CE_Loss
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def get_optimizer(name, model, lr, weight_decay=0, **kwargs):
    """
    获取优化器的工厂方法。

    参数:
    - name (str): 优化器名称，如 'Adam', 'AdamW', 'SGD', 'RMSprop' 等。
    - model (torch.nn.Module): 模型实例，用于获取参数。
    - lr (float): 学习率。
    - weight_decay (float): 权重衰减。
    - kwargs (dict): 传递给优化器的其他参数。

    返回:
    - torch.optim.Optimizer: 优化器实例。
    """
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


class WarmupCosineScheduler(_LRScheduler):
    """
    自定义学习率调度器：Warmup + 余弦衰减。

    参数:
    - optimizer: 优化器实例。
    - warmup_epochs: Warmup 阶段的 epoch 数量。
    - max_epochs: 总的 epoch 数量。
    - eta_min: 最低学习率。
    - last_epoch: 上次的 epoch 索引。
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前 epoch 计算学习率。
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]
