import numpy as np
import torch

def calculate_confusion_matrix(preds, labels, num_classes, ignored_classes=None):
    """
    计算预测与标签的混淆矩阵，用于计算IoU和其他分割指标。
    """
    if ignored_classes is None:
        ignored_classes = []

    # Flatten预测值和标签
    preds = preds.view(-1)
    labels = labels.view(-1)

    # 去除忽略类
    if ignored_classes:
        mask = torch.isin(labels, torch.tensor(ignored_classes).to(labels.device), invert=True)
        preds = preds[mask]
        labels = labels[mask]

    # 使用Numpy进行高效的混淆矩阵计算
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    cm = np.bincount(num_classes * labels_np + preds_np, minlength=num_classes**2).reshape(num_classes, num_classes)
    cm = torch.tensor(cm, dtype=torch.int64, device=preds.device)

    return cm
def compute_iou(cm):
    """
    计算每个类别的IoU
    """
    intersection = cm.diagonal()
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection.float() / union.clamp(min=1)  # 防止除以零
    return iou

def compute_dice(cm):
    """
    计算每个类别的Dice Coefficient
    """
    intersection = cm.diagonal()
    dice = (2 * intersection.float()) / (cm.sum(dim=1) + cm.sum(dim=0)).clamp(min=1)  # 防止除以零
    return dice

def compute_pixel_accuracy(cm):
    """
    计算像素级准确度
    """
    correct = cm.diagonal().sum()
    total = cm.sum()
    return correct.float() / total.clamp(min=1)  # 防止除以零

def compute_mean_accuracy(cm):
    """
    计算每个类别的准确率的平均值
    """
    accuracy = cm.diagonal().float() / cm.sum(dim=1).clamp(min=1)  # 防止除以零
    return accuracy.mean()

def compute_frequency_weighted_iou(cm):
    """
    计算频率加权的IoU
    """
    frequency = cm.sum(dim=1).float() / cm.sum().float()
    iou = compute_iou(cm)
    return (frequency * iou).sum()