import torch


import torch

def calculate_iou(preds, labels, num_classes, ignored_classes=None):
    """
    计算分割任务的 IoU，支持忽略某些类别
    :param preds: 预测的张量，形状为 [batch_size, H, W]
    :param labels: 真实标签张量，形状为 [batch_size, H, W]
    :param num_classes: 分类数
    :param ignored_classes: 需要忽略的类别索引列表 (可选)
    :return: 每个类别的 IoU 和平均 IoU
    """
    # 如果没有指定忽略的类别，默认为空列表
    if ignored_classes is None:
        ignored_classes = []

    # 确保 preds 和 labels 是整数类型
    preds = preds.to(torch.int64)
    labels = labels.to(torch.int64)

    # 确保标签值在合法范围内
    assert preds.min() >= 0 and preds.max() < num_classes, f"Preds values are out of range: {preds.min()} - {preds.max()}"
    assert labels.min() >= 0 and labels.max() < num_classes, f"Labels values are out of range: {labels.min()} - {labels.max()}"

    iou_per_class = []
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)

    for class_idx in range(num_classes):
        # 跳过被忽略的类别
        if class_idx in ignored_classes:
            iou_per_class.append(None)
            continue

        intersection = (preds_onehot[:, class_idx] & labels_onehot[:, class_idx]).sum().item()
        union = (preds_onehot[:, class_idx] | labels_onehot[:, class_idx]).sum().item()
        iou = intersection / union if union > 0 else 0
        iou_per_class.append(iou)

    # 计算忽略类别后的平均 IoU
    valid_iou = [iou for iou in iou_per_class if iou is not None]
    mean_iou = sum(valid_iou) / len(valid_iou) if valid_iou else 0

    return iou_per_class, mean_iou


def calculate_dice(preds, labels, num_classes, ignored_classes=None):
    """
    计算分割任务的 Dice 系数，支持忽略某些类别
    :param preds: 预测的张量，形状为 [batch_size, H, W]
    :param labels: 真实标签张量，形状为 [batch_size, H, W]
    :param num_classes: 分类数
    :param ignored_classes: 需要忽略的类别索引列表 (可选)
    :return: 每个类别的 Dice 和平均 Dice
    """
    # 如果没有指定忽略的类别，默认为空列表
    if ignored_classes is None:
        ignored_classes = []

    # 确保 preds 和 labels 是整数类型
    preds = preds.to(torch.int64)
    labels = labels.to(torch.int64)

    # 确保标签值在合法范围内
    assert preds.min() >= 0 and preds.max() < num_classes, f"Preds values are out of range: {preds.min()} - {preds.max()}"
    assert labels.min() >= 0 and labels.max() < num_classes, f"Labels values are out of range: {labels.min()} - {labels.max()}"

    dice_per_class = []
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)

    for class_idx in range(num_classes):
        # 跳过被忽略的类别
        if class_idx in ignored_classes:
            dice_per_class.append(None)
            continue

        intersection = (preds_onehot[:, class_idx] & labels_onehot[:, class_idx]).sum().item()
        dice = (2 * intersection) / (preds_onehot[:, class_idx].sum() + labels_onehot[:, class_idx].sum() + 1e-6)
        dice_per_class.append(dice)

    # 计算忽略类别后的平均 Dice
    valid_dice = [dice for dice in dice_per_class if dice is not None]
    mean_dice = sum(valid_dice) / len(valid_dice) if valid_dice else 0

    return dice_per_class, mean_dice

