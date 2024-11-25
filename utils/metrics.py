import torch

def calculate_iou(preds, labels, num_classes):
    """
    计算分割任务的 IoU
    :param preds: 预测的张量，形状为 [batch_size, H, W]
    :param labels: 真实标签张量，形状为 [batch_size, H, W]
    :param num_classes: 分类数
    :return: 每个类别的 IoU 和平均 IoU
    """
    iou_per_class = []
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)

    for class_idx in range(num_classes):
        intersection = (preds_onehot[:, class_idx] & labels_onehot[:, class_idx]).sum().item()
        union = (preds_onehot[:, class_idx] | labels_onehot[:, class_idx]).sum().item()
        iou = intersection / union if union > 0 else 0
        iou_per_class.append(iou)

    mean_iou = sum(iou_per_class) / num_classes
    return iou_per_class, mean_iou


def calculate_dice(preds, labels, num_classes):
    """
    计算分割任务的 Dice 系数
    :param preds: 预测的张量，形状为 [batch_size, H, W]
    :param labels: 真实标签张量，形状为 [batch_size, H, W]
    :param num_classes: 分类数
    :return: 每个类别的 Dice 和平均 Dice
    """
    dice_per_class = []
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)

    for class_idx in range(num_classes):
        intersection = (preds_onehot[:, class_idx] & labels_onehot[:, class_idx]).sum().item()
        dice = (2 * intersection) / (preds_onehot[:, class_idx].sum() + labels_onehot[:, class_idx].sum() + 1e-6)
        dice_per_class.append(dice)

    mean_dice = sum(dice_per_class) / num_classes
    return dice_per_class, mean_dice
