import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存检查点函数"""
    print(f"Saving checkpoint to {filepath}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, strict=False):
    """
    加载检查点函数。

    参数:
    - filepath (str): 检查点文件路径。
    - model (torch.nn.Module): 要加载权重的模型。
    - optimizer (torch.optim.Optimizer, 可选): 如果需要恢复优化器状态，可传入。
    - strict (bool): 是否严格匹配模型和检查点的参数大小。

    返回:
    - epoch (int): 检查点中保存的 epoch。
    - loss (float): 检查点中保存的损失。
    """
    if not os.path.isfile(filepath):
        print(f"[Warning] No checkpoint found at '{filepath}'")
        return None, None

    print(f"Loading checkpoint from '{filepath}'")
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # 加载模型权重
    model_state_dict = checkpoint.get('model_state_dict', None)
    if model_state_dict:
        current_model_state_dict = model.state_dict()
        mismatched_keys = []

        # 过滤不匹配的权重
        filtered_state_dict = {}
        successfully_loaded_layers = []  # 用于保存成功加载的层名
        for k, v in model_state_dict.items():
            if k in current_model_state_dict and v.size() == current_model_state_dict[k].size():
                filtered_state_dict[k] = v
                successfully_loaded_layers.append(k)  # 记录加载成功的层名
            else:
                mismatched_keys.append(k)

        # 加载权重
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Model loaded with {len(filtered_state_dict)} matching keys.")
        print(f"Successfully loaded layers: {', '.join(successfully_loaded_layers)}")
        if mismatched_keys:
            print(f"[Warning] {len(mismatched_keys)} keys skipped due to size mismatch: {mismatched_keys}")
    else:
        print("[Error] No 'model_state_dict' found in checkpoint!")

    # 加载优化器状态（如果提供了优化器）
    if optimizer:
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        if optimizer_state_dict:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"[Error] Failed to load optimizer state: {e}")
        else:
            print("[Warning] No 'optimizer_state_dict' found in checkpoint.")

    # 获取 epoch 和损失值
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    if loss is not None:
        print(f"Checkpoint loaded. Epoch: {epoch}, Loss: {loss:.4f}")
    else:
        print(f"Checkpoint loaded. Epoch: {epoch}, Loss not found in checkpoint.")

    return epoch, loss

def save_epoch_predictions(images, labels, preds, out_dir, epoch_idx, mean, std, num_classes):
    """
    Save a large image for a given epoch, showing images, labels, and predictions.
    The labels and predictions are mapped to RGB using a colormap.
    """
    # Process images: Inverse normalize the images (denormalization)
    images_np = process_image(images, mean, std)

    # Convert the labels and predictions to numpy arrays
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # Normalize the labels and predictions to [0, 1] range for colormap
    norm = Normalize(vmin=0, vmax=num_classes - 1)

    # Create a colormap (e.g., 'tab20' for a large number of classes)
    cmap = cm.get_cmap('tab20', num_classes)

    # Get number of batches
    num_batches = len(images)

    # Create a large canvas for the plot
    fig, axes = plt.subplots(num_batches, 3, figsize=(12, num_batches * 4))

    # If there is only one batch, axes will be a 1D array, so we need to handle that case
    if num_batches == 1:
        axes = np.expand_dims(axes, axis=0)  # Make it 2D

    # Iterate over each batch and display image, label, and prediction
    for batch_idx in range(num_batches):
        # Display the original image
        axes[batch_idx, 0].imshow(images_np[batch_idx])
        axes[batch_idx, 0].set_title(f'Image {batch_idx}')
        axes[batch_idx, 0].axis('off')

        # Map label indices to RGB using the colormap
        label_rgb = cmap(norm(labels_np[batch_idx]))  # shape [H, W, 4] (RGBA)
        axes[batch_idx, 1].imshow(label_rgb)
        axes[batch_idx, 1].set_title(f'Label {batch_idx}')
        axes[batch_idx, 1].axis('off')

        # Map prediction indices to RGB using the colormap
        pred_rgb = cmap(norm(preds_np[batch_idx]))  # shape [H, W, 4] (RGBA)
        axes[batch_idx, 2].imshow(pred_rgb)
        axes[batch_idx, 2].set_title(f'Pred {batch_idx}')
        axes[batch_idx, 2].axis('off')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(out_dir, f'epoch_{epoch_idx}_preds.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def process_image(images, mean, std):
    """
    Inverse normalize the image tensors to their original pixel range.

    :param images: Tensor of shape (B, C, H, W)
    :param mean: Mean used for normalization (list or tensor)
    :param std: Standard deviation used for normalization (list or tensor)
    :return: Denormalized images as numpy arrays
    """
    # 如果 mean 和 std 是列表，转换成 Tensor
    if isinstance(mean, list):
        mean = torch.tensor(mean).float()
    if isinstance(std, list):
        std = torch.tensor(std).float()

    # 获取 images 所在的设备
    device = images.device

    # 将 mean 和 std 移动到与 images 相同的设备
    mean = mean.to(device)
    std = std.to(device)

    # 进行逆归一化
    images = images * std[None, :, None, None] + mean[None, :, None, None]

    # 转换为 numpy
    return images.cpu().numpy().transpose(0, 2, 3, 1)  # Change to (B, H, W, C)