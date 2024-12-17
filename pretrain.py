import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model.net import net
from utils.metrics import *
from torchinfo import summary
from utils.weight_init import weights_init
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from configs.net_v0_tiny_imagenet import config, train_loader, val_loader
from utils.modelsave import save_checkpoint, load_checkpoint, seed_everything
from utils.loss_optimizer import get_optimizer, WarmupCosineScheduler


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 反归一化处理
mean = np.array([0.485, 0.456, 0.406])  # 归一化时的均值
std = np.array([0.229, 0.224, 0.225])  # 归一化时的标准差


def create_activation_hook(writer, epoch, batch_idx):
    def activation_hook(module, input, output):
        # 获取输出张量
        activation_values = output.detach()

        # 计算激活值的均值、标准差、最大值、最小值
        mean = activation_values.mean().item()
        std = activation_values.std().item()
        max_val = activation_values.max().item()
        min_val = activation_values.min().item()

        # 计算激活为零的比例
        neg_activation_rate = (activation_values < 0).sum().item() / activation_values.numel()

        # 计算非零激活的比例
        pos_activation_rate = (activation_values >= 0).sum().item() / activation_values.numel()

        # 为每一层生成唯一名称，包括父模块的名称和层的名称
        parent_module_name = module.__module__.split('.')[-1]  # 获取父模块的名称（例如：nn.Module）
        layer_name = f"{parent_module_name}_{module.__class__.__name__}"

        # 将这些统计信息写入 TensorBoard
        writer.add_scalar(f"activations/{layer_name}/mean", mean, global_step=epoch * len(writer.log_dir) + batch_idx)
        writer.add_scalar(f"activations/{layer_name}/std", std, global_step=epoch * len(writer.log_dir) + batch_idx)
        writer.add_scalar(f"activations/{layer_name}/max", max_val, global_step=epoch * len(writer.log_dir) + batch_idx)
        writer.add_scalar(f"activations/{layer_name}/min", min_val, global_step=epoch * len(writer.log_dir) + batch_idx)
        writer.add_scalar(f"activations/{layer_name}/neg_activation_rate", neg_activation_rate,
                          global_step=epoch * len(writer.log_dir) + batch_idx)
        writer.add_scalar(f"activations/{layer_name}/pos_activation_rate", pos_activation_rate,
                          global_step=epoch * len(writer.log_dir) + batch_idx)

        # 记录激活值的直方图
        writer.add_histogram(f"activations/{layer_name}/histogram", activation_values,
                             global_step=epoch * len(writer.log_dir) + batch_idx)

    return activation_hook


def Classification_train(model, train_loader, val_loader, device, config):
    """
    训练分类任务的函数，支持早期停止、模型保存和TensorBoard日志。
    """
    epochs = config["train_epoch"]
    save_dir = config['save_dir']
    save_interval = config['save_interval']
    patience = config['patience']
    out_dir = config['out_img_dir']
    accumulation_steps = config['accumulation_steps']
    fp16 = config['fp16']

    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 获取优化器和学习率调度器
    optimizer = get_optimizer(name=config['optimizer'], model=model, lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=config['warmup_epochs'],
                                      max_epochs=epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失

    # 如果启用混合精度训练，则使用GradScaler
    scaler = GradScaler() if fp16 else None

    model.to(device)

    # 初始化TensorBoard日志
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 加载检查点
    best_accuracy = 0.0
    # start_epoch, _ = load_checkpoint(config['best_checkpoint'], model, optimizer)
    start_epoch, _ = load_checkpoint(config['pre_train'], model, optimizer)
    start_epoch = start_epoch if start_epoch is not None else 0

    no_improve_epochs = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                images, labels = images.to(device), labels.to(device)

                # 正向传播：如果启用混合精度，使用autocast
                if fp16:
                    with autocast():
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                if fp16:
                    # 使用混合精度时，反向传播需要梯度缩放
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()  # 在每个batch开始前清空梯度
                else:
                    # 不使用混合精度时，正常反向传播
                    loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()  # 在每个batch开始前清空梯度

                epoch_loss += loss.item()

                # 计算准确率
                _, preds = torch.max(outputs, dim=1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=correct_preds / total_preds)

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_accuracy = correct_preds / total_preds

        print(f'Epoch [{epoch + 1}/{epochs}] train --- Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_accuracy:.4f}')
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

        # 验证步骤
        val_loss, val_accuracy, val_images, val_outputs, val_labels = \
            validate_classification(model, val_loader, loss_fn, device, epoch, config)

        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存最好的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config['best_checkpoint'])
            print(f"Best model saved at epoch {epoch + 1} with Accuracy {best_accuracy:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 早期停止：如果验证集准确率在指定次数内没有改善，停止训练
        if no_improve_epochs >= patience:
            print(f"Stopping training early at epoch {epoch + 1} due to no improvement.")
            break

    writer.close()
    print("Training complete.")


def validate_classification(model, data_loader, loss_fn, device, epoch, config):
    """
    Validation function for classification tasks. Computes accuracy and saves sample predictions.
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    val_images, val_labels, val_outputs = None, None, None

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # 计算准确率
            _, preds = torch.max(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

            # 保存最后一批次的数据（可以选择保存任何一批次的数据）
            val_images, val_labels, val_outputs = images, labels, preds

    # 计算平均损失和准确率
    avg_loss = running_loss / len(data_loader)
    avg_accuracy = correct_preds / total_preds

    return avg_loss, avg_accuracy, val_images, val_outputs, val_labels



if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    seed_everything()
    model = net("v0", num_classes=config['num_classes'], input_size=config['input_shape'], is_pretraining=True).to(
        device)
    # 初始化权重
    weights_init(model, init_type='kaiming', init_gain=0.1, bias_init='normal')
    model.eval()
    print("Model Summary:")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size",
                   "output_size",
                   "num_params",
                   "params_percent",
                   "trainable"],
        depth=10 # Control the depth of details in the output
    )
    Classification_train(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
