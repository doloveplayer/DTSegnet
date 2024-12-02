import os
import numpy as np
from tqdm import tqdm
from model.net import net
from utils.metrics import *
from torchinfo import summary
from utils.weight_init import weights_init
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from configs.net_v0 import config, train_loader, val_loader
from utils.modelsave import save_checkpoint, load_checkpoint, seed_everything, save_epoch_predictions
from utils.loss_optimizer import get_loss_function, get_optimizer, WarmupCosineScheduler

from model.segformer.segformer import SegFormer

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 反归一化处理
mean = np.array([0.485, 0.456, 0.406])  # 归一化时的均值
std = np.array([0.229, 0.224, 0.225])  # 归一化时的标准差


def Segmentation_train(model, train_loader, val_loader, device, config):
    """
    训练分割任务的函数，支持早期停止、模型保存和TensorBoard日志。
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
    loss_fn = get_loss_function(name=config['loss_function'])

    # 如果启用混合精度训练，则使用GradScaler
    scaler = GradScaler() if fp16 else None

    model.to(device)

    # 初始化TensorBoard日志
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 加载检查点
    best_iou = 0.0
    start_epoch, _ = load_checkpoint(config['best_checkpoint'], model, optimizer)
    start_epoch = start_epoch if start_epoch is not None else 0

    no_improve_epochs = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        iou_scores = []
        dice_scores = []
        iou_per_class = [[] for _ in range(config['num_classes'])]  # 初始化每个类别的 IoU 存储列表
        dice_per_class = [[] for _ in range(config['num_classes'])]  # 初始化每个类别的 Dice 存储列表

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                images, labels = images.to(device), labels.to(device)
                if accumulation_steps <= 1:
                    optimizer.zero_grad()  # 在每个batch开始前清空梯度
                # 正向传播：如果启用混合精度，使用autocast
                if fp16:
                    with autocast():
                        outputs = model(images)
                        labels = labels.squeeze(1)  # 移除标签的单通道维度
                        loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
                else:
                    outputs = model(images)
                    labels = labels.squeeze(1)  # 移除标签的单通道维度
                    loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])

                # 梯度累积
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

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

                # 计算IoU和Dice
                preds = outputs.argmax(dim=1)
                iou_class, iou = calculate_iou(preds, labels, config['num_classes'], ignored_classes=[0])
                dice_class, dice = calculate_dice(preds, labels, config['num_classes'], ignored_classes=[0])

                iou_scores.append(iou)
                dice_scores.append(dice)

                # 保存每个类别的IoU和Dice
                for class_idx in range(config['num_classes']):
                    iou_per_class[class_idx].append(iou_class[class_idx])
                    dice_per_class[class_idx].append(dice_class[class_idx])

                tepoch.set_postfix(loss=loss.item(), miou=iou, dice=dice)

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_iou = sum(iou_scores) / len(iou_scores)
        avg_dice = sum(dice_scores) / len(dice_scores)

        avg_iou_per_class = [sum(iou) / len(iou) for iou in iou_per_class]
        avg_dice_per_class = [sum(dice) / len(dice) for dice in dice_per_class]

        print(
            f'Epoch [{epoch + 1}/{epochs}] train --- Loss: {avg_epoch_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}')
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('IoU/train', avg_iou, epoch)
        writer.add_scalar('Dice/train', avg_dice, epoch)

        # 记录每个类别的IoU和Dice到TensorBoard
        for class_idx in range(config['num_classes']):
            writer.add_scalar(f'train_IoU/class_{class_idx}', avg_iou_per_class[class_idx], epoch)
            writer.add_scalar(f'train_Dice/class_{class_idx}', avg_dice_per_class[class_idx], epoch)

        # 验证步骤
        val_loss, val_iou, val_dice, val_iou_per_class, val_dice_per_class, val_images, val_outputs, val_labels = \
            (validate_segmentation(model, val_loader, loss_fn, device, epoch, config))

        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)

        # 记录每个类别的IoU和Dice到TensorBoard
        for class_idx in range(config['num_classes']):
            writer.add_scalar(f'val_IoU/class_{class_idx}', val_iou_per_class[class_idx], epoch)
            writer.add_scalar(f'val_Dice/class_{class_idx}', val_dice_per_class[class_idx], epoch)

        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存最好的模型
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config['best_checkpoint'])
            print(f"Best model saved at epoch {epoch + 1} with IoU {best_iou:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 早期停止：如果验证集IoU在指定次数内没有改善，停止训练
        if no_improve_epochs >= patience:
            print(f"Stopping training early at epoch {epoch + 1} due to no improvement.")
            break

    writer.close()
    print("Training complete.")


def validate_segmentation(model, data_loader, loss_fn, device, epoch, config):
    """
    Validation function for segmentation tasks. Computes IoU, Dice, and saves sample predictions.
    """
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []
    iou_per_class = [[] for _ in range(config['num_classes'])]  # 初始化每个类别的 IoU 存储列表
    dice_per_class = [[] for _ in range(config['num_classes'])]  # 初始化每个类别的 Dice 存储列表

    val_images, val_labels, val_outputs = None, None, None

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            labels = labels.squeeze(1)  # 去掉标签的单通道维度
            loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
            running_loss += loss.item()

            # 计算IoU和Dice
            preds = outputs.argmax(dim=1)
            iou_class, iou = calculate_iou(preds, labels, config['num_classes'], ignored_classes=[0])
            dice_class, dice = calculate_dice(preds, labels, config['num_classes'], ignored_classes=[0])

            iou_scores.append(iou)  # 保存当前batch的IoU
            dice_scores.append(dice)  # 保存当前batch的Dice

            # 保存每个类别的IoU和Dice
            for class_idx in range(config['num_classes']):
                iou_per_class[class_idx].append(iou_class[class_idx])
                dice_per_class[class_idx].append(dice_class[class_idx])

            # 保存最后一批次的数据（可以选择保存任何一批次的数据）
            val_images, val_labels, val_outputs = images, labels, preds

    # 计算平均损失、IoU和Dice
    avg_loss = running_loss / len(data_loader)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)

    avg_iou_per_class = [sum(iou) / len(iou) for iou in iou_per_class]
    avg_dice_per_class = [sum(dice) / len(dice) for dice in dice_per_class]

    return avg_loss, avg_iou, avg_dice, avg_iou_per_class, avg_dice_per_class, val_images, val_outputs, val_labels


if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    seed_everything()
    model = net("v0", num_classes=config['num_classes'], input_size=config['input_shape']).to(device)
    weights_init(model)

    # model_seg = SegFormer(phi="b0").to(device)
    # weights_init(model_seg)

    # model_ = net("v0", num_classes=21, input_size=(512, 512)).to(device)
    # model_ = SegFormer(phi="b0").to(device)
    # model_.eval()
    # print("Model Summary:")
    # summary(
    #     model_,
    #     input_size=(1, 3, 512, 512),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     # depth=3  # Control the depth of details in the output
    # )
    Segmentation_train(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
