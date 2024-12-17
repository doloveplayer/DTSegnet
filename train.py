import os
from tqdm import tqdm
from model.net import net
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from configs.net_v0 import config, train_loader, val_loader
from utils.modelsave import save_checkpoint, load_checkpoint, seed_everything
from utils.loss_optimizer import get_loss_function, get_optimizer, WarmupCosineScheduler
from utils.weight_init import weights_init
from utils.metrics import *

from torch.autograd import profiler

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Segmentation_train(model, train_loader, val_loader, device, config):
    log_interval = 10
    cm_update_interval = 20
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
        confusion_matrix = torch.zeros(config['num_classes'], config['num_classes'], dtype=torch.int64)

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (images, labels, _) in enumerate(tepoch):
                images, labels = images.to(device), labels.to(device)

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

                # 反向传播
                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 梯度累积
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(tepoch)):
                    if fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()  # 在累积后的最后一个 batch 或 epoch 结束时清零梯度

                epoch_loss += loss.item()
                # 每隔几个批次更新混淆矩阵
                if (batch_idx + 1) % cm_update_interval == 0 or (batch_idx + 1 == len(tepoch)):
                    preds = outputs.argmax(dim=1)
                    cm = calculate_confusion_matrix(preds, labels, config['num_classes'], config['ignored_classes'],
                                                    device=device)
                    confusion_matrix += cm
                    tmp_iou = compute_iou(confusion_matrix)

                # 每隔几个批次记录日志
                if (batch_idx + 1) % log_interval == 0:
                    tepoch.set_postfix(loss=loss.item(), iou=tmp_iou.mean())

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        avg_epoch_loss = epoch_loss / len(train_loader)

        # 计算指标
        iou = compute_iou(confusion_matrix)
        dice = compute_dice(confusion_matrix)
        pixel_accuracy = compute_pixel_accuracy(confusion_matrix)
        mean_accuracy = compute_mean_accuracy(confusion_matrix)
        freq_weighted_iou = compute_frequency_weighted_iou(confusion_matrix)

        print(f'Epoch [{epoch + 1}/{epochs}] train --- Loss: {avg_epoch_loss:.4f}, '
              f'IoU: {iou.mean():.4f}, Dice: {dice.mean():.4f}, '
              f'Pixel Accuracy: {pixel_accuracy:.4f}, Mean Accuracy: {mean_accuracy:.4f}, '
              f'Freq Weighted IoU: {freq_weighted_iou:.4f}')

        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('IoU/train', iou.mean(), epoch)
        writer.add_scalar('Dice/train', dice.mean(), epoch)
        writer.add_scalar('Pixel Accuracy/train', pixel_accuracy, epoch)
        writer.add_scalar('Mean Accuracy/train', mean_accuracy, epoch)
        writer.add_scalar('Freq Weighted IoU/train', freq_weighted_iou, epoch)

        # 验证步骤
        val_loss, val_iou, val_dice, val_pixel_accuracy, val_mean_accuracy, val_freq_weighted_iou = \
            validate_segmentation(model, val_loader, loss_fn, device, epoch, config)

        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, '
              f'IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, '
              f'Pixel Accuracy: {val_pixel_accuracy:.4f}, Mean Accuracy: {val_mean_accuracy:.4f}, '
              f'Freq Weighted IoU: {val_freq_weighted_iou:.4f}')

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('Pixel Accuracy/val', val_pixel_accuracy, epoch)
        writer.add_scalar('Mean Accuracy/val', val_mean_accuracy, epoch)
        writer.add_scalar('Freq Weighted IoU/val', val_freq_weighted_iou, epoch)

        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, epoch, save_dir, "best_checkpoint.pth")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


def validate_segmentation(model, val_loader, loss_fn, device, epoch, config):
    """
    用于验证模型的表现。
    """
    model.eval()
    val_loss = 0.0
    confusion_matrix = torch.zeros(config['num_classes'], config['num_classes'], dtype=torch.int64)

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Validating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = labels.squeeze(1)

            loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            cm = calculate_confusion_matrix(preds, labels, config['num_classes'], ignored_classes=[0])
            confusion_matrix += cm

    avg_val_loss = val_loss / len(val_loader)

    # 计算验证指标
    iou = compute_iou(confusion_matrix)
    dice = compute_dice(confusion_matrix)
    pixel_accuracy = compute_pixel_accuracy(confusion_matrix)
    mean_accuracy = compute_mean_accuracy(confusion_matrix)
    freq_weighted_iou = compute_frequency_weighted_iou(confusion_matrix)

    return avg_val_loss, iou.mean(), dice.mean(), pixel_accuracy, mean_accuracy, freq_weighted_iou


if __name__ == '__main__':
    print(torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    seed_everything()
    model = net("v0", num_classes=config['num_classes'], input_size=config['input_shape']).to(device)
    # 初始化权重
    weights_init(model, init_type='kaiming', init_gain=0.1, bias_init='normal')
    model.freeze_layers(freeze_encoder=False, freeze_decoder=False, freeze_fusion=False, freeze_biattention=False)

    # for batch in train_loader:
    #     visualize_batch(batch, save_dir="./", filename="batch_image.png")
    #     break

    # model_seg = SegFormer(phi="b0").to(device)
    # weights_init(model_seg)

    # model_ = net("v0", num_classes=config['num_classes'], input_size=config['input_shape']).to(device)
    # # 初始化权重
    # weights_init(model_, init_type='kaiming', init_gain=0.1, bias_init='normal')
    # from model.segformer.segformer import SegFormer
    # model = SegFormer(phi="b0").to(device)
    model.eval()
    print("Model Summary:")
    summary(
        model,
        input_size=(1, 3, 512, 512),
        col_names=["input_size",
                   "output_size",
                   "num_params",
                   "params_percent",
                   "trainable"],
        depth=10  # Control the depth of details in the output
    )
    # Segmentation_train(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
