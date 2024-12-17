import os
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

from torch.autograd import profiler

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # model.freeze_layers(freeze_encoder=True, freeze_decoder=False, freeze_fusion=False, freeze_biattention=True)

    # 初始化TensorBoard日志
    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    # 加载检查点
    best_iou = 0.0
    start_epoch, _ = load_checkpoint(config['best_checkpoint'], model, optimizer)
    # start_epoch, _ = load_checkpoint("./checkpoints/net_v0_tiny_imgnet/checkpoint_epoch_340.pth", model, optimizer)
    start_epoch = start_epoch if start_epoch is not None else 0

    no_improve_epochs = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        iou_scores = []
        dice_scores = []

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

                # 计算IoU和Dice
                preds = outputs.argmax(dim=1)
                _, iou = calculate_iou(preds, labels, config['num_classes'], ignored_classes=[0])
                _, dice = calculate_dice(preds, labels, config['num_classes'], ignored_classes=[0])

                iou_scores.append(iou)
                dice_scores.append(dice)

                tepoch.set_postfix(loss=loss.item(), miou=iou, dice=dice)

                # Example usage
                # save_epoch_predictions(
                #     images=images,
                #     labels=labels,
                #     preds=preds,
                #     out_dir=out_dir,
                #     epoch_idx=epoch,
                #     mean=[0.485, 0.456, 0.406],  # Example mean
                #     std=[0.229, 0.224, 0.225],  # Example std
                #     num_classes=21  # For example, 21 classes in segmentation
                # )

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_iou = sum(iou_scores) / len(iou_scores)
        avg_dice = sum(dice_scores) / len(dice_scores)

        print(
            f'Epoch [{epoch + 1}/{epochs}] train --- Loss: {avg_epoch_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}')
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('IoU/train', avg_iou, epoch)
        writer.add_scalar('Dice/train', avg_dice, epoch)

        # 验证步骤
        val_loss, val_iou, val_dice, val_images, val_outputs, val_labels = \
            (validate_segmentation(model, val_loader, loss_fn, device, epoch, config))

        unique_preds = torch.unique(val_outputs)
        print(f"Unique preds in this batch: {unique_preds.cpu().numpy()}")
        unique_labels = torch.unique(val_labels)
        print(f"Unique labels in this batch: {unique_labels.cpu().numpy()}")

        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)

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

    val_images, val_labels, val_outputs = None, None, None

    with torch.no_grad():
        for images, labels, _ in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            labels = labels.squeeze(1)  # 去掉标签的单通道维度
            loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
            running_loss += loss.item()

            # 计算IoU和Dice
            preds = outputs.argmax(dim=1)
            _, iou = calculate_iou(preds, labels, config['num_classes'], ignored_classes=[0])
            _, dice = calculate_dice(preds, labels, config['num_classes'], ignored_classes=[0])

            iou_scores.append(iou)  # 保存当前batch的IoU
            dice_scores.append(dice)  # 保存当前batch的Dice

            # 保存最后一批次的数据（可以选择保存任何一批次的数据）
            val_images, val_labels, val_outputs = images, labels, preds

    # 计算平均损失、IoU和Dice
    avg_loss = running_loss / len(data_loader)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)

    return avg_loss, avg_iou, avg_dice, val_images, val_outputs, val_labels


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

    # for batch in train_loader:
    #     visualize_batch(batch, save_dir="./", filename="batch_image.png")
    #     break

    # model_seg = SegFormer(phi="b0").to(device)
    # weights_init(model_seg)

    # model_ = net("v0", num_classes=config['num_classes'], input_size=config['input_shape']).to(device)
    # # 初始化权重
    # weights_init(model_, init_type='kaiming', init_gain=0.1, bias_init='normal')
    # model_ = SegFormer(phi="b0").to(device)
    # model.eval()
    # print("Model Summary:")
    # summary(
    #     model,
    #     input_size=(1, 3, 224, 224),
    #     col_names=["input_size",
    #                "output_size",
    #                "num_params",
    #                "params_percent",
    #                "trainable"],
    #     depth=10 # Control the depth of details in the output
    # )
    Segmentation_train(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
