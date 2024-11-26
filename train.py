import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.net import net
from utils.metrics import *
from utils.weight_init import weights_init
from torchinfo import summary
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from configs.net_v0_VOC import config, train_loader, val_loader
from utils.modelsave import save_checkpoint, load_checkpoint, seed_everything
from utils.loss_optimizer import get_loss_function, get_optimizer, WarmupCosineScheduler

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Segmentation_train(model, train_loader, val_loader, device, config):
    """
    Training function for segmentation tasks with early stopping, checkpointing, and TensorBoard support.
    """
    epochs = config["train_epoch"]
    save_dir = config['save_dir']
    save_interval = config['save_interval']
    patience = config['patience']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = get_optimizer(name=config['optimizer'], model=model, lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=config['warmup_epochs'],
                                      max_epochs=epochs, eta_min=1e-6)
    loss_fn = get_loss_function(name=config['loss_function'])

    fp16 = config['fp16']
    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    model.to(device)

    writer = SummaryWriter(log_dir=config['logs_dir'], comment=config['comment'])

    best_iou = 0.0
    start_epoch, _ = load_checkpoint(config['best_checkpoint'], model, optimizer)
    if start_epoch is None:
        start_epoch = 0

    no_improve_epochs = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        iou_scores = []
        dice_scores = []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                if not fp16:
                    optimizer.zero_grad()
                    outputs = model(images)
                    labels = labels.squeeze(1)  # 去掉标签的单通道维度
                    loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
                    loss.backward()
                    optimizer.step()
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(images)
                        labels = labels.squeeze(1)  # 去掉标签的单通道维度
                        loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_loss += loss.item()

                # Compute IoU and Dice
                preds = outputs.argmax(dim=1)
                _, iou = calculate_iou(preds, labels, config['num_classes'])
                _, dice = calculate_dice(preds, labels, config['num_classes'])

                iou_scores.append(iou)  # mean IoU
                dice_scores.append(dice)  # mean Dice

                tepoch.set_postfix(loss=loss.item(), miou=iou, dice=dice)

                # Visualize the images, labels, and predictions in TensorBoard
                writer.add_image('Inputs/batch', make_grid(images, normalize=True, scale_each=True), epoch)
                writer.add_image('Labels/batch', make_grid(labels.float(), normalize=True, scale_each=True), epoch)

                # Ensure that predictions are float for make_grid
                writer.add_image('Predictions/batch', make_grid(preds.float(), normalize=True, scale_each=True), epoch)

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

        # Validation step
        val_loss, val_iou, val_dice, val_images, val_outputs, val_labels = validate_segmentation(
            model, val_loader, loss_fn, device, epoch)

        print(f'Epoch [{epoch + 1}/{epochs}] val --- Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)

        # Save predictions to TensorBoard
        if val_images is not None:
            # 修正：将 val_outputs 和 val_labels 转换为浮点型，并扩展通道
            writer.add_image('Inputs/val', make_grid(val_images, normalize=True, scale_each=True), epoch)
            writer.add_image('Labels/val', make_grid(val_labels.float(), normalize=True, scale_each=True),
                             epoch)
            writer.add_image('Predictions/val',
                             make_grid(val_outputs.float(), normalize=True, scale_each=True), epoch)

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config['best_checkpoint'])
            print(f"Best model saved at epoch {epoch + 1} with IoU {best_iou:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Stopping training early at epoch {epoch + 1} due to no improvement.")
            break

    writer.close()
    print("Training complete.")


def validate_segmentation(model, data_loader, loss_fn, device, epoch):
    """
    Validation function for segmentation tasks. Computes IoU, Dice, and saves sample predictions.
    """
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []

    val_images, val_labels, val_outputs = None, None, None

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            labels = labels.squeeze(1)  # 去掉标签的单通道维度
            loss = loss_fn(outputs, labels.long(), cls_weights=config['cls_weights'])
            running_loss += loss.item()

            # Compute IoU and Dice
            preds = outputs.argmax(dim=1)
            _, iou = calculate_iou(preds, labels, config['num_classes'])
            _, dice = calculate_dice(preds, labels, config['num_classes'])

            iou_scores.append(iou)  # mean IoU
            dice_scores.append(dice)  # mean Dice

            if val_images is None:
                val_images, val_labels, val_outputs = images, labels, preds

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

    weights_init(model)

    model_ = net("v0", num_classes=25, input_size=(512, 512)).to(device)
    model_.eval()
    print("Model Summary:")
    summary(
        model_,
        input_size=(1, 3, 512, 512),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        # depth=3  # Control the depth of details in the output
    )
    # Segmentation_train(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
