import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.load_data import VOC2012SegmentationDataset, SynchronizedRandomCrop, visualize_batch
import torchvision.transforms as transforms

# 配置参数
config = {
    'save_dir': './checkpoints/net_v0_voc2012',
    'logs_dir': './logs/net_v0_voc2012',
    'best_checkpoint': './checkpoints/net_v0_voc2012/net_v0_voc2012_best.pth',
    'out_img_dir': './output/net_v0_voc2012',
    'comment': "net_v0_voc2012",
    'accumulation_steps': 1,
    'num_classes': 21,
    'input_shape': (224, 224),
    'cls_weights': np.ones([21], np.float32),
    'train_batch': 1,
    'train_epoch': 1000,
    'num_workers': 1,
    'learning_rate': 1e-4,
    'warmup_epochs': 10,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 100,
    'dataset_path': r'D:\deeplearning\segformer\VOCdevkit',  # 数据集路径
    'loss_function': 'FocalLoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False,  # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义数据增强
transform_sync = SynchronizedRandomCrop(size=(224, 224))  # 确保这个方法正确同步裁剪图像和标签
transform_img = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])


# 获取训练数据加载器
def get_train_dataloader():
    dataset = VOC2012SegmentationDataset(
        root=config['dataset_path'],
        image_set='train',
        transform=transform_sync,
        target_transform=transform_img
    )
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 获取验证数据加载器
def get_val_dataloader():
    dataset = VOC2012SegmentationDataset(
        root=config['dataset_path'],
        image_set='val',
        transform=transform_sync,
        target_transform=transform_img
    )
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 创建数据加载器
train_loader = get_train_dataloader()
val_loader = get_val_dataloader()

if __name__ == '__main__':
    for batch in train_loader:
        visualize_batch(batch)
        break
