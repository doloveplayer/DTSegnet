import os
import torch
from torch.utils.data import DataLoader
from dataset.load_data import SegmentationDataset
import torchvision.transforms as transforms

# 配置参数
config = {
    'save_dir': './checkpoints/net_v0',
    'logs_dir': './logs/net_v0',
    'best_checkpoint': './checkpoints/net_v0/net_v0_best.pth',
    'out_img_dir': './output/net_v0',
    'comment': "net_v0",
    'train_batch': 1,
    'train_epoch': 100,
    'num_workers': 1,
    'learning_rate': 3e-4,
    'warmup_epochs': 5,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 5,
    'train_img_path': r'E:\data\train\img_1024',  # 数据集路径
    'train_label_path': r'E:\data\train\mask_1024',
    'test_img_path': r'E:\data\test\img_1024',  # 数据集路径
    'test_label_path': r'E:\data\test\mask_1024',
    'loss_function': 'CrossEntropyLoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False,  # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义图像的预处理操作
transform = transforms.Compose([
    transforms.RandomResizedCrop(512),  # 随机裁剪并调整大小到 224x224
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])


def get_train_dataloader():
    dataset = SegmentationDataset(features_dir=config['train_img_path'], labels_dir=config['train_label_path'],
                                  transform=transform)  # 实例化数据集
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 创建数据集和数据加载器
def get_val_dataloader():
    dataset = SegmentationDataset(features_dir=config['test_img_path'], labels_dir=config['test_label_path'],
                                  transform=transform)  # 实例化数据集
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


train_loader = get_train_dataloader()  # 创建 DataLoader 实例
val_loader = get_val_dataloader()  # 创建 DataLoader 实例
