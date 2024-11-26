import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset.load_data import VOC2012SegmentationDataset
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation

# 配置参数
config = {
    'save_dir': './checkpoints/net_v0_voc2012',
    'logs_dir': './logs/net_v0_voc2012',
    'best_checkpoint': './checkpoints/net_v0_voc2012/net_v0_voc2012_best.pth',
    'out_img_dir': './output/net_v0_voc2012',
    'comment': "net_v0_voc1012",
    'num_classes': 21,
    'input_shape': (256, 256),
    'cls_weights': np.ones([21], np.float32),
    'train_batch': 4,
    'train_epoch': 100,
    'num_workers': 1,
    'learning_rate': 1e-5,
    'warmup_epochs': 5,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 5,
    'dataset_path': r'D:\deeplearning\segformer\VOCdevkit',  # 数据集路径
    'loss_function1': 'CELoss',  # 损失函数
    'loss_function2': 'DiceLoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False,  # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
# 定义图像的预处理操作
transform = transforms.Compose([
    transforms.RandomResizedCrop(256),  # 随机裁剪并调整大小到 224x224
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])


def get_train_dataloader():
    # 加载VOC2012数据集
    print("data path : ", config['dataset_path'])
    dataset = VOC2012SegmentationDataset(root=config['dataset_path'], image_set='train', transform=transform)
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 创建数据集和数据加载器
def get_val_dataloader():
    print("data path : ", config['dataset_path'])
    dataset = VOC2012SegmentationDataset(root=config['dataset_path'], image_set='val', transform=transform)
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


train_loader = get_train_dataloader()  # 创建 DataLoader 实例
val_loader = get_val_dataloader()  # 创建 DataLoader 实例
