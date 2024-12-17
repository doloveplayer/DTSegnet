import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from dataset.load_data import TinyImageNet, SynchronizedRandomCrop
import torchvision.transforms as transforms

# 配置参数
config = {
    'save_dir': './checkpoints/net_v0_tiny_imgnet',
    'logs_dir': './logs/net_v0_tiny_imgnet',
    'best_checkpoint': './checkpoints/net_v0_tiny_imgnet/net_v0_tiny_imgnet_best.pth',
    'out_img_dir': './output/net_v0_tiny_imgnet',
    'pre_train': 'checkpoints/net_v0_tiny_imgnet/checkpoint_epoch_340.pth',
    'comment': "net_v0_tiny_imgnet",
    'accumulation_steps': 4,
    'num_classes': 200,
    'input_shape': (64, 64),
    'cls_weights': np.ones([200], np.float32),
    'train_batch': 128,
    'train_epoch': 1000,
    'num_workers': 1,
    'learning_rate': 1e-3,
    'warmup_epochs': 10,
    'weight_decay': 0,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 100,
    'dataset_path': r'D:\deeplearning\data\tiny-imagenet-200',  # 数据集路径
    'loss_function': 'CELoss',  # 损失函数
    'optimizer': 'AdamW',  # 优化器
    'fp16': False,  # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义数据增强
transform_img = transforms.Compose([
    transforms.RandomCrop((64, 64)),  # 确保这个方法正确同步裁剪图像和标签
    transforms.RandAugment(num_ops=9,
                           magnitude=9),
    transforms.ToTensor(),  # 将图像转换为Tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])

transform_val = transforms.Compose([
    transforms.RandomCrop((64, 64)),  # 确保这个方法正确同步裁剪图像和标签
    transforms.ToTensor(),  # 将图像转换为Tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])


# 获取训练数据加载器
def get_train_dataloader():
    dataset = TinyImageNet(config["dataset_path"], train=True, transform=transform_img)
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 获取验证数据加载器
def get_val_dataloader():
    dataset = TinyImageNet(config["dataset_path"], train=False, transform=transform_val)
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 创建数据加载器
train_loader = get_train_dataloader()
val_loader = get_val_dataloader()

if __name__ == '__main__':
    # 加载数据集
    root = config["dataset_path"]  # 更改为你本地数据集的路径
    dataset = TinyImageNet(root=root, train=True, transform=transform_img)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


    # 可视化一批图像
    def visualize_batch(dataloader):
        data_iter = iter(dataloader)
        images, labels = next(data_iter)

        # 获取标签和对应的类名
        labels = dataset.return_label(labels)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2行4列的网格
        axes = axes.ravel()

        for i in range(8):
            axes[i].imshow(images[i].permute(1, 2, 0))  # 转换为HWC格式显示
            axes[i].set_title(f"Label: {labels[i]}")
            axes[i].axis('off')

        plt.show()


    # 调用可视化函数
    visualize_batch(dataloader)
