import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dataset.load_data import PotsdamDataset, SynchronizedRandomCrop
from collections import Counter

# 配置参数
config = {
    'save_dir': './checkpoints/net_v0_pots',
    'logs_dir': './logs/net_v0_pots',
    'best_checkpoint': './checkpoints/net_v0/net_v0_pots_best.pth',
    'out_img_dir': './output/net_v0_pots',
    'comment': "net_v0_pots",
    'accumulation_steps': 1,
    'num_classes': 6,
    'input_shape': (512, 512),
    'cls_weights': [1.12611100e-07, 1.55022628e-07, 1.60020193e-07, 2.31721509e-07,
                    9.99998404e-01, 9.36419051e-07],
    'train_batch': 1,
    'train_epoch': 1000,
    'num_workers': 1,
    'learning_rate': 1e-4,
    'warmup_epochs': 10,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'save_interval': 5,
    'patience': 100,
    'train_img_path': r'D:\deeplearning\GeoSeg-main\data\Potsdam\train\images_512',  # 数据集路径
    'train_label_path': r'D:\deeplearning\GeoSeg-main\data\Potsdam\train\masks_512',
    'test_img_path': r'D:\deeplearning\GeoSeg-main\data\Potsdam\test\images_512',  # 数据集路径
    'test_label_path': r'D:\deeplearning\GeoSeg-main\data\Potsdam\test\masks_512',
    'loss_function': 'CELoss',  # 损失函数
    'optimizer': 'SGD',  # 优化器
    'fp16': False,  # 混合精度
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# 定义数据增强
transform_sync = SynchronizedRandomCrop(size=(512, 512))  # 确保这个方法正确同步裁剪图像和标签
transform_img = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])


# 获取训练数据加载器
def get_train_dataloader():
    dataset = PotsdamDataset(
        features_dir=config['train_img_path'],
        labels_dir=config['train_label_path'],
        transform=transform_sync,
        target_transform=transform_img
    )
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


# 获取验证数据加载器
def get_val_dataloader():
    dataset = PotsdamDataset(
        features_dir=config['test_img_path'],
        labels_dir=config['test_label_path'],
        transform=transform_sync,
        target_transform=transform_img
    )
    return DataLoader(dataset, batch_size=config['train_batch'], shuffle=True, num_workers=config['num_workers'],
                      persistent_workers=True, pin_memory=True)


train_loader = get_train_dataloader()  # 创建 DataLoader 实例
val_loader = get_val_dataloader()  # 创建 DataLoader 实例


# 计算每个类别的样本数量
def calculate_class_weights(train_loader, num_classes):
    class_counts = Counter()
    for images, labels, _ in train_loader:
        for label in labels:
            class_counts.update(label.view(-1).cpu().numpy())

    # 转换为 NumPy 数组
    class_counts = np.array([class_counts[i] for i in range(num_classes)])
    print("Class counts:", class_counts)

    # 计算类别权重
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights)
    print("Class weights:", class_weights)

    return class_weights


# 计算每个样本的权重
def calculate_sample_weights(train_loader, class_weights):
    sample_weights = []
    for images, labels, _ in train_loader:
        for label in labels:
            label_weights = class_weights[label.view(-1).cpu().numpy()]
            sample_weight = label_weights.mean()  # 你可以选择其他聚合方式，如中位数或最大值
            sample_weights.append(sample_weight)

    sample_weights = np.array(sample_weights)
    print("Sample weights:", sample_weights)
    return sample_weights


# 重新创建训练数据加载器
def get_train_dataloader_with_sampler(train_loader, sampler):
    dataset = train_loader.dataset
    return DataLoader(
        dataset=dataset,
        batch_size=config['train_batch'],
        sampler=sampler,
        num_workers=config['num_workers'],
        persistent_workers=True,
        pin_memory=True
    )


if __name__ == '__main__':
    class_weights = calculate_class_weights(train_loader, config['num_classes'])

    sample_weights = calculate_sample_weights(train_loader, class_weights)

    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # train_loader = get_train_dataloader_with_sampler(train_loader, sampler)
