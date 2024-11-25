import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义类
CLASSES = (
    'industrial area',
    'paddy field',
    'irrigated field',
    'dry cropland',
    'garden land',
    'arbor forest',
    'shrub forest',
    'park',
    'natural meadow',
    'artificial meadow',
    'river',
    'urban residential',
    'lake',
    'pond',
    'fish pond',
    'snow',
    'bareland',
    'rural residential',
    'stadium',
    'square',
    'road',
    'overpass',
    'railway station',
    'airport',
    'unlabeled'
)

# 定义调色板
PALETTE = [
    [200, 0, 0],  # industrial area
    [0, 200, 0],  # paddy field
    [150, 250, 0],  # irrigated field
    [150, 200, 150],  # dry cropland
    [200, 0, 200],  # garden land
    [150, 0, 250],  # arbor forest
    [150, 150, 250],  # shrub forest
    [200, 150, 200],  # park
    [250, 200, 0],  # natural meadow
    [200, 200, 0],  # artificial meadow
    [0, 0, 200],  # river
    [250, 0, 150],  # urban residential
    [0, 150, 200],  # lake
    [0, 200, 250],  # pond
    [150, 200, 250],  # fish pond
    [250, 250, 250],  # snow
    [200, 200, 200],  # bareland
    [200, 150, 150],  # rural residential
    [250, 200, 150],  # stadium
    [150, 150, 0],  # square
    [250, 150, 150],  # road
    [250, 150, 0],  # overpass
    [250, 200, 250],  # railway station
    [200, 150, 0],  # airport
    [0, 0, 0]  # unlabeled
]


class SegmentationDataset(Dataset):
    def __init__(self, features_dir, labels_dir, transform=None, target_size=(256, 256)):
        """
        初始化分割数据集
        :param features_dir: 特征图像的目录
        :param labels_dir: 标签图像的目录
        :param transform: 图像变换（应用于特征图像）
        :param target_size: 标签图像裁剪大小
        """
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
        self.label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
        self.image_files.sort()
        self.label_files.sort()
        self.label_transform = transforms.Compose([
            transforms.Resize(target_size),  # 调整特征图像大小
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        feature_file = self.image_files[idx]
        label_file = self.label_files[idx]

        feature_path = os.path.join(self.features_dir, feature_file)
        label_path = os.path.join(self.labels_dir, label_file)

        # 读取特征和标签图像
        feature = Image.open(feature_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # 对特征图像应用变换
        if self.transform:
            feature = self.transform(feature)
            label = self.label_transform(label)
        return feature, label
