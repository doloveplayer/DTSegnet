import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms.functional import resized_crop

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

class SynchronizedRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        # 获取随机裁剪参数
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        # 同步裁剪图像和标签
        img = resized_crop(img, i, j, h, w, self.size)
        label = resized_crop(label, i, j, h, w, self.size)
        return img, label

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
            transforms.RandomResizedCrop(256),  # 随机裁剪并调整大小到 224x224
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
            label = torch.tensor(np.array(self.label_transform(label)), dtype=torch.long)  # 标签转换为tensor
        # unique_label = torch.unique(label)
        # print(f"Unique label in this data: {unique_label.cpu().numpy()}")
        return feature, label


class VOC2012SegmentationDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(self.root, 'VOC2012', 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'VOC2012', 'SegmentationClass')

        self.image_set_file = os.path.join(self.root, 'VOC2012', 'ImageSets', 'Segmentation', f'{image_set}.txt')

        with open(self.image_set_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        label_path = os.path.join(self.label_dir, f'{img_id}.png')

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # 同步裁剪和调整
        if self.transform:
            img, label = self.transform(img, label)

        # 转换为张量和归一化
        if self.target_transform:
            img = self.target_transform(img)

        label = torch.tensor(np.array(label), dtype=torch.long)
        label[label == 255] = 0  # 忽略255类

        return img, label
