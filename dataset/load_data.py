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
    def __init__(self, root, image_set='train', transform=None, target_size=(256, 256)):
        """
        :param root: VOC2012 数据集路径（包含 VOCdevkit 文件夹）
        :param image_set: 数据集划分（'train' 或 'val'）
        :param transform: 预处理（可选）
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform

        # 图像和标签的文件路径
        self.image_dir = os.path.join(self.root, 'VOC2012', 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'VOC2012', 'SegmentationClass')

        # 划分txt文件的路径
        self.image_set_file = os.path.join(self.root, 'VOC2012', 'ImageSets', 'Segmentation', f'{image_set}.txt')

        self.label_transform = transforms.Compose([
            transforms.Resize(target_size),  # 调整特征图像大小
        ])

        # 从txt文件中读取所有的图像ID
        with open(self.image_set_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取图像ID
        img_id = self.img_ids[idx]

        # 获取图像和标签的文件路径
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        label_path = os.path.join(self.label_dir, f'{img_id}.png')

        # 读取图像和标签
        img = Image.open(img_path).convert('RGB')  # 转换为RGB模式
        label = Image.open(label_path)  # 标签是PNG格式，通常是单通道

        # 应用预处理（如果有）
        if self.transform:
            img = self.transform(img)
            label = torch.tensor(np.array(self.label_transform(label)), dtype=torch.long)  # 标签转换为tensor
            label[label == 255] = 0  # 将255替换为-1，表示忽略

        unique_label = torch.unique(label)
        # print(f"Unique label in this data: {unique_label.cpu().numpy()}")

        return img, label
