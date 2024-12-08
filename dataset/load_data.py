import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resized_crop

# Function to map label indices to colors based on the PALETTE
def label_to_color_image(label):
    label = label.numpy()  # Convert to numpy for easier indexing
    color_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(len(PALETTE_pots)):
        color_image[label == i] = PALETTE_pots[i]
    return color_image

# Function to visualize a batch of images and labels
def visualize_batch(batch):
    features, labels = batch

    feature = features[0]  # (C, H, W) tensor
    label = labels[0]  # (H, W) tensor

    feature = feature.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    feature = np.clip(feature, 0, 1)  # Ensure the image is in range [0, 1]

    label_color = label_to_color_image(label)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(feature)
    ax[0].set_title("Feature Image")
    ax[0].axis('off')

    ax[1].imshow(label_color)
    ax[1].set_title("Label Image")
    ax[1].axis('off')

    plt.show()

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

CLASSES_pots = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE_pots = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]


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


class PotsdamDataset(Dataset):
    def __init__(self, features_dir, labels_dir, transform=None, target_transform=None):
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
        self.target_transform = target_transform
        self.image_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
        self.label_files = [f for f in os.listdir(labels_dir) if f.endswith('.png')]
        # print(len(self.image_files))
        # print(len(self.label_files))
        self.image_files.sort()
        self.label_files.sort()

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

        # 同步裁剪和调整
        if self.transform:
            img, label = self.transform(feature, label)

        # 转换为张量和归一化
        if self.target_transform:
            feature = self.target_transform(feature)

        label = torch.tensor(np.array(label), dtype=torch.long)
        unique_label = torch.unique(label)
        print(f"Unique label in this data: {unique_label.cpu().numpy()}")

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
        label[label == 255] = 0  # 忽略255类 白色当作背景

        # unique_label = torch.unique(label)
        # print(f"Unique label in this data: {unique_label.cpu().numpy()}")

        return img, label


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
