import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resized_crop

vod_dict = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}
fbp_dict = {0: 'industrial area', 1: 'paddy field', 2: 'irrigated field', 3: 'dry cropland', 4: 'garden land',
            5: 'arbor forest',
            6: 'shrub forest', 7: 'park', 8: 'natural meadow', 9: 'artificial meadow', 10: 'river',
            11: 'urban residential',
            12: 'lake', 13: 'pond', 14: 'fish pond', 15: 'snow', 16: 'bareland', 17: 'rural residential', 18: 'stadium',
            19: 'square', 20: 'road', 21: 'overpass', 22: 'railway station', 23: 'airport', 24: 'unlabeled'}

fbp_Palette_Dict = {0: [200, 0, 0], 1: [0, 200, 0], 2: [150, 250, 0], 3: [150, 200, 150], 4: [200, 0, 200],
                    5: [150, 0, 250],
                    6: [150, 150, 250], 7: [200, 150, 200], 8: [250, 200, 0], 9: [200, 200, 0], 10: [0, 0, 200],
                    11: [250, 0, 150],
                    12: [0, 150, 200], 13: [0, 200, 250], 14: [150, 200, 250], 15: [250, 250, 250], 16: [200, 200, 200],
                    17: [200, 150, 150], 18: [250, 200, 150], 19: [150, 150, 0], 20: [250, 150, 150], 21: [250, 150, 0],
                    22: [250, 200, 250], 23: [200, 150, 0], 24: [0, 0, 0]}

postdam_dict = {0: 'ImSurf', 1: 'Building', 2: 'LowVeg', 3: 'Tree', 4: 'Car', 5: 'Clutter'}

postdam_Palette_Dict = {0: [255, 255, 255], 1: [0, 0, 255], 2: [0, 255, 255], 3: [0, 255, 0], 4: [255, 204, 0],
                        5: [255, 0, 0]}


class SynchronizedRandomCrop:
    def __init__(self, size):
        """
        初始化同步裁剪类
        :param size: 输出裁剪图像和标签的目标尺寸
        """
        self.size = size

    def __call__(self, img, label):
        """
        对图像和标签应用同步裁剪
        :param img: 输入图像
        :param label: 输入标签
        :return: 裁剪后的图像和标签
        """
        # 获取随机裁剪参数
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(1.0, 1.0)
        )

        # 同步裁剪图像和标签
        img = resized_crop(img, i, j, h, w, self.size)
        label = resized_crop(label, i, j, h, w, self.size)

        return img, label

class PotsdamDataset(Dataset):
    def __init__(self, features_dir, labels_dir, transform=None, target_transform=None):
        """
        Initialize the dataset with feature and label directories.
        :param features_dir: Directory containing feature images.
        :param labels_dir: Directory containing label images.
        :param transform: Transform to apply to feature images.
        :param target_transform: Transform to apply to label images.
        """
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = [f for f in os.listdir(features_dir) if f.endswith('.tif')]
        self.label_files = [f for f in os.listdir(labels_dir) if f.endswith('.tif')]
        print(len(self.image_files), len(self.label_files))
        self.image_files.sort()
        self.label_files.sort()

    def __len__(self):
        return len(self.image_files)

    def rgb_to_grayscale(self, rgb_mask):
        """
        Convert the RGB mask to a single-channel grayscale mask.
        :param rgb_mask: The RGB mask image.
        :return: Grayscale mask (single-channel tensor).
        """
        grayscale_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)

        # Map RGB values to label indices using postdam_Palette_Dict
        for class_idx, color in postdam_Palette_Dict.items():
            mask = np.all(rgb_mask == color, axis=-1)
            grayscale_mask[mask] = class_idx

        return grayscale_mask

    def __getitem__(self, idx):
        feature_file = self.image_files[idx]
        label_file = self.label_files[idx]

        feature_path = os.path.join(self.features_dir, feature_file)
        label_path = os.path.join(self.labels_dir, label_file)

        feature = Image.open(feature_path).convert("RGB")
        label_rgb = Image.open(label_path).convert("RGB")

        if self.transform:
            feature, label_rgb = self.transform(feature, label_rgb)

        label_rgb_np = np.array(label_rgb)
        label_grayscale = self.rgb_to_grayscale(label_rgb_np)

        label = torch.tensor(label_grayscale, dtype=torch.long)

        if self.target_transform:
            feature = self.target_transform(feature)

        unique_label = torch.unique(label)
        print(f"Unique labels in this batch: {unique_label.cpu().numpy()}")

        return feature, label, label_rgb_np


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

        # 初始化类别统计字典
        self.class_counts = {class_name: 0 for class_name in vod_dict.keys()}

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
        label[label == 255] = 0  # 忽略255类，白色当作背景

        # 更新类别统计
        unique, counts = torch.unique(label, return_counts=True)
        for class_idx, count in zip(unique, counts):
            class_name = [key for key, value in vod_dict.items() if value == class_idx.item()]
            if class_name:
                self.class_counts[class_name[0]] += count.item()

        return img, label

    def get_class_counts(self):
        return self.class_counts


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
