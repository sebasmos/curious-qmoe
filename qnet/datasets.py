# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, fold_data, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(".png")])
        self.labels = self._load_labels(fold_data)

    def _load_labels(self, fold_data):
        # Match image names with their corresponding classes
        labels = {}
        for index, row in fold_data.iterrows():
            labels[row['image_name']] = row['class']  # Use 'class' from the fold_data
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image_name = os.path.basename(img_path)
        label = self.labels[image_name]  # Get the label from the dictionary
        return image, label  # Returning the image and its label

    
def build_dataset_fold(args, fold_dir, fold_data):
    transform = build_transform(is_train=False, args=args)  # Set False for evaluation
    dataset = ImageDataset(root_dir=fold_dir, fold_data=fold_data, transform=transform)

    print(f"Dataset built with {len(dataset)} images from {fold_dir}.")

    # Return DataLoader instead of dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return dataloader


def build_unlabeled_dataset_mnist(split_type, args):
    """
    Builds the dataset based on the split type (test, in this case).

    Parameters:
        split_type (str): Specifies the type of dataset (test).
        args: Additional arguments containing the path and transformation settings.

    Returns:
        dataset: The built dataset with image and filename.
    """
    transform = build_transform(split_type == 'train', args)

    dataset = UnlabeledImageDataset(args.data_path, transform=transform)

    print(f"{split_type.capitalize()} dataset built with {len(dataset)} images from {args.data_path}.")

    return dataset

def build_unlabeled_dataset(split_type, args):
    """
    Builds the dataset based on the split type (test, in this case).

    Parameters:
        split_type (str): Specifies the type of dataset (test).
        args: Additional arguments containing the path and transformation settings.

    Returns:
        dataset: The built dataset with image and filename.
    """
    transform = build_transform(split_type == 'train', args)

    # Only for the 'test' set, as per your description
    if split_type == 'test':
        root = os.path.join(args.data_path, 'test')
    else:
        raise ValueError(f"Unknown split_type: {split_type}. Only 'test' is supported in this scenario.")

    dataset = UnlabeledImageDataset(root, transform=transform)

    print(f"{split_type.capitalize()} dataset built with {len(dataset)} images from {root}.")

    return dataset
    
# Custom dataset for images without labels
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List of all image filenames in the root_dir
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image path
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])
        # Load image
        image = Image.open(img_name).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Return the image and its filename (instead of labels)
        return image, self.image_filenames[idx]

def specific_dataset(args):
    
    split_type = args.specific_data_path.split("/")[-1] 
    
    transform = build_transform(split_type == split_type, args)
    
    root = args.specific_data_path
    
    dataset = UnlabeledImageDataset(root, transform=transform)
    
    print(f"{split_type.capitalize()} dataset built with {len(dataset)} images from {root}.")

    return dataset


def build_datasetv2(args):
    """
    Builds the dataset based on the split type (train, val, or test).

    Parameters:
        split_type (str): Specifies the type of dataset ('train', 'val', 'test').
        args: Additional arguments containing the path and transformation settings.

    Returns:
        dataset: The built dataset.
    """

    split_type = args.specific_data_path.split("/")[-1] 
    print(split_type)
    
    transform = build_transform(split_type == split_type, args)
    
    root = args.specific_data_path
    
    dataset = datasets.ImageFolder(root, transform=transform)

    print(f"{split_type.capitalize()} dataset built with {len(dataset)} images from {root}.")

    return dataset


def build_dataset(split_type, args):
    """
    Builds the dataset based on the split type (train, val, or test).

    Parameters:
        split_type (str): Specifies the type of dataset ('train', 'val', 'test').
        args: Additional arguments containing the path and transformation settings.

    Returns:
        dataset: The built dataset.
    """
    transform = build_transform(split_type == 'train', args)

    # Set the appropriate folder based on split_type ('train', 'val', or 'test')
    if split_type == 'train':
        root = os.path.join(args.data_path, 'train')
    elif split_type == 'val':
        root = os.path.join(args.data_path, 'val')
    elif split_type == 'test':
        root = os.path.join(args.data_path, 'test')
    else:
        raise ValueError(f"Unknown split_type: {split_type}. Must be 'train', 'val', or 'test'.")

    dataset = datasets.ImageFolder(root, transform=transform)

    print(f"{split_type.capitalize()} dataset built with {len(dataset)} images from {root}.")

    return dataset

def build_dataset_old(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            # transforms.RandomResizedCrop(args.input_size),
            create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            interpolation='bicubic',
            # re_prob=args.reprob,
            # re_mode=args.remode,
            # re_count=args.recount,
            mean=mean,
            std=std,
            )
        ])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
