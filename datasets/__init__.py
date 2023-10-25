"""
Some codes are partially adapted from
https://github.com/AaltoML/generative-inverse-heat-dissipation/blob/main/scripts/datasets.py
"""

import pickle
import numpy as np
from PIL import Image
import blobfile as bf

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, MNIST

from datasets.lsun import LSUN
from datasets.quadratic import QuadraticDataset
from datasets.melbourne import MelbourneDataset
from datasets.gridwatch import GridwatchDataset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def data_scaler(data):
    return data * 2. - 1.


def data_inverse_scaler(data):
    return (data + 1.) / 2.

def get_dataset(config):
    if config.data.modality == '1D':
        if config.data.dataset == "Quadratic":
            dataset = QuadraticDataset(num_data=config.data.num_data,
                                    num_points=config.data.dimension,
                                    seed=42)
            test_dataset = QuadraticDataset(num_data=config.data.num_data,
                                            num_points=config.data.dimension,
                                            seed=43)
        elif config.data.dataset == "Melbourne":
            dataset = MelbourneDataset(phase='train')
            test_dataset = MelbourneDataset(phase='test')
        elif config.data.dataset == "Gridwatch":
            dataset = GridwatchDataset(phase='train', seed=87)
            test_dataset = GridwatchDataset(phase='test', seed=85)
        else:
            raise NotImplementedError(f"Unknown dataset: {config.data.dataset}")
    else:
        if config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose(
                [transforms.Resize(config.data.image_size), transforms.ToTensor()]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [transforms.Resize(config.data.image_size), transforms.ToTensor()]
            )

        # Downloading torchvision dataset is set to be False due to malfunctioning using multiprocessing
        if config.data.dataset == "CIFAR10":
            dataset = CIFAR10(
                './data/cifar10',
                train=True,
                download=False,
                transform=tran_transform,
            )
            test_dataset = CIFAR10(
                './data/cifar10_test',
                train=False,
                download=False,
                transform=test_transform,
            )

        elif config.data.dataset == "MNIST":
            mnist_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor()
                ]
            )
            dataset = MNIST(
                './data/mnist',
                train=True,
                download=False,
                transform=mnist_transform,
            )
            test_dataset = MNIST(
                './data/mnist_test',
                train=False,
                download=False,
                transform=test_transform,
            )

        elif config.data.dataset == "LSUN":
            train_folder = "{}_train".format(config.data.category)
            val_folder = "{}_val".format(config.data.category)

            if config.data.random_flip:
                dataset = LSUN(
                    root='./data/lsun',
                    classes=[train_folder],
                    transform=transforms.Compose(
                        [
                            transforms.Resize(config.data.image_size),
                            transforms.CenterCrop(config.data.image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                        ]
                    ),
                )
            else:
                dataset = LSUN(
                    root='./data/lsun',
                    classes=[train_folder],
                    transform=transforms.Compose(
                        [
                            transforms.Resize(config.data.image_size),
                            transforms.CenterCrop(config.data.image_size),
                            transforms.ToTensor(),
                        ]
                    ),
                )

            test_dataset = LSUN(
                root='./data/lsun',
                classes=[val_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        elif config.data.dataset == "FFHQ":
            dataset = load_data(data_dir='./data/ffhq_dataset/images1024x1024',
                                image_size=config.data.image_size,
                                random_flip=config.data.random_flip)

            test_dataset = load_data(data_dir='./data/ffhq_dataset/images1024x1024',
                                image_size=config.data.image_size,
                                random_flip=False)

        elif config.data.dataset == 'AFHQ':
            dataset = load_data(data_dir='./data/afhq',
                        image_size=config.data.image_size,
                        random_flip=config.data.random_flip)
            test_dataset = load_data(data_dir='./data/afhq',
                                image_size=config.data.image_size,
                                random_flip=False)

        else:
            dataset, test_dataset = None, None
    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def load_data(
        *, data_dir, image_size, class_cond=False, random_flip=True):
    """
    NOTE: Change to original function, returns the Pytorch dataloader, not a generator
    For a dataset, create a dataloader over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                                       label. If classes are not available and this is true, an
                                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        random_flip=random_flip
    )
    return dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1,
                 random_flip=True):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        # Random horizontal flip
        if self.random_flip:
            if np.random.rand() > 0.5:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution,
                  crop_x: crop_x + self.resolution]
        # Changed here so that not centered at zero
        # arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.astype(np.float32) / 255

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
