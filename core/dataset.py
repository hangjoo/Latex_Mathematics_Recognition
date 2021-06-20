import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
from albumentations.pytorch.transforms import ToTensorV2

from .utils import get_transforms


def split_gt(groundtruth, proportion=1.0, test_percent=None, max_seq_len=None):
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data = []
        for line in fd:
            img_path, gt = line.split("\t")
            img_path = img_path.strip()
            gt = gt.strip()
            if max_seq_len is None:
                data.append([img_path, gt])
            elif len(gt.split(" ")) < max_seq_len:
                data.append([img_path, gt])
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]

    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, transform=None, rgb=3):
        """
        Args
            data: A list that includes an image name and raw latex text. E.g) [["/{img_path}/train_00001.jpg", "4 \\times 7 = 2 8"], ...]
            tokenizer: A Tokenizer class instance. Used for converting token to id or contrary.
            transform: Pytorch transforms to apply on images.
            rgb: If set 3, image would be loaded as 3 channels(RGB), else if grayscale.
        """
        super(TrainDataset, self).__init__()
        self.transform = get_transforms(transform)
        self.rgb = rgb
        self.tokenizer = tokenizer
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": sent,
                    "encoded": [
                        self.tokenizer.token_to_id[self.tokenizer.START_TOKEN],
                        *self.tokenizer.encode(sent),
                        self.tokenizer.token_to_id[self.tokenizer.END_TOKEN],
                    ],
                },
            }
            for p, sent in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        if self.rgb:  # RGB
            image = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        else:  # Grayscale
            image = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)

        # rotate 90 degrees clockwise if aspect ratio is smaller than 0.65.
        aspect_ratio = image.shape[1] / image.shape[0]
        if aspect_ratio < 0.65:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # apply transforms.
        if self.transform:
            image = self.transform(image=image)["image"]

        if not self.rgb:
            image /= 255.

        # to tensor(channels, height, width).
        image = ToTensorV2()(image=image)["image"]

        return {"path": item["path"], "truth": item["truth"], "image": image}

    @staticmethod
    def collate_fn(data):
        max_len = max([len(d["truth"]["encoded"]) for d in data])
        # Padding with -1, will later be replaced with the PAD token
        padded_encoded = [d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1] for d in data]
        return {
            "path": [d["path"] for d in data],
            "image": torch.stack([d["image"] for d in data], dim=0),
            "truth": {"text": [d["truth"]["text"] for d in data], "encoded": torch.tensor(padded_encoded)},
        }


class EvalDataset(Dataset):
    def __init__(self, data, tokenizer, transform=None, rgb=3):
        super(EvalDataset, self).__init__()
        self.transform = get_transforms(transform)
        self.rgb = rgb
        self.tokenizer = tokenizer
        self.data = [
            {
                "path": p,
                "img_name": img_name,
                "truth": {
                    "text": sent,
                    "encoded": [
                        self.tokenizer.token_to_id[self.tokenizer.START_TOKEN],
                        *self.tokenizer.encode(sent),
                        self.tokenizer.token_to_id[self.tokenizer.END_TOKEN],
                    ],
                },
            }
            for p, img_name, sent in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]

        if self.rgb:  # RGB
            image = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        else:  # Grayscale
            image = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)

        # rotate 90 degrees clockwise if aspect ratio is smaller than 0.65.
        aspect_ratio = image.shape[0] / image.shape[1]
        if aspect_ratio < 0.65:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # apply transforms.
        if self.transform:
            image = self.transform(image=image)["image"]

        if not self.rgb:
            image = image / 255.

        # to tensor(channels, height, width).
        image = ToTensorV2()(image=image)["image"]

        return {"path": item["path"], "img_name": item["img_name"], "truth": item["truth"], "image": image}

    @staticmethod
    def collate_fn(data):
        max_len = max([len(d["truth"]["encoded"]) for d in data])
        # Padding with -1, will later be replaced with the PAD token
        padded_encoded = [d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1] for d in data]
        return {
            "path": [d["path"] for d in data],
            "img_name": [d["img_name"] for d in data],
            "image": torch.stack([d["image"] for d in data], dim=0),
            "truth": {"text": [d["truth"]["text"] for d in data], "encoded": torch.tensor(padded_encoded)},
        }


def dataset_loader(config, tokenizer):
    # Read data
    train_data, valid_data = [], []
    if config.data.random_split:
        for i, path in enumerate(config.data.train.path):
            prop = 1.0
            if len(config.data.dataset_proportions) > i:
                prop = config.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, test_percent=config.data.test_proportions, max_seq_len=config.train_config.max_seq_len)
            train_data += train
            valid_data += valid
    else:
        for i, path in enumerate(config.data.train.path):
            prop = 1.0
            if len(config.data.dataset_proportions) > i:
                prop = config.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(config.data.valid.path):
            valid = split_gt(path, max_seq_len=config.train_config.max_seq_len)
            valid_data += valid

    train_transform = config.data.train.transforms
    valid_transform = config.data.valid.transforms if not config.data.random_split else train_transform

    # Load data
    train_dataset = TrainDataset(train_data, tokenizer, transform=train_transform, rgb=config.data.rgb)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
        num_workers=config.train_config.num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataset = TrainDataset(valid_data, tokenizer, transform=valid_transform, rgb=config.data.rgb)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=config.train_config.num_workers,
        collate_fn=valid_dataset.collate_fn,
    )

    return train_loader, valid_loader
