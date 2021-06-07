import os
import random
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
from albumentations.pytorch.transforms import ToTensorV2

from .utils import get_transforms


def split_gt(groundtruth, proportion=1.0, test_percent=None):
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data = []
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]

    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data


def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1] for d in data]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {"text": [d["truth"]["text"] for d in data], "encoded": torch.tensor(padded_encoded)},
    }


def collate_eval_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1] for d in data]
    return {
        "path": [d["path"] for d in data],
        "file_path": [d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {"text": [d["truth"]["text"] for d in data], "encoded": torch.tensor(padded_encoded)},
    }


class DefaultDataset(Dataset):
    def __init__(
        self, data, tokenizer, crop=False, transform=None, rgb=3,
    ):
        """
        Args
            data: A list that includes an image name and raw latex text. E.g) [["/{img_path}/train_00001.jpg", "4 \\times 7 = 2 8"], ...]
            tokenizer: A Tokenizer class instance. Used for converting token to id or contrary.
            transform: Pytorch transforms to apply on images.
            rgb: If set 3, image would be loaded as 3 channels(RGB), else if grayscale.
        """
        super(DefaultDataset, self).__init__()
        self.transform = get_transforms(transform)
        self.crop = crop
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

        # apply transforms.
        if self.transform:
            image = self.transform(image=image)["image"]

        # to tensor(channels, height, width).
        image = ToTensorV2()(image=image)["image"]

        return {"path": item["path"], "truth": item["truth"], "image": image}


class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self, groundtruth, token_to_id, id_to_token, crop=False, transform=None, rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path": p1,
                "truth": {"text": truth, "encoded": [self.token_to_id[START], *encode_truth(truth, self.token_to_id), self.token_to_id[END]]},
            }
            for p, p1, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "file_path": item["file_path"], "truth": item["truth"], "image": image}


def dataset_loader(config, tokenizer):
    # Read data
    train_data, valid_data = [], []
    if config.data.random_split:
        for i, path in enumerate(config.data.train.path):
            prop = 1.0
            if len(config.data.dataset_proportions) > i:
                prop = config.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, config.data.test_proportions)
            train_data += train
            valid_data += valid

        # Load data
        train_dataset = DefaultDataset(train_data, tokenizer, crop=config.data.crop, transform=config.data.train.transforms, rgb=config.data.rgb)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train_config.batch_size, shuffle=True, num_workers=config.train_config.num_workers, collate_fn=collate_batch,
        )

        valid_dataset = DefaultDataset(valid_data, tokenizer, crop=config.data.crop, transform=config.data.train.transforms, rgb=config.data.rgb)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.train_config.batch_size, shuffle=False, num_workers=config.train_config.num_workers, collate_fn=collate_batch,
        )
    else:
        for i, path in enumerate(config.data.train.path):
            prop = 1.0
            if len(config.data.dataset_proportions) > i:
                prop = config.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(config.data.valid.path):
            valid = split_gt(path)
            valid_data += valid

        # Load data
        train_dataset = DefaultDataset(train_data, tokenizer, crop=config.data.crop, transform=config.data.train.transforms, rgb=config.data.rgb)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train_config.batch_size, shuffle=True, num_workers=config.data.train_config.num_workers, collate_fn=collate_batch,
        )

        valid_dataset = DefaultDataset(valid_data, tokenizer, crop=config.data.crop, transform=config.data.valid.transforms, rgb=config.data.rgb)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.train_config.batch_size, shuffle=False, num_workers=config.data.train_config.num_workers, collate_fn=collate_batch,
        )

    return train_loader, valid_loader
