import argparse
import os
from tqdm import tqdm
import random
import csv

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from core.checkpoint import load_checkpoint
from core.dataset import EvalDataset
from core.flags import Flags
from core.utils import set_random_seed
from core.builder import get_model


def main(parser):
    is_cuda = torch.cuda.is_available()

    # load checkpoint.
    ckpt = load_checkpoint(parser.checkpoint, cuda=is_cuda)

    # load config from checkpoint.
    config = Flags(ckpt["configs"]).get()

    # fix random seed.
    set_random_seed(config.seed)

    device = torch.device("cuda" if is_cuda else "cpu")

    dummy_gt = "\\sin " * parser.max_sequence  # set maximum inference sequence
    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]

    # TODO: Define & Apply test transforms.
    tokenizer = ckpt["tokenizer"]
    transform = config.data.test.transforms
    test_dataset = EvalDataset(test_data, tokenizer, crop=False, transform=transform, rgb=config.data.rgb)
    test_data_loader = DataLoader(
        test_dataset, batch_size=parser.batch_size, shuffle=False, num_workers=config.train_config.num_workers, collate_fn=test_dataset.collate_fn,
    )
    print(
        "[+] Data\n", "The number of test samples : {}\n".format(len(test_dataset)),
    )

    model = get_model(config, tokenizer).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    results = []
    for d in tqdm(test_data_loader):
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)

        output = model(input, expected, False, 0.0)
        decoded_values = output.transpose(1, 2)
        _, sequence = torch.topk(decoded_values, 1, dim=1)
        sequence = sequence.squeeze(1)
        sequence_str = [tokenizer.decode(sequence_, do_eval=True) for sequence_ in sequence]
        for path, predicted in zip(d["img_name"], sequence_str):
            results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", dest="checkpoint", default="./log/attention_50/best_score.pth", type=str, help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence", dest="max_sequence", default=300, type=int, help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", default=8, type=int, help="batch size when doing inference",
    )

    eval_dir = os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/")
    file_path = os.path.join(eval_dir, "eval_dataset/input.txt")
    parser.add_argument(
        "--file_path", dest="file_path", default=file_path, type=str, help="file path when doing inference",
    )

    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "submit")
    parser.add_argument(
        "--output_dir", dest="output_dir", default=output_dir, type=str, help="output directory",
    )

    parser = parser.parse_args()
    main(parser)
