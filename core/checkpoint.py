import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "lr": [],

    "grad_norm": [],
    "train_loss": [],
    "train_symbol_acc": [],
    "train_sent_acc": [],
    "train_wer": [],
    "train_score": [],

    "valid_loss": [],
    "valid_symbol_acc": [],
    "valid_sent_acc": [],
    "valid_wer": [],
    "valid_score": [],

    "model_state": {},
    "optim_state": {},
    "configs": None,
    "tokenizer": None,
}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix="", base_name=""):
    if base_name:
        filename = f"{base_name}.pth"
    else:
        filename = f"{checkpoint['epoch']:0>4}.pth"
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))


def write_tensorboard(
    writer,
    epoch,
    grad_norm,
    train_loss,
    train_symbol_accuracy,
    train_sentence_accuracy,
    train_wer,
    validation_loss,
    validation_symbol_accuracy,
    validation_sentence_accuracy,
    validation_wer,
    model,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_symbol_accuracy", train_symbol_accuracy, epoch)
    writer.add_scalar("train_sentence_accuracy", train_sentence_accuracy, epoch)
    writer.add_scalar("train_wer", train_wer, epoch)
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_symbol_accuracy", validation_symbol_accuracy, epoch)
    writer.add_scalar("validation_sentence_accuracy", validation_sentence_accuracy, epoch)
    writer.add_scalar("validation_wer", validation_wer, epoch)
    writer.add_scalar("grad_norm", grad_norm, epoch)

    for name, param in model.encoder.named_parameters():
        writer.add_histogram("encoder/{}".format(name), param.detach().cpu().numpy(), epoch)
        if param.grad is not None:
            writer.add_histogram("encoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch)

    for name, param in model.decoder.named_parameters():
        writer.add_histogram("decoder/{}".format(name), param.detach().cpu().numpy(), epoch)
        if param.grad is not None:
            writer.add_histogram("decoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch)
