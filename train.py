import os
import argparse
import time
import shutil
import yaml
from tqdm import tqdm
from psutil import virtual_memory

import numpy as np
import torch
import torch.nn as nn

from core.dataset import dataset_loader
from core.builder import get_model, get_loss, get_optimizer, get_scheduler
from core.Tokenizer import Tokenizer

from core.flags import Flags
from core.checkpoint import default_checkpoint, load_checkpoint, save_checkpoint
from core.utils import set_random_seed
from core.metrics import word_error_rate, sentence_acc

import wandb


def main(config_file):
    """
    Train math formula recognition model
    """
    config = Flags(config_file).get()

    # init wandb logger
    wandb_params = config.wandb._asdict()
    wandb_config = {
        "model": config.model.type,
        "loss": config.loss.type,
        "optimizer": config.optimizer.type,
        "transforms": config.data.train.transforms,
        "rgb": config.data.rgb,
        "batch_size": config.train_config.batch_size,
        "num_epochs": config.train_config.num_epochs,
        "teacher_forcing": config.train_config.teacher_forcing_ratio,
        "max_grad_norm": config.train_config.max_grad_norm,
        "random_seed": config.seed,
    }
    wandb.init(config=wandb_config, **wandb_params)

    # set random seed
    set_random_seed(config.seed)

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(config.model.type, device))

    # print system environments
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    # load checkpoint and print result
    if config.checkpoint != "":
        ckpt = load_checkpoint(config.checkpoint, cuda=is_cuda)
        print(
            "[+] Checkpoint\n",
            f"Resuming from epoch : {ckpt['epoch']}\n",
        )
    else:
        ckpt = default_checkpoint

    # get data
    if ckpt["tokenizer"]:
        tokenizer = ckpt["tokenizer"]
    else:
        tokenizer = Tokenizer(config.data.token_paths)
    train_loader, valid_loader = dataset_loader(config, tokenizer)
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_loader.dataset)),
        "The number of validation samples : {}\n".format(len(valid_loader.dataset)),
        "The number of classes : {}\n".format(len(tokenizer.token_to_id)),
    )

    # get model, loss
    model = get_model(config, tokenizer).to(device)
    if ckpt["model_state"]:
        model.load_state_dict(ckpt["model_state"])
    model.train()

    criterion = get_loss(config)
    params_to_optimise = [param for param in model.parameters() if param.requires_grad]
    print(
        "[+] Model\n",
        f"Type: {config.model.type}\n",
        f"Model parameters: {format(sum(p.numel() for p in params_to_optimise), ',')}\n",
    )
    print("[+] Loss")
    for k, v in config.loss._asdict():
        print(f" {k}: {v}")
    print()

    # get optimizer
    optimizer = get_optimizer(config, params_to_optimise)
    if ckpt["optim_state"]:
        optimizer.load_state_dict(ckpt["optim_state"])
    # for param_group in optimizer.param_groups:
    #     param_group["initial_lr"] = config.optimizer.lr
    print("[+] Optimizer")
    for k, v in config.optimizer._asdict():
        print(f" {k}: {v}")
    print()

    # get scheduler
    scheduler = get_scheduler(config, optimizer)
    if scheduler:
        print("[+] Scheduler")
        for k, v in config.scheduler._asdict():
            print(f" {k}: {v}")
        print()

    # log
    os.makedirs(config.prefix, exist_ok=True)
    if not os.path.exists(config.prefix):
        os.makedirs(config.prefix)
    log_file = open(os.path.join(config.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(config.prefix, "train_config.yaml"))
    wandb.save(glob_str=os.path.join(config.prefix, "train_config.yaml"))
    wandb.watch(models=model, criterion=criterion, log="all")

    # train model
    best_score = 0
    for epoch_i in range(ckpt["epoch"], config.train_config.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch_i + 1, end=config.train_config.num_epochs, epoch=epoch_i + 1, pad=len(str(config.train_config.num_epochs)),
        )

        # train
        train_result = run_epoch(
            tokenizer,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch_text,
            config.train_config.teacher_forcing_ratio,
            config.train_config.max_grad_norm,
            device,
            train=True,
        )

        # validation
        valid_result = run_epoch(
            tokenizer,
            valid_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch_text,
            config.train_config.teacher_forcing_ratio,
            config.train_config.max_grad_norm,
            device,
            train=False,
        )

        # epoch results.
        epoch_lr = scheduler["scheduler"].get_lr()
        if isinstance(epoch_lr, list):
            epoch_lr = epoch_lr[-1]
        grad_norm = train_result["grad_norm"]

        train_loss = train_result["loss"]
        train_symbol_acc = train_result["correct_symbols"] / train_result["total_symbols"]
        train_sent_acc = train_result["sent_acc"] / train_result["num_sent_acc"]
        train_wer = train_result["wer"] / train_result["num_wer"]
        train_score = 0.9 * train_sent_acc + 0.1 * (1 - train_wer)

        valid_loss = valid_result["loss"]
        valid_symbol_acc = valid_result["correct_symbols"] / valid_result["total_symbols"]
        valid_sent_acc = valid_result["sent_acc"] / valid_result["num_sent_acc"]
        valid_wer = valid_result["wer"] / valid_result["num_wer"]
        valid_score = 0.9 * valid_sent_acc + 0.1 * (1 - valid_wer)

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # update checkpoint.
        ckpt["epoch"] = epoch_i + 1
        ckpt["lr"].append(epoch_lr)
        ckpt["grad_norm"].append(grad_norm)
        ckpt["train_loss"].append(train_loss)
        ckpt["train_symbol_acc"].append(train_symbol_acc)
        ckpt["train_sent_acc"].append(train_sent_acc)
        ckpt["train_wer"].append(train_wer)
        ckpt["train_score"].append(train_score)
        ckpt["valid_loss"].append(valid_loss)
        ckpt["valid_symbol_acc"].append(valid_symbol_acc)
        ckpt["valid_sent_acc"].append(valid_sent_acc)
        ckpt["valid_wer"].append(valid_wer)
        ckpt["valid_score"].append(valid_score)
        ckpt["model_state"] = model.state_dict()
        ckpt["optim_state"] = optimizer.state_dict()
        ckpt["configs"] = config_dict
        ckpt["tokenizer"] = tokenizer

        # save checkpoint.
        save_checkpoint(ckpt, prefix=config.prefix)

        # save best score checkpoint.
        if valid_score > best_score:
            best_score = valid_score
            save_checkpoint(ckpt, dir=".", prefix=config.prefix, base_name="best_score")
            wandb.save(glob_str=os.path.join(config.prefix, "best_score.pth"))

        # log write.
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch_i % config.train_config.print_interval == 0 or epoch_i == config.train_config.num_epochs - 1:
            output_string = (
                f"{epoch_text}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Train Symbol Accuracy = {train_symbol_acc:.4f}, "
                f"Train Sentence Accuracy = {train_sent_acc:.4f}, "
                f"Train WER = {train_wer:.4f}, "
                f"Train Score = {valid_score:.4f}"
                f"Valid Loss = {valid_loss:.4f}, "
                f"Valid Symbol Accuracy = {valid_symbol_acc:.4f}, "
                f"Valid Sentence Accuracy = {valid_sent_acc:.4f}, "
                f"Valid WER = {valid_wer:.4f}, "
                f"Valid Score = {valid_score:.4f} "
                f"lr = {epoch_lr:.4e} "
                f"(time elapsed {elapsed_str})"
            )
            print(output_string)
            log_file.write(output_string + "\n")
            wandb.log(
                {
                    "epoch": epoch_i + 1,
                    "lr": epoch_lr,
                    "train/symbol_acc": train_symbol_acc,
                    "train/sentence_acc": train_sent_acc,
                    "train/wer": train_wer,
                    "train/loss": train_loss,
                    "train/score": train_score,
                    "valid/symbol_acc": valid_symbol_acc,
                    "valid/sentence_acc": valid_sent_acc,
                    "valid/wer": valid_wer,
                    "valid/loss": valid_loss,
                    "valid/score": valid_score,
                }
            )


def run_epoch(
    tokenizer, data_loader, model, criterion, optimizer, scheduler, epoch_text, teacher_forcing_ratio, max_grad_norm, device, train=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"), total=len(data_loader.dataset), dynamic_ncols=True, leave=False,
    ) as pbar:
        for d in data_loader:
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = tokenizer.token_to_id[tokenizer.PAD_TOKEN]

            output = model(input, expected, train, teacher_forcing_ratio)

            decoded_values = output.transpose(1, 2)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)

            loss = criterion(decoded_values, expected[:, 1:])

            if train:
                optim_params = [p for param_group in optimizer.param_groups for p in param_group["params"]]
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients, it returns the total norm of all parameters
                grad_norm = nn.utils.clip_grad_norm_(optim_params, max_norm=max_grad_norm)
                grad_norms.append(grad_norm)

                # cycle
                optimizer.step()
                if scheduler and scheduler["type"] == "iter":
                    scheduler["scheduler"].step()

            losses.append(loss.item())

            expected_str = [tokenizer.decode(expected_, do_eval=True) for expected_ in expected]
            sequence_str = [tokenizer.decode(sequence_, do_eval=True) for sequence_ in sequence]
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

    if train and scheduler and scheduler["type"] == "epoch":
        scheduler["scheduler"].step()

    expected = [tokenizer.decode(expected_, do_eval=False) for expected_ in expected]
    sequence = [tokenizer.decode(sequence_, do_eval=False) for sequence_ in sequence]
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc,
    }
    if train:
        try:
            result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
        except:
            result["grad_norm"] = np.mean(grad_norms)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", dest="config_file", default="configs/Default_SATRN.yaml", type=str, help="Path of configuration file",
    )
    parser = parser.parse_args()
    main(parser.config_file)
