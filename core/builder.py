# models
from .build.model import Attention
from .build.model import SATRN

# losses
from torch.nn import CrossEntropyLoss

# optimizers
from torch.optim import SGD, Adam, adadelta

# schedulers
from torch.optim.lr_scheduler import StepLR
from .build.scheduler import CircularLRBeta

model_list = {
    "Attention": Attention,
    "SATRN": SATRN
}

loss_list = {
    "CrossEntropyLoss": CrossEntropyLoss,
}

optim_list = {
    "SGD": SGD,
    "Adam": Adam,
    "adadelta": adadelta,
}

iter_scheduler_list = {
    "CircularLRBeta": CircularLRBeta,
}
epoch_scheduler_list = {
    "StepLR": StepLR,
}


def get_model(config, tokenizer, *args, **kwagrs):
    model_name = config.model.type

    if model_name in model_list:
        model = model_list[model_name](config, tokenizer)
    else:
        raise NotImplementedError

    return model


def get_loss(config, *args, **kwagrs):
    loss_name = config.loss.type

    if loss_name in loss_list:
        loss_fn = loss_list[loss_name](*args, **kwagrs)
    else:
        raise NotImplementedError

    return loss_fn


def get_optimizer(config, params, *args, **kwargs):
    optim_args = config.optimizer._asdict()
    optim_name = optim_args.pop("type")

    if optim_name in optim_list:
        optimizer = optim_list[optim_name](params=params, *args, **kwargs, **optim_args)
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(config, optimizer, *args, **kwargs):
    scheduler_args = config.scheduler._asdict()
    scheduler_name = scheduler_args.pop("type")

    if scheduler_name in iter_scheduler_list:
        scheduler = iter_scheduler_list[scheduler_name](optimizer=optimizer, *args, **kwargs, **scheduler_args)
        scheduler_type = "iter"
    elif scheduler_name in epoch_scheduler_list:
        scheduler = epoch_scheduler_list[scheduler_name](optimizer=optimizer, *args, **kwargs, **scheduler_args)
        scheduler_type = "epoch"
    elif not scheduler_name:
        return None
    else:
        raise NotImplementedError

    return {"scheduler": scheduler, "type": scheduler_type}
