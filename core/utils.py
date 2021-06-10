import random
import numpy as np
import torch

import albumentations as A


def set_random_seed(seed):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(transforms):
    if transforms[0] == None:
        return None
    
    transform_list = [eval(f"A.{transform}") for transform in transforms]

    return A.Compose(transforms=transform_list)
