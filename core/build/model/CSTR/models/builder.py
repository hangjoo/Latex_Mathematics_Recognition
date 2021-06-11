import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import build_from_cfg
from .registry import MODELS


def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)

    return model
