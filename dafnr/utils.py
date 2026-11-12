import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> torch.Generator:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int):

    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str, overrides=None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[leaf] = parsed
    return cfg


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
