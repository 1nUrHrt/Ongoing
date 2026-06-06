from .InteractionDataset import InteractionDataset
from .DrugDataset import DrugDataset
from EarlyStop import DefaultEarlyStop


import os
from typing import Literal
from .SubDrugDataset import SubDrugDataset
import Encoder as Encoder
import Classifier
import Optimizer
import LrScheduler
import Criterion
import EarlyStop

import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def get_encoder(config):
    model_type = config["type"]
    if model_type not in Encoder.__all__:
        raise NotImplementedError("Enocoder {} not supported.".format(model_type))
    return getattr(Encoder, model_type)(**config["params"])


def get_classifier(config):
    model_type = config["type"]
    if model_type not in Classifier.__all__:
        raise NotImplementedError("Classifier {} not supported.".format(model_type))
    return getattr(Classifier, model_type)(**config["params"])


def get_optimizer(config, model_parameters):
    optimizer_type = config["type"]
    if optimizer_type not in Optimizer.__all__:
        raise NotImplementedError("Optimizer {} not supported.".format(optimizer_type))
    return getattr(Optimizer, optimizer_type)(model_parameters, **config["params"])


def get_scheduler(config, optimizer):
    scheduler_type = config["type"]
    if scheduler_type not in LrScheduler.__all__:
        raise NotImplementedError(
            "LrScheduler {} not supported.".format(scheduler_type)
        )
    return getattr(LrScheduler, scheduler_type)(optimizer, **config["params"])


def get_criterion(config):
    criterion_type = config["type"]
    if criterion_type not in Criterion.__all__:
        raise NotImplementedError("Criterion {} not supported.".format(criterion_type))
    return getattr(Criterion, criterion_type)(**config["params"])


def get_early_stop(config):
    early_stop_type = config["type"]
    if early_stop_type not in EarlyStop.__all__:
        raise NotImplementedError("EarlyStop {} not supported.".format(early_stop_type))
    return getattr(EarlyStop, early_stop_type)(**config["params"])


def load_dataset(base_dir: str):

    def get_path(file_name):
        return os.path.join(base_dir, file_name)

    drug_set = DrugDataset(os.path.join(base_dir, "drug.csv"))

    train_set = SubDrugDataset(get_path("train_set.csv"), drug_set)
    val_set = SubDrugDataset(get_path("val_set.csv"), drug_set)
    test_set = SubDrugDataset(get_path("test_set.csv"), drug_set)
    train_itc = InteractionDataset(get_path("train.csv"))
    val_itc = InteractionDataset(get_path("val.csv"))
    test_itc = InteractionDataset(get_path("test.csv"))
    return train_set, train_itc, val_set, val_itc, test_set, test_itc


# COLORS = {
#     "info": "\033[92m",  # 绿色
#     "debug": "\033[94m",  # 蓝色
#     "warning": "\033[93m",  # 黄色
#     "error": "\033[91m",  # 红色
#     "reset": "\033[0m",
# }


# def wrapper_text(text, mode):
#     return f"{COLORS[mode]}{text}{COLORS['reset']}"


__all__ = [
    "InteractionDataset",
    "DrugDataset",
    "DefaultEarlyStop",
    "get_encoder",
    "get_classifier",
    "get_optimizer",
    "get_scheduler",
    "load_dataset",
    "SubDrugDataset",
    "Timer",
]
