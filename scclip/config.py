import torch
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import List, Union
import os

from pytorch_lightning.utilities.argparse import (
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)


# class BaseConfig(PretrainedConfig):
# from dataclasses import dataclass, field
# @dataclass

# from pytorch_lightning.strategies import DDPStrategy
trainer_defaults = {
    "accelerator": "gpu",
    "devices": 1,
    "strategy": "ddp_find_unused_parameters_false",  # 'strategy': 'dp', #DDPStrategy(find_unused_parameters=True), # , #'ddp',
    "auto_select_gpus": True,
    # 'max_steps': 30000,
    "max_epochs": 100,
    "gradient_clip_val": 1,  # 5
    "num_sanity_val_steps": 0,
    "log_graph": True,
    # 'default_root_dir': os.path.dirname(os.path.dirname(__file__)) + '/results/clip/',
    "default_root_dir": os.path.join(os.path.expanduser("~"), "results/"),
    # 'default_root_dir': os.path.join(os.path.expanduser('~'), 'output/'),
    # 'weights_summary':"full",
    # 'deterministic':True,
    "check_val_every_n_epoch": 10,
    "enable_progress_bar": False,
    "precision": 16,
}

trainer_m1 = {
    "accelerator": "cpu",
    "devices": 1,
    "strategy": "dp",  # 'strategy': 'dp', #DDPStrategy(find_unused_parameters=True), # , #'ddp',
    "auto_select_gpus": True,
    # 'max_steps': 300,
    "max_epochs": 1,
    "gradient_clip_val": 1,  # 5
    "num_sanity_val_steps": 0,
    "log_graph": True,
    "default_root_dir": os.path.dirname(os.path.dirname(__file__)) + "/results/clip/",
    # 'weights_summary':"full",
    # 'deterministic':True,
    "check_val_every_n_epoch": 1,
    "precision": 16,
    # 'fast_dev_run': True
}


class PretrainedConfig(PretrainedConfig):
    @classmethod
    def from_argparse_args(cls, args, **kwargs):  # -> Any:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser):  # -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls):  # -> Namespace:
        return parse_env_variables(cls)

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):  # -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)


def get_optim_config(optim_type="mae_pretrain"):
    configs = {
        "mae_pretrain": {
            "optimizer": "AdamW",
            "learning_rate": 1.5e-4,
            "weight_decay": 0.05,
            "momentum": (0.9, 0.95),
            "schedule": "cosine",
            "warmup_epochs": 40,
            "warmup_steps": 2500,
        },
        "mae_finetune": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.05,
            "momentum": (0.9, 0.999),
            "schedule": "cosine",
            "warmup_epochs": 5,
            "warmup_steps": 2500,
            "training_epochs": 100,
        },
        "vit_pretrain": {
            "optimizer": "Adam",
            "learning_rate": 3e-3,
            "weight_decay": 0.3,
            "warmup_steps": 10000,
            "schedule": "cosine",
        },
        "vit_finetune": {},
        "clip_pretrain": {
            "optimizer": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 0.2,
            "momentum": (0.9, 0.98),
            "schedule": "cosine",
            "warmup_steps": 2000,
        },
        "clip_finetune": {},
        "test_pretrain": {
            "optimizer": "AdamW",
            "learning_rate": 1.5e-4,
            "weight_decay": 0.05,
            "momentum": (0.9, 0.95),
            "schedule": "cosine",
            "warmup_epochs": 40,
            "warmup_steps": 10000,
        },
        "test2_pretrain": {
            "optimizer": "AdamW",
            "learning_rate": 1e-5,
            "weight_decay": 0.05,
            "momentum": (0.9, 0.95),
            "schedule": "cosine",
            "warmup_epochs": 40,
            "warmup_steps": 2500,
        },
    }
    return configs[optim_type]


def get_model_config(model_size):
    configs = {
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 1024,
            "decoder_num_hidden_layers": 16,
            "decoder_intermediate_size": 4096,
            # patch_size:int=64, # to adjust
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "decoder_num_attention_heads": 16,
            "decoder_hidden_size": 512,
            "decoder_num_hidden_layers": 8,
            "decoder_intermediate_size": 2048,
            # patch_size:int=64, # to adjust
        },
        "small": {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "decoder_num_attention_heads": 8,
            "decoder_hidden_size": 256,
            "decoder_num_hidden_layers": 6,
            "decoder_intermediate_size": 512,
            # 'hidden_dropout_prob':0.0,
            # 'attention_probs_dropout_prob':0.0,
            # patch_size:int=64, # to adjust
        },
        "mini": {
            "hidden_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "decoder_num_attention_heads": 4,
            "decoder_hidden_size": 64,
            "decoder_num_hidden_layers": 3,
            "decoder_intermediate_size": 256,
            # patch_size:int=64, # to adjust
        },
        "test": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 32,
            "decoder_num_attention_heads": 1,
            "decoder_hidden_size": 8,
            "decoder_num_hidden_layers": 1,
            "decoder_intermediate_size": 32,
            # patch_size:int=64, # to adjust
        },
    }
    return configs[model_size]
