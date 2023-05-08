#!/usr/bin/env python
"""
# Author: Lei Xiong
# Contact: jsxlei@gmail.com
# File Name: RegulationNetwork.py
# Created Time : Tue 25 Jan 2022 07:20:34 PM EST
# Description:

"""

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping

from scclip.data import MixDataModule
from scclip.clip import CLIPModel
from scclip.vit import ViTConfig
from scclip.callback import Monitor
from scclip.config import PretrainedConfig, get_model_config

import os
import argparse


from pathlib import Path
HOME = Path.home()
print('Start', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clip')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', type=str, default=None)
    # DataModule
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--backed", action='store_true', default=False)
    parser.add_argument("--split", default=0.9)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dist', type=str, default=None)
    parser.add_argument('--peak_dist', type=int, default=10_000)
    parser.add_argument('--experiment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mod', type=str, default='multiome')
    
    # Module
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--use_imputed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--requires_grad', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--norm_features', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--version", type=str, default='v2')
    
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--fast_dev_run", action='store_true', default=False)
    
    # parser.add_argument('--version', type=str, default='v2')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--no_scalex', action='store_true', default=False)
    parser.add_argument('--use_val', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--test_data', type=str, default='ADRD')
    parser.add_argument('--cell_type', type=str, default='cell_type')
    parser.add_argument('--use_seq', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--temp', type=float, default=2.6592)
    parser.add_argument('--num_patches', type=int, default=128)

    # parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    seed_everything(args.seed)


    if args.checkpoint is not None:
        model = CLIPModel.load_from_checkpoint(args.checkpoint) 
        args.default_root_dir = args.checkpoint.split('lightning_logs/')[0]
    else:   
        dm = MixDataModule(
            data_dir = args.data_dir,
            modality=args.mod, 
            batch_size=args.batch_size,
            linked = args.dist,
            split = args.split,
            n_top_genes = args.n_top_genes,
            binary = args.binary,
            use_seq = args.use_seq,
        )


        model_config = get_model_config('small')
        atac_config = ViTConfig(**{
            'modality':'atac', 
            'num_patches':args.num_patches, 
            'feature_size': dm.dataset.mdata.mod['atac'].shape[1],
            'attention_probs_dropout_prob': args.dropout,
            'hidden_dropout_prob': args.dropout,
            **model_config
        })


        rna_config = ViTConfig(**{
            'modality':'rna', 
            'num_patches':args.num_patches, 
            'attention_probs_dropout_prob': args.dropout,
            'hidden_dropout_prob': args.dropout,
            'feature_size':dm.dataset.mdata.mod['rna'].shape[1], 
            **model_config
        })

        config = PretrainedConfig(
            projection_dim=args.projection_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            requires_grad=args.requires_grad,
            logit_scale=args.temp,
        )

        model = CLIPModel(
            config, 
            atac_config=atac_config, 
            rna_config=rna_config,
        ) 

        # print(model, flush=True)

        # out_dir
        args.default_root_dir = f'results/{args.data_dir}/vit_{args.temp}_{args.requires_grad}'
        os.makedirs(args.default_root_dir, exist_ok=True)
        print('default_root_dir:', args.default_root_dir, flush=True)


        # trainer
        logger = TensorBoardLogger(save_dir=args.default_root_dir, default_hp_metric=False, version='')
        callbacks = [
            Monitor(dm), 
            # LearningRateMonitor(logging_interval='epoch'),
            # EarlyStopping(monitor="loss/val", patience=10)
        ] 
        trainer = Trainer(
            callbacks=callbacks,
            accelerator = 'gpu',
            devices = 1,
            gradient_clip_val = 5,
            num_sanity_val_steps = 0,
            logger=logger,
            max_steps=args.max_steps,
            fast_dev_run=args.fast_dev_run,
        )

        # fit
        trainer.fit(model, dm) 





       
        