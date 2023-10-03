#!/usr/bin/env python

from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import os

# from scalex.metrics import silhouette_score, batch_entropy_mixing_score
from .plot import plot_heatmap, plot_umap, sort_index_by_classes, plot_paired_umap


# from .utils import get_attention_maps, default
from torch.utils.data import Subset, DataLoader


class Monitor(Callback):
    def __init__(self, dm, metric="euclidean"):
        """
        # monitor function to do three things
        # 1. Monitor the attention maps
        # 2. Monitor the latent
        # 3. Monitor the general feature embedding
        """
        super().__init__()

        self.dataset = dm.dataset
        val_dl = dm.val_dataloader()
        self.example_array = next(iter(val_dl))
        self.example_array["cell_type"] = self.dataset.le.inverse_transform(
            self.example_array["cell_type"]
        )
        self.sort_by_labels()

        self.metric = metric

    def sort_by_labels(self):
        indices = sort_index_by_classes(self.example_array["cell_type"])
        for k, v in self.example_array.items():
            if k == "batch":
                if self.dataset.n_batch > 1:
                    v = self.dataset.batch_le.inverse_transform(v)
                else:
                    v = np.array(v)
            self.example_array[k] = v[indices]

    def to_device(self, device="cpu"):
        for k, v in self.example_array.items():
            if not isinstance(v, np.ndarray):
                self.example_array[k] = v.to(device)

    def on_train_epoch_end(self, trainer, pl_module):
        self.show_umap(pl_module, show=False)
        self.show_corr(pl_module, show=False)

    def save_checkpoint(self, trainer):
        trainer.save_checkpoint(
            os.path.join(
                trainer.logger.log_dir,
                "checkpoints",
                "epoch={}-step={}.ckpt".format(
                    trainer.current_epoch, trainer.global_step
                ),
            )
        )

    @torch.no_grad()
    def get_output(self, pl_module):
        atac_embeds, rna_embeds = pl_module(
            self.example_array["atac"], self.example_array["rna"]
        )

        atac_embeds = atac_embeds.detach().cpu().numpy()
        rna_embeds = rna_embeds.detach().cpu().numpy()
        concat_embeds = np.concatenate([atac_embeds, rna_embeds])
        concat_labels = np.concatenate(
            [self.example_array["cell_type"], self.example_array["cell_type"]]
        )
        domain_labels = np.array(
            ["atac"] * len(atac_embeds) + ["rna"] * len(rna_embeds)
        )
        obs = pd.DataFrame(
            [concat_labels, domain_labels], index=["cell_type", "modality"]
        ).T
        paired = AnnData(concat_embeds, obs=obs)

        return paired

    @torch.no_grad()
    def show_umap(
        self, pl_module, metric=None, show=True, norm=False, score=False, **kwargs
    ):
        metric = self.metric if metric is None else metric
        device = pl_module.device
        self.to_device(device)
        if not show:
            experiment = pl_module.logger.experiment
            # current_step = pl_module.global_step
            current_step = pl_module.current_epoch

        paired = self.get_output(pl_module)

        # fig = plot_umap(paired, color=paired.obs.columns, metric=metric, show=False, **kwargs)
        fig = plot_paired_umap(
            paired, color=paired.obs.columns, metric=metric, show=False, **kwargs
        )
        if score:
            print(compute_metrics(paired=paired, use_rep="X_umap"), flush=True)
        if show:
            plt.title("umap")
            plt.show()
            plt.close()
        else:
            experiment.add_figure("umap", fig, current_step)

        # fig = plot_paired_scatter(paired.obsm['X_umap'])
        # if show:
        #     plt.title('umap_paired')
        #     plt.show()
        #     plt.close()
        # else:
        #     experiment.add_figure('umap_paired', fig, current_step)

        # norm
        # paired.X = paired.X / np.linalg.norm(paired.X, axis=-1, keepdims=True)

        # fig = plot_umap(paired, color=paired.obs.columns, metric=metric, show=False, **kwargs)
        # if score:
        #     print(compute_metrics(paired=paired, use_rep='X_umap'), flush=True)
        # if show:
        #     plt.title('umap_norm')
        #     plt.show()
        #     plt.close()
        # else:
        #     experiment.add_figure('umap_norm', fig, current_step)

        # fig = plot_paired_scatter(paired.obsm['X_umap'])
        # if show:
        #     plt.title('umap_paired_norm')
        #     plt.show()
        #     plt.close()
        # else:
        #     experiment.add_figure('umap_paired_norm', fig, current_step)

    @torch.no_grad()
    def show_corr(self, pl_module, show=True, metric=None, norm=False):
        metric = self.metric if metric is None else metric
        device = pl_module.device
        self.to_device(device)
        if not show:
            experiment = pl_module.logger.experiment
            current_step = pl_module.global_step

        paired = self.get_output(pl_module)
        if norm:
            paired.X = paired.X / np.linalg.norm(paired.X, axis=-1, keepdims=True)

        X = pairwise_distances(
            paired.X, metric=metric
        )  # [:len(atac_embeds), len(atac_embeds):]
        fig = plot_heatmap(
            X,
            obs_names=paired.obs["cell_type"],
            var_names=paired.obs["cell_type"],
            sort=False,
            cmap="RdBu",
        )

        if show:
            plt.title("corr")
            plt.show()
            plt.close()
        else:
            experiment.add_figure("corr", fig, current_step)

    @torch.no_grad()
    def show_ipot(self, pl_module, show=True):
        pass

    @torch.no_grad()  # affected by model
    def show_predict(self, pl_module, show=True):  # takes 1s
        """
        Show training process in on_train_epoch_end
        """

        if not show:
            experiment = pl_module.logger.experiment
            global_step = pl_module.global_step

        for name, array in self.example_array.items():
            pred = (
                pl_module(array[0].to(pl_module.device))
                .to("cpu")
                .detach()
                .numpy()
                .squeeze()
            )

            fig = plot_heatmap(
                pred[:, self.index[name]["gene"]],
                obs_names=self.obs_names[name],
                vmax=5.5,
            )
            if show:
                plt.title("predicted/{}".format(name))
                plt.show()
            else:
                experiment.add_figure("predicted/{}".format(name), fig, current_epoch)

            score = [
                scipy.stats.pearsonr(pred[:, i], array[1][:, i])[0]
                for i in range(array[1].shape[1])
            ]
            g = sns.displot(score)
            g.set_axis_labels("Frequency", "Pearson coef")
            plt.xlim(-1, 1)
            plt.ylim(0, 500)
            if show:
                plt.show()
            else:
                experiment.add_figure(
                    "pearson_coef/{}".format(name), plt.gcf(), current_epoch
                )

    @torch.no_grad()
    def show_attention_maps(self, pl_module, show=True):  # takes 41.6s
        """ """
        # get attention_maps
        _ = pl_module(self.example_input_array[0])

        fig = plot_maps(
            attn_maps,
            n_col=self.n_sample,
            title=self.example_input_names,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            # vmax=0.1
        )

        if show:
            plt.show()
        else:
            pl_module.logger.experiment.add_figure(
                "attention/" + name, fig, pl_module.current_epoch
            )
