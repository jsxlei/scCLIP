
r"""
Performance evaluation metrics
"""

from typing import Any, Mapping, Optional, TypeVar, Union, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scglue.metrics import (
    foscttm, 
    mean_average_precision,
    normalized_mutual_info,
    avg_silhouette_width,
    graph_connectivity,
    seurat_alignment_score,
    avg_silhouette_width_batch
)
import time
# from .typehint import RandomState
# from .utils import get_rs


def compute_metrics(paired=None, atac=None, rna=None, use_rep='X', is_paired=True, verbose=False):
    if isinstance(paired, AnnData):
        if use_rep != 'X':
            combined_latent = paired.obsm[use_rep]
        else:
            combined_latent = paired.X
        combined_cell_type = paired.obs['cell_type'].values
        combined_domain = paired.obs['modality'].values
        if 'batch' in paired.obs:
            combined_batch = paired.obs['batch'].values
        else:
            combined_batch = np.array(['batch']*paired.shape[0])
        
        atac_latent = combined_latent[:paired.shape[0]//2]
        rna_latent = combined_latent[paired.shape[0]//2:]
        # combined_uni = paired.
    else:
        combined_latent = np.concatenate([atac['latent'], rna['latent']])
        combined_cell_type = np.concatenate([atac['cell_type'], rna['cell_type']])
        combined_domain = np.concatenate([atac['domain'], rna['domain']])
        combined_batch = np.concatenate([atac['batch'], rna['batch']])
        atac_latent = atac['latent']
        rna_latent = rna['latent']
        # combined_uni = np.concatenate([atac['X'], rna['X']])

    # metrics = {
    #     "foscttm": np.concatenate(foscttm(atac_latent, rna_latent)).mean().item(),
    #     "mean_average_precision": mean_average_precision(combined_latent, combined_cell_type),
    #     # "normalized_mutual_info": normalized_mutual_info(combined_latent, combined_cell_type),
    #     "avg_silhouette_width": avg_silhouette_width(combined_latent, combined_cell_type),
    #     "graph_connectivity": graph_connectivity(combined_latent, combined_cell_type),
    #     "seurat_alignment_score": seurat_alignment_score(combined_latent, combined_domain, random_state=0),
    #     "avg_silhouette_width_domain": avg_silhouette_width_batch(combined_latent, combined_domain, combined_cell_type),
    #     # "neighbor_conservation": scglue.metrics.neighbor_conservation(combined_latent, combined_uni, combined_domain)
    # }
    # metrics = {}
    # import time
    if verbose: t0 = time.time()
    # t0 = time.time()
    metrics = {}
    if is_paired:
        metrics["foscttm"] = np.concatenate(foscttm(atac_latent, rna_latent)).mean().item() ; 
        if verbose: print("foscttm", time.time()-t0); t0 = time.time()
    metrics["mean_average_precision"] = mean_average_precision(combined_latent, combined_cell_type) ; 
    if verbose: print("map", time.time()-t0); t0 = time.time()
    # metrics["normalized_mutual_info"] = normalized_mutual_info(combined_latent, combined_cell_type); print("nmi", time.time()-t0); t0 = time.time()
    metrics["avg_silhouette_width"] = avg_silhouette_width(combined_latent, combined_cell_type) #; 
    if verbose: print("asw", time.time()-t0); t0 = time.time()
    metrics["graph_connectivity"] = graph_connectivity(combined_latent, combined_cell_type) #; 
    if verbose: print("gc", time.time()-t0); t0 = time.time()
    metrics["seurat_alignment_score"] = seurat_alignment_score(combined_latent, combined_domain, random_state=0) #; 
    if verbose: print("sas", time.time()-t0); t0 = time.time()
    metrics["avg_silhouette_width_domain"] = avg_silhouette_width_batch(combined_latent, combined_domain, combined_cell_type) #; 
    if verbose: print("aswd", time.time()-t0); t0 = time.time()
    if len(np.unique(combined_batch)) > 1:
        metrics['avg_silhouette_width_batch'] = avg_silhouette_width_batch(combined_latent, combined_batch, combined_cell_type)
    return metrics



import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: 

    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities 
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.gpus = gpus
        self.classes = classes
        self.knn_k = knn_k
        self.knn_t = knn_t

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if self.gpus > 0:
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, self.feature_bank, self.targets_bank, self.classes, self.knn_k, self.knn_t)
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)
    
    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)
            

def evaluate(prediction_test, sol_test, scores_path="scores/scores.txt"):
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "a+") as f:
        if type(prediction_test.X) != numpy.ndarray:
            X = prediction_test.X.toarray()
        else:
            X = prediction_test.X
        X = torch.tensor(X)

        Xsol = torch.tensor(sol_test.X.toarray())
        Xsol.argmax(1)
        # Order the columns of the prediction matrix so that the perfect prediction is the identity matrix
        X = X[:, Xsol.argmax(1)]

        labels = torch.arange(X.shape[0])
        forward_accuracy = (torch.argmax(X, dim=1) == labels).float().mean().item()
        backward_accuracy = (torch.argmax(X, dim=0) == labels).float().mean().item()
        avg_accuracy = 0.5 * (forward_accuracy + backward_accuracy)
        print("top1 forward acc:", forward_accuracy)
        print("top1 forward acc:", forward_accuracy, file=f)
        print("top1 backward acc:", backward_accuracy)
        print("top1 backward acc:", backward_accuracy, file=f)
        print("top1 avg acc:", avg_accuracy)
        print("top1 avg acc:", avg_accuracy, file=f)

        _, top_indexes_forward = X.topk(5, dim=1)
        _, top_indexes_backward = X.topk(5, dim=0)
        l_forward = labels.expand(5, X.shape[0]).T
        l_backward = l_forward.T
        top5_forward_accuracy = (
            torch.any(top_indexes_forward == l_forward, 1).float().mean().item()
        )
        top5_backward_accuracy = (
            torch.any(top_indexes_backward == l_backward, 0).float().mean().item()
        )
        top5_avg_accuracy = 0.5 * (top5_forward_accuracy + top5_backward_accuracy)

        print("top5 forward acc:", top5_forward_accuracy)
        print("top5 forward acc:", top5_forward_accuracy, file=f)
        print("top5 backward acc:", top5_backward_accuracy)
        print("top5 backward acc:", top5_backward_accuracy, file=f)
        print("top5 avg acc:", top5_avg_accuracy)
        print("top5 avg acc:", top5_avg_accuracy, file=f)

        logits_row_sums = X.clip(min=0).sum(dim=1)
        top1_competition_metric = (
            X.clip(min=0).diagonal().div(logits_row_sums).mean().item()
        )
        print("top1 competition metric:", top1_competition_metric)
        print("top1 competition metric:", top1_competition_metric, file=f)

        # For soft predictions, the competition score can be made equal to the forward accuracy (or backward accuracy) by
        # putting 1 at the max of each row (or each column) and 0 elsewhere
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        top1_competition_metric = (
            hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()
        )
        print("top1 competition metric for soft predictions:", top1_competition_metric)
        print(
            "top1 competition metric for soft predictions:",
            top1_competition_metric,
            file=f,
        )

        # FOSCTTM
        foscttm = (X > torch.diag(X)).float().mean().item()
        foscttm_x = (X > torch.diag(X)).float().mean(axis=1).mean().item()
        foscttm_y = (X > torch.diag(X)).float().mean(axis=0).mean().item()
        print("foscttm:", foscttm, "foscttm_x:", foscttm_x, "foscttm_y:", foscttm_y)
        print(
            "foscttm:",
            foscttm,
            "foscttm_x:",
            foscttm_x,
            "foscttm_y:",
            foscttm_y,
            file=f,
        )

