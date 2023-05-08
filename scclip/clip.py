import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import os

import numpy as np
import copy
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling, ModelOutput
from transformers import PreTrainedModel


from .lightning import LitModule
from .vit import ViTConfig, ViTModel

import time



# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale, requires_grad=requires_grad)

    def forward(self, atac_embeds, rna_embeds):
        # normalized features
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_atac = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale
        logits_per_rna = logits_per_atac.T

        loss = clip_loss(logits_per_atac) 

        return loss, logits_per_atac #, logits_per_rna 


def kl_div(mu, var):
    return kl_divergence(
        Normal(mu, var.sqrt()),
        Normal(torch.zeros_like(mu),torch.ones_like(var))
    ).sum(dim=1).mean()


def rna_output(x):
    return torch.log1p(x.softmax(dim=-1) * 1e4)

    
def sinkhorn(out):
    Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def compute_similarities(I_emb, T_emb):
    sim_ii, sim_tt = I_emb @ I_emb.t(), T_emb @ T_emb.t()
    sim_it, sim_ti = I_emb @ T_emb.t(), T_emb @ I_emb.t()
    return sim_ii, sim_tt, sim_it, sim_ti




@dataclass
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for atac-rna similarity.
        logits_per_atac:(`torch.FloatTensor` of shape `(atac_batch_size, rna_batch_size)`):
            The scaled dot product scores between `atac_embeds` and `rna_embeds`. This represents the atac-rna
            similarity scores.
        logits_per_rna:(`torch.FloatTensor` of shape `(rna_batch_size, atac_batch_size)`):
            The scaled dot product scores between `rna_embeds` and `atac_embeds`. This represents the rna-atac
            similarity scores.
        rna_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The rna embeddings obtained by applying the projection layer to the pooled output of [`CLIPRnaModel`].
        atac_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The atac embeddings obtained by applying the projection layer to the pooled output of [`CLIPAtacModel`].
        rna_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPRnaModel`].
        atac_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPAtacModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_atac: torch.FloatTensor = None
    logits_per_rna: torch.FloatTensor = None
    atac_embeds: torch.FloatTensor = None
    rna_embeds: torch.FloatTensor = None
    atac_outputs: BaseModelOutputWithPooling = None
    rna_outputs: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["rna_model_output", "atac_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )





class CLIPModel(PreTrainedModel, LitModule):

    def __init__(
        self,
        config,
        atac_config,
        rna_config,
    ):
        super().__init__(config)
        self.args = config
        self.save_hyperparameters()

        self.atac_model = ViTModel(atac_config)
        self.rna_model = ViTModel(rna_config)

        self.atac_embed_dim = self.atac_model.config.hidden_size
        self.rna_embed_dim = self.rna_model.config.hidden_size


        self.atac_projection = nn.Linear(self.atac_embed_dim, config.projection_dim, bias=False)
        self.rna_projection = nn.Linear(self.rna_embed_dim, config.projection_dim, bias=False)

        self.criterion = CLIPLoss(self.config.logit_scale, requires_grad=self.config.requires_grad)


        print(f'atac_num_patches: {self.atac_model.embeddings.num_patches}', flush=True)
        print(f'rna_num_patches: {self.rna_model.embeddings.num_patches}', flush=True)


    def forward(
        self,
        atac_values=None,
        rna_values=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        atac_outputs = self.atac_model(
            atac_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        atac_embeds = atac_outputs[1]  # pooler_output
        if self.atac_projection:
            atac_embeds = self.atac_projection(atac_embeds)


        # rna_embeds = self.rna_model(rna_values)
        rna_outputs = self.rna_model(
            rna_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        rna_embeds = rna_outputs[1]
        if self.rna_projection:
            rna_embeds = self.rna_projection(rna_embeds)

                
        # loss = logits_per_atac = logits_per_rna = None
        atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)
        
        return atac_embeds, rna_embeds
        # loss, logits_per_atac, logits_per_rna = self.criterion(
        #     atac_embeds, rna_embeds
        # )

        # return CLIPOutput(
        #     loss=loss,
        #     logits_per_atac=logits_per_atac,
        #     logits_per_rna=logits_per_rna,
        #     rna_embeds=rna_embeds,
        #     atac_embeds=atac_embeds,
        #     rna_outputs=rna_outputs,
        #     atac_outputs=atac_outputs,
        # )


    def _step(self, batch, batch_idx, mode):
        # output = self(batch['atac'], batch['rna'])
        # loss = output.loss
        # log_dict = {f'clip_loss/{mode}': loss}
        atac_embeds, rna_embeds = self(batch['atac'], batch['rna'])
        loss, similarity = self.criterion(atac_embeds, rna_embeds)
        
        with torch.no_grad():
            # similarity = output.logits_per_atac
            batch_size = similarity.shape[0]
            acc_x = torch.sum(torch.argmax(similarity, dim=1) == torch.arange(batch_size).to(similarity.device)) / batch_size
            acc_y = torch.sum(torch.argmax(similarity, dim=0) == torch.arange(batch_size).to(similarity.device)) / batch_size
            foscttm_x = (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
            foscttm_y = (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
            matchscore_x = similarity.softmax(dim=1).diag().mean().item()
            matchscore_y = similarity.softmax(dim=0).diag().mean().item()
            
            acc = (acc_x + acc_y)/2
            foscttm = (foscttm_x + foscttm_y)/2
            matchscore = (matchscore_x + matchscore_y)/2
            
        log_dict = {
            f'acc/{mode}': acc,
            f'foscttm/{mode}': foscttm,
            f'matchscore/{mode}': matchscore,
            f'loss/{mode}': loss,
        }

        # logit_scale learnable
        if self.config.requires_grad:
            log_dict.update({'logit_scale': self.criterion.logit_scale})

        if mode == 'predict':
            return atac_embeds, rna_embeds, log_dict

        # log_dict.update({f'loss/{mode}': loss})
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)


    def get_rna_features(
        self,
        rna_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        rna_outputs = self.rna_model(
            rna_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        rna_features = rna_outputs[1]
        # return pooled_output

        if self.rna_projection:
            rna_features = self.rna_projection(rna_features)
        
        # # normalized features
        # rna_features = rna_features / rna_features.norm(dim=-1, keepdim=True)

        return rna_features


    def get_atac_features(
        self,
        atac_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        atac_outputs = self.atac_model(
            atac_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        atac_features = atac_outputs[1]  # pooled_output
        # return pooled_output

        if self.atac_projection:
            atac_features = self.atac_projection(atac_features)

        # normalized features
        # atac_features = atac_features / atac_features.norm(dim=-1, keepdim=True)

        return atac_features
