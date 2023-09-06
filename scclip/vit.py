import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTModel, PreTrainedModel, PretrainedConfig, ViTPreTrainedModel
from transformers.models.vit.modeling_vit import (
    ViTEncoder,
    ViTPooler,
    BaseModelOutputWithPooling,
    ViTLayer,
)  # , MaskedLMOutput
from transformers.utils import ModelOutput
from dataclasses import dataclass
from einops import rearrange


@dataclass
class MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cls_token: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


class ViTConfig(PretrainedConfig):
    model_type = "vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        feature_size=30111,
        num_patches=128,
        image_size=224,
        # patch_size=16,
        mask_ratio=0.2,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.feature_size = feature_size
        self.mask_ratio = mask_ratio
        # self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride


class MAEConfig(PretrainedConfig):
    model_type = "mae"

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = "gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        feature_size=None,  # new_added
        num_patches=128,
        # patch_size:int=64, # to adjust
        qkv_bias=True,
        decoder_num_attention_heads=8,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=2048,
        mask_ratio: float = 0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        feature_sizes = {"rna": 36601, "atac": 1154464}  # tabula 58870

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.feature_size = feature_size  # if feature_size is not None else feature_sizes[modality]# new added
        self.num_patches = num_patches
        # self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = PatchEmbeddings(config)

        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.num_patches = num_patches
        self.patch_size = self.patch_embeddings.patch_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        # batch_size, num_channels, height, width = pixel_values.shape
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(
            pixel_values
        )  # , interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, config):  # feature_size, patch_size, embed_dim):
        super().__init__()
        feature_size, num_patches, embed_dim = (
            config.feature_size,
            config.num_patches,
            config.hidden_size,
        )
        patch_size = math.ceil(
            feature_size / num_patches
        )  # ; print(patch_size); import pdb; pdb.set_trace()
        pad_size = num_patches * patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(
            x.shape[0], self.num_patches, self.patch_size
        )
        x = self.projection(x)
        return x


from copy import deepcopy
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoderOutput


class ViTDecoder(nn.Module):
    def __init__(self, config, patch_size):
        super().__init__()
        num_patches = config.num_patches
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            patch_size,
            bias=True,  # config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        # self.initialize_weights(num_patches)

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        hidden_states = self.decoder_embed(hidden_states)

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, head_mask=None, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(
                v
                for v in [logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# class ViTModel(ViTPreTrainedModel):
#     def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
#         super().__init__(config)
#         self.config = config


class ViTModel(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            # head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTMLM(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
        self.decoder = ViTDecoder(config, patch_size=self.vit.embeddings.patch_size)

        self.mask_ratio = config.mask_ratio
        self.config = config

        self.post_init()

    def forward(
        self,
        values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:]  # batch_size, num_patches, hidden_size

        reconstructed = self.decoder(sequence_output)[0].view(values.shape[0], -1)[
            :, : values.shape[1]
        ]

        reconstruction_loss = nn.functional.mse_loss(
            reconstructed, values, reduction="none"
        )
        masked_loss = None
        if bool_masked_pos is not None:
            mask = bool_masked_pos.repeat_interleave(self.patch_size)[: values.shape[1]]
            masked_loss = (reconstruction_loss * mask).sum() / (
                (mask.sum() + 1e-5) * values.shape[0]
            )
        else:
            masked_loss = reconstruction_loss.mean()
            # reconstruction_loss = nn.functional.mse_loss(reconstructed, values, reduction='none')
            # masked_loss = nn.functional.binary_cross_entropy_with_logits(reconstructed, values)

        # reconstructed = reconstructed.sigmoid()
        if not return_dict:
            output = (reconstructed,) + outputs[1:]

        return MaskedLMOutput(
            loss=masked_loss,
            logits=reconstructed,
            cls_token=outputs[0][:, 0],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _step(self, batch, batch_idx, mode):
        if self.mask_ratio > 0 and mode == "train":
            bool_masked_pos = torch.rand((1, self.config.num_patches)) < self.mask_ratio
            bool_masked_pos = bool_masked_pos.to(batch.device)
        else:
            bool_masked_pos = None
        output = self(batch, bool_masked_pos=bool_masked_pos)

        log_dict = {
            f"loss/{mode}": output.loss,
            # TODO: add back in pearson R calculation w/o regnet package
            # f'corr/{mode}': pearsonr(batch, output.logits).nanmean()
        }

        if mode == "predict":
            return {"latent": output.cls_token, "log_dict": log_dict}  # , output.logits

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return output.loss


if __name__ == "__main__":
    config = MAEConfig(feature_size=1000)
    model = ViTMLM(config)

    inputs = torch.randn(10, 1000)
    out = model(inputs)

    # import pdb; pdb.set_trace()
