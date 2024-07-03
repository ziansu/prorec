# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch (Assembly-Source Code) DualEncoder model."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPooling
)
from transformers.models.roberta.modeling_roberta import (
    RobertaOutput
)
from . import LongelmConfig, LongelmModel
from .configuration_dualencoder import DualEncoderConfig


logger = logging.get_logger(__name__)

# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class CASPOutput(ModelOutput):  # TODO: modify this part later
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_source:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_assembly:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        source_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The source code embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        assembly_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The assembly embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        source_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        assembly_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_source: torch.FloatTensor = None
    logits_per_assembly: torch.FloatTensor = None
    source_embeds: torch.FloatTensor = None
    assembly_embeds: torch.FloatTensor = None
    source_model_output: BaseModelOutputWithPooling = None
    assembly_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["source_model_output", "assembly_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class DualEncoderModel(PreTrainedModel):
    config_class = DualEncoderConfig
    base_model_prefix = 'dual_encoder'

    def __init__(
        self,
        config: Optional[DualEncoderConfig] = None,
        assembly_model: Optional[PreTrainedModel] = None,
        source_model: Optional[PreTrainedModel] = None,
    ):
        if config is None and (assembly_model is None or source_model is None):
            raise ValueError("Either a configuration or an assembly model and a text model has to be provided")

        if config is None:
            config = DualEncoderConfig.from_assembly_source_configs(assembly_model.config, source_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise  ValueError(f"config: {config} has to be of type {self.config_class}")
            
        # initialize with config
        super().__init__(config)

        if assembly_model is None:
            config.assembly_config = LongelmConfig.from_dict(config.assembly_config)
            assembly_model = LongelmModel(config.assembly_config)
        
        if source_model is None:
            source_model = AutoModel.from_config(config.source_config)

        self.assembly_model = assembly_model
        # NOTE: remove projection head of codet5p-110m-embedding model (temp)
        if hasattr(source_model, 'encoder') and hasattr(source_model, 'proj'):
            self.source_model = source_model.encoder
        else:
            self.source_model = source_model

        # make sure that the individual model's config refers to the shared config
        # so that the udpates to the config will be synced
        self.assembly_model.config = self.config.assembly_config
        self.source_model.config = self.config.source_config

        self.assembly_embed_dim = config.assembly_config.hidden_size
        self.source_embed_dim = config.source_config.hidden_size
        self.projection_dim = config.projection_dim

        self.assembly_projection = nn.Linear(self.assembly_embed_dim, self.projection_dim, bias=False)
        self.source_projection = nn.Linear(self.source_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    def get_source_features(
        self,
        input_ids=None,
        attention_mask=None,
        # position_ids=None,
        # token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            source_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`)
        """
        source_outputs = self.source_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = source_outputs[0][:, 0, :]
        source_features = self.source_projection(pooled_output)

        return source_features
    
    def get_assembly_features(
        self,
        input_ids=None,
        attention_mask=None,
        graph_attention_mask=None,
        relative_node_positions=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            assembly_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`)
        """
        assembly_output = self.assembly_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_node_positions=relative_node_positions,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_outputs = assembly_output[1]
        assembly_features = self.assembly_projection(pooled_outputs)

        return assembly_features
    
    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        # source_position_ids: Optional[torch.LongTensor] = None,
        # source_token_type_ids: Optional[torch.LongTensor] = None,

        assembly_input_ids: Optional[torch.LongTensor] = None,
        assembly_attention_mask: Optional[torch.Tensor] = None,
        assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assembly_outputs = self.assembly_model(
            input_ids=assembly_input_ids,
            attention_mask=assembly_attention_mask,
            graph_attention_mask=assembly_graph_attention_mask,
            relative_node_positions=assembly_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE: brute-force pooling for T5Encoder
        # source_embeds = source_outputs[1]
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        assembly_embeds = assembly_outputs[1]
        assembly_embeds = self.assembly_projection(assembly_embeds)

        # normalize features
        z1 = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        z2 = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logits_per_source = torch.matmul(source_embeds, assembly_embeds.t()) * logit_scale
        logits_per_source = torch.matmul(z1, z2.t()) * logit_scale
        logits_per_assembly = logits_per_source.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_source)

        if not return_dict:
            output = (logits_per_source, logits_per_assembly, source_embeds, assembly_embeds, source_outputs, assembly_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return CASPOutput(
            loss=loss,
            logits_per_source=logits_per_source,
            logits_per_assembly=logits_per_assembly,
            source_embeds=source_embeds,
            assembly_embeds=assembly_embeds,
            source_model_output=source_outputs,
            assembly_model_output=assembly_outputs
        )
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)


class MomentumDualEncoderModel(DualEncoderModel):

    def __init__(
        self,
        config: Optional[DualEncoderConfig] = None,
        assembly_model: Optional[PreTrainedModel] = None,
        source_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(
            config=config,
            assembly_model=assembly_model,
            source_model=source_model
        )
        self.K = config.K
        self.m = config.m  # TODO: different m for each encoder

        self.assembly_model_k = deepcopy(self.assembly_model)
        for param_k in self.assembly_model_k.parameters():
            param_k.requires_grad = False
        self.source_model_k = deepcopy(self.source_model)
        for param_k in self.source_model_k.parameters():
            param_k.requires_grad = False
        
        # create the queues
        self.register_buffer("source_queue", torch.randn(config.projection_dim, self.K))
        self.source_queue = nn.functional.normalize(self.source_queue, dim=0)
        self.register_buffer("assembly_queue", torch.randn(config.projection_dim, self.K))
        self.assembly_queue = nn.functional.normalize(self.assembly_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_unpdate_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # update source model
        for param_q, param_k in zip(
            self.source_model.parameters(), self.source_model_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
        # update assembly model
        for param_q, param_k in zip(
            self.assembly_model.parameters(), self.assembly_model_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, source_keys, assembly_keys):
        # gather keys before updating queue
        source_keys = concat_all_gather(source_keys)
        assembly_keys = concat_all_gather(assembly_keys)
    
        batch_size = source_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.source_queue[:, ptr : ptr + batch_size] = source_keys.T
        self.assembly_queue[:, ptr : ptr + batch_size] = assembly_keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        # source_position_ids: Optional[torch.LongTensor] = None,
        # source_token_type_ids: Optional[torch.LongTensor] = None,

        assembly_input_ids: Optional[torch.LongTensor] = None,
        assembly_attention_mask: Optional[torch.Tensor] = None,
        assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assembly_outputs = self.assembly_model(
            input_ids=assembly_input_ids,
            attention_mask=assembly_attention_mask,
            graph_attention_mask=assembly_graph_attention_mask,
            relative_node_positions=assembly_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE: brute-force pooling for T5Encoder
        # source_embeds = source_outputs[1]
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        assembly_embeds = assembly_outputs[1]
        assembly_embeds = self.assembly_projection(assembly_embeds)

        # normalize features
        q1 = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        q2 = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        # compute key features
        with torch.no_grad():
            if self.training:
                self._momentum_unpdate_key_encoder()

            k1 = self.source_model_k(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0][:, 0, :]
            k1 = self.source_projection(k1)
            k1 = k1 / k1.norm(dim=-1, keepdim=True)

            k2 = self.assembly_model_k(
                input_ids=assembly_input_ids,
                attention_mask=assembly_attention_mask,
                graph_attention_mask=assembly_graph_attention_mask,
                relative_node_positions=assembly_relative_node_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[1]
            k2 = self.assembly_projection(k2)
            k2 = k2 / k2.norm(dim=-1, keepdim=True)

        # positive logits
        src_l_pos = torch.einsum("nc,nc->n", [q1, k2]).unsqueeze(-1)
        asm_l_pos = torch.einsum("nc,nc->n", [q2, k1]).unsqueeze(-1)
        # negative logits
        src_l_neg = torch.einsum("nc,ck->nk", [q1, self.assembly_queue.clone().detach()])
        asm_l_neg = torch.einsum("nc,ck->nk", [q2, self.source_queue.clone().detach()])

        # logits: Nx(1+K)
        src_logits = torch.cat([src_l_pos, src_l_neg], dim=1)
        asm_logits = torch.cat([asm_l_pos, asm_l_neg], dim=1)

        # apply scale
        logit_scale = self.logit_scale.exp()
        src_logits = logit_scale * src_logits
        asm_logits = logit_scale * asm_logits

        # labels: positive key indicators
        src_labels = torch.zeros(src_logits.shape[0], dtype=torch.long).cuda()
        asm_labels = torch.zeros(asm_logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if self.training:
            self._dequeue_and_enqueue(
                source_keys=k1, 
                assembly_keys=k2
            )

        # compute loss
        src_loss = nn.functional.cross_entropy(src_logits, src_labels)
        asm_loss = nn.functional.cross_entropy(asm_logits, asm_labels)
        loss = (src_loss + asm_loss) / 2.0

        return CASPOutput(
            loss=loss,
            logits_per_source=src_logits,
            logits_per_assembly=asm_logits,
            source_embeds=k1,
            assembly_embeds=k2,
        )


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class DualEncoderRanker(MomentumDualEncoderModel):

    def forward(
        self,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,

        assembly_input_ids: Optional[torch.LongTensor] = None,
        assembly_attention_mask: Optional[torch.Tensor] = None,
        assembly_graph_attention_mask: Optional[torch.Tensor] = None,
        assembly_relative_node_positions: Optional[torch.LongTensor] = None,

        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        target_scores: Optional[torch.FloatTensor] = None,
        log_target: Optional[bool] = False,

        return_loss: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CASPOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        source_outputs = self.source_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            # token_type_ids=source_token_type_ids,
            # position_ids=source_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assembly_outputs = self.assembly_model(
            input_ids=assembly_input_ids,
            attention_mask=assembly_attention_mask,
            graph_attention_mask=assembly_graph_attention_mask,
            relative_node_positions=assembly_relative_node_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE: brute-force pooling for T5Encoder
        source_embeds = source_outputs[0][:, 0, :]
        source_embeds = self.source_projection(source_embeds)

        assembly_embeds = assembly_outputs[1]
        assembly_embeds = self.assembly_projection(assembly_embeds)

        # normalize features
        e_src = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        e_asm = assembly_embeds / assembly_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        k = target_scores.shape[1]  # e_asm: (batch_size, embed_dim), e_src: (batch_size * k, embed_dim)
        e_src = e_src.view(e_asm.shape[0], k, e_asm.shape[1])   # (batch_size, k, embed_dim)
        logits_per_asm = torch.matmul(e_asm[:, None, :], e_src.transpose(1, 2)).\
                                                        squeeze() * logit_scale
        log_input = nn.functional.log_softmax(logits_per_asm, dim=1)

        # shape correctness
        assert(log_input.shape == target_scores.shape)

        loss = None
        if return_loss:
            loss = nn.functional.kl_div(log_input, target_scores, log_target=log_target)   # NOTE: log_target is False by default

        if not return_dict:
            output = (None, logits_per_asm, source_embeds, assembly_embeds, source_outputs, assembly_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return CASPOutput(
            loss=loss,
            logits_per_source=None,
            logits_per_assembly=logits_per_asm,
            source_embeds=source_embeds,
            assembly_embeds=assembly_embeds,
            source_model_output=source_outputs,
            assembly_model_output=assembly_outputs
        )