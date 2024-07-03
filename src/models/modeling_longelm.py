from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    logging
)
logger = logging.get_logger(__name__)

from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaOutput,
    RobertaSelfOutput,
    RobertaPooler,
    RobertaLMHead,
    RobertaClassificationHead
)

from .configuration_longelm import LongelmConfig


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class LongelmEmbeddings(nn.Module):
    """
    Absolute indexing for local context.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute" or self.position_embedding_type == "mixed":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LongelmSparseAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # sparse attention info
        self.block_size = config.block_size
        self.global_memory_size = config.global_memory_size
        self.node_size = config.node_size
        self.max_relative_distance = config.max_relative_position_embeddings

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        if self.position_embedding_type == "mixed":
            self.relative_attention_num_buckets = config.max_relative_position_embeddings + 1   # NOTE: bucket is just one unit relative distance here
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_attention_heads)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def _local_block_attention(
        self,

        local_query_layer: torch.Tensor,
        local_key_layer: torch.Tensor,
        local_value_layer: torch.Tensor,

        block_node_key_layer: torch.Tensor,
        block_node_value_layer: torch.Tensor,

        global_key_layer: torch.Tensor,
        global_value_layer: torch.Tensor,

        local_attention_mask: torch.FloatTensor,

        head_mask: Optional[torch.FloatTensor] = None
    ):
        # block self-attention
        block_self_attention_scores = torch.matmul(local_query_layer, local_key_layer.transpose(-1, -2))
        # scale
        block_self_attention_scores = block_self_attention_scores / math.sqrt(self.attention_head_size)
        # add mask
        block_self_attention_scores = block_self_attention_scores + local_attention_mask

        # block-node cross-attention
        block_node_cross_attention_scores = torch.matmul(local_query_layer, block_node_key_layer.transpose(-1, -2))
        # scale
        block_node_cross_attention_scores = block_node_cross_attention_scores / math.sqrt(self.attention_head_size)

        # NOTE: check whether broadcast (batch_size, num_heads, 1, global_size, attn_head_size)
        global_key_layer = global_key_layer[:, :, None, :, :]
        global_value_layer = global_value_layer[:, :, None, :, :]
        # block-global cross-attention
        block_global_cross_attention_scores = torch.matmul(local_query_layer, global_key_layer.transpose(-1, -2))
        # scale
        block_global_cross_attention_scores = block_global_cross_attention_scores / math.sqrt(self.attention_head_size)


        # compute attention_probs
        attention_probs = F.softmax(
            torch.cat(
                (block_self_attention_scores, block_node_cross_attention_scores, block_global_cross_attention_scores),
                dim=-1  # check
            ),
            dim=-1
        )

        attention_probs = self.dropout(attention_probs)

        # mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(
            attention_probs,
            torch.cat(
                (local_value_layer, block_node_value_layer, 
                 global_value_layer.expand(
                     size=local_value_layer.shape[:3] + global_value_layer.shape[-2:]
                 )
                ),
                dim=-2
            )
        )

        context_layer = context_layer.view(context_layer.shape[:2] + (-1, self.attention_head_size))\
                                        .permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs

    def _compute_bias(self, relative_positions: torch.LongTensor, max_distance: int):
        relative_positions = torch.where(relative_positions >= 0, relative_positions, max_distance)    # NOTE: avoid IndexError, node -1 positions will be masked out later
        relative_positions = torch.where(relative_positions < max_distance, relative_positions, max_distance-1)
        torch._assert(torch.max(relative_positions) <= max_distance, "relative position exceeds max distance")
        values = self.relative_attention_bias(relative_positions)    # shape (batch_size, query_length, key_length, num_heads)
        values = values.permute([0, 3, 1, 2])  # shape (batch_size, num_heads, query_length, key_length)
        return values

    def _node_masked_biased_attention(
        self,

        local_key_layer: torch.Tensor,
        local_value_layer: torch.Tensor,

        block_node_query_layer: torch.Tensor,
        node_key_layer: torch.Tensor,
        node_value_layer: torch.Tensor,

        global_key_layer: torch.Tensor,
        global_value_layer: torch.Tensor,

        local_attention_mask: torch.FloatTensor,
        graph_attention_mask: torch.FloatTensor,
        relative_positions: torch.LongTensor,
        max_distance: int,

        head_mask: Optional[torch.FloatTensor] = None
    ):
        
        # NOTE: check whether broadcast (batch_size, num_heads, 1, num_blocks * node_size, attn_head_size)
        node_key_layer = node_key_layer[:, :, None, :, :]
        node_value_layer = node_value_layer[:, :, None, :, :]
        # node masked biased self-attention
        relative_bias = self._compute_bias(relative_positions, max_distance=max_distance)
        node_self_attention_scores = torch.matmul(block_node_query_layer, 
                                            node_key_layer.transpose(-1, -2))
        # add bias
        relative_bias = relative_bias.view(node_self_attention_scores.shape)
        node_self_attention_scores = node_self_attention_scores + relative_bias
        # scale
        node_self_attention_scores = node_self_attention_scores / math.sqrt(self.attention_head_size)
        # add mask, view as (batch_size, 1, num_blocks, node_size, num_blocks * node_size)
        graph_attention_mask = graph_attention_mask.view(relative_bias.shape[:1] + (1,) + relative_bias.shape[2:])
        node_self_attention_scores = node_self_attention_scores + graph_attention_mask

        # node block cross attention
        node_block_cross_attention_scores = torch.matmul(block_node_query_layer, local_key_layer.transpose(-1, -2))
        # scale
        node_block_cross_attention_scores = node_block_cross_attention_scores / math.sqrt(self.attention_head_size)
        # NOTE: check, add mask, should work similarily for keys
        node_block_cross_attention_scores = node_block_cross_attention_scores + local_attention_mask

        # NOTE: check whether broadcast (batch_size, num_heads, 1, global_size, attn_head_size)
        global_key_layer = global_key_layer[:, :, None, :, :]
        global_value_layer = global_value_layer[:, :, None, :, :]
        # node global cross attention
        node_global_cross_attention_scores = torch.matmul(block_node_query_layer, global_key_layer.transpose(-1, -2))
        # scale
        node_global_cross_attention_scores = node_global_cross_attention_scores / math.sqrt(self.attention_head_size)


        # compute attention_probs
        attention_probs = F.softmax(
            torch.cat(
                (node_self_attention_scores, node_block_cross_attention_scores, node_global_cross_attention_scores),
                dim=-1
            ),
            dim=-1
        )

        attention_probs = self.dropout(attention_probs)

        # mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(
            attention_probs,
            torch.cat(
                (node_value_layer.expand(
                    size=local_value_layer.shape[:3] + node_value_layer.shape[-2:]
                ), 
                 local_value_layer,
                 global_value_layer.expand(
                     size=local_value_layer.shape[:3] + global_value_layer.shape[-2:]
                 )),
                dim=-2
            )
        )

        context_layer = context_layer.view(context_layer.shape[:2] + (-1, self.attention_head_size))\
                                        .permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


    def _global_attention(
        self,

        local_key_layer: torch.Tensor,
        local_value_layer: torch.Tensor,

        node_key_layer: torch.Tensor,
        node_value_layer: torch.Tensor,

        global_query_layer: torch.Tensor,
        global_key_layer: torch.Tensor,
        global_value_layer: torch.Tensor,
        
        attention_mask: torch.FloatTensor,

        head_mask: Optional[torch.FloatTensor] = None
    ):
        local_key_layer = local_key_layer.view(local_key_layer.shape[:2] + (-1, self.attention_head_size))
        local_value_layer = local_value_layer.view(local_value_layer.shape[:2] + (-1, self.attention_head_size))
        # global local cross-attention
        global_local_cross_attention_scores = torch.matmul(global_query_layer,
                                            local_key_layer.transpose(-1, -2))
        # scale
        global_local_cross_attention_scores = global_local_cross_attention_scores / math.sqrt(self.attention_head_size)
        # mask
        global_local_cross_attention_scores = global_local_cross_attention_scores + attention_mask

        # global node cross-attention
        global_node_cross_attention_scores = torch.matmul(global_query_layer,
                                            node_key_layer.transpose(-1, -2))
        # scale
        global_node_cross_attention_scores = global_node_cross_attention_scores / math.sqrt(self.attention_head_size)

        # global self-attention
        global_self_attention_scores = torch.matmul(global_query_layer,
                                            global_key_layer.transpose(-1, -2))
        # scale
        global_self_attention_scores = global_self_attention_scores / math.sqrt(self.attention_head_size)


        # attention probs
        attention_probs = F.softmax(
            torch.cat(
                (global_local_cross_attention_scores, global_node_cross_attention_scores, global_self_attention_scores),
                dim=-1
            ),
            dim=-1
        )

        attention_probs = self.dropout(attention_probs)

        # mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(
            attention_probs,
            torch.cat(
                (local_value_layer, node_value_layer, global_value_layer),
                dim=-2
            )
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # split hidden_states
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_blocks = (seq_len - self.global_memory_size) // (self.block_size + self.node_size)
        assert seq_len == (num_blocks * (self.block_size + self.node_size) + self.global_memory_size)
        global_hidden_states = hidden_states[:, -self.global_memory_size:]
        node_hidden_states = hidden_states[:, -self.global_memory_size-num_blocks*self.node_size:-self.global_memory_size]
        hidden_states = hidden_states[:, :num_blocks * self.block_size]

        # ignore past key values first

        local_query_layer = self.transpose_for_scores(self.query(hidden_states))
        local_key_layer = self.transpose_for_scores(self.key(hidden_states))
        local_value_layer = self.transpose_for_scores(self.value(hidden_states))
        # view as blocks
        local_query_layer = local_query_layer.view(batch_size, self.num_attention_heads, num_blocks, self.block_size, self.attention_head_size)
        local_key_layer = local_key_layer.view(batch_size, self.num_attention_heads, num_blocks, self.block_size, self.attention_head_size)
        local_value_layer = local_value_layer.view(batch_size, self.num_attention_heads, num_blocks, self.block_size, self.attention_head_size)

        node_query_layer = self.transpose_for_scores(self.query(node_hidden_states))
        node_key_layer = self.transpose_for_scores(self.key(node_hidden_states))
        node_value_layer = self.transpose_for_scores(self.value(node_hidden_states))
        # view as node blocks
        block_node_query_layer = node_query_layer.view(batch_size, self.num_attention_heads, num_blocks, self.node_size, self.attention_head_size)
        block_node_key_layer = node_key_layer.view(batch_size, self.num_attention_heads, num_blocks, self.node_size, self.attention_head_size)
        block_node_value_layer = node_value_layer.view(batch_size, self.num_attention_heads, num_blocks, self.node_size, self.attention_head_size)

        global_query_layer = self.transpose_for_scores(self.query(global_hidden_states))
        global_key_layer = self.transpose_for_scores(self.key(global_hidden_states))
        global_value_layer = self.transpose_for_scores(self.value(global_hidden_states))

        local_attention_mask = attention_mask.view(batch_size, 1, num_blocks, 1, self.block_size)

        local_context_layer, local_attention_probs = self._local_block_attention(
            local_query_layer=local_query_layer,
            local_key_layer=local_key_layer,
            local_value_layer=local_value_layer,
            block_node_key_layer=block_node_key_layer,
            block_node_value_layer=block_node_value_layer,
            global_key_layer=global_key_layer,
            global_value_layer=global_value_layer,
            local_attention_mask=local_attention_mask,
            head_mask=head_mask
        )
        node_context_layer, node_attention_probs = self._node_masked_biased_attention(
            local_key_layer=local_key_layer,
            local_value_layer=local_value_layer,
            block_node_query_layer=block_node_query_layer,
            node_key_layer=node_key_layer,
            node_value_layer=node_value_layer,
            global_key_layer=global_key_layer,
            global_value_layer=global_value_layer,
            local_attention_mask=local_attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_positions=relative_node_positions,
            head_mask=head_mask,
            max_distance=self.max_relative_distance
        )
        global_context_layer, global_attention_probs = self._global_attention(
            local_key_layer=local_key_layer,
            local_value_layer=local_value_layer,
            node_key_layer=node_key_layer,
            node_value_layer=node_value_layer,
            global_query_layer=global_query_layer,
            global_key_layer=global_key_layer,
            global_value_layer=global_value_layer,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        context_layer = torch.cat((local_context_layer, node_context_layer, global_context_layer), dim=1)
        if output_attentions:
            attention_probs = (local_attention_probs, node_attention_probs, global_attention_probs)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LongelmSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LongelmAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = LongelmSparseAttention(config, position_embedding_type=position_embedding_type)
        self.output = LongelmSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            graph_attention_mask,
            relative_node_positions,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LongelmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LongelmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # if not self.is_decoder:
            #     raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
            raise NotImplementedError
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            graph_attention_mask,
            relative_node_positions,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LongelmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LongelmLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    graph_attention_mask,
                    relative_node_positions,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    graph_attention_mask,
                    relative_node_positions,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                # next_decoder_cache += (layer_outputs[-1],)
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LongelmPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LongelmConfig
    base_model_prefix = "longelm"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LongelmEncoder):
            module.gradient_checkpointing = value


class LongelmPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.global_memory_size = config.global_memory_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the [CLS] tokens.
        global_token_tensor = hidden_states[:, -1]  # TODO: try average
        pooled_output = self.dense(global_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongelmModel(LongelmPretrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = LongelmEmbeddings(config)
        self.encoder = LongelmEncoder(config)

        self.pooler = LongelmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # def get_extended_local_attention_mask(
    #     self,
    #     attention_mask: torch.Tensor,
    #     device: torch.device = None,
    #     dtype: torch.float = None
    # ) -> torch.Tensor:
    #     """
    #     Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    #     Arguments:
    #         attention_mask (`torch.Tensor`):
    #             Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
    #         input_shape (`Tuple[int]`):
    #             The shape of the input to the model.

    #     Returns:
    #         `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    #     """
    #     if dtype is None:
    #         dtype = self.dtype

    #     assert attention_mask.dim() == 2    # (batch_size, seq_len)

    #     if self.config.is_decoder:
    #         raise NotImplementedError
    #     else:
    #         extended_attention_mask = attention_mask[:, None, None, :]
        
    #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    #     # masked positions, this operation will create a tensor which is 0.0 for
    #     # positions we want to attend and the dtype's smallest value for masked positions.
    #     # Since we are adding it to the raw scores before the softmax, this is
    #     # effectively the same as removing these entirely.
    #     extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    #     extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    #     return extended_attention_mask

    # def get_graph_attention_mask(
    #     self,
    #     relative_node_positions: torch.Tensor,
    #     dtype: torch.float = None,
    # ) -> torch.Tensor:
    #     graph_attention_mask = (relative_node_positions >= 0).float()   # with diagonal values
    #     graph_attention_mask = (1.0 - graph_attention_mask) * torch.finfo(dtype).min
    #     return graph_attention_mask

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph_attention_mask: Optional[torch.Tensor] = None,
        relative_node_positions: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # if token_type_ids is None:
        #     if hasattr(self.embeddings, "token_type_ids"):
        #         buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape=None)
        # graph_attention_mask: torch.Tensor = self.get_graph_attention_mask(relative_node_positions)
        graph_attention_mask = self.get_extended_attention_mask(
            attention_mask=graph_attention_mask, input_shape=None)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            # token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_node_positions=relative_node_positions,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LongelmEPHead(nn.Module):
    """Head for edge prediction."""

    def __init__(self, config):
        super().__init__()
        self.add_projection = config.ep_add_linear_projection
        self.global_memory_size = config.global_memory_size
        self.node_size = config.node_size
        self.block_size = config.block_size

        if self.add_projection:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        "compute edge probability"

        seq_len = hidden_states.shape[1]
        num_blocks = (seq_len - self.global_memory_size) // (self.block_size + self.node_size)
        assert seq_len == (num_blocks * (self.block_size + self.node_size) + self.global_memory_size)
        features = hidden_states[:, -self.global_memory_size-num_blocks*self.node_size:-self.global_memory_size]

        if self.add_projection:
            x = self.dense(features)
        else:
            x = features

        m = torch.bmm(x, x.transpose(2, 1))
        return m.view(x.shape[0], -1)   # (B, L * L)


@dataclass
class MaskedLMWithEdgePredOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    edge_prediction_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    ep_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LongelmForMaskedLMWithEdgePrediction(LongelmPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        
        self.longelm = LongelmModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.ep_head = LongelmEPHead(config)

        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ep_bce_weights: Optional[torch.FloatTensor] = None,
        ep_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        ep_labels:
            Labels for computing the edge prediction loss.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        batch_size = input_ids.shape[0]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.longelm(
            input_ids,
            attention_mask=attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_node_positions=relative_node_positions,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        edge_prediction_scores = self.ep_head(sequence_output)

        masked_lm_loss = None
        
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        edge_prediction_loss = None

        if (ep_labels is not None) and (ep_bce_weights is not None):
            ep_labels = ep_labels.to(edge_prediction_scores.device)
            ep_bce_weights = ep_bce_weights.to(edge_prediction_scores.device)
            # https://discuss.pytorch.org/t/masking-binary-cross-entropy-loss/61065
            edge_prediction_loss = F.binary_cross_entropy_with_logits(
                input=edge_prediction_scores,
                target=ep_labels.view(batch_size, -1),
                weight=ep_bce_weights.view(batch_size, -1)
            )

            # scale the ep loss to match the scale of masked_lm_loss
            edge_prediction_loss = edge_prediction_loss * ep_labels.shape[1] ** 2 / ep_bce_weights.sum() * batch_size

        loss = None
        if (masked_lm_loss is not None) and (edge_prediction_loss is not None):
            loss = masked_lm_loss + edge_prediction_loss

        if not return_dict:
            output = (prediction_scores, edge_prediction_scores) + outputs[2:]
            return ((loss, masked_lm_loss, edge_prediction_loss) + output) \
                    if loss is not None else output

        return MaskedLMWithEdgePredOutput(
            loss=loss,
            masked_lm_loss=masked_lm_loss,
            edge_prediction_loss=edge_prediction_loss,
            logits=prediction_scores,
            ep_logits=edge_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BinSimOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None    
    embs_wo_pooler:Optional[torch.FloatTensor] = None    


class LongelmForBinSim(LongelmPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        # if margin is not a field of config:
        if not hasattr(config, 'margin'):
            self.margin = 0.5
            print("Do not have margin in config")
            print("Set it to 0.5")
        else:
            self.margin = config.margin
            print("Set margin to %.3f" % self.margin)
        self.longelm = LongelmModel(config, add_pooling_layer=True)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        graph_attention_mask: Optional[torch.FloatTensor] = None,
        relative_node_positions: Optional[torch.LongTensor] = None,        
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,        
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,        
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
                
        batch_size = input_ids.shape[0]
        view_size = input_ids.shape[1]
        input_ids = input_ids.view(batch_size * view_size, -1)
        attention_mask = attention_mask.view(
            batch_size * view_size, *attention_mask.shape[2:]
        )
        relative_node_positions = relative_node_positions.view(
            batch_size * view_size, *relative_node_positions.shape[2:]
        )
        outputs = self.longelm(
            input_ids,
            attention_mask=attention_mask,
            graph_attention_mask=graph_attention_mask,
            relative_node_positions=relative_node_positions,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        embs = outputs.pooler_output
        embs = embs.view(batch_size, view_size, -1)

        first_emb = embs[:, 0, :]
        pos_emb = embs[:, 1, :]
        neg_emb = embs[:, 2, :]
        pos_sim = F.cosine_similarity(first_emb, pos_emb)
        neg_sim = F.cosine_similarity(first_emb, neg_emb)
        
        loss = torch.mean((neg_sim - pos_sim + self.margin).clamp(min=1e-6))
        embs_reshaped = embs.view(batch_size, view_size, -1)
        
        embs_wo_pooler = outputs.last_hidden_state[:, -1, :]    # NOTE: only for global 1
        embs_wo_pooler = embs_wo_pooler.view(batch_size, view_size, -1)
        return BinSimOutput(
            loss=loss,
            embeddings=embs_reshaped,
            embs_wo_pooler=embs_wo_pooler
        )
