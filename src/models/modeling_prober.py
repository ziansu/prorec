import os
from os import PathLike
from . import LongelmConfig, LongelmModel
from .configuration_prober import SrcProberConfig

import torch
import torch.nn as nn
from dataclasses import dataclass
from peft import PeftModelForCausalLM
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from typing import Optional, Union, Tuple, Dict, List, Callable, Any


@dataclass
class SrcProberCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    asm_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SrcProberMultiModalProjector(nn.Module):
    def __init__(self, config: SrcProberConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.asm_encoder_config.hidden_size, config.src_lm_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.src_lm_config.hidden_size, config.src_lm_config.hidden_size, bias=True)

    def forward(self, bin_features):
        hidden_states = self.linear_1(bin_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states



class SrcProberForConditionalGeneration(PreTrainedModel):
    config_class = SrcProberConfig
    base_model_prefix = 'src_prober'

    def __init__(
        self,
        config=None,
        asm_encoder=None,
        src_language_model=None,
    ):
        super().__init__(config)

        self.config = config
        if asm_encoder:
            self.asm_encoder = asm_encoder
        else:
            self.asm_encoder = LongelmModel(config.asm_encoder_config, add_pooling_layer=False)
        self.projection = SrcProberMultiModalProjector(config)
        if src_language_model == 'empty':
            self.src_language_model = None
        elif src_language_model:
            self.src_language_model = src_language_model
        else:
            self.src_language_model = AutoModelForCausalLM.from_config(config.src_lm_config)
        # TODO: understand proper pad token id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # freeze encoder
        for p in self.asm_encoder.parameters():
            p.requires_grad = False
        # freeze lm
        if self.src_language_model:
            for p in self.src_language_model.parameters():
                p.requires_grad = False
        
        # Initialize weights and apply final processing
        # self.post_init()

    def state_dict(self):
        "avoid loading `src_language_model`'s parameters in `_load_pretrained_model`"
        filtered_state_dict = {}
        for k, v in super().state_dict().items():
            if 'src_language_model' not in k:
                filtered_state_dict[k] = v
        return filtered_state_dict
    
    def get_input_embeddings(self) -> nn.Module:
        return self.src_language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module):
        return self.src_language_model.set_input_embeddings(value)
    
    def set_decoder(self, decoder):
        self.src_language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.src_language_model.get_decoder()

    def tie_weights(self):
        if self.src_language_model:
            return self.src_language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.src_language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        # self.config.text_config.vocab_size = model_embeds.num_embeddings
        # self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_asm_features(
        self,
        asm_features: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        num_asms, asm_seq_len, embed_dim = asm_features.shape
        batch_size, seq_len = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))   # bool
        # 1. Create a mask to know where special asm tokens are
        special_asm_token_mask = input_ids == self.config.asm_token_index
        num_special_asm_tokens = torch.sum(special_asm_token_mask, dim=-1)  # NOTE: should be all 1
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_asm_tokens.max() * (asm_seq_len - 1)) + seq_len
        batch_indices, non_asm_indices = torch.where(input_ids != self.config.asm_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged asm-src sequence.
        # `special_asm_token_mask` identifies asm tokens. Each asm token will be replaced by `nb_asm_tokens_per_asm - 1` text tokens
        # `torch.cumsum` computes how each asm token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_asm_token_mask * (asm_seq_len - 1) + 1), -1) - 1
        nb_asm_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_asm_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_asm_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the asm model or the language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device
        target_device = inputs_embeds.device
        batch_indices, non_asm_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_asm_indices.to(target_device),
            text_to_overwrite.to(target_device)
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_asm_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_asm_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_asm_indices]

        # 5. Fill the embeddings corresponding to the asm. Anything that is zeros needs filling.
        asm_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        asm_to_overwrite &= asm_to_overwrite.cumsum(-1) - 1 >= nb_asm_pad[:, None].to(target_device)

        if asm_to_overwrite.sum() != asm_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of asm tokens is {torch.sum(special_asm_token_mask)} while"
                f" the number of asm given to the model is {num_asms}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[asm_to_overwrite] = asm_features.contiguous().reshape(-1, embed_dim).to(target_device, dtype=final_embedding.dtype)
        final_attention_mask |= asm_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        asm_input_ids: torch.LongTensor = None,
        asm_attention_mask: Optional[torch.Tensor] = None,
        asm_graph_attention_mask: Optional[torch.Tensor] = None,
        asm_relative_node_positions: Optional[torch.Tensor] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        asm_feature_select_strategy: Optional[str] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        asm_feature_select_strategy = (
            asm_feature_select_strategy
            if asm_feature_select_strategy is not None
            else self.config.asm_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. get text input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. merge text and asms
            if asm_input_ids is not None and input_ids.shape[1] != 1:
                # encode assembly
                encoder_outputs = self.asm_encoder(
                    input_ids=asm_input_ids,
                    attention_mask=asm_attention_mask,
                    graph_attention_mask=asm_graph_attention_mask,
                    relative_node_positions=asm_relative_node_positions,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=return_dict,
                )
                if asm_feature_select_strategy == 'all':
                    asm_features = encoder_outputs[0]
                elif asm_feature_select_strategy == 'nodes':
                    asm_features = encoder_outputs[0][:, -201:-1]  # FIXME: use config
                elif asm_feature_select_strategy == 'nodes+global':
                    asm_features = encoder_outputs[0][:, -201:]
                else:
                    raise ValueError(f"Unexpected select features strategy: {self.config.asm_feature_select_strategy}")
                # project embeddings
                asm_features = self.projection(asm_features)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_asm_features(
                    asm_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)

            # In case input_ids.shape[1] == 1 & past_key_values != None, we are in the case of 
            # generation with cache
            elif past_key_values is not None and asm_input_ids is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.src_language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return SrcProberCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        asm_input_ids=None,
        asm_attention_mask=None,
        asm_graph_attention_mask=None,
        asm_relative_node_positions=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing inputs_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.asm_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "asm_input_ids": asm_input_ids,
                "asm_attention_mask": asm_attention_mask,
                "asm_graph_attention_mask": asm_graph_attention_mask,
                "asm_relative_node_positions": asm_relative_node_positions,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.src_language_model._reorder_cache(*args, **kwargs)
    
    def save_pretrained(
        self, 
        save_directory: str | PathLike, 
        is_main_process: bool = True, 
        state_dict: Dict | None = None, 
        save_function: Callable[..., Any] = torch.save, 
        push_to_hub: bool = False, 
        max_shard_size: int | str = "5GB", 
        safe_serialization: bool = True, 
        variant: str | None = None, token: str | bool | None = None, 
        save_peft_format: bool = True, 
        **kwargs
    ):
        if isinstance(self.src_language_model, PeftModelForCausalLM):
            self.src_language_model.save_pretrained(
                # os.path.join(save_directory, 'peft'), 
                save_directory, 
                is_main_process, 
                pusb_to_hub=push_to_hub, 
                max_shard_size=max_shard_size, 
                variant=variant, 
                token=token, 
                save_peft_format=save_peft_format, 
                **kwargs
            )

        # remove src_language_model related parameters in state_dict
        if not state_dict:
            model_to_save = unwrap_model(self)
            state_dict = model_to_save.state_dict()

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if 'src_language_model' not in k:
                filtered_state_dict[k] = v
        
        return super().save_pretrained(
            save_directory, 
            is_main_process, 
            filtered_state_dict, 
            save_function, 
            push_to_hub, 
            max_shard_size, 
            safe_serialization, 
            variant, 
            token, 
            save_peft_format, 
            **kwargs
        )
    
    def load_adapter(
        self, 
        peft_model_id: str | None = None, 
        adapter_name: str | None = None, 
        revision: str | None = None, 
        token: str | None = None, 
        device_map: str | None = "auto", 
        max_memory: str | None = None, 
        offload_folder: str | None = None, 
        offload_index: int | None = None, 
        peft_config: Dict[str, Any] = None, 
        adapter_state_dict: Dict[str, torch.Tensor] | None = None, 
        adapter_kwargs: Dict[str, Any] | None = None
    ) -> None:
        pass