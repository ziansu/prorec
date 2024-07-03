from typing import Any, Dict
from .configuration_longelm import LongelmConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig


class SrcProberConfig(PretrainedConfig):

    model_type = 'rad-prober'
    is_composition = True

    def __init__(
        self,
        ignore_index=-100,
        asm_token_index=32000,  # index of <asm_token>
        projector_hidden_act="gelu",
        asm_feature_select_strategy="all",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.asm_token_index = asm_token_index
        self.projector_hidden_act = projector_hidden_act

        if asm_feature_select_strategy not in ["all", "nodes", "nodes+global"]:
            raise ValueError(
                "asm_feature_select_strategy should be one of 'all', 'nodes'."
                f"Got: {asm_feature_select_strategy}"
            )
        self.asm_feature_select_strategy = asm_feature_select_strategy

        if "asm_encoder_config" not in kwargs:
            raise ValueError("`asm_encoder_config` can not be `None`.")

        if "src_lm_config" not in kwargs:
            raise ValueError("`src_lm_config` can not be `None`.")

        asm_encoder_config = kwargs.pop("asm_encoder_config")
        src_lm_config = kwargs.pop("src_lm_config")

        self.asm_encoder_config = asm_encoder_config
        self.src_lm_config = src_lm_config

        if isinstance(self.asm_encoder_config, dict):
            self.asm_encoder_config = LongelmConfig.from_dict(self.asm_encoder_config)

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()