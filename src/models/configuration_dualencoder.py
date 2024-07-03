from .configuration_longelm import LongelmConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig


class DualEncoderConfig(PretrainedConfig):
    
    model_type = "assembly-source-dual-encoder"
    is_composition = True

    def __init__(
        self,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs
    ):

        super().__init__(**kwargs)

        if "assembly_config" not in kwargs:
            raise ValueError("`assembly_config` can not be `None`.")

        if "source_config" not in kwargs:
            raise ValueError("`source_config` can not be `None`.")
        
        assembly_config = kwargs.pop("assembly_config")
        source_config = kwargs.pop("source_config")

        self.assembly_config = assembly_config
        self.source_config = source_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_assembly_source_configs(
        cls,
        assembly_config: PretrainedConfig,
        source_config: PretrainedConfig,
        **kwargs
    ):
        return cls(assembly_config=assembly_config.to_dict(), source_config=source_config.to_dict(), **kwargs)
    

class MomentumDualEncoderConfig(DualEncoderConfig):

    def __init__(
        self,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        K=65536,
        m=0.999,
        **kwargs
    ):
        super().__init__(projection_dim, logit_scale_init_value, **kwargs)
        self.K = K
        self.m = m