from transformers.configuration_utils import PretrainedConfig


class LongelmConfig(PretrainedConfig):

    model_type = "longelm"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_relative_position_embeddings=8,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        position_embedding_type="mixed",
        use_cache=True,
        classifier_dropout=None,
        ep_add_linear_projection=False,
        global_memory_size=1,
        node_size=1,
        block_size=10,
        max_blocks=400,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position_embeddings = max_relative_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.ep_add_linear_projection = ep_add_linear_projection

        self.global_memory_size = global_memory_size
        self.node_size = node_size
        self.block_size = block_size
        self.max_blocks = max_blocks

        # NOTE: cannot assert here because config will be saved as a diff from default config
        # assert self.max_position_embeddings == (self.node_size + self.block_size) * self.max_blocks + global_memory_size
