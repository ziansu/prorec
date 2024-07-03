import json
import logging
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union, Tuple, Optional, Dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SignatureScorer(object):

    def __init__(
        self,
        dual_encoder: PreTrainedModel = None,
        assembly_tokenizer: PreTrainedTokenizer = None,
        source_tokenizer: PreTrainedTokenizer = None,
        device: str = None,
        force_dataparallel: bool = False
    ):
        
        self.dual_encoder = dual_encoder
        self.assembly_tokenizer = assembly_tokenizer
        self.source_tokenizer = source_tokenizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.force_dataparallel = force_dataparallel
        if force_dataparallel:
            import torch.nn as nn
            assert device == "cuda"
            self.dual_encoder = self.dual_encoder.to(device)
            self.dual_encoder.assembly_model = \
                nn.DataParallel(self.dual_encoder.assembly_model)
            self.dual_encoder.source_model = \
                nn.DataParallel(self.dual_encoder.source_model)

    def encode(
        self,
        examples: List[str],
        batch_size: int = 64,
        encoder_type: str = None,
        normalize_to_unit: bool = False,
        return_numpy: bool = False,
    ):
        if not self.force_dataparallel:
            self.dual_encoder = self.dual_encoder.to(self.device)
        embedding_list = []
        encoding_function_name = f"get_{encoder_type}_features"
        
        with torch.no_grad():
            total_batch = len(examples) // batch_size + (1 if len(examples) % batch_size > 0 else 0)
            # for batch_id in tqdm(range(total_batch)):
            for batch_id in range(total_batch):
                if encoder_type == 'source':
                    inputs = self.source_tokenizer(
                        examples[batch_id*batch_size: (batch_id+1)*batch_size],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                elif encoder_type == 'assembly':
                    inputs = self.assembly_tokenizer.batch_inst_encode(
                        examples[batch_id*batch_size: (batch_id+1)*batch_size],
                    )
                else:
                    raise ValueError("Unknown encoder type.")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = getattr(self.dual_encoder, encoding_function_name)(
                    **inputs,
                    return_dict=True
                )
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if return_numpy and not isinstance(embeddings, np.ndarray):
            return embeddings.numpy()
        
        return embeddings
    
    def score(
        self,
        queries: List[str],
        candidates: List[str],
        encode_batch_size=64
    ):
        assert len(candidates) % len(queries) == 0
        batch_size = len(queries)

        query_vecs = self.encode(
            examples=queries,
            batch_size=encode_batch_size,
            encoder_type='assembly',
            normalize_to_unit=True,
            return_numpy=False
        )
        candidate_vecs = self.encode(
            examples=candidates,
            batch_size=encode_batch_size,
            encoder_type='source',
            normalize_to_unit=True,
            return_numpy=False
        )
        candidate_vecs = candidate_vecs.view(batch_size, -1, candidate_vecs.shape[-1])
        # print('candidate vecs shape:', candidate_vecs.shape)
        scores = torch.matmul(query_vecs.unsqueeze(1), candidate_vecs.transpose(-1, -2)).squeeze(1)
        # print('scores shape:', scores.shape)
        return scores

    def get_top_n(
        self,
        queries,
        candidates,
        n=3,
        encode_batch_size=64,
    ):
        scores = self.score(queries, candidates, encode_batch_size)
        ordered_scores, indices = torch.sort(scores, dim=-1, descending=True)
        # print(ordered_scores.shape, indices.shape)
        
        scores = []
        top_candidates = []
        n_cand = len(candidates) // len(queries)
        for qid, (ordered_score, indice) in enumerate(zip(ordered_scores, indices)):
            scores.append(str(ordered_score[:n].tolist()))
            top_candidates.append(str([candidates[qid * n_cand + i.item()] for i in indice[:n]]))
        return scores, top_candidates


import sys
sys.path.append('../')
from models import (
    MomentumDualEncoderConfig,
    MomentumDualEncoderModel,
    LongelmModel,
    LongelmTokenizer,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)


def load_scorer(
    model_args,
    force_dataparallel=False
):

    config = MomentumDualEncoderConfig.from_pretrained(
        model_args.dualencoder_name_or_path,
        subfolder=model_args.dualencoder_subfolder,
        cache_dir=model_args.cache_dir,
    )
    assert config.K == 4096

    # load codet5p-embedding
    source_model = AutoModel.from_pretrained(
        model_args.source_model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    source_model.config.use_cache = False  # only decoder should have use_cache as True

    # load casp
    config.source_config = source_model.config

    dual_encoder = MomentumDualEncoderModel.from_pretrained(
        model_args.dualencoder_name_or_path,
        subfolder=model_args.dualencoder_subfolder,
        cache_dir=model_args.cache_dir,
        config=config,
        source_model=source_model
    )
    print('PE shape of CASP assembly encoder:', dual_encoder.assembly_model.embeddings.position_embeddings.weight.shape)

    # load tokenizers
    assembly_tokenizer = LongelmTokenizer.from_pretrained(
        model_args.assembly_model_name_or_path,
        cache_dir=model_args.cache_dir,
        # NOTE: currently using default parameters
    )
    assembly_tokenizer.max_blocks = assembly_tokenizer.max_blocks - 1   # FIXME: this is only temp fix of max pe problem
    source_tokenizer = AutoTokenizer.from_pretrained(
        model_args.source_model_name_or_path,
        cache_dir=model_args.cache_dir,
        # token=model_args.token,
        trust_remote_code=True
    )

    # retriever
    scorer = SignatureScorer(
        dual_encoder=dual_encoder,
        assembly_tokenizer=assembly_tokenizer,
        source_tokenizer=source_tokenizer,
        device='cuda',
        # device='cpu'    # debug
        force_dataparallel=force_dataparallel
    )

    return scorer