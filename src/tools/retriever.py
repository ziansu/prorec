import faiss
import logging
import json
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union, Tuple, Optional, Dict

from datasets import load_dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossModalRetriever(object):

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
        
        self.index = None
    
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
            for batch_id in tqdm(range(total_batch)):
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

    def load_index(
        self,
        index_path: str
    ):
        # load embeddings
        embeddings = np.load(os.path.join(index_path, 'key_embeddings.npy'))

        # load keys
        with open(os.path.join(index_path, 'keys.pkl'), 'rb') as f:
            keys = pickle.load(f)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = {"keys": keys, "index": index}

    def build_index(
        self,
        keys_or_file_path: Union[str, List[str]],
        encoder_type: str = 'source',
        batch_size: int = 64,
        index_path: str = None
    ):
        "Build index using FAISS"
        if isinstance(keys_or_file_path, str):
            raise NotImplementedError
        else:
            keys = keys_or_file_path

        logger.info("Encoding embeddings for keys...")
        embeddings = self.encode(
            examples=keys,
            batch_size=batch_size,
            encoder_type=encoder_type,
            normalize_to_unit=True,
            return_numpy=True
        )

        logger.info("Building index...")
        self.index = {"keys": keys}

        index = faiss.IndexFlatIP(embeddings.shape[1]) # TODO: acceleration options if too slow
        index.add(embeddings)
        self.index["index"] = index
        logger.info("Finished building index.")

        # cache
        if index_path:
            if not os.path.exists(index_path):
                os.mkdir(index_path)
            # store embeddings
            np.save(os.path.join(index_path, 'key_embeddings.npy'), embeddings)
            
            # store keys
            with open(os.path.join(index_path, 'keys.pkl'), 'wb') as f:
                pickle.dump(self.index["keys"], f)

    def search(
        self,
        queries: List[str],
        encoder_type: str = None,
        threshold: float = 0.6,
        top_k: int = 5,
    ):
        "Dynamically search for document given built index"
        assert self.index is not None

        query_vecs = self.encode(
            examples=queries,
            batch_size=64,
            encoder_type=encoder_type,
            normalize_to_unit=True,
            return_numpy=True
        )

        D, I = self.index["index"].search(query_vecs, top_k)

        def pack_single_result(dist, idx):
            results = [(self.index["keys"][i], s) for i, s in zip(idx, dist) if s >= threshold]
            return results
        
        if isinstance(queries, list):
            combined_results = []
            for i in range(len(queries)):
                results = pack_single_result(D[i], I[i])
                combined_results.append(results)
            return combined_results
        else:
            return pack_single_result(D[0], I[0])
        
    def search_for_index(
        self,
        queries: List[str],
        encoder_type: str = None,
        threshold: float = 0.6,
        top_k: int = 5,
    ):
        "Dynamically search for document given built index"
        assert self.index is not None

        query_vecs = self.encode(
            examples=queries,
            batch_size=64,
            encoder_type=encoder_type,
            normalize_to_unit=True,
            return_numpy=True
        )

        D, I = self.index["index"].search(query_vecs, top_k)
        
        if isinstance(queries, list):
            combined_results = []
            for i in range(len(queries)):
                results = (D[i], I[i])
                combined_results.append(results)
            return combined_results
        else:
            return (D[0], I[0])

    @staticmethod
    def retrieve(
        source_embed,
        target_embed,
        source_id_map,
        target_id_map,
        top_k
    ):
        print(top_k, source_embed.shape[0])
        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        # print(f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['index'] = source_idx
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['retrieved_index'] = retrieved_index
            results[source_id]['score'] = dist.tolist()

        return results, D, I
    
    @staticmethod
    def retrieve_from_file(
        source_embed_file,
        target_embed_file,
        source_id_file,
        target_id_file,
        pool_size,
        top_k,
        save_file,
    ):
        with open(source_id_file, 'r') as f:
            source_id_map = {}
            for idx, line in enumerate(f.readlines()[:pool_size]):
                source_id_map[idx] = line.strip()

        with open(target_id_file, 'r') as f:
            target_id_map = {}
            for idx, line in enumerate(f.readlines()[:pool_size]):
                target_id_map[idx] = line.strip()

        source_embed = np.load(source_embed_file + '.npy')
        target_embed = np.load(target_embed_file + '.npy')
        assert (len(source_id_map) == source_embed.shape[0])
        assert (len(target_id_map) == target_embed.shape[0])
        indexer = faiss.IndexFlatIP(target_embed.shape[1])
        indexer.add(target_embed)
        print(
            f'source embedding shape: {source_embed.shape}, target embedding shape: {target_embed.shape}')
        D, I = indexer.search(source_embed, top_k)

        results = {}
        for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
            source_id = source_id_map[source_idx]
            results[source_id] = {}
            retrieved_target_id = [target_id_map[x] for x in retrieved_index]
            results[source_id]['retrieved'] = retrieved_target_id
            results[source_id]['score'] = dist.tolist()

        with open(save_file, 'w+') as f:
            json.dump(results, f, indent=2)

        return results


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


def load_retriever(
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
    retriever = CrossModalRetriever(
        dual_encoder=dual_encoder,
        assembly_tokenizer=assembly_tokenizer,
        source_tokenizer=source_tokenizer,
        device='cuda',
        # device='cpu'    # debug
        force_dataparallel=force_dataparallel
    )

    return retriever