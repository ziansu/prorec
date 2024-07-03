from copy import deepcopy
import networkx as nx
from networkx.algorithms.shortest_paths import floyd_warshall_numpy
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import time


class LongelmTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        node_size=1,
        block_size=8,
        max_blocks=200,
        global_memory_size=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_size = node_size
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.global_memory_size = global_memory_size

        self.inst_token = '<INST>'

    def create_relative_node_positions(
        self,
        block_count,
        deps
    ):
        # filter out out-of-range dependencies
        assert block_count <= self.max_blocks
        remaining_data_dep = []
        for source_node_id, target_node_id in deps:
            if source_node_id >= block_count or target_node_id >= block_count:  # support truncation
                continue
            else:
                remaining_data_dep.append((source_node_id, target_node_id))

        # graph construction
        graph = nx.Graph()
        graph.add_nodes_from(range(block_count))
        graph.add_edges_from(remaining_data_dep)

        # dense graph all pairs shortest path length
        spl_matrix = floyd_warshall_numpy(graph)
        spl_matrix[spl_matrix == np.inf] = -1
        spl_matrix = torch.tensor(
            spl_matrix, dtype=torch.long)

        # TODO: expand as node size and reshape as final matrix

        return spl_matrix

    def inst_encode(
        self,
        code: List[Tuple[int, str]],
        deps: List[Tuple[int, int]],
        return_extra_info=False,
    ):
        tokens = []
        block_count = 0
        special_tokens_mask = []
        
        for inst_id, instruction in code:   # NOTE: this can be accelerated by batch encode
            if inst_id >= self.max_blocks:
                break
            instruction_tokens = self.tokenize(instruction)
            block_count += 1

            # pad or truncate block
            instruction_tokens = instruction_tokens[:self.block_size]
            tokens += instruction_tokens
            special_tokens_mask += [0] * len(instruction_tokens)
            if len(instruction_tokens) < self.block_size:
                tokens += [self.pad_token] * (self.block_size - len(instruction_tokens))
                special_tokens_mask += [1] * (self.block_size - len(instruction_tokens))

        # pad blocks to max_blocks
        # (NOTE: in practice, just make sure each instance in batch has same block_count)
        if block_count <= self.max_blocks:
            tokens += [self.pad_token] * self.block_size * (self.max_blocks - block_count)
            special_tokens_mask += [1] * self.block_size * (self.max_blocks - block_count)

        attention_mask = (~torch.tensor(special_tokens_mask, dtype=torch.bool)).int()

        # nodes
        tokens += [self.inst_token] * self.max_blocks * self.node_size
        special_tokens_mask += [1] * self.max_blocks * self.node_size
        
        # global memory
        tokens += [self.cls_token] * self.global_memory_size
        special_tokens_mask += [1] * self.global_memory_size

        # convert tokens to ids
        input_ids = self.convert_tokens_to_ids(tokens)

        relative_node_positions = self.create_relative_node_positions(
            self.max_blocks,
            deps
        )

        if return_extra_info:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'special_tokens_mask': torch.tensor(special_tokens_mask, dtype=torch.int), \
                'relative_node_positions': relative_node_positions
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long), \
                'attention_mask': attention_mask, \
                'relative_node_positions': relative_node_positions
            }
    
    def batch_inst_encode(
        self,
        examples,
        max_transitions=None,
    ):
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'graph_attention_mask': [],
            'relative_node_positions': []
        }
        
        for example in examples:
            if isinstance(example['code'], str):
                encoded = self.inst_encode(eval(example['code']), eval(example['data_dep']))
            else:
                encoded = self.inst_encode(example['code'], example['data_dep'])
            batch['input_ids'].append(encoded['input_ids'])
            batch['attention_mask'].append(encoded['attention_mask'])
            batch['graph_attention_mask'].append(encoded['relative_node_positions'] >= 0)
            batch['relative_node_positions'].append(encoded['relative_node_positions'])
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'graph_attention_mask': torch.stack(batch['graph_attention_mask']),
            'relative_node_positions': torch.stack(batch['relative_node_positions'])
        }