import sys
sys.path.append('../')
from tools.scorer import load_scorer


from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, DatasetDict
import json
import logging
import numpy as np
import os
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed,
)
from typing import Optional
import warnings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dualencoder_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dualencoder_subfolder: str = field(
        default=None, metadata={
            "help": "Subfolder of the model checkpoint, e.g. `checkpoint-1000`."
        }
    )
    source_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained model name or path if not the same as model_name"}
    )
    assembly_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained model name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    token: bool = field(
        default=False,
        metadata={
            "help": (
                "huggingface token"
            )
        },
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default='PurCL/lmpa-prober-new-dev'
    )
    probed_dataset_name: str = field(
        default='new_verbose_generation_results_diverse.json'
    )
    filtered_dataset_name: str = field(
        default=None
    )
    query_batch_size: int = field(
        default=64
    )
    encode_batch_size: int = field(
        default=32
    )
    output_dir: str = field(
        default=".cache/binary-source"
    )
    save_file: str = field(
        default='retrieval_results.json'
    )
    cpu: Optional[bool] = field(
        default=False
    )
    top_k: Optional[int] = field(
        default=3
    )

    def __post_init__(self):
        pass


def post_process(original_data_split, probed_data_split):
    "(in-place) post-processing of generated signatures, maybe move to probing later"
    
    strip_len = len('<s><asm_token>\n')
    processed_split = {
        'src': [],
        'codeart': [],
        'probed_sources': []
    }

    print('start post-processing...')
    for example, res in zip(tqdm(original_data_split), probed_data_split):
        # assert example['src'] == res['src']
        processed_split['src'].append(example['src'])  # FIXME: remove this in final test
        processed_split['codeart'].append(example['codeart'])
        new_probed_sources = []
        for ps in res['probed_sources']:
            new_ps = ps[strip_len:]
            if '{' in new_ps:
                new_ps = new_ps.split('{')[0]
            new_probed_sources.append(new_ps)
        processed_split['probed_sources'].append(new_probed_sources)
    return Dataset.from_dict(processed_split)


def score_and_filter_probed_signature(
    scorer,
    model_args,
    data_args,
    examples: Dataset
):
    "evaluate performance on generated signatures"

    # outer batch size (just to make sure inner batch is full and memory is not overloaded)
    batch_size = data_args.query_batch_size
    total_batch = len(examples) // batch_size + \
        (0 if len(examples) % batch_size == 0 else 1)

    filtered = {
        'candidate_signatures': [],
        'scores': [],
    }
    for bid in tqdm(range(total_batch)):
        batch_queries = []
        batch_candidates = []
        batch_srcs = []
        for example in examples.select(range(bid * batch_size, \
                                        min((bid+1) * batch_size, len(examples)))):
            batch_queries.append(eval(example['codeart']))
            batch_candidates += example['probed_sources']
            batch_srcs.append(example['src'])
        # use scorer to get results
        scores, top_candidates = scorer.get_top_n(
            queries=batch_queries,
            candidates=batch_candidates,
            n=data_args.top_k,
            encode_batch_size=data_args.encode_batch_size
        )

        filtered['scores'] += scores
        filtered['candidate_signatures'] += top_candidates

    results = []
    for scores, cands, example in zip(filtered['scores'], filtered['candidate_signatures'], examples):
        results.append({
            'src': example['src'],
            'top_candidates': cands
        })
    with open('scored_signatures.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(len(examples), len(filtered['scores']), len(''))
    examples = examples.add_column(name='signature_scores', column=filtered['scores'])
    examples = examples.add_column(name='candidate_signatures', column=filtered['candidate_signatures'])
    examples = examples.remove_columns('probed_sources')

    return examples


def main():
    set_seed(42)
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # load scorer
    scorer = load_scorer(
        model_args,
        force_dataparallel=False
    )

    # load dataset
    original_dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=model_args.cache_dir
    )
    if data_args.probed_dataset_name.endswith('.json'):
        with open(data_args.probed_dataset_name, 'r') as f:
            examples = json.load(f)
        probed_dataset = {'train': examples}
    elif data_args.probed_dataset_name.startswith('PurCL/'):
        probed_dataset = load_dataset(
            data_args.probed_dataset_name,
            cache_dir=model_args.cache_dir
        )
    else:
        raise NotImplementedError
    
    # in-place post-processing
    processed_split = post_process(original_dataset['validation'], probed_dataset['train'])

    # filter
    filtered_split = score_and_filter_probed_signature(
        scorer,
        model_args,
        data_args,
        processed_split
        # processed_split.select(range(65)) # debug
    )

    print(filtered_split)

    if data_args.filtered_dataset_name:
        DatasetDict({'validation': filtered_split}).push_to_hub(
            repo_id=data_args.filtered_dataset_name,
            private=True,
            token=model_args.token
        )
    else:
        print('Not uploading.')

if __name__ == '__main__':
    main()