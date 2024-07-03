from tools import load_retriever
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser
)
from tqdm import tqdm
from typing import Optional

import os
import sys


@dataclass
class ModelArguments:
    dualencoder_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    dualencoder_subfolder: Optional[str] = field(
        default=None, metadata={"help": "Subfolder of the dual encoder"}
    )
    source_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    assembly_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    index_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to load the index."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    top_k: Optional[int] = field(
        default=3, metadata={"help": "Top-k relevant source code functions to retrieve"}
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    batch_size: Optional[int] = field(
        default=64
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # load matched dataset
    matched_dataset = load_dataset(
        data_args.dataset_name,
        cache_dir=model_args.cache_dir,
    )

    # load retriever
    retriever = load_retriever(model_args, force_dataparallel=True)

    # prepare keys (source code), get rid of duplicates
    keys = set()
    for example in tqdm(matched_dataset['train']):
        keys.add(example['src'])
    keys = list(keys)
    print('Total number of unique keys: {}'.format(len(keys)))

    # asynchronous retrieval
    retriever.build_index(
        keys_or_file_path=keys,
        encoder_type='source',
        batch_size=data_args.batch_size,
        index_path=model_args.index_dir,
    )


if __name__ == '__main__':
    main()