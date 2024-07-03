from tools import load_retriever
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    HfArgumentParser
)
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
    augmented_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the augmented dataset."}
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
    retriever.load_index(model_args.index_dir)

    def transforms(examples):

        examples['retrieved_index'] = retriever.search_for_index(
            [eval(ca) for ca in examples['codeart']],
            encoder_type='assembly',
            threshold=0.4,
            top_k=model_args.top_k
        )
        return examples

    # columns = ['codeart']
    columns = []

    # ra_train = matched_dataset['train'].select(range(100000)).map(
    ra_train = matched_dataset['train'].map(
        transforms,
        batched=True,
        batch_size=data_args.batch_size,
        remove_columns=columns,
        desc="Running direct binary-to-source retrieval on train dataset"
    )
    if 'validation' in matched_dataset:
        # ra_valid = matched_dataset['validation'].select(range(10000)).map(
        ra_valid = matched_dataset['validation'].map(
            transforms,
            batched=True,
            batch_size=data_args.batch_size,
            remove_columns=columns,
            desc="Running direct binary-to-source retrieval on validation dataset"
        )
    else:
        # ra_valid = matched_dataset['test'].select(range(10000)).map(
        ra_valid = matched_dataset['test'].map(
            transforms,
            batched=True,
            batch_size=data_args.batch_size,
            remove_columns=columns,
            desc="Running direct binary-to-source retrieval on validation dataset"
        )
    # print(matched_dataset)
    DatasetDict({'train': ra_train, 'validation': ra_valid}).save_to_disk(
        os.path.join(model_args.cache_dir, 'lmpa-direct-ra'),
        max_shard_size='1GB'
    )


if __name__ == '__main__':
    main()