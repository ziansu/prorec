#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""
from models import (
    DualEncoderConfig,
    MomentumDualEncoderConfig,
    DualEncoderModel,
    MomentumDualEncoderModel,
    LongelmConfig,
    LongelmModel,
    LongelmTokenizer,
)

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.35.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    
    assembly_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    source_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    # config_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # )
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )
    # image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_assembly_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the assembly model parameters or not."}
    )
    freeze_source_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the source model parameters or not."}
    )

    use_momentum_encoder: bool = field(
        default=True, metadata={"help": "Whether to use momentum encoder for contrastive learning"}
    )
    queue_size: int = field(
        default=128, metadata={"help": "The size of the queue"}
    )
    momentum: float = field(
        default=0.999, metadata={"help": "Momentum"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    # image_column: Optional[str] = field(
    #     default="image_path",
    #     metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    # )
    # caption_column: Optional[str] = field(
    #     default="caption",
    #     metadata={"help": "The name of the column in the datasets containing the image captions."},
    # )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # NOTE: brute force modification
    if model_args.assembly_model_name_or_path == 'None':
        model_args.assembly_model_name_or_path = None


    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=model_args.token,
        )
        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                # streaming=data_args.streaming,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                # streaming=data_args.streaming,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 5. Load pretrained model, tokenizer
    
    # load config
    if model_args.assembly_model_name_or_path:
        assembly_config = LongelmConfig.from_pretrained(
            model_args.assembly_model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code
        )
    else:
        model_args.global_memory_size = 1
        model_args.node_size = 1
        model_args.block_size = 8
        model_args.max_blocks = 400
        model_args.num_hidden_layers = 6
        model_args.max_relative_position_embeddings = 8

        assembly_tokenizer = LongelmTokenizer.from_pretrained(
            './tokenizer/3m/',
            block_size=model_args.block_size,
            node_size=model_args.node_size,
            max_blocks=model_args.max_blocks,
            global_memory_size=model_args.global_memory_size,
        )

        assembly_config = LongelmConfig(
            max_position_embeddings=(model_args.node_size + model_args.block_size) *
            model_args.max_blocks + model_args.global_memory_size + (assembly_tokenizer.pad_token_id + 1),   # avoid IndexError of positional embedding due to padding idx
            max_relative_position_embeddings=model_args.max_relative_position_embeddings,
            global_memory_size=model_args.global_memory_size,
            node_size=model_args.node_size,
            block_size=model_args.block_size,
            max_blocks=model_args.max_blocks,
            ep_add_linear_projection=False,
            num_hidden_layers=model_args.num_hidden_layers
        )
        
    source_config = AutoConfig.from_pretrained(
        model_args.source_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )
    source_config.use_cache = False  # only decoder should have use_cache as True

    # load tokenizer
    if model_args.assembly_model_name_or_path:
        assembly_tokenizer = LongelmTokenizer.from_pretrained(
            model_args.assembly_model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            global_memory_size=assembly_config.global_memory_size,
            node_size=assembly_config.node_size,
            block_size=assembly_config.block_size,
            max_blocks=assembly_config.max_blocks,
        )
    source_tokenizer = AutoTokenizer.from_pretrained(
        model_args.source_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )

    # load pretrained model
    if model_args.assembly_model_name_or_path:
        assembly_encoder = LongelmModel.from_pretrained(
            model_args.assembly_model_name_or_path,
            add_pooling_layer=True,     # NOTE: using default pooler projection
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code
        )
    else:
        assembly_encoder = LongelmModel(assembly_config)
    source_encoder = AutoModel.from_pretrained(
        model_args.source_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # build dual encoder
    if model_args.use_momentum_encoder:
        config = MomentumDualEncoderConfig(
            projection_dim=512,
            logit_scale_init_value=2.6592,
            K=model_args.queue_size,
            m=model_args.momentum,
            assembly_config=assembly_config,
            source_config=source_config,
        )
        model = MomentumDualEncoderModel(
            config=config,
            assembly_model=assembly_encoder,
            source_model=source_encoder,
        )
    else:
        config = DualEncoderConfig(
            projection_dim=512,
            logit_scale_init_value=2.6592,
            assembly_config=assembly_config,
            source_config=source_config,
        )
        model = DualEncoderModel(
            config=config,
            assembly_model=assembly_encoder,
            source_model=source_encoder,
        )
    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    # TODO: freeze params

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    # 6. Get the column names for input/target.
    source_column_name = 'src'
    assembly_column_name = 'codeart'

    # 7. Preprocessing the datasets.

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))

    def collate_fn(examples):
        # NOTE: currently process on the fly
        sources = [example[source_column_name] for example in examples]
        sources = [src.split('{')[0] for src in sources]
        # print(sources)
        # assert 0
        source_inputs = source_tokenizer(
            sources,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        assembly_inputs = assembly_tokenizer.batch_inst_encode(
            [eval(example[assembly_column_name]) for example in examples]
        )
        return {
            'source_input_ids': source_inputs['input_ids'],
            'source_attention_mask': source_inputs['attention_mask'],
            'assembly_input_ids': assembly_inputs['input_ids'],
            'assembly_attention_mask': assembly_inputs['attention_mask'],
            'assembly_graph_attention_mask': assembly_inputs['graph_attention_mask'],
            'assembly_relative_node_positions': assembly_inputs['relative_node_positions'],
            'return_loss': True
        }

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )
    
    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        source_tokenizer.save_pretrained(training_args.output_dir)
        assembly_tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "contrastive-image-text-modeling"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()