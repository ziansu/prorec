from models import (
    MomentumDualEncoderConfig,
    MomentumDualEncoderModel,
    LongelmConfig,
    LongelmModel,
    LongelmTokenizer,
    SrcProberConfig,
    SrcProberForConditionalGeneration
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
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    
    dualencoder_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    dualencoder_subfolder: str = field(
        metadata={"help": "Subfolder of checkpoint"}
    )
    asm_tokenizer_name_or_path: str = field(
        metadata={"help": "Path to pretrained assembly tokenizer"}
    )
    src_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    asm_feature_select_strategy: str = field(
        metadata={"help": "['all', 'nodes']"}
    )
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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
    de_config = MomentumDualEncoderConfig.from_pretrained(
        model_args.dualencoder_name_or_path,
        subfolder=model_args.dualencoder_subfolder,
        cache_dir=model_args.cache_dir,
    )
    assert de_config.K == 4096
    # load codet5p-embedding
    ct5p_embedding_model = AutoModel.from_pretrained(
        "Salesforce/codet5p-110m-embedding",
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    ct5p_embedding_model.config.use_cache = False  # only decoder should have use_cache as True

    # load casp
    de_config.source_config = ct5p_embedding_model.config

    dual_encoder = MomentumDualEncoderModel.from_pretrained(
        model_args.dualencoder_name_or_path,
        subfolder=model_args.dualencoder_subfolder,
        cache_dir=model_args.cache_dir,
        config=de_config,
        source_model=ct5p_embedding_model
    )

    asm_encoder = dual_encoder.assembly_model
    asm_encoder_config = dual_encoder.config.assembly_config
    asm_encoder.pooler = None   # remove pooler
    asm_tokenizer = LongelmTokenizer.from_pretrained(
        model_args.asm_tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        global_memory_size=asm_encoder_config.global_memory_size,
        node_size=asm_encoder_config.node_size,
        block_size=asm_encoder_config.block_size,
        max_blocks=asm_encoder_config.max_blocks,
    )
    # asm_tokenizer.max_blocks -= 1

    src_tokenizer = AutoTokenizer.from_pretrained(
        model_args.src_model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    # to use 4bit use `load_in_4bit=True` instead
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    src_model = AutoModelForCausalLM.from_pretrained(
        model_args.src_model_name_or_path,
        cache_dir=model_args.cache_dir,
        quantization_config=quantization_config,
    )
    src_tokenizer.add_tokens('<asm_token>')

    # build prober
    config = SrcProberConfig(
        ignore_index=-100,
        asm_token_index=src_tokenizer.convert_tokens_to_ids('<asm_token>'),
        asm_feature_select_strategy=model_args.asm_feature_select_strategy,
        asm_encoder_config=asm_encoder_config,
        src_lm_config=src_model.config
    )
    model = SrcProberForConditionalGeneration(
        config=config,
        asm_encoder=asm_encoder,
        src_language_model=src_model
    )
    model.resize_token_embeddings(len(src_tokenizer))
    print_trainable_parameters(model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

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

    if "starcoder" in model_args.src_model_name_or_path \
        or "CodeLlama" in model_args.src_model_name_or_path:
        src_tokenizer.pad_token = src_tokenizer.eos_token   # NOTE: check if correct
        model.config.pad_token_id = src_tokenizer.convert_tokens_to_ids(src_tokenizer.pad_token)
        def collate_fn(examples):
            # NOTE: currently process on the fly
            sources = [example['src'] for example in examples]
            sources = ['<asm_token>\n' + s for s in sources]    # '\n' or ' ' or ''
            source_inputs = src_tokenizer(
                sources,
                max_length=data_args.max_seq_length,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            assembly_inputs = asm_tokenizer.batch_inst_encode(
                [eval(example['codeart']) for example in examples]
            )
            return {
                'input_ids': source_inputs['input_ids'],
                'attention_mask': source_inputs['attention_mask'],
                'labels': source_inputs['input_ids'],
                'asm_input_ids': assembly_inputs['input_ids'],
                'asm_attention_mask': assembly_inputs['attention_mask'],
                'asm_graph_attention_mask': assembly_inputs['graph_attention_mask'],
                'asm_relative_node_positions': assembly_inputs['relative_node_positions'],
            }
    elif "llama" in model_args.src_model_name_or_path:
        raise NotImplementedError
    else:
        raise NotImplementedError("Code LMs other than Starcoder, Llama are not supported.")
    
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
        src_tokenizer.save_pretrained(training_args.output_dir)
        asm_tokenizer.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()