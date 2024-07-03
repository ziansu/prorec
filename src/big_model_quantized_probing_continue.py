# continue probing of generated signatures

from models import (
    LongelmTokenizer,
    SrcProberConfig,
    SrcProberForConditionalGeneration
)

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, Dict, List

from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.inference import prepare_pippy, PartialState
from accelerate.utils import gather_object

import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    GenerationConfig,
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
    
    prober_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    prober_subfolder: str = field(
        metadata={"help": "Subfolder of checkpoint"}
    )
    asm_tokenizer_name_or_path: str = field(
        metadata={"help": "Path to pretrained assembly tokenizer"}
    )
    src_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
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
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization options for base source code language model: 4bit, 8bit, None"}
    )


@dataclass
class DataInferenceArguments:
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
    output_file: Optional[str] = field(
        default=None, metadata={"help": "The output file name."}
    )
    num_return_sequences: Optional[int] = field(
        default=5, metadata={"help": "Number of returned sequences per example."}
    )
    max_new_tokens: Optional[int] = field(
        default=100, metadata={"help": "Maximum new tokens to generate."}
    )
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams for beam search."}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_k: Optional[int] = field(
        default=50, metadata={"help": "Top-k for random sampling."}
    )
    top_p: Optional[float] = field(
        default=1.0, metadata={"help": "Top-p for nucleus sampling."}
    )

    # split for parallel inference
    parallel_inference_split: Optional[str] = field(
        default='1/1', metadata={"help": "split_id/total_splits"}
    )

    max_inference_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of samples for inference."}
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


def batch_inference(
    model,
    asm_tokenizer,
    src_tokenizer,
    dataset_split,
    batch_size,
    generation_config,
    process_index,
    n_candidates=5,
) -> List[Dict]:
    
    num_return_sequences = generation_config.num_return_sequences
    src_tokenizer.pad_token = src_tokenizer.eos_token  # check correctness
    
    results = []
    n_batch = len(dataset_split) // batch_size + (1 if len(dataset_split) % batch_size > 0 else 0)
    for i in tqdm(range(n_batch)):
        examples = dataset_split[i * batch_size: (i+1) * batch_size]
        raw_asms = []
        for example in examples:
            for _ in range(n_candidates):
                raw_asms.append(eval(example['codeart']))
        asm_inputs = asm_tokenizer.batch_inst_encode(raw_asms)
        # source_inputs = ['<asm_token> '] * len(examples)
        source_inputs = ['<asm_token>\n' + sig for example in examples \
            for sig in eval(example['candidate_signatures'])[:n_candidates]]
        # print(source_inputs)
        assert(len(raw_asms) == len(source_inputs))
        source_inputs = src_tokenizer(
            source_inputs,
            return_tensors='pt',
            padding='longest',
        )
        input_dict = {
            'input_ids': source_inputs['input_ids'],
            'attention_mask': source_inputs['attention_mask'],
            'asm_input_ids': asm_inputs['input_ids'],
            'asm_attention_mask': asm_inputs['attention_mask'],
            'asm_graph_attention_mask': asm_inputs['graph_attention_mask'],
            'asm_relative_node_positions': asm_inputs['relative_node_positions'],
        }
        input_dict = {k: v.to(f'cuda:{process_index}') for k, v in input_dict.items()}
        outputs = model.generate(
            **input_dict,
            generation_config=generation_config
        )
        decoded_outputs =  src_tokenizer.batch_decode(outputs)
        for j, example in enumerate(examples):
            js = {
                'oracle_source': example['src'],
                'probed_sources': \
                    decoded_outputs[j * num_return_sequences * n_candidates: \
                                    (j+1) * num_return_sequences * n_candidates]
            }
            results.append(js)

    return results


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataInferenceArguments, TrainingArguments))
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
                # use_auth_token=True if model_args.use_auth_token else None,
                # streaming=data_args.streaming,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
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

    # 5. load model and tokenizer
    accelerator = Accelerator()
    src_tokenizer = AutoTokenizer.from_pretrained(
        model_args.src_model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    # to use 4bit use `load_in_4bit=True` instead
    if model_args.quantization == '4bit':
        # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    elif model_args.quantization == '8bit':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
        raise NotImplementedError("Sharding not implemented yet.")
    src_model = AutoModelForCausalLM.from_pretrained(
        model_args.src_model_name_or_path,
        cache_dir=model_args.cache_dir,
        quantization_config=quantization_config,
    )
    src_tokenizer.add_tokens('<asm_token>')
    src_tokenizer.pad_token = src_tokenizer.eos_token
    src_model.resize_token_embeddings(len(src_tokenizer))
    config = SrcProberConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.prober_name_or_path,
        src_lm_config=src_model.config,
        subfolder=model_args.prober_subfolder,
        cache_dir=model_args.cache_dir
    )
    model = SrcProberForConditionalGeneration.from_pretrained(
        model_args.prober_name_or_path,
        config=config,
        src_language_model="empty",
        subfolder=model_args.prober_subfolder,
        cache_dir=model_args.cache_dir,
        # device_map={"": accelerator.process_index},
        torch_dtype=torch.float16
        # torch_dtype=torch.bfloat16
    )
    model.src_language_model = src_model
    model = model.eval()
    model = model.to('cuda')

    asm_tokenizer = LongelmTokenizer.from_pretrained(
        model_args.asm_tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        global_memory_size=config.asm_encoder_config.global_memory_size,
        node_size=config.asm_encoder_config.node_size,
        block_size=config.asm_encoder_config.block_size,
        max_blocks=config.asm_encoder_config.max_blocks,
    )

    generation_config = GenerationConfig(
        bos_token_id=src_tokenizer.convert_tokens_to_ids('<asm_token>'),
        eos_token_id=src_tokenizer.eos_token_id,
        pad_token_id=src_tokenizer.eos_token_id,
        max_new_tokens=data_args.max_new_tokens,
        use_cache=True,
        do_sample=True,
        # manipulation of model output logits
        temperature=data_args.temperature,
        top_k=data_args.top_k,
        top_p=data_args.top_p,

        # define output variables
        num_return_sequences=data_args.num_return_sequences,
        output_scores=False,
        return_dict_in_generate=False,

        num_beams=data_args.num_beams,
    )

    # TODO: adjust data loading here
    if 'train' in dataset and 'validation' in dataset:
        examples_all = [e for e in dataset['train']] + [e for e in dataset['validation']]
    elif 'train' in dataset:
        examples_all = [e for e in dataset['train']]
    elif 'validation' in dataset:
        examples_all = [e for e in dataset['validation']]
    elif 'test' in dataset:
        examples_all = [e for e in dataset['test']]
    else:
        assert 0

    if data_args.max_inference_samples is not None:
        examples_all = examples_all[:data_args.max_inference_samples]

    parallel_inference_split_index = int(data_args.parallel_inference_split.split('/')[0])
    parallel_inference_num_splits = int(data_args.parallel_inference_split.split('/')[1])
    parallel_inference_split_size = len(examples_all) // parallel_inference_num_splits + \
                        (1 if len(examples_all) % parallel_inference_num_splits > 0 else 0)
    examples_all = examples_all[(parallel_inference_split_index-1)*parallel_inference_split_size:\
                                parallel_inference_split_index*parallel_inference_split_size]

    # print(examples_all[0]['src'][:500])

    # prober_dataset = load_dataset(
    #     'PurCL/lmpa-prober-new-dev',
    #     cache_dir=model_args.cache_dir,
    #     data_dir=data_args.data_dir,
    #     token=model_args.token,
    # )
    # print('*' * 40)
    # print(prober_dataset['validation'][0]['src'][:500])

    # exit()

    with accelerator.split_between_processes(examples_all) as examples:
        results = batch_inference(
            model,
            asm_tokenizer,
            src_tokenizer,
            examples,
            batch_size=training_args.per_device_eval_batch_size,
            generation_config=generation_config,
            process_index=accelerator.process_index
        )

    results_gathered = gather_object(results)

    if accelerator.is_main_process:
        # TODO: post-processing

        with open(os.path.join(training_args.output_dir, data_args.output_file), 'w') as f:
            json.dump(results_gathered, f, indent=2)


if __name__ == '__main__':
    main()