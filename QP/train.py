import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from collections import namedtuple

import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from data import DatasetArguments, prepare_dataset
import json

check_min_version("4.6.0")
logger = logging.getLogger(__name__)


@dataclass
class TaskArguments:
    run_tags: str = field(
        default='',
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    input_columns: str = field(
        default='["label"]',
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    label_columns: str = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    #conda install -c huggingface transformers

args_config = {
    'model': ModelArguments, 
    'data': DatasetArguments, 
    'task': TaskArguments,
    'train': TrainingArguments
}
parser = HfArgumentParser(args_config.values())
Args = namedtuple('Args', args_config.keys())
def parse_args():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = Args(*parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])))
    else:
        args = Args(*parser.parse_args_into_dataclasses())
    return args

def main():

    from datasets import set_caching_enabled
    set_caching_enabled(False)

    config = parse_args()
    print(config.train.run_name)
    if config.train.run_name and config.train.run_name != config.train.output_dir:
        import clearml
        task = clearml.Task.init(project_name="tslm/tslm-gen", task_name=f"{config.train.run_name}", tags=[tag for tag in config.task.run_tags.replace(' ', '').split(',') if tag])    
        task.set_resource_monitor_iteration_timeout(0)
        parse_args()

    # Detecting last checkpoint.
    checkpoint = None
    if os.path.isdir(config.train.output_dir) and config.train.do_train and not config.train.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(config.train.output_dir)
        if last_checkpoint is None and len(os.listdir(config.train.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({config.train.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and config.train.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            checkpoint = last_checkpoint
        else:
            checkpoint = config.train.resume_from_checkpoint


    datasets = prepare_dataset(config.data, logger)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {config.train.local_rank}, device: {config.train.device}, n_gpu: {config.train.n_gpu}"
        + f"distributed training: {bool(config.train.local_rank != -1)}, 16-bits training: {config.train.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(config.train.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Training/evaluation parameters {config.train}")

    set_seed(config.train.seed)
    print(config.task.input_columns)
    inputs = json.loads(config.task.input_columns)
    inputs.sort()
    print('Input Columns:', inputs)
    
    labels = json.loads(config.task.label_columns)
    labels.sort()
    print('Label Columns:', labels)

    num_labels = len(labels)
    is_regression = True

    model_config = AutoConfig.from_pretrained(
        config.model.config_name if config.model.config_name else config.model.model_name_or_path,
        num_labels=num_labels,
        problem_type='regression',
        finetuning_task='custom',
        cache_dir=config.model.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
        cache_dir=config.model.cache_dir,
        use_fast=config.model.use_fast_tokenizer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        from_tf=bool(".ckpt" in config.model.model_name_or_path),
        config=model_config,
        cache_dir=config.model.cache_dir,
    )


    if config.task.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False


    if config.task.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({config.task.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    
    max_seq_length = min(config.task.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            tuple(examples[col] for col in inputs)
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        
        result["label"] = list(zip(*tuple(examples[col] for col in labels)))
        return result

    if config.train.do_predict:
        predict_dataset = datasets['validation']
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not config.task.overwrite_cache)
    
    # Log a few random samples from the training set:
    if config.train.do_train:
        for index in random.sample(range(len(datasets['train'])), 3):
            logger.info(f"Sample {index} of the training set: {datasets['train'][index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if config.task.pad_to_max_length:
        data_collator = default_data_collator
    elif config.train.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=config.train,
        train_dataset=datasets['train'] if config.train.do_train else None,
        eval_dataset=datasets['validation'] if config.train.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if config.train.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        metrics["train_samples"] = len(datasets['train'])

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if config.train.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=datasets['validation'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if config.train.do_predict:
        logger.info("*** Predict ***")
        predict_dataset.remove_columns(labels)
        predictions = trainer.predict(datasets['validation'], metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        results = []
        for orig, pred in zip(predict_dataset, predictions):
            input_data = {col:orig[col] for col in inputs}
            pred_data = {col:p for col, p in zip(labels, pred)}
            results.append({**input_data, **pred_data})
        results_file = os.path.join(config.train.output_dir, f"predict_results.csv.gz")
        pd.DataFrame(results).to_csv(results_file, index=False)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
