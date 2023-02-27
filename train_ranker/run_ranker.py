import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
# from datasets import load_dataset, load_metric
from data_utils.dataset import ReRankingDataset_train, ReRankingDataset_eval
from model_utils import UnixcoderRanker
from trainer_utils.trainer import Trainer
from data_utils.data_collator import DataCollatorForReranking_train,DataCollatorForReranking_eval
from data_utils.metric_utils import compute_ranked_pass_k
from trainer_utils.training_args import TrainingArguments

import transformers
from transformers import (
    HfArgumentParser,
    PretrainedConfig,
    set_seed,
)
from transformers import RobertaConfig, RobertaTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch
# from unixcoder import UniXcoder
from unixcoder import UniXcoder

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    
    step1_max_num : Optional[int] = field(
        default=3,
        metadata={
            "help": "step1 bank length"
        },
    )
    
    step2_max_num : Optional[int] = field(
        default=3,
        metadata={
            "help": "step2 bank length"
        },
    )

    max_source_length : Optional[int] = field(
        default=400,
        metadata={
            "help": "max source length"
        },
    )

    max_candidate_length : Optional[int] = field(
        default=600,
        metadata={
            "help": "max candidate length"
        },
    )

    train_data_path: Optional[str] = field(
        default="/home2/partdata"
    )

    dev_data_path: Optional[str] = field(
        default="/home2/datasets/partdata/eval.pkl"
    )

    test_data_path: Optional[str] = field(
        default=None
    )
    
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="microsoft/unixcoder-base",metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    loss_type: str = field(
        default="contrastive"
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    position_extend_way: str = field(
        default='normal',
        metadata={
            "help": "to initialize the new position embedding weights from normal (normal) "
            "or copying from trained position embedding of the original model (copys)"
        },
    )
    model_type: str = field(
        default='unixcoder'
    )
    temperature : Optional[float] = field(
        default=0.05,
    )
    



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if model_args.model_type.lower() == 'unixcoder':
        config = RobertaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            # cache_dir="huggingcace/unixcoder",
        )
        config.is_decoder = True
        setattr(config, "loss_type", model_args.loss_type)
        setattr(config, "model_type", model_args.model_type)
        setattr(config, "temperature", model_args.temperature)
        
        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            # cache_dir="huggingcace/unixcoder",
        )
        tokenizer.add_tokens(["<mask0>"],special_tokens=True)
        model = UnixcoderRanker.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            # cache_dir="huggingcace/unixcoder",
        )


    if data_args.max_source_length + data_args.max_candidate_length + 7 > config.max_position_embeddings:
        # How to understand + 7: 
        # the total max input length is data_args.max_source_length + data_args.max_candidate_length + 5 (cls+mode+eos+source+seq+target+eos)
        # and for roberta positional ids starts for 2 (the first position is 2, padding is 1, 0 is unused)
        # therefore total number for new position embedding is data_args.max_source_length + data_args.max_candidate_length + 7
        model._resize_position_embedding(data_args.max_source_length + data_args.max_candidate_length + 7, extend_way = model_args.position_extend_way)
        config.max_position_embeddings = data_args.max_source_length + data_args.max_candidate_length + 7


        
    
    if training_args.do_train:
        if data_args.train_data_path is None:
            raise ValueError("There should be train_data_path")
        data_file = data_args.train_data_path
        train_dataset = ReRankingDataset_train(data_file, tokenizer=tokenizer, args = data_args)  
        train_dataset.read_data(1)  
        
    if training_args.do_eval:
        if data_args.dev_data_path is None:
            raise ValueError("There should be dev_data_path")
        data_file = data_args.dev_data_path    
        eval_dataset = ReRankingDataset_eval(data_file,args = data_args)

    if training_args.do_predict:
        if data_args.test_data_path is None:
            raise ValueError("There should be dev_data_path")
        data_file = data_args.test_data_path
        test_dataset = ReRankingDataset_eval(data_file, args = data_args)




    data_collator = DataCollatorForReranking_train(tokenizer, model_type = model_args.model_type)
    eval_data_collator = DataCollatorForReranking_eval(tokenizer, model_type = model_args.model_type)
    compute_metrics = compute_ranked_pass_k()
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        data_collator_eval = eval_data_collator   
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")

        metrics = predict_results.metrics

        metrics["predict_samples"] =len(test_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)







def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()