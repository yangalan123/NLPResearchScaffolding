# Author: Chenghao Yang
# Based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# and https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoConfig
# seq2seq training_args is a subclass of training_args
from transformers import AutoTokenizer, default_data_collator, DataCollatorForSeq2Seq

from data_utils import create_dataset, preprocess_function_clf, preprocess_function_gen
from utils import setup_logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    preprocessing_num_workers: Optional[int] = field(
        default=10,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    task_family_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task_family (e.g., glue)"},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on under task_family (e.g., mnli)"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    max_seq_length: int = field(
        default=768,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    text_column: Optional[str] = field(default='text', metadata={
        "help": "The name of the column in the datasets containing the full texts."})
    label_column: Optional[str] = field(default='label', metadata={
        "help": "The name of the column in the datasets containing the full labels."})
    # for generation tasks only
    reference_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the reference (e.g., summary, translation)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=300,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "temperature for decoding"
            )
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": (
                "top_k for decoding"
            )
        },
    )
    top_p: Optional[float] = field(
        default=0.95,
        metadata={
            "help": (
                "top_p for decoding"
            )
        },
    )
    min_length: Optional[int] = field(
        default=250,
        metadata={
            "help": (
                "min_length for decoding"
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "max number of training samples being used"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "max number of evaluation samples being used"
            )
        },
    )

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            # raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
            assert self.dataset_name is not None or (
                    self.task_name is not None and self.task_family_name is not None), "Need a dataset to train the model"
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv",
                                       "json"], f"`train_file`({train_extension}) should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), f"`validation_file` should have the same extension (csv or json, now {validation_extension}!={train_extension}) as `train_file`."


@dataclass
class ScriptArguments:
    """
    Script-related Arguments
    """
    model_name_or_path: Optional[str] = field(
        default="/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-13b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    exp_name: Optional[str] = field(
        default="llama2-7b-peft-768",
        metadata={"help": "The name of the experiment, used to name the output directory."},
    )
    logger_file_path: Optional[str] = field(
        default="./log/train.log",
        metadata={"help": "The name of the logger file."},
    )
    mode: Optional[str] = field(
        default="classification",
        metadata={"help": "The mode of the experiment, either classification or generation."},
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "The name of the cache directory."},
    )
    prediction_file_path: Optional[str] = field(
        default="./prediction.txt",
        metadata={"help": "The name of the prediction output file."},
    )

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.logger_file_path), exist_ok=True)
        assert self.mode in ["classification",
                             "generation"], f"mode should be either classification or generation, now {self.mode} is not supported."


if __name__ == '__main__':
    # see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments for a complete
    # list of arguments
    parser = HfArgumentParser((ScriptArguments, DataArguments, Seq2SeqTrainingArguments))
    script_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = setup_logger(script_args.logger_file_path, training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.warning(f"Training/evaluation parameters {training_args}")
    logger.warning(f"Script parameters {script_args}")
    logger.warning(f"Data parameters {data_args}")

    # Step-0: create tokenizer (strings -> token ids)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # for llama specifically
    if "llama" in script_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
    # step-1: load dataset
    dataset = create_dataset(data_args=data_args, script_args=script_args, training_args=training_args, logger=logger)
    # step-2: preprocessing
    # get metadata information
    column_names = dataset["train"].column_names
    if script_args.mode == "classification":
        label_list = dataset["train"].unique(data_args.label_column)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    if script_args.mode == "generation":
        dataset = dataset.map(
            lambda examples: preprocess_function_gen(examples, data_args=data_args, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    elif script_args.mode == "classification":
        dataset = dataset.map(
            lambda examples: preprocess_function_clf(examples, data_args=data_args, tokenizer=tokenizer),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        raise NotImplementedError(f"mode {script_args.mode} is not supported")
    # step-3: create the model, first at CPU
    if script_args.mode == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            num_labels=num_labels,
        )
        # logging.warning(f"model name: {script_args.model_name_or_path} loaded as AutoModelForSequenceClassification")
    elif script_args.mode == "generation":
        try:
            # decoder-only
            model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
        except:
            # seq2seq
            model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path)
    # step-4: create the trainer
    if script_args.mode == "classification":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else dataset['test'],
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    else:
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'] if training_args.do_train else None,
            eval_dataset=dataset['validation'] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        train_dataset = dataset['train']
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        eval_dataset = dataset['validation'] if "validation" in dataset else dataset['test']
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = dataset['test']
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams,
            temperature=data_args.temperature,
            top_k=data_args.top_k,
            top_p=data_args.top_p,
            min_new_tokens=data_args.min_length
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        reshaped_predictions = predict_results.predictions.reshape(-1, predict_results.predictions.shape[-1])
        if trainer.is_world_process_zero():
            predictions = tokenizer.batch_decode(
                reshaped_predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            with open(script_args.prediction_file_path, "w") as writer:
                writer.write("\n\n\n".join(predictions))

    kwargs = {"finetuned_from": script_args.model_name_or_path, "tasks": data_args.task_name}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
