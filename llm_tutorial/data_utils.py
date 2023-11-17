from datasets import load_dataset
def create_dataset(data_args, script_args, training_args, logger):
    if data_args.task_name is not None and data_args.task_family_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.task_family_name,
            data_args.task_name,
            cache_dir=script_args.cache_dir,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=script_args.cache_dir,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
                print(f"set up test files : {data_args.test_file}")
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=script_args.cache_dir,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=script_args.cache_dir,
            )
    return raw_datasets

def example_input_fn_clf(examples, data_args):
    return examples[data_args.text_column]


def preprocess_function_clf(examples, tokenizer, data_args, label_to_id=None, padding="max_length", compose_input_fn=example_input_fn_clf):
    # extract the input from batched samples
    # used with normal trainer
    input_examples = compose_input_fn(examples, data_args)
    # Tokenize the texts
    result = tokenizer(input_examples, padding=padding, max_length=data_args.max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and data_args.label_column in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[data_args.label_column]]
    if data_args.label_column != "label" and "label" in examples:
        result["label"] = examples[data_args.label_column]
        result['original_label'] = examples['label']
    return result

def example_input_fn_gen(examples, data_args, prefix="Please Generate a Response: "):
    inputs, targets = [], []
    for i in range(len(examples[data_args.text_column])):
        if examples[data_args.text_column][i] and examples[data_args.reference_column][i]:
            inputs.append(examples[data_args.text_column][i])
            targets.append(examples[data_args.reference_column][i])

    inputs = [prefix + inp for inp in inputs]
    return inputs, targets
def preprocess_function_gen(examples, tokenizer, data_args, padding='max_length', compose_input_fn=example_input_fn_gen):
    # remove pairs where at least one record is None
    # have to be used with seq2seq trainer
    inputs, targets = compose_input_fn(examples, data_args)

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
