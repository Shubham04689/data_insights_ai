#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Fine-tuning and Deployment Script
-------------------------------------
This script fine-tunes a pre-trained language model from Hugging Face
on a custom dataset and deploys the resulting model back to Hugging Face Hub.
"""


import os
import json
import logging
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from huggingface_hub import login, HfApi
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#########################################################
#                  CONFIGURATION VARIABLES               #
#########################################################

# Model Parameters
MODEL_CONFIG = {
    "base_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Base model to fine-tune
    "new_model_name": "ca-finetune-deepseek-r1-1.5b",                # Name for your fine-tuned model
    "model_revision": "main",                                # Model revision to use (branch/tag)
    "use_lora": True,                                        # Whether to use LoRA for parameter-efficient fine-tuning
    "use_4bit": True,                                        # Use 4-bit quantization
    "use_8bit": False,                                       # Use 8-bit quantization
    "use_fp16": True,                                        # Use fp16 precision
}

# LoRA Configuration (only used if use_lora is True)
LORA_CONFIG = {
    "r": 16,                                                # Rank
    "lora_alpha": 32,                                       # Alpha parameter
    "lora_dropout": 0.05,                                   # Dropout probability
    "bias": "none",                                         # Bias type
    "task_type": "CAUSAL_LM",                               # Task type
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Modules to apply LoRA to
}

# Dataset Parameters
DATASET_CONFIG = {
    "data_dir": "dataset",                            # Root directory containing dataset subdirectories
    "jsonl_dir": "jsonl",                                   # Directory containing JSONL files
    "metadata_dir": "metadata",                             # Directory containing metadata JSON files
    "text_dir": "text",                                     # Directory containing text files
    "combined_dataset_file": "combined_dataset.json",       # Combined dataset file
    "train_test_split": 0.1,                                # Proportion of data to use for testing
    "max_length": 1024,                                     # Maximum sequence length
    "text_column": "text",                                  # Default column name for text data
    "instruction_column": "instruction",                    # Column name for instruction (if using instruction tuning)
    "response_column": "response",                          # Column name for response (if using instruction tuning)
    "use_instruction_format": True,                         # Whether to use instruction tuning format
    "instruction_template": "[INST] {instruction} [/INST]", # Template for instructions
    "combined_template": "[INST] {instruction} [/INST] {response}", # Template for combining instruction and response
}

# Training Parameters
TRAINING_CONFIG = {
    "output_dir": "./results",                              # Directory to save results
    "num_train_epochs": 3,                                  # Number of training epochs
    "per_device_train_batch_size": 2,                       # Training batch size
    "per_device_eval_batch_size": 2,                        # Evaluation batch size
    "gradient_accumulation_steps": 8,                       # Gradient accumulation steps
    "save_steps": 100,                                      # Save checkpoint every X steps
    "save_total_limit": 3,                                  # Limit total saved checkpoints
    "evaluation_strategy": "steps",                         # Evaluation strategy
    "eval_steps": 100,                                      # Evaluate every X steps
    "warmup_steps": 100,                                    # Number of warmup steps
    "weight_decay": 0.01,                                   # Weight decay
    "learning_rate": 1e-4,                                  # Learning rate
    "fp16": True,                                           # Use mixed precision (fp16)
    "bf16": False,                                          # Use bfloat16
    "logging_steps": 10,                                    # Log every X steps
    "report_to": "tensorboard",                             # Where to report results
}
# Hugging Face Hub Parameters
HF_CONFIG = {
    "hf_token": "",    # Your Hugging Face token (set to None to use environment variable)
    "push_to_hub": True,                                    # Whether to push the model to Hugging Face Hub
    "repository_id": None,                                  # Repository ID (defaults to MODEL_CONFIG["new_model_name"])
    "private": False,                                       # Whether the repository should be private
}

#########################################################
#                   UTILITY FUNCTIONS                    #
#########################################################

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATASET_CONFIG["dataset_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    model_save_dir = os.path.join(TRAINING_CONFIG["output_dir"], MODEL_CONFIG["new_model_name"])
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Directories created: {DATASET_CONFIG['dataset_dir']}, {TRAINING_CONFIG['output_dir']}, {model_save_dir}")
    return model_save_dir

def authenticate_huggingface():
    """Authenticate with Hugging Face."""
    token = HF_CONFIG["hf_token"] or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No Hugging Face token provided. Please set HF_CONFIG['hf_token'] or the HF_TOKEN environment variable.")

    login(token=token)
    logger.info("Successfully authenticated with Hugging Face Hub")

def setup_directories():
    """Create necessary directories if they don't exist."""
    # Ensure output directory exists
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    model_save_dir = os.path.join(TRAINING_CONFIG["output_dir"], MODEL_CONFIG["new_model_name"])
    os.makedirs(model_save_dir, exist_ok=True)

    # Check if dataset directories exist
    data_dir = DATASET_CONFIG["data_dir"]
    if not os.path.exists(data_dir):
        logger.warning(f"Main data directory '{data_dir}' does not exist. Creating it.")
        os.makedirs(data_dir, exist_ok=True)

    # Check subdirectories
    jsonl_dir = os.path.join(data_dir, DATASET_CONFIG["jsonl_dir"])
    metadata_dir = os.path.join(data_dir, DATASET_CONFIG["metadata_dir"])
    text_dir = os.path.join(data_dir, DATASET_CONFIG["text_dir"])

    if not os.path.exists(jsonl_dir):
        logger.warning(f"JSONL directory '{jsonl_dir}' does not exist. Creating it.")
        os.makedirs(jsonl_dir, exist_ok=True)

    if not os.path.exists(metadata_dir):
        logger.warning(f"Metadata directory '{metadata_dir}' does not exist. Creating it.")
        os.makedirs(metadata_dir, exist_ok=True)

    if not os.path.exists(text_dir):
        logger.warning(f"Text directory '{text_dir}' does not exist. Creating it.")
        os.makedirs(text_dir, exist_ok=True)

    logger.info(f"Directory setup complete. Using model save directory: {model_save_dir}")
    return model_save_dir

def authenticate_huggingface():
    """Authenticate with Hugging Face."""
    token = HF_CONFIG["hf_token"] or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No Hugging Face token provided. Please set HF_CONFIG['hf_token'] or the HF_TOKEN environment variable.")

    login(token=token)
    logger.info("Successfully authenticated with Hugging Face Hub")

def load_jsonl_files():
    """Load all JSONL files from the JSONL directory."""
    jsonl_dir = os.path.join(DATASET_CONFIG["data_dir"], DATASET_CONFIG["jsonl_dir"])
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No JSONL files found in {jsonl_dir}")
        return None

    datasets = []
    for file_path in jsonl_files:
        logger.info(f"Loading JSONL file: {file_path}")
        try:
            dataset = load_dataset('json', data_files=file_path)['train']
            datasets.append(dataset)
            logger.info(f"Loaded {len(dataset)} examples from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")

    if not datasets:
        return None

    # Combine all datasets
    combined_dataset = concatenate_datasets(datasets)
    logger.info(f"Combined {len(combined_dataset)} examples from {len(datasets)} JSONL files")
    return combined_dataset

def load_text_files():
    """Load all text files from the text directory."""
    text_dir = os.path.join(DATASET_CONFIG["data_dir"], DATASET_CONFIG["text_dir"])
    text_files = glob.glob(os.path.join(text_dir, "*.txt"))

    if not text_files:
        logger.warning(f"No text files found in {text_dir}")
        return None

    all_texts = []
    for file_path in text_files:
        logger.info(f"Loading text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split by paragraphs
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            all_texts.extend(paragraphs)
            logger.info(f"Loaded {len(paragraphs)} paragraphs from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")

    if not all_texts:
        return None

    # Create dataset
    dataset = Dataset.from_dict({DATASET_CONFIG["text_column"]: all_texts})
    logger.info(f"Created dataset with {len(dataset)} examples from text files")
    return dataset

def load_metadata_files():
    """Load all metadata JSON files from the metadata directory."""
    metadata_dir = os.path.join(DATASET_CONFIG["data_dir"], DATASET_CONFIG["metadata_dir"])
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))

    if not metadata_files:
        logger.warning(f"No metadata files found in {metadata_dir}")
        return None

    all_metadata = []
    for file_path in metadata_files:
        logger.info(f"Loading metadata file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # If metadata is a list, extend all_metadata
            if isinstance(metadata, list):
                all_metadata.extend(metadata)
                logger.info(f"Loaded {len(metadata)} metadata entries from {os.path.basename(file_path)}")
            # If metadata is a dict, append it
            elif isinstance(metadata, dict):
                all_metadata.append(metadata)
                logger.info(f"Loaded 1 metadata entry from {os.path.basename(file_path)}")
            else:
                logger.warning(f"Unexpected metadata format in {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")

    if not all_metadata:
        return None

    # Try to convert to dataset
    try:
        # Check if all items have the same keys for proper conversion
        keys = set(all_metadata[0].keys())
        all_same_keys = all(set(item.keys()) == keys for item in all_metadata)

        if all_same_keys:
            # Convert list of dicts to dict of lists for Dataset.from_dict
            data_dict = {k: [item[k] for item in all_metadata] for k in keys}
            dataset = Dataset.from_dict(data_dict)
            logger.info(f"Created dataset with {len(dataset)} examples from metadata files")
            return dataset
        else:
            logger.warning("Metadata entries have different keys, which may cause issues")
            # Try to find common keys
            common_keys = set.intersection(*[set(item.keys()) for item in all_metadata])
            if common_keys:
                data_dict = {k: [item.get(k) for item in all_metadata] for k in common_keys}
                dataset = Dataset.from_dict(data_dict)
                logger.info(f"Created dataset with {len(dataset)} examples using common keys: {common_keys}")
                return dataset
            else:
                logger.error("No common keys found in metadata entries")
                return None
    except Exception as e:
        logger.error(f"Error creating dataset from metadata: {str(e)}")
        return None

def load_combined_dataset():
    """Load the combined dataset JSON file."""
    combined_path = os.path.join(DATASET_CONFIG["data_dir"], DATASET_CONFIG["combined_dataset_file"])

    if not os.path.exists(combined_path):
        logger.warning(f"Combined dataset file not found: {combined_path}")
        return None

    try:
        logger.info(f"Loading combined dataset from: {combined_path}")
        with open(combined_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different possible formats
        if isinstance(data, list):
            # List of examples
            if all(isinstance(item, dict) for item in data):
                # List of dictionaries
                keys = set(data[0].keys())
                all_same_keys = all(set(item.keys()) == keys for item in data)

                if all_same_keys:
                    data_dict = {k: [item[k] for item in data] for k in keys}
                    dataset = Dataset.from_dict(data_dict)
                else:
                    # Try with common keys
                    common_keys = set.intersection(*[set(item.keys()) for item in data])
                    data_dict = {k: [item.get(k) for item in data] for k in common_keys}
                    dataset = Dataset.from_dict(data_dict)
            else:
                logger.error("Unsupported data format in combined dataset")
                return None
        elif isinstance(data, dict):
            # Already in the format for Dataset.from_dict
            dataset = Dataset.from_dict(data)
        else:
            logger.error("Unsupported data format in combined dataset")
            return None

        logger.info(f"Loaded combined dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading combined dataset: {str(e)}")
        return None

def merge_datasets(datasets):
    """Merge multiple datasets, handling column mismatches."""
    if not datasets or len(datasets) == 0:
        return None

    if len(datasets) == 1:
        return datasets[0]

    # Find common columns across all datasets
    common_columns = set(datasets[0].column_names)
    for ds in datasets[1:]:
        common_columns = common_columns.intersection(set(ds.column_names))

    logger.info(f"Found {len(common_columns)} common columns across datasets: {common_columns}")

    if DATASET_CONFIG["text_column"] not in common_columns and common_columns:
        # Try to identify a suitable text column from common columns
        potential_text_columns = [col for col in common_columns if "text" in col.lower() or
                                 "content" in col.lower() or "body" in col.lower()]

        if potential_text_columns:
            text_column = potential_text_columns[0]
            logger.info(f"Using '{text_column}' as the text column")
            DATASET_CONFIG["text_column"] = text_column
        else:
            # Pick the first string column as text column
            for col in common_columns:
                if isinstance(datasets[0][col][0], str):
                    logger.info(f"Using '{col}' as the text column")
                    DATASET_CONFIG["text_column"] = col
                    break

    # Ensure we have a text column
    if DATASET_CONFIG["text_column"] not in common_columns:
        logger.warning(f"Text column '{DATASET_CONFIG['text_column']}' not found in common columns")

        # Create a new column combining available text-like columns
        for i, ds in enumerate(datasets):
            # Find potential text columns
            text_cols = [col for col in ds.column_names if
                         isinstance(ds[col][0], str) and len(ds[col][0]) > 10]

            if text_cols:
                logger.info(f"Creating text column in dataset {i+1} by combining: {text_cols}")

                def combine_text_columns(example):
                    combined = " ".join([example[col] for col in text_cols])
                    return {DATASET_CONFIG["text_column"]: combined}

                datasets[i] = ds.map(combine_text_columns)
            else:
                logger.error(f"No suitable text columns found in dataset {i+1}")
                return None

    # Now merge the datasets
    try:
        # If using instruction format, prepare the data
        if DATASET_CONFIG["use_instruction_format"]:
            prepared_datasets = []

            for ds in datasets:
                # Check if this dataset has the instruction and response columns
                has_instruction = DATASET_CONFIG["instruction_column"] in ds.column_names
                has_response = DATASET_CONFIG["response_column"] in ds.column_names

                if has_instruction and has_response:
                    # Format the text using the template
                    def format_instruction(example):
                        formatted = DATASET_CONFIG["combined_template"].format(
                            instruction=example[DATASET_CONFIG["instruction_column"]],
                            response=example[DATASET_CONFIG["response_column"]]
                        )
                        return {DATASET_CONFIG["text_column"]: formatted}

                    prepared_ds = ds.map(format_instruction)
                    prepared_datasets.append(prepared_ds)
                elif DATASET_CONFIG["text_column"] in ds.column_names:
                    # Just use the text column as is
                    prepared_datasets.append(ds)
                else:
                    logger.warning(f"Dataset missing required columns for instruction format")

            combined_dataset = concatenate_datasets(prepared_datasets)
        else:
            # Just concatenate the datasets
            combined_dataset = concatenate_datasets(datasets)

        logger.info(f"Successfully merged {len(datasets)} datasets with {len(combined_dataset)} total examples")
        return combined_dataset
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        return None


def load_and_process_dataset():
    """Load and process datasets from various sources."""
    # Try to load from each source
    jsonl_dataset = load_jsonl_files()
    text_dataset = load_text_files()
    metadata_dataset = load_metadata_files()
    combined_dataset = load_combined_dataset()

    # Collect all non-None datasets
    available_datasets = [ds for ds in [jsonl_dataset, text_dataset, metadata_dataset, combined_dataset] if ds is not None]

    if not available_datasets:
        raise ValueError("No valid datasets found in any of the specified locations")

    # Merge all available datasets
    dataset = merge_datasets(available_datasets)

    if dataset is None:
        raise ValueError("Failed to merge datasets properly")

    # Split dataset into train/test
    split_dataset = dataset.train_test_split(test_size=DATASET_CONFIG["train_test_split"])
    logger.info(f"Dataset split into {len(split_dataset['train'])} training examples and {len(split_dataset['test'])} test examples")

    # Save a sample of the dataset for reference
    sample_size = min(10, len(dataset))
    sample = dataset.select(range(sample_size))
    sample_path = os.path.join(TRAINING_CONFIG["output_dir"], "dataset_sample.json")

    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample.to_dict(), f, indent=2)

    logger.info(f"Saved dataset sample to {sample_path}")

    return split_dataset

def preprocess_function(examples, tokenizer):
    """Tokenize the text data."""
    # Assume we're working with a column called 'text', adjust as needed
    text_column = "text"
    if text_column not in examples:
        # Try to find any text column in the dataset
        text_columns = [col for col in examples.keys() if isinstance(examples[col][0], str) and len(examples[col][0]) > 0]
        if text_columns:
            text_column = text_columns[0]
            logger.info(f"Using '{text_column}' as the text column")
        else:
            raise ValueError("No text column found in the dataset")

    # Tokenize the texts
    tokenized_inputs = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=DATASET_CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt"
    )

    # If the model expects labels, set them equal to the input_ids
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

    return tokenized_inputs

def prepare_model():
    """Load and prepare the model for fine-tuning."""
    model_name = MODEL_CONFIG["base_model_name"]
    revision = MODEL_CONFIG["model_revision"]

    # Load tokenizer
    logger.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        padding_side="right"
    )

    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up loading configuration
    load_kwargs = {
        "revision": revision,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if MODEL_CONFIG["use_fp16"] else torch.float32,
    }

    # Add quantization parameters if needed
    if MODEL_CONFIG["use_4bit"]:
        load_kwargs.update({
            "quantization_config": {
                "quant_method": "bitsandbytes_4bit",  # Specify the quantization method
                "bnb_4bit_compute_dtype": torch.float16 if MODEL_CONFIG["use_fp16"] else torch.float32,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        })


    elif MODEL_CONFIG["use_8bit"]:
        load_kwargs.update({"load_in_8bit": True})

    # Load model
    logger.info(f"Loading model: {model_name} with configuration: {load_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Apply LoRA if requested
    if MODEL_CONFIG["use_lora"]:
        logger.info("Applying LoRA configuration to model")
        if MODEL_CONFIG["use_4bit"] or MODEL_CONFIG["use_8bit"]:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=LORA_CONFIG["task_type"],
            target_modules=LORA_CONFIG["target_modules"]
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

def train_model(model, tokenizer, dataset):
    """Train the model on the prepared dataset."""
    model_save_dir = os.path.join(TRAINING_CONFIG["output_dir"], MODEL_CONFIG["new_model_name"])

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        evaluation_strategy=TRAINING_CONFIG["evaluation_strategy"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG["bf16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        report_to=TRAINING_CONFIG["report_to"],
        push_to_hub=HF_CONFIG["push_to_hub"],
        hub_model_id=HF_CONFIG["repository_id"] or MODEL_CONFIG["new_model_name"],
        hub_private_repo=HF_CONFIG["private"],
    )

    # Preprocess the dataset
    logger.info("Preprocessing the dataset")

    # For processing in batches to avoid memory issues
    def batch_preprocess(examples):
        return preprocess_function(examples, tokenizer)

    tokenized_train = dataset["train"].map(
        batch_preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    tokenized_test = dataset["test"].map(
        batch_preprocess,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're fine-tuning a causal language model, not a masked one
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting model training")
    train_result = trainer.train()

    # Save model, tokenizer, and training metrics
    logger.info(f"Saving model to {model_save_dir}")
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Run evaluation and save results
    logger.info("Running evaluation")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Push to Hub if configured
    if HF_CONFIG["push_to_hub"]:
        logger.info(f"Pushing model to Hugging Face Hub as {HF_CONFIG['repository_id'] or MODEL_CONFIG['new_model_name']}")
        trainer.push_to_hub()

    return model_save_dir

def test_model(model_path, tokenizer):
    """Test the fine-tuned model on a sample input."""
    logger.info(f"Testing model from {model_path}")

    # For LoRA models, we need to load them correctly
    if MODEL_CONFIG["use_lora"]:
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["base_model_name"],
            torch_dtype=torch.float16 if MODEL_CONFIG["use_fp16"] else torch.float32,
            trust_remote_code=True
        )

        # Load and merge the LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)

        # You might want to merge weights for inference
        model = model.merge_and_unload()
    else:
        # Load the full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if MODEL_CONFIG["use_fp16"] else torch.float32,
            trust_remote_code=True
        )

    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Test with a sample prompt
    test_prompt = "This is a test of the fine-tuned model. It should complete this sentence with domain-specific knowledge:"
    logger.info(f"Testing with prompt: {test_prompt}")

    generated_text = generator(test_prompt)
    logger.info(f"Generated text: {generated_text[0]['generated_text']}")

    return generated_text

#########################################################
#                    MAIN FUNCTION                      #
#########################################################

def main():
    """Main function to orchestrate the fine-tuning process."""
    logger.info("Starting the LLM fine-tuning process")

    try:
        # Setup directories
        model_save_dir = setup_directories()

        # Authenticate with Hugging Face if needed
        if HF_CONFIG["push_to_hub"]:
            authenticate_huggingface()

        # Load and process dataset
        dataset = load_and_process_dataset()

        # Prepare model and tokenizer
        model, tokenizer = prepare_model()

        # Train model
        trained_model_path = train_model(model, tokenizer, dataset)

        # Test the fine-tuned model
        test_results = test_model(trained_model_path, tokenizer)

        # Print summary
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved to: {trained_model_path}")
        if HF_CONFIG["push_to_hub"]:
            repo_id = HF_CONFIG["repository_id"] or MODEL_CONFIG["new_model_name"]
            logger.info(f"Model pushed to Hugging Face Hub: {repo_id}")

        # Save a summary of the fine-tuning process
        summary = {
            "model_name": MODEL_CONFIG["new_model_name"],
            "base_model": MODEL_CONFIG["base_model_name"],
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_stats": {
                "train_size": len(dataset["train"]),
                "test_size": len(dataset["test"]),
            },
            "training_config": TRAINING_CONFIG,
            "model_config": MODEL_CONFIG,
            "use_lora": MODEL_CONFIG["use_lora"],
        }

        with open(os.path.join(TRAINING_CONFIG["output_dir"], "fine_tuning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"An error occurred during fine-tuning: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    main()