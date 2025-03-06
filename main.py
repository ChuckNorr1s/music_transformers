"""
Production-ready audio classification training pipeline.

This script provides two subcommands:
    1. prepare: Load your audio dataset from a folder (using the "audiofolder" loader)
       and save it in Arrow format to disk.
    2. train:   Load a pre-saved dataset from disk and fine-tune a pretrained audio classification model.
"""

import argparse
import logging
import os
import sys
import numpy as np

from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def prepare_dataset_cmd(args):
    """Load dataset using the audiofolder loader and save to disk."""
    try:
        logging.info("Loading dataset from audiofolder using data_dir: %s", args.data_dir)
        dataset = load_dataset("audiofolder", data_dir=args.data_dir)
        logging.info("Saving dataset to disk at: %s", args.output_dir)
        dataset.save_to_disk(args.output_dir)
        # Log a sample from the dataset (from the 'train' split)
        logging.info("Dataset sample: %s", dataset["train"][-1])
    except Exception as e:
        logging.error("Failed to prepare dataset: %s", e)
        sys.exit(1)


def prepare_dataset(dataset, feature_extractor, max_duration, num_proc):
    """Preprocess the dataset with the feature extractor using configurable processes."""
    def preprocess_function(examples):
        # Extract audio arrays from the 'audio' field and apply feature extraction
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )
        return inputs

    logging.info("Preprocessing dataset with max_duration=%s and num_proc=%d", max_duration, num_proc)
    dataset_encoded = dataset.map(
        preprocess_function,
        remove_columns=["audio"],
        batched=True,
        batch_size=100,
        num_proc=num_proc,
    )
    return dataset_encoded


def create_label_maps(dataset_encoded):
    """Create mapping dictionaries for labels."""
    id2label_fn = dataset_encoded["train"].features["label"].int2str
    label_names = dataset_encoded["train"].features["label"].names
    id2label = {str(i): id2label_fn(i) for i in range(len(label_names))}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    logging.info("Found %d labels: %s", num_labels, id2label)
    return id2label, label2id, num_labels


def compute_metrics(eval_pred):
    """Compute accuracy for a batch of predictions."""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def train_model(args):
    """Load a dataset from disk, preprocess it, and train the model."""
    try:
        logging.info("Loading dataset from disk: %s", args.dataset_disk)
        dataset = load_from_disk(args.dataset_disk)
    except Exception as e:
        logging.error("Failed to load dataset from disk: %s", e)
        sys.exit(1)

    try:
        logging.info("Splitting dataset with test size: %f", args.test_size)
        dataset = dataset["train"].train_test_split(seed=args.seed, shuffle=True, test_size=args.test_size)
    except Exception as e:
        logging.error("Failed during dataset split: %s", e)
        sys.exit(1)

    try:
        logging.info("Loading feature extractor for model: %s", args.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_id, do_normalize=True, return_attention_mask=True
        )
    except Exception as e:
        logging.error("Failed to load feature extractor: %s", e)
        sys.exit(1)

    sampling_rate = feature_extractor.sampling_rate
    logging.info("Casting 'audio' column with sampling_rate: %d", sampling_rate)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Preprocess the dataset using configurable number of processes
    dataset_encoded = prepare_dataset(dataset, feature_extractor, args.max_duration, args.num_proc)

    # Create label maps
    id2label, label2id, num_labels = create_label_maps(dataset_encoded)

    try:
        logging.info("Loading model: %s", args.model_id)
        model = AutoModelForAudioClassification.from_pretrained(
            args.model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        sys.exit(1)

    model_name = args.model_id.split("/")[-1]
    output_dir = f"{model_name}-finetuned-{args.experiment_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    try:
        logging.info("Starting training...")
        trainer.train()
        logging.info("Training complete!")
        trainer.save_model()
        logging.info("Model saved to: %s", output_dir)
    except Exception as e:
        logging.error("Training failed: %s", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Audio Classification Pipeline")
    subparsers = parser.add_subparsers(title="subcommands", required=True, dest="command")

    # Subcommand for dataset preparation
    parser_prepare = subparsers.add_parser("prepare", help="Prepare and save dataset from audio folder")
    parser_prepare.add_argument("--data_dir", type=str, default="./dataset", help="Directory containing raw audio data")
    parser_prepare.add_argument("--output_dir", type=str, default="./music", help="Directory to save the prepared dataset")

    # Subcommand for training
    parser_train = subparsers.add_parser("train", help="Train the audio classification model")
    parser_train.add_argument("--dataset_disk", type=str, default="./music", help="Path to the prepared dataset on disk")
    parser_train.add_argument("--model_id", type=str, default="ntu-spml/distilhubert", help="Pretrained model id")
    parser_train.add_argument("--test_size", type=float, default=0.1, help="Test split size")
    parser_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser_train.add_argument("--max_duration", type=float, default=600.0, help="Maximum duration (seconds) for each audio sample")
    parser_train.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser_train.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser_train.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser_train.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser_train.add_argument("--logging_steps", type=int, default=5, help="Steps between logging updates")
    parser_train.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision training")
    parser_train.add_argument("--push_to_hub", action="store_true", help="Push the final model to the Hugging Face hub")
    parser_train.add_argument("--experiment_name", type=str, default="finetuning", help="Suffix for output directory name")
    parser_train.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for dataset mapping")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "prepare":
        prepare_dataset_cmd(args)
    elif args.command == "train":
        train_model(args)
    else:
        logging.error("Unknown command")
        sys.exit(1)


if __name__ == "__main__":
    main()
