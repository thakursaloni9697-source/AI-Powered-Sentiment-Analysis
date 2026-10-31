"""
Fine-tuning entrypoint for DistilBERT sentiment classification.
Coordinates data preparation, EDA visualization, model training, and post-training evaluation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from src.utils import set_seed, plot_eda_figures, plot_training_history
from src.dataset import load_or_create_dataset, split_and_save_data, SentimentDataset
from src.preprocessing import get_tokenizer
from src.model import get_model, count_parameters
from src.trainer import DistilBertTrainer
from evaluate import evaluate_model


def run_training_pipeline(
    sample_size: int = 5000,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    device: str = config.DEVICE
):
    """
    Executes the end-to-end DistilBERT fine-tuning pipeline.
    """
    # 1. Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    print(f"Random seed set to {config.RANDOM_SEED}")
    print(f"Target Compute Device: {device.upper()}")

    # 2. Acquire and prepare dataset
    df = load_or_create_dataset(sample_size=sample_size, random_state=config.RANDOM_SEED)

    # 3. Exploratory Data Analysis & Plots
    print("\nGenerating Exploratory Data Analysis (EDA) visualizations...")
    plot_eda_figures(df, text_col="text", label_col="label")
    print(f"EDA plots saved to '{config.RESULTS_DIR}'.")

    # 4. Stratified Train / Val / Test splitting
    print("\nSplitting dataset into stratified Train / Validation / Test sets...")
    train_df, val_df, test_df = split_and_save_data(df, random_state=config.RANDOM_SEED)

    # 5. Tokenizer & Dataset Loaders
    print(f"\nInitializing DistilBERT Tokenizer ('{config.MODEL_NAME}')...")
    tokenizer = get_tokenizer(config.MODEL_NAME)

    train_dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    val_dataset = SentimentDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)

    # 6. Initialize Model
    print(f"\nInitializing DistilBERT Architecture ('{config.MODEL_NAME}') with 2 output labels...")
    model = get_model(config.MODEL_NAME, num_labels=config.NUM_CLASSES, device=device)
    param_info = count_parameters(model)
    print(f"Model Parameters: Total={param_info['total']:,} | Trainable={param_info['trainable']:,}")

    # 7. Train & Fine-Tune Model
    trainer = DistilBertTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=config.MODEL_SAVE_DIR
    )
    history = trainer.train()

    # 8. Generate Loss & Accuracy Curves
    print("\nGenerating Training Progression Curves...")
    plot_training_history(history)
    print(f"Training curves saved to '{config.RESULTS_DIR}'.")

    # 9. Final Model Evaluation on Unseen Test Set
    print("\nExecuting Model Evaluation on Holdout Test Set...")
    metrics = evaluate_model(
        model_dir=config.MODEL_SAVE_DIR,
        test_csv=config.TEST_DATA_PATH,
        device=device
    )

    print("\n=======================================================")
    print("DISTILBERT SENTIMENT ANALYSIS PIPELINE COMPLETE")
    print(f"Final Holdout Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Final Holdout Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Saved Model Directory:       {config.MODEL_SAVE_DIR}")
    print(f"Saved Results Directory:     {config.RESULTS_DIR}")
    print("=======================================================\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune DistilBERT for Sentiment Analysis")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=5000, help="Total dataset sample size")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    run_training_pipeline(
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
