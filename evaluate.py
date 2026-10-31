"""
Model evaluation module for fine-tuned DistilBERT on unseen test data.
Generates metrics, classification reports, confusion matrix, and ROC curves.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import SentimentDataset
from src.model import load_trained_model
from src.utils import (
    calculate_metrics,
    save_metrics_and_report,
    plot_confusion_matrix,
    plot_roc_curve
)


def evaluate_model(
    model_dir: Path = config.MODEL_SAVE_DIR,
    test_csv: Path = config.TEST_DATA_PATH,
    batch_size: int = config.EVAL_BATCH_SIZE,
    device: str = config.DEVICE
) -> Dict[str, Any]:
    """
    Evaluates fine-tuned model against holdout test dataset.
    Computes all standard classification metrics and produces publication plots.
    """
    if not test_csv.exists():
        raise FileNotFoundError(f"Test data file '{test_csv}' does not exist. Run 'python train.py' first.")

    print(f"\n=======================================================")
    print(f"Evaluating DistilBERT Model on Test Set: {test_csv}")
    print(f"Device: {device.upper()}")
    print(f"=======================================================\n")

    # Load Model and Tokenizer
    model, tokenizer = load_trained_model(model_dir=model_dir, device=device)

    # Load Test Data
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df)} test samples.")

    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities

    # Calculate metrics
    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_probs=all_probs
    )

    # Save metrics JSON & text report
    save_metrics_and_report(metrics, config.RESULTS_DIR)

    # Generate evaluation plots
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=config.CONFUSION_MATRIX_PLOT)
    plot_roc_curve(all_labels, all_probs, save_path=config.ROC_CURVE_PLOT)

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 55)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 55)
    print("\nDetailed Classification Report:")
    print(metrics["classification_report_text"])
    print(f"Plots and metrics saved to: {config.RESULTS_DIR}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned DistilBERT Model")
    parser.add_argument("--model_dir", type=str, default=str(config.MODEL_SAVE_DIR), help="Model weights directory")
    parser.add_argument("--test_data", type=str, default=str(config.TEST_DATA_PATH), help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=config.EVAL_BATCH_SIZE, help="Inference batch size")
    args = parser.parse_args()

    evaluate_model(
        model_dir=Path(args.model_dir),
        test_csv=Path(args.test_data),
        batch_size=args.batch_size
    )
