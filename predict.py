"""
Inference engine for fine-tuned DistilBERT sentiment classification.
Supports single text queries, batch inference, CSV processing, and token explainability.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import config
from src.model import load_trained_model, explain_prediction
from src.preprocessing import clean_text


def predict_sentiment(
    text: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    device: str = config.DEVICE,
    include_explanation: bool = False
) -> Dict[str, Any]:
    """
    Predicts sentiment for a single input text string.
    
    Args:
        text: Raw input review or tweet.
        model: Pre-loaded DistilBERT model (if None, loaded from disk).
        tokenizer: Pre-loaded DistilBERT tokenizer (if None, loaded from disk).
        device: Device to run inference on.
        include_explanation: Whether to compute gradient-based token attributions.
        
    Returns:
        Dictionary containing label, confidence, probabilities, and attributions.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "text": text,
            "cleaned_text": "",
            "label": "Unknown",
            "confidence": 0.0,
            "probabilities": {"Negative": 0.5, "Positive": 0.5},
            "error": "Input text is empty or contains only invalid characters."
        }

    if model is None or tokenizer is None:
        model, tokenizer = load_trained_model(device=device)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LEN,
        padding=True
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    predicted_id = int(torch.argmax(logits, dim=-1).item())
    confidence = float(probs[predicted_id])
    label_name = config.ID2LABEL[predicted_id]

    result = {
        "text": text,
        "cleaned_text": cleaned,
        "label": label_name,
        "label_id": predicted_id,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Negative": round(float(probs[0]), 4),
            "Positive": round(float(probs[1]), 4)
        }
    }

    if include_explanation:
        try:
            attributions = explain_prediction(cleaned, model, tokenizer, device=device)
            result["attributions"] = attributions
        except Exception as e:
            result["attributions"] = []
            result["explanation_error"] = str(e)

    return result


def predict_batch(
    texts: List[str],
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    batch_size: int = config.EVAL_BATCH_SIZE,
    device: str = config.DEVICE
) -> List[Dict[str, Any]]:
    """
    Performs high-throughput batched sentiment classification.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_trained_model(device=device)

    results = []
    cleaned_texts = [clean_text(t) for t in texts]

    model.eval()
    for i in range(0, len(cleaned_texts), batch_size):
        batch_slice = cleaned_texts[i:i + batch_size]
        original_slice = texts[i:i + batch_size]

        # Handle empty strings safely
        valid_indices = [idx for idx, t in enumerate(batch_slice) if len(t) > 0]
        valid_texts = [batch_slice[idx] for idx in valid_indices]

        if not valid_texts:
            for orig in original_slice:
                results.append({
                    "text": orig,
                    "label": "Neutral",
                    "confidence": 0.5,
                    "prob_negative": 0.5,
                    "prob_positive": 0.5
                })
            continue

        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LEN,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        valid_out_map = {}
        for sub_i, orig_i in enumerate(valid_indices):
            pred_id = int(np.argmax(probs[sub_i]))
            valid_out_map[orig_i] = {
                "label": config.ID2LABEL[pred_id],
                "confidence": round(float(probs[sub_i][pred_id]), 4),
                "prob_negative": round(float(probs[sub_i][0]), 4),
                "prob_positive": round(float(probs[sub_i][1]), 4),
            }

        for idx, orig in enumerate(original_slice):
            if idx in valid_out_map:
                res = valid_out_map[idx]
                res["text"] = orig
                results.append(res)
            else:
                results.append({
                    "text": orig,
                    "label": "Unknown",
                    "confidence": 0.5,
                    "prob_negative": 0.5,
                    "prob_positive": 0.5
                })

    return results


def process_csv_file(
    input_file: Union[str, Path],
    text_column: str,
    output_file: Optional[Union[str, Path]] = None,
    batch_size: int = config.EVAL_BATCH_SIZE
) -> pd.DataFrame:
    """
    Reads a CSV, predicts sentiments for the designated text column,
    and returns an augmented DataFrame.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")

    print(f"Running batch prediction for {len(df)} records from {input_path}...")
    texts = df[text_column].astype(str).tolist()
    
    preds = predict_batch(texts, batch_size=batch_size)
    
    df["predicted_sentiment"] = [p["label"] for p in preds]
    df["confidence"] = [p["confidence"] for p in preds]
    df["prob_positive"] = [p["prob_positive"] for p in preds]
    df["prob_negative"] = [p["prob_negative"] for p in preds]

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT Sentiment Analysis CLI Inference")
    parser.add_argument("--text", type=str, help="Single review or tweet to analyze")
    parser.add_argument("--file", type=str, help="Path to input CSV file for batch inference")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name in CSV")
    parser.add_argument("--output", type=str, help="Output CSV path for batch predictions")
    parser.add_argument("--explain", action="store_true", help="Include token attribution explainability")
    
    args = parser.parse_args()

    if args.text:
        res = predict_sentiment(args.text, include_explanation=args.explain)
        print("\n" + "=" * 50)
        print("DistilBERT Sentiment Prediction Result")
        print("=" * 50)
        print(f"Input Text:    {res['text']}")
        print(f"Sentiment:     {res['label']}")
        print(f"Confidence:    {res['confidence'] * 100:.2f}%")
        print(f"Probabilities: Positive={res['probabilities']['Positive']:.4f}, Negative={res['probabilities']['Negative']:.4f}")
        
        if args.explain and "attributions" in res:
            print("\nToken Attributions:")
            for attr in res["attributions"]:
                print(f"  {attr['token']:<15} {attr['normalized_score']:+.4f} ({attr['direction']})")
        print("=" * 50)
        
    elif args.file:
        process_csv_file(args.file, text_column=args.text_col, output_file=args.output)
    else:
        print("Usage: python predict.py --text 'Your review here' OR python predict.py --file data.csv --text_col text")
