"""
Configuration settings for the Sentiment Analysis with Fine-Tuned DistilBERT project.
Centralizes paths, hyperparameters, model constants, and device configuration.
"""

import os
from pathlib import Path
import torch

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent

# Ensure Matplotlib config directory is writable
MPL_DIR = BASE_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

# Keep Hugging Face downloads enabled unless offline mode is explicitly requested.
HF_OFFLINE = False
for variable in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
):
    if HF_OFFLINE:
        os.environ[variable] = "1"
    else:
        os.environ.pop(variable, None)

# Directory Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODEL_SAVE_DIR = MODELS_DIR / "distilbert-sentiment"

# Create project directories used during training and inference.
for directory in (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SCREENSHOTS_DIR,
    NOTEBOOKS_DIR,
    MODEL_SAVE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_PATH = DATA_DIR / "dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
VAL_DATA_PATH = DATA_DIR / "val.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Model constants
MODEL_NAME = "distilbert-base-uncased"
NUM_CLASSES = 2
LABEL2ID = {"Negative": 0, "Positive": 1}
ID2LABEL = {0: "Negative", 1: "Positive"}

# Tokenization and Model Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
RANDOM_SEED = 42

# Hardware configuration
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
PIN_MEMORY = USE_GPU

if not USE_GPU:
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    except Exception:
        pass

NUM_WORKERS = 0

# Evaluation metric filenames
METRICS_FILE = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / "classification_report.txt"
TRAINING_HISTORY_FILE = RESULTS_DIR / "training_history.json"

# Plot filenames
CONFUSION_MATRIX_PLOT = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT = RESULTS_DIR / "roc_curve.png"
TRAINING_LOSS_PLOT = RESULTS_DIR / "loss_curves.png"
ACCURACY_CURVE_PLOT = RESULTS_DIR / "accuracy_curves.png"
CLASS_DIST_PLOT = RESULTS_DIR / "class_distribution.png"
LENGTH_DIST_PLOT = RESULTS_DIR / "length_distribution.png"
WORD_FREQ_PLOT = RESULTS_DIR / "word_frequency.png"
