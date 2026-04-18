import os
import copy
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_CSV = "/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/dataset_updated/annotations_clean/train.csv"
VAL_CSV   = "/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/dataset_updated/annotations_clean/val.csv"
TEST_CSV  = "/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/dataset_updated/annotations_clean/test.csv"

BASE_OUTPUT_DIR = "/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/output_single_model"
RUN_NAME = "mobilenet_v3_large_head_frozen"

MODEL_NAME = "mobilenet_v3_large"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 20
PATIENCE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Threshold is tuned on validation later
DEFAULT_THRESHOLD = 0.5

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
torch.backends.cudnn.benchmark = True

# =========================================================
# DATASET
# =========================================================
class TornadoBinaryDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(float(row["binary_label"]), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "filepath": img_path
        }

# =========================================================
# AUDIT HELPERS
# =========================================================
def audit_split(df: pd.DataFrame, split_name: str):
    print("\n" + "=" * 80)
    print(f"AUDIT: {split_name}")
    print("=" * 80)
    print("Rows:", len(df))
    print("\nLabel counts:")
    print(df["binary_label"].value_counts().sort_index())
    print("\nClass counts:")
    print(df["class_name"].value_counts())

    missing = df[~df["filepath"].map(lambda x: Path(x).exists())]
    print("\nMissing files:", len(missing))
    if len(missing) > 0:
        missing.to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_missing_files.csv"), index=False)

def audit_overlap(train_df, val_df, test_df):
    for df in [train_df, val_df, test_df]:
        df["basename"] = df["filepath"].map(lambda x: Path(x).name)

    train_names = set(train_df["basename"])
    val_names = set(val_df["basename"])
    test_names = set(test_df["basename"])

    train_val = sorted(train_names & val_names)
    train_test = sorted(train_names & test_names)
    val_test = sorted(val_names & test_names)

    overlap_report = {
        "train_val_overlap_count": len(train_val),
        "train_test_overlap_count": len(train_test),
        "val_test_overlap_count": len(val_test),
        "train_val_examples": train_val[:50],
        "train_test_examples": train_test[:50],
        "val_test_examples": val_test[:50],
    }

    with open(os.path.join(OUTPUT_DIR, "split_overlap_report.json"), "w") as f:
        json.dump(overlap_report, f, indent=2)

    print("\n" + "=" * 80)
    print("SPLIT OVERLAP AUDIT (BASENAME LEVEL)")
    print("=" * 80)
    print(json.dumps(overlap_report, indent=2))

# =========================================================
# TRANSFORMS
# =========================================================
def build_transforms(image_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
        ),
        transforms.RandomRotation(degrees=5),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform

# =========================================================
# DATALOADERS
# =========================================================
def build_dataloaders(train_df, val_df, test_df):
    train_transform, eval_transform = build_transforms(IMAGE_SIZE)

    train_dataset = TornadoBinaryDataset(train_df, transform=train_transform)
    val_dataset   = TornadoBinaryDataset(val_df, transform=eval_transform)
    test_dataset  = TornadoBinaryDataset(test_df, transform=eval_transform)

    pin_memory = DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

# =========================================================
# MODEL
# =========================================================
def build_model():
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier with stronger dropout
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
        nn.Hardswish(),
        nn.Dropout(0.6),
        nn.Linear(in_features, 1)
    )

    return model

# =========================================================
# METRICS
# =========================================================
def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan

    return metrics, y_pred

def tune_threshold(y_true, y_prob, out_csv):
    rows = []
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.10, 0.91, 0.02):
        metrics, _ = compute_binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append({
            "threshold": float(thr),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thr)

    thresh_df = pd.DataFrame(rows)
    thresh_df.to_csv(out_csv, index=False)

    return best_threshold, thresh_df

# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true, y_prob = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.detach().cpu().numpy().flatten().tolist())

    metrics, _ = compute_binary_metrics(y_true, y_prob, threshold=DEFAULT_THRESHOLD)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics

def evaluate_labeled(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    y_true, y_prob = [], []
    filepaths = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            filepaths.extend(batch["filepath"])

    metrics, y_pred = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = running_loss / max(len(loader), 1)

    pred_df = pd.DataFrame({
        "filepath": filepaths,
        "true_label": np.array(y_true).astype(int),
        "pred_prob": np.array(y_prob, dtype=float),
        "pred_label": y_pred.astype(int),
    })

    return metrics, pred_df, np.array(y_true).astype(int), np.array(y_prob, dtype=float)

# =========================================================
# PLOTTING
# =========================================================
def plot_training_curves(history_df, output_dir):
    if history_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_f1"], label="Train F1")
    plt.plot(history_df["epoch"], history_df["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_roc_auc"], label="Train ROC-AUC")
    plt.plot(history_df["epoch"], history_df["val_roc_auc"], label="Val ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix_figure(y_true, y_pred, output_path, class_names=("Non-Tornado", "Tornado")):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# MAIN
# =========================================================
def main():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print("Using device:", DEVICE)
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    audit_split(train_df, "train")
    audit_split(val_df, "val")
    audit_split(test_df, "test")
    audit_overlap(train_df.copy(), val_df.copy(), test_df.copy())

    train_label_counts = train_df["binary_label"].value_counts().to_dict()
    num_neg = train_label_counts.get(0, 0)
    num_pos = train_label_counts.get(1, 0)
    pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0

    print("\nNegative:", num_neg)
    print("Positive:", num_pos)
    print("pos_weight:", pos_weight_value)

    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df)

    model = build_model().to(DEVICE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\nModel:", MODEL_NAME)
    print("Trainable params:", trainable_params)
    print("Total params:", total_params)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -1.0
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None
    best_val_pred_df = None
    best_val_y_true = None
    best_val_y_prob = None
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_idx = epoch + 1

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics, val_pred_df, val_y_true, val_y_prob = evaluate_labeled(
            model, val_loader, criterion, DEVICE, threshold=DEFAULT_THRESHOLD
        )

        row = {
            "epoch": epoch_idx,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "train_roc_auc": float(train_metrics["roc_auc"]) if not pd.isna(train_metrics["roc_auc"]) else np.nan,
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_roc_auc": float(val_metrics["roc_auc"]) if not pd.isna(val_metrics["roc_auc"]) else np.nan,
        }
        history.append(row)

        print(
            f"Epoch {epoch_idx:03d} | "
            f"LR {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1']:.4f} | "
            f"Train AUC {train_metrics['roc_auc']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | "
            f"Val AUC {val_metrics['roc_auc']:.4f}"
        )

        current_score = val_metrics["roc_auc"] if not pd.isna(val_metrics["roc_auc"]) else -val_metrics["loss"]
        scheduler.step(current_score)

        if current_score > best_val_auc:
            best_val_auc = current_score
            best_epoch = epoch_idx
            best_state = copy.deepcopy(model.state_dict())
            best_train_metrics = copy.deepcopy(train_metrics)
            best_val_metrics = copy.deepcopy(val_metrics)
            best_val_pred_df = val_pred_df.copy()
            best_val_y_true = val_y_true.copy()
            best_val_y_prob = val_y_prob.copy()
            patience_counter = 0

            torch.save(best_state, os.path.join(OUTPUT_DIR, "best_mobilenet_v3_large.pth"))
            best_val_pred_df.to_csv(os.path.join(OUTPUT_DIR, "best_val_predictions_default_threshold.csv"), index=False)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch_idx}")
            break

    elapsed = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
    plot_training_curves(history_df, OUTPUT_DIR)

    # -----------------------------------------------------
    # Tune threshold on validation
    # -----------------------------------------------------
    best_threshold, thresh_df = tune_threshold(
        best_val_y_true,
        best_val_y_prob,
        os.path.join(OUTPUT_DIR, "threshold_search.csv")
    )
    print("\nBest validation threshold:", best_threshold)

    # Save validation predictions with tuned threshold
    tuned_val_metrics, tuned_val_pred = compute_binary_metrics(best_val_y_true, best_val_y_prob, threshold=best_threshold)
    best_val_pred_df["pred_label_tuned"] = tuned_val_pred
    best_val_pred_df.to_csv(os.path.join(OUTPUT_DIR, "best_val_predictions_tuned_threshold.csv"), index=False)

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------
    model.load_state_dict(best_state)

    test_metrics, test_pred_df, test_y_true, test_y_prob = evaluate_labeled(
        model, test_loader, criterion, DEVICE, threshold=best_threshold
    )
    test_pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    plot_confusion_matrix_figure(
        y_true=test_pred_df["true_label"].values,
        y_pred=test_pred_df["pred_label"].values,
        output_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        class_names=("Non-Tornado", "Tornado")
    )

    summary = {
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "best_epoch": best_epoch,
        "runtime_sec": elapsed,
        "best_threshold_from_val": best_threshold,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_train_loss": best_train_metrics["loss"] if best_train_metrics else None,
        "best_train_accuracy": best_train_metrics["accuracy"] if best_train_metrics else None,
        "best_train_precision": best_train_metrics["precision"] if best_train_metrics else None,
        "best_train_recall": best_train_metrics["recall"] if best_train_metrics else None,
        "best_train_f1": best_train_metrics["f1"] if best_train_metrics else None,
        "best_train_roc_auc": best_train_metrics["roc_auc"] if best_train_metrics else None,
        "best_val_loss_default_thr": best_val_metrics["loss"] if best_val_metrics else None,
        "best_val_accuracy_default_thr": best_val_metrics["accuracy"] if best_val_metrics else None,
        "best_val_precision_default_thr": best_val_metrics["precision"] if best_val_metrics else None,
        "best_val_recall_default_thr": best_val_metrics["recall"] if best_val_metrics else None,
        "best_val_f1_default_thr": best_val_metrics["f1"] if best_val_metrics else None,
        "best_val_roc_auc": best_val_metrics["roc_auc"] if best_val_metrics else None,
        "best_val_f1_tuned_thr": tuned_val_metrics["f1"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "final_summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\nClassification Report:\n")
        f.write(classification_report(
            test_pred_df["true_label"],
            test_pred_df["pred_label"],
            target_names=["non-tornado", "tornado"],
            digits=4
        ))

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()