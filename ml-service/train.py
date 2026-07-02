"""Train the lesion classifier on ISIC 2019 (MEL vs NV) and EXPORT the artifacts the
inference service needs:
  - model/model.pt            -> best classifier state_dict
  - model/normalization.json  -> mean, std, feature_names, threshold, input_dim,
                                 architecture (hidden_dims, dropout), version, metrics

Key improvements over the original training:
  - stratified train / validation / test split
  - class-imbalance handling via pos_weight in BCEWithLogitsLoss
  - weight decay + dropout + BatchNorm for regularization
  - early stopping on validation ROC-AUC (restores the best epoch)
  - DECISION THRESHOLD chosen on the validation set (Youden's J or a target recall)
    instead of a blind 0.5 -> directly improves melanoma sensitivity
  - full test metrics: accuracy, precision, recall (sensitivity), specificity, F1,
    ROC-AUC and confusion matrix
  - optional feature cache (.npz) so re-training does not re-extract features

Dataset:
  1. --data-dir /path/to/extracted/isic-2019   (manual download, NO Kaggle API key)
  2. omit it  ->  kagglehub (needs KAGGLE_USERNAME / KAGGLE_KEY)

Usage:
    python train.py --data-dir /data/isic-2019 --epochs 200 --target-recall 0.85
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)

from src.dataset import ISIC2019Dataset
from src.feature_dataset import build_features_dataset, normalize_with_stats
from src.classifier import Classifier

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
VERSION = os.environ.get("MODEL_VERSION", "isic2019-mlp-v2")
HIDDEN_DIMS = (256, 128, 64)


def download_dataset():
    import kagglehub
    print("Downloading ISIC 2019 dataset...")
    path = kagglehub.dataset_download("andrewmvd/isic-2019")
    print("Dataset downloaded to:", path)
    return path


def load_features(args):
    if args.cache and os.path.exists(args.cache):
        print("Loading cached features from", args.cache)
        d = np.load(args.cache, allow_pickle=True)
        return d["X"], d["Y"], list(d["feature_names"])

    dataset_path = args.data_dir if args.data_dir else download_dataset()
    images_dir = os.path.join(dataset_path, "ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
    csv_file = os.path.join(dataset_path, "ISIC_2019_Training_GroundTruth.csv")
    if not os.path.isdir(images_dir) or not os.path.isfile(csv_file):
        raise FileNotFoundError(
            f"Expected ISIC 2019 layout under '{dataset_path}'.\n"
            f"  images: {images_dir}\n  labels: {csv_file}"
        )
    dataset = ISIC2019Dataset(root_dir=images_dir, csv_file=csv_file, transform=None)
    print(f"Extracting features from {len(dataset)} images (this is the slow part)...")
    X, Y, feature_names = build_features_dataset(dataset)
    if args.cache:
        np.savez_compressed(args.cache, X=X, Y=Y, feature_names=np.array(feature_names, dtype=object))
        print("Cached features to", args.cache)
    return X, Y, feature_names


def pick_threshold(y_true, y_prob, strategy="youden", target_recall=None, target_precision=None):
    """Decision-threshold selection, chosen on the validation set.

    - "target_recall"    : lowest threshold that reaches >= target_recall (old default
                            behaviour when --target-recall is passed; maximizes melanoma
                            sensitivity, at the cost of many false positives)
    - "target_precision" : lowest threshold that reaches >= target_precision
    - "max_f1"            : threshold maximizing F1 (precision/recall balance) - DEFAULT.
                            Raises precision substantially over Youden's J without
                            requiring a hand-picked target.
    - "youden"            : Youden's J (max TPR - FPR); tends to favor recall on
                            imbalanced data like this one.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if strategy == "target_recall" or target_recall is not None:
        ok = np.where(tpr >= target_recall)[0]
        if len(ok):
            return float(thr[ok[0]])
        return float(thr[int(np.argmax(tpr - fpr))])
    if strategy == "target_precision" or target_precision is not None:
        for t in sorted(thr):
            pred = (y_prob >= t).astype(int)
            if precision_score(y_true, pred, zero_division=0) >= target_precision:
                return float(t)
        return float(thr.max())
    if strategy == "max_f1":
        f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thr]
        return float(thr[int(np.argmax(f1s))])
    j = tpr - fpr                      # Youden's J
    return float(thr[int(np.argmax(j))])


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print(f"  accuracy   : {accuracy_score(y_true, y_pred):.3f}")
    print(f"  precision  : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  recall/sens: {recall_score(y_true, y_pred, zero_division=0):.3f}  (melanoma)")
    print(f"  specificity: {spec:.3f}")
    print(f"  f1         : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  roc_auc    : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  confusion  : TN={tn} FP={fp} FN={fn} TP={tp}")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--cache", type=str, default="model/features.npz",
                   help="npz cache for extracted features (set '' to disable)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--target-recall", type=float, default=None,
                   help="if set, pick the threshold that reaches this melanoma recall "
                        "(implies --threshold-strategy target_recall)")
    p.add_argument("--target-precision", type=float, default=None,
                   help="if set, pick the threshold that reaches this precision "
                        "(implies --threshold-strategy target_precision)")
    p.add_argument("--threshold-strategy", type=str, default="max_f1",
                   choices=["youden", "max_f1", "target_recall", "target_precision"],
                   help="how to pick the decision threshold on the validation set "
                        "(default: max_f1, which raises precision over Youden's J)")
    p.add_argument("--pos-weight-scale", type=float, default=1.0,
                   help="multiplier on the class-imbalance pos_weight used by "
                        "BCEWithLogitsLoss; <1.0 makes the model less eager to predict "
                        "melanoma, trading recall for precision")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y, feature_names = load_features(args)
    print("Feature matrix:", X.shape, "| features:", len(feature_names),
          "| melanoma:", int(Y.sum()), "nevus:", int(len(Y) - Y.sum()))

    # Stratified split: train / val / test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, Y, test_size=args.test_ratio, stratify=Y, random_state=args.seed)
    val_size = args.val_ratio / (1.0 - args.test_ratio)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=args.seed)

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    X_tr_n = normalize_with_stats(X_tr, mean, std)
    X_val_n = normalize_with_stats(X_val, mean, std)
    X_test_n = normalize_with_stats(X_test, mean, std)

    Xtr = torch.tensor(X_tr_n, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xval = torch.tensor(X_val_n, dtype=torch.float32)
    Xtest = torch.tensor(X_test_n, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    model = Classifier(input_dim=Xtr.shape[1], hidden_dims=HIDDEN_DIMS, dropout=args.dropout)
    num_pos = float(y_tr.sum())
    num_neg = float(len(y_tr) - num_pos)
    pos_weight = torch.tensor([(num_neg / max(num_pos, 1.0)) * args.pos_weight_scale])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xval)).numpy()
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, no_improve = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - val ROC-AUC: {val_auc:.3f} (best {best_auc:.3f})")
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} (best val ROC-AUC {best_auc:.3f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xval)).numpy()
        test_prob = torch.sigmoid(model(Xtest)).numpy()

    threshold = pick_threshold(y_val, val_prob, args.threshold_strategy,
                                args.target_recall, args.target_precision)
    evaluate("VALIDATION", y_val, val_prob, threshold)
    test_metrics = evaluate("TEST", y_test, test_prob, threshold)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
    with open(os.path.join(MODEL_DIR, "normalization.json"), "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_names": feature_names,
            "threshold": threshold,
            "input_dim": int(Xtr.shape[1]),
            "hidden_dims": list(HIDDEN_DIMS),
            "dropout": args.dropout,
            "version": VERSION,
            "metrics": test_metrics,
        }, f, indent=2)
    print(f"\nSaved artifacts to {MODEL_DIR}/ (version={VERSION})")


if __name__ == "__main__":
    main()
