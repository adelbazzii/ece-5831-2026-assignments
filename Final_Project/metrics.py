# test set metrics

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from train_utils import eval_epoch


def evaluate(model, loader, criterion, device, class_names):
    loss, acc, probs, labels = eval_epoch(model, loader, criterion, device)
    preds = probs.argmax(1)
    classes = np.arange(len(class_names))

    cm = confusion_matrix(labels, preds, labels=classes)
    report = classification_report(
        labels,
        preds,
        labels=classes,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=classes, zero_division=0
    )

    results = {
        "test_loss": loss,
        "test_acc": acc,
        "test_top3_acc": top_k_accuracy_score(labels, probs, k=3, labels=classes),
        "test_top5_acc": top_k_accuracy_score(labels, probs, k=5, labels=classes),
        # average of the per-class recalls, so small classes count as much as big ones
        "test_balanced_acc": balanced_accuracy_score(labels, preds),
        "test_macro_f1": f1.mean(),
        "per_class_precision": dict(zip(class_names, precision)),
        "per_class_recall": dict(zip(class_names, recall)),
        "per_class_f1": dict(zip(class_names, f1)),
        "most_confused": most_confused(cm, class_names),
    }
    return results, cm, report, probs, labels


def most_confused(cm, class_names, n=10):
    # largest off-diagonal counts in the confusion matrix
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    top = np.argsort(errors, axis=None)[::-1][:n]
    pairs = []
    for i, j in zip(*np.unravel_index(top, cm.shape)):
        pairs.append(
            {"true": class_names[i], "pred": class_names[j], "count": int(errors[i, j])}
        )
    return pairs


def train_val_gap(history):
    # gap at the epoch whose weights were kept (lowest val loss)
    # note: train acc is measured with augmentation and dropout on, so the
    # real gap is a bit bigger than this
    best = int(np.argmin(history["val_loss"]))
    return {
        "best_epoch": best + 1,
        "train_acc": history["train_acc"][best],
        "val_acc": history["val_acc"][best],
        "acc_gap": history["train_acc"][best] - history["val_acc"][best],
        "train_loss": history["train_loss"][best],
        "val_loss": history["val_loss"][best],
        "loss_gap": history["val_loss"][best] - history["train_loss"][best],
    }
