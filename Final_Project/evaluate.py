# evaluates a pre-trained model

import json
import os

import models
import plots
import torch
from data import class_weights, make_loaders, split_data
from metrics import evaluate, train_val_gap
from torch import nn
from torchvision import datasets
from train_utils import get_device

DATA_DIR = "dataset"
MODEL_NAME = "SeaCNNv1"


def run_test(model, loaders, criterion, device, class_names, history, model_name):
    # computes all test metrics and saves them + the plots to the model folder
    # (train.py calls this too after training)
    results, cm, report, probs, labels = evaluate(
        model, loaders["test"], criterion, device, class_names
    )
    results["train_val_gap"] = train_val_gap(history)

    print(f"\ntest loss: {results['test_loss']:.4f}")
    print(f"top-1 acc: {results['test_acc']:.4f}")
    print(f"top-3 acc: {results['test_top3_acc']:.4f}")
    print(f"top-5 acc: {results['test_top5_acc']:.4f}")
    print(f"balanced acc: {results['test_balanced_acc']:.4f}")
    print(f"macro F1: {results['test_macro_f1']:.4f}\n")
    print(report)
    print("most confused (true -> predicted):")
    for pair in results["most_confused"]:
        print(f"  {pair['true']} -> {pair['pred']}: {pair['count']}")
    gap = results["train_val_gap"]
    print(
        f"\nbest epoch {gap['best_epoch']}: train acc {gap['train_acc']:.4f}, "
        f"val acc {gap['val_acc']:.4f}, gap {gap['acc_gap']:.4f}"
    )

    with open(os.path.join(model_name, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    plots.plot_confusion_matrix(
        cm, class_names, model_name, os.path.join(model_name, "confusion_matrix.png")
    )
    plots.plot_per_class_f1(
        results, model_name, os.path.join(model_name, "per_class_f1.png")
    )
    plots.plot_precision_recall(
        results, model_name, os.path.join(model_name, "precision_recall.png")
    )
    plots.plot_sample_predictions(
        model,
        loaders["test"],
        class_names,
        device,
        os.path.join(model_name, "sample_predictions.png"),
    )
    plots.plot_confident_mistakes(
        probs,
        labels,
        loaders["test"].dataset,
        class_names,
        os.path.join(model_name, "confident_mistakes.png"),
    )
    plots.plot_gradcam(
        model,
        loaders["test"],
        class_names,
        device,
        os.path.join(model_name, "gradcam.png"),
    )
    plots.plot_first_layer_filters(
        model, model_name, os.path.join(model_name, "first_layer_filters.png")
    )
    plots.plot_feature_maps(
        model,
        loaders["test"],
        model_name,
        device,
        os.path.join(model_name, "feature_maps.png"),
    )
    print("saved results to", model_name)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(MODEL_NAME, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_NAME, "history.json")) as f:
        history = json.load(f)

    device = get_device()
    print("device:", device)

    # data
    folder = datasets.ImageFolder(DATA_DIR)
    paths, labels = zip(*folder.samples)
    class_names = folder.classes
    num_classes = len(class_names)
    splits = split_data(
        paths, labels, val_frac=0.15, test_frac=0.15, seed=config["seed"]
    )
    loaders = make_loaders(
        splits,
        config["img_size"],
        batch_size=config["batch_size"],
    )

    weights = class_weights(splits["train"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config["label_smoothing"]
    )

    model = getattr(models, MODEL_NAME)(num_classes).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_NAME, "best.pt"), map_location=device)
    )

    run_test(model, loaders, criterion, device, class_names, history, MODEL_NAME)


if __name__ == "__main__":
    main()
