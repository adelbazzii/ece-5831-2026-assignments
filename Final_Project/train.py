# trains the model and evaluates it

import json
import os
import random

import models
import numpy as np
import plots
import torch
from data import class_weights, make_loaders, split_data
from evaluate import run_test
from torch import nn
from torchinfo import summary
from torchvision import datasets
from train_utils import fit, get_device

DATA_DIR = "dataset"
MODEL_NAME = "SeaCNNv1"

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
LABEL_SMOOTHING = 0.1


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("device:", device)
    os.makedirs(MODEL_NAME, exist_ok=True)

    # data
    folder = datasets.ImageFolder(DATA_DIR)
    paths, labels = zip(*folder.samples)
    class_names = folder.classes
    num_classes = len(class_names)
    splits = split_data(paths, labels, val_frac=0.15, test_frac=0.15, seed=SEED)
    print(f"{len(paths)} images, {num_classes} classes")
    print(
        f"train: {len(splits['train'])}, val: {len(splits['val'])}, "
        f"test: {len(splits['test'])}"
    )

    loaders = make_loaders(splits, IMG_SIZE, batch_size=BATCH_SIZE)
    plots.plot_class_distribution(
        splits["train"], class_names, os.path.join(MODEL_NAME, "class_distribution.png")
    )
    plots.plot_augmented_batch(
        loaders["train"], class_names, os.path.join(MODEL_NAME, "augmented_samples.png")
    )

    # weighted loss because the classes are imbalanced
    weights = class_weights(splits["train"], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # model
    # look up the class in models.py with the same name as MODEL_NAME
    model = getattr(models, MODEL_NAME)(num_classes).to(device)
    # layer table
    model_summary = summary(
        model, input_size=(1, 3, IMG_SIZE, IMG_SIZE), device=device, verbose=0
    )
    print(model_summary)
    with open(os.path.join(MODEL_NAME, "model_summary.txt"), "w") as f:
        f.write(str(model_summary))

    config = {
        "model": MODEL_NAME,
        "params": model_summary.trainable_params,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "label_smoothing": LABEL_SMOOTHING,
        "class_weights": True,
        "seed": SEED,
        "device": str(device),
    }
    with open(os.path.join(MODEL_NAME, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = fit(
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        device,
        EPOCHS,
        os.path.join(MODEL_NAME, "best.pt"),
        patience=PATIENCE,
    )
    with open(os.path.join(MODEL_NAME, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plots.plot_history(
        history, MODEL_NAME, os.path.join(MODEL_NAME, "learning_curves.png")
    )

    # testing
    run_test(model, loaders, criterion, device, class_names, history, MODEL_NAME)


if __name__ == "__main__":
    main()
