# evaluation plots

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from data import unnormalize
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn


def save(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_distribution(samples, class_names, save_path):
    counts = Counter(y for _, y in samples)
    order = np.argsort([counts[i] for i in range(len(class_names))])
    plt.figure(figsize=(10, 6))
    plt.barh([class_names[i] for i in order], [counts[i] for i in order])
    plt.title("Training images per class")
    plt.xlabel("Images")
    save(save_path)


def plot_augmented_batch(loader, class_names, save_path):
    imgs, labels = next(iter(loader))
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(unnormalize(imgs[i]))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    save(save_path)


def plot_history(history, model_name, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.title(f"{model_name} loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Training accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.title(f"{model_name} accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    save(save_path)


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    _, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap="Blues",
        include_values=False,
        xticks_rotation="vertical",
        im_kw={"vmin": 0, "vmax": 1},
    )
    ax.set_title(f"{model_name} confusion matrix (normalized)")
    save(save_path)


def plot_per_class_f1(results, model_name, save_path):
    names = list(results["per_class_f1"].keys())
    f1 = np.array(list(results["per_class_f1"].values()))
    order = np.argsort(f1)
    macro_f1 = results["test_macro_f1"]

    plt.figure(figsize=(10, 6))
    plt.barh([names[i] for i in order], f1[order])
    plt.axvline(
        macro_f1, color="gray", linestyle="--", label=f"macro F1 = {macro_f1:.3f}"
    )
    plt.xlabel("F1")
    plt.title(f"{model_name} F1 per class (test set)")
    plt.legend()
    save(save_path)


def plot_sample_predictions(model, loader, class_names, device, save_path):
    model.eval()
    imgs, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(imgs.to(device)).argmax(1).cpu()

    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(unnormalize(imgs[i]))
        true_name = class_names[labels[i]]
        pred_name = class_names[preds[i]]
        color = "green" if true_name == pred_name else "red"
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=9)
        plt.axis("off")
    save(save_path)


def plot_precision_recall(results, model_name, save_path):
    names = list(results["per_class_f1"].keys())
    precision = np.array(list(results["per_class_precision"].values()))
    recall = np.array(list(results["per_class_recall"].values()))
    # same order as the F1 plot
    order = np.argsort(list(results["per_class_f1"].values()))
    y = np.arange(len(names))

    plt.figure(figsize=(10, 8))
    plt.barh(y - 0.2, precision[order], height=0.4, label="Precision")
    plt.barh(y + 0.2, recall[order], height=0.4, label="Recall")
    plt.yticks(y, [names[i] for i in order])
    plt.xlim(0, 1)
    plt.title(f"{model_name} precision and recall per class (test set)")
    plt.legend()
    save(save_path)


def plot_confident_mistakes(probs, labels, dataset, class_names, save_path):
    # the 10 wrong predictions the model was most sure about
    preds = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(preds != labels)[0]
    wrong = wrong[np.argsort(conf[wrong])[::-1]][:10]

    plt.figure(figsize=(15, 6))
    for k, i in enumerate(wrong):
        img, _ = dataset[i]
        plt.subplot(2, 5, k + 1)
        plt.imshow(unnormalize(img))
        plt.title(
            f"True: {class_names[labels[i]]}\n"
            f"Pred: {class_names[preds[i]]} ({conf[i]:.2f})",
            color="red",
            fontsize=9,
        )
        plt.axis("off")
    save(save_path)


def grad_cam(model, x):
    # heatmap of where the last conv layer's activations pushed the
    # predicted class score up
    layer = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    saved = {}

    def save_grad(grad):
        saved["grad"] = grad

    def forward_hook(module, inputs, output):
        saved["act"] = output
        output.register_hook(save_grad)

    handle = layer.register_forward_hook(forward_hook)
    model.eval()
    out = model(x)
    handle.remove()

    preds = out.argmax(1)
    model.zero_grad()
    out.gather(1, preds[:, None]).sum().backward()
    model.zero_grad()

    # weight each channel by its average gradient
    weights = saved["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * saved["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), preds.cpu()


def plot_gradcam(model, loader, class_names, device, save_path, n=6):
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n], labels[:n]
    cams, preds = grad_cam(model, imgs.to(device))

    plt.figure(figsize=(3 * n, 6.5))
    for i in range(n):
        img = unnormalize(imgs[i])
        true_name = class_names[labels[i]]
        pred_name = class_names[preds[i]]
        color = "green" if true_name == pred_name else "red"

        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=9)
        plt.axis("off")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(img)
        plt.imshow(cams[i], cmap="jet", alpha=0.5)
        plt.axis("off")
    plt.suptitle("Grad-CAM (red = regions that drove the prediction)")
    save(save_path)


def plot_first_layer_filters(model, model_name, save_path):
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.detach().cpu()
    # rescale each filter to 0-1 so it can be shown as an RGB image
    low = w.amin(dim=(1, 2, 3), keepdim=True)
    high = w.amax(dim=(1, 2, 3), keepdim=True)
    w = (w - low) / (high - low + 1e-8)

    cols = 8
    rows = int(np.ceil(len(w) / cols))
    plt.figure(figsize=(cols * 1.2, rows * 1.3))
    for i in range(len(w)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(w[i].permute(1, 2, 0).numpy(), interpolation="nearest")
        plt.axis("off")
    plt.suptitle(f"{model_name} first layer filters")
    save(save_path)


def plot_feature_maps(model, loader, model_name, device, save_path, n=17):
    # output of the first ReLU (after the first conv + batchnorm) for one image
    img = next(iter(loader))[0][:1]
    relu = next(m for m in model.modules() if isinstance(m, nn.ReLU))
    saved = {}

    def forward_hook(module, inputs, output):
        saved["maps"] = output.detach().cpu().clone()

    handle = relu.register_forward_hook(forward_hook)
    model.eval()
    with torch.no_grad():
        model(img.to(device))
    handle.remove()
    maps = saved["maps"][0]

    cols = 6
    rows = int(np.ceil((n + 1) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    plt.subplot(rows, cols, 1)
    plt.imshow(unnormalize(img[0]))
    plt.title("input", fontsize=9)
    plt.axis("off")
    for i in range(min(n, len(maps))):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(maps[i], cmap="viridis")
        plt.title(f"channel {i}", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"{model_name} first layer feature maps")
    save(save_path)
