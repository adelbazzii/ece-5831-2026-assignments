# training loop

import copy
import time

import torch
from tqdm import tqdm


def get_device():
    # use the mac gpu if it's available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    n = 0
    bar = tqdm(loader, desc="train", leave=False)
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
        bar.set_postfix(loss=f"{total_loss / n:.4f}")
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
            # keep the score for every class (needed for top-k accuracy)
            all_probs.append(out.softmax(1).cpu())
            all_labels.append(y.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / n, correct / n, probs, labels


def fit(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
    patience=10,
):
    # trains and keeps the weights with the lowest val loss
    # stops early if val loss hasn't improved for `patience` epochs
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = eval_epoch(model, loaders["val"], criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        print(
            f"epoch {epoch}/{epochs}: train loss {train_loss:.4f}, "
            f"train acc {train_acc:.4f}, val loss {val_loss:.4f}, "
            f"val acc {val_acc:.4f}, lr {lr:.2e} ({time.time() - start:.0f}s)"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"stopping early, no improvement for {patience} epochs")
                break

    model.load_state_dict(best_weights)
    return history
