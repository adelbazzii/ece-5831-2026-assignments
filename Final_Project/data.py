# data processing

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# imagenet mean/std
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def split_data(paths, labels, val_frac=0.15, test_frac=0.15, seed=42):
    # stratify so train/val/test all have the same class balance
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_frac / (1 - test_frac),
        stratify=train_labels,
        random_state=seed,
    )
    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }


def class_weights(samples, num_classes):
    # bigger weight for classes with fewer images (turtle has ~4x more than seal)
    labels = [y for _, y in samples]
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(w).float()


def get_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, test_tf


class SeaAnimalDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # a few images are RGBA or grayscale
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_loaders(splits, img_size, batch_size=64, num_workers=8):
    train_tf, test_tf = get_transforms(img_size)
    train_set = SeaAnimalDataset(splits["train"], train_tf)
    val_set = SeaAnimalDataset(splits["val"], test_tf)
    test_set = SeaAnimalDataset(splits["test"], test_tf)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": True,
    }
    return {
        "train": DataLoader(train_set, shuffle=True, drop_last=True, **args),
        "val": DataLoader(val_set, **args),
        "test": DataLoader(test_set, **args),
    }


def unnormalize(img):
    # tensor back to a normal image for plots
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
