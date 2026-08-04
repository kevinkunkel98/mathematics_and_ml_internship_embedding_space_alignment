"""
@article{shinoda2024petface,
  title={PetFace: A Large-Scale Dataset and Benchmark for Animal Identification},
  author={Shinoda, Risa and Shiohara, Kaede},
  journal={arXiv preprint arXiv:2407.13555},
  year={2024}
}
This module contains data loading utilities for the PetFace dataset.

File Structure:
- data/cat/
    - cat.csv  # CSV file with columns: 'Name', 'Breed'
    - <cat_id>/
        - 00.jpg
        - 01.jpg
        - ...
"""

import os
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms
from typing import Tuple

class PetFaceDataset(Dataset):
    def __init__(
        self,
        df,
        img_root,
        label_encoder,
        transform=None,
        df_name="df",
        visualize_breed_distribution=False,
    ):

        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.le = label_encoder
        self.df_name = df_name

        self.df["Breed_encoded"] = self.le.transform(self.df["Breed"])

        # Visualize breed distribution
        # -------------------------------------------------------------------------------------------
        if visualize_breed_distribution:
            breed_order = self.le.classes_
            breed_counts = self.df["Breed"].value_counts()
            breed_counts = breed_counts.reindex(breed_order, fill_value=0)

            plt.figure(figsize=(8, 12))
            plt.barh(breed_counts.index, breed_counts.values)
            plt.title("Breed Distribution df " + self.df_name)
            plt.xlabel("Count")
            plt.ylabel("Breed")
            plt.tight_layout()
            plt.show()
        # -------------------------------------------------------------------------------------------

        # Build list of cats with their image paths and labels
        self.cats = []
        for _, row in self.df.iterrows():
            folder_path = os.path.join(img_root, str(row["Name"]))
            images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            if len(images) > 0:
                self.cats.append({"images": images, "label": row["Breed_encoded"]})

    def __len__(self):
        return len(self.cats)

    def __getitem__(self, idx):
        cat = self.cats[idx]
        # Randomly select ONE image inside the folder
        img_path = random.choice(cat["images"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, cat["label"], img_path


def test_breed_coverage(train_set, val_set, test_set, le):
    """
    Test that every breed appears at least once in each dataset split.

    Args:
        train_set, val_set, test_set: PetFaceDataset instances
        le: LabelEncoder fitted on the full dataset
    """
    all_breeds = set(le.classes_)

    def breeds_in_dataset(dataset):
        labels = [cat["label"] for cat in dataset.cats]
        return set(le.inverse_transform(labels))

    splits = {"Train": train_set, "Validation": val_set, "Test": test_set}

    passed = True
    for name, dataset in splits.items():
        breeds = breeds_in_dataset(dataset)
        missing = all_breeds - breeds
        if missing:
            print(f"[FAIL] {name} split is missing breeds: {missing}")
            passed = False
        else:
            print(f"[PASS] {name} split contains all breeds.")

    if passed:
        print("\nAll splits contain every breed at least once.")
    else:
        print("\nSome splits are missing breeds. Consider per-breed splitting.")


def load_petface_data(
    batch_size,
    label_path,
    test_size=0.2,
    val_size=0.1,
    strip_percent=4,
    visualize_breed_distribution=False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:

    df = pd.read_csv(label_path, dtype={"Name": str})

    df = df[df["Breed"] != "Unknown"]
    df = df[df["Breed"] != "Mixed Breed"]
    df = df.reset_index(drop=True)

    # TODO Breed should be evenly distributed
    if strip_percent > 0:
        n = int(len(df) * strip_percent / 100)
        df = df.iloc[:n].reset_index(drop=True)
        print(f"Stripped down dataframe to {len(df)} entries.")

    # Remove breeds with fewer than 3 examples (required for 3-way split)
    df = df.groupby("Breed").filter(lambda x: len(x) >= 3).reset_index(drop=True)
    n_breeds = df['Breed'].nunique()
    print(
        f"Dataframe after removing 'Unknown' and 'Mixed Breed' has {len(df)} entries and {n_breeds} unique breeds.\n"
    )
    
    stripped_label_path = label_path.replace(".csv", "_stripped.csv")
    df.to_csv(stripped_label_path, index=False)

    le = LabelEncoder()
    le.fit(df["Breed"])

    train_list = []
    val_list = []
    test_list = []

    for breed, group in df.groupby("Breed"):
        if len(group) < 3:
            raise ValueError(
                f"Breed '{breed}' has fewer than 3 examples, which is not allowed for 3-way split. Should be handeled earlier."
            )
        temp_df, test_df = train_test_split(group, test_size=test_size, random_state=42)

        train_df, val_df = train_test_split(
            temp_df, test_size=val_size, random_state=42
        )

        train_list.append(train_df)
        val_list.append(val_df)
        test_list.append(test_df)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
        ]
    )

    """
    Pixel nach ToTensor() sind in [0, 1]
    Mit mean=0.5 und std=0.5 ⇒ Wertebereich wird [-1, 1]
    Mean ≈ 0, Variance ≈ 1
    Die Aktivierungsfunktionen (ReLU, Tanh etc.) arbeiten oft stabiler, wenn Inputs um Null liegen.
    xnorm = (x - mean) / std
    Für mean=0.5, std=0.5:
    xnorm = (x - 0.5) / 0.5 = 2x - 1
    """
    train_set = PetFaceDataset(
        train_df,
        "data/cat/",
        le,
        transform,
        "train",
        visualize_breed_distribution=visualize_breed_distribution,
    )
    val_set = PetFaceDataset(
        val_df,
        "data/cat/",
        le,
        transform,
        "val",
        visualize_breed_distribution=visualize_breed_distribution,
    )
    test_set = PetFaceDataset(
        test_df,
        "data/cat/",
        le,
        transform,
        "test",
        visualize_breed_distribution=visualize_breed_distribution,
    )

    test_breed_coverage(train_set, val_set, test_set, le)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_breeds

# TODO generalize load data function to other datasets as well


def plot_breed_distribution(df: pd.DataFrame, label_encoder, split_name: str = "", figsize: tuple = (10, 12)) -> plt.Figure:
    """
    Create and return a breed distribution plot.
    
    Args:
        df: DataFrame with Breed column
        label_encoder: LabelEncoder fitted on the full dataset
        split_name: Name of the split (e.g., "train", "val", "test") for title
        figsize: Figure size (default: (10, 12))
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    breed_order = label_encoder.classes_
    breed_counts = df["Breed"].value_counts()
    breed_counts = breed_counts.reindex(breed_order, fill_value=0)
    
    fig = plt.figure(figsize=figsize)
    plt.barh(breed_counts.index, breed_counts.values, color='steelblue')
    
    title = f"Breed Distribution"
    if split_name:
        title += f" ({split_name})"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Breed", fontsize=12)
    plt.tight_layout()
    
    return fig


def main():
    train_loader, val_loader, test_loader, _ = load_petface_data(
        batch_size=32,
        label_path="./data/cat/cat.csv",
        strip_percent=4,
        visualize_breed_distribution=True,
    )
    i = 0
    for images, labels, img_paths, folder_paths in train_loader:
        for img_tensor, label_idx, folder_path in zip(images, labels, folder_paths):
            breed_name = train_loader.dataset.le.inverse_transform([label_idx.item()])[
                0
            ]
            img = img_tensor.permute(1, 2, 0).numpy()
            i = i + 1
            plt.imshow(img)
            plt.title(f"Breed: {breed_name}, Path: {folder_path}")
            plt.show()
            break
        print(f"Processed {i} images, from every batch one.")


if __name__ == "__main__":
    main()
