from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.scratch_tokenizer import CharTokenizer


class ScratchTextDataset(Dataset):
    def __init__(
        self,
        contents: list[str],
        labels: list[int],
        tokenizer: CharTokenizer,
        max_length: int,
    ):
        self.contents = contents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.tokenizer.encode(
            self.contents[idx],
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_cols = {"content", "label"}
    if not required_cols.issubset(frame.columns):
        raise ValueError(f"{csv_path} 缺少必须字段: {required_cols}")
    return frame[["content", "label"]].dropna()


def build_label_mapping(train_df: pd.DataFrame):
    unique_labels = sorted(train_df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def prepare_dataloader(
    df: pd.DataFrame,
    label2id: dict,
    tokenizer: CharTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
):
    labels = [label2id[label] for label in df["label"].tolist()]
    dataset = ScratchTextDataset(
        contents=df["content"].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
