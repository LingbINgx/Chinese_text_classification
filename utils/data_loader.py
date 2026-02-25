
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

import sys
sys.path.append("..")
from utils.preprocessing import clean_text, encode_text


class TextClassificationDataset(Dataset):
    def __init__(self, contents, labels, tokenizer, max_length):
        super().__init__()
        self.contents = [clean_text(item) for item in contents]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        encoded = encode_text(self.contents[idx], self.tokenizer, self.max_length)
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item



class ScratchTextDataset(Dataset):
    def __init__(self, contents, labels, tokenizer, max_length):
        super().__init__()
        self.contents = [clean_text(item) for item in contents]
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
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item
        

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


def prepare_dataloader(df, label2id, tokenizer, max_length, batch_size, shuffle=False, dataset_cls=TextClassificationDataset):
    labels = [label2id[label] for label in df["label"].tolist()]
    
    if dataset_cls == TextClassificationDataset:
        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        dataset = dataset_cls(
            contents=df["content"].tolist(),
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    
    elif dataset_cls == ScratchTextDataset:
        dataset = dataset_cls(
            contents=df["content"].tolist(),
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    