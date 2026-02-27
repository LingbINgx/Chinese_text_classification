
import re
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding



def clean_text(text: str):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[\t\r]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text)

    return text.strip()

def encode_text(text: str, tokenizer, max_length: int = 256):
    text = clean_text(text)
    return tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt",
    )


class TransformerTextDataset(Dataset):
    # 这使用预训练Transformer模型
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

    def get_data_loader(self, batch_size, shuffle=False):
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")
        dataset = self.__class__(
            contents=self.contents,
            labels=self.labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        res = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
        return res


class ScratchTextDataset(Dataset):
    # 使用从头训练的文本分类模型
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
    
    def get_data_loader(self, batch_size, shuffle=False): 
        dataset = self.__class__(
            contents=self.contents,
            labels=self.labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        res = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return res
        

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


def prepare_dataloader(df, label2id, tokenizer, max_length, batch_size,  dataset_cls="transformer", shuffle=False):
    labels = [label2id[label] for label in df["label"].tolist()]
    
    dataset_map = {
        "transformer": TransformerTextDataset,
        "scratch": ScratchTextDataset,
    }
    if dataset_cls not in dataset_map:
        raise ValueError(f"Unknown dataset_cls: {dataset_cls}")
    dataset_cls = dataset_map[dataset_cls]
    
    res = dataset_cls(
        contents=df["content"].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    ).get_data_loader(batch_size=batch_size, shuffle=shuffle) 
    return res