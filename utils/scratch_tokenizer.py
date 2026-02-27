import json
import re
from collections import Counter
from pathlib import Path
import logging

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[\t\r]", "", text)
    text = text.lower()
    text = re.sub(r" +", " ", text)
    return text.strip()


class CharTokenizer:
    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is None:
            vocab = {
                self.PAD: 0,
                self.UNK: 1,
                self.CLS: 2,
                self.SEP: 3,
            }
        self.vocab = vocab
        self.id2token = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK]

    @property
    def cls_id(self) -> int:
        return self.vocab[self.CLS]

    @property
    def sep_id(self) -> int:
        return self.vocab[self.SEP]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_vocab(
        self,
        texts: list[str],
        min_freq: int = 1,
        max_vocab_size: int | None = 12000,
    ):
        counter = Counter()
        for text in texts:
            text = clean_text(text)
            counter.update(list(text))

        tokens = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
        tokens.sort(key=lambda x: (-x[1], x[0]))

        base_size = 4
        if max_vocab_size is not None:
            limit = max(max_vocab_size - base_size, 0)
            tokens = tokens[:limit]

        self.vocab = {
            self.PAD: 0,
            self.UNK: 1,
            self.CLS: 2,
            self.SEP: 3,
        }
        for token, _ in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.id2token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        text = clean_text(text)
        content_limit = max(max_length - 2, 0)
        token_ids = [self.vocab.get(ch, self.unk_id) for ch in text[:content_limit]]
        token_ids = [self.cls_id] + token_ids + [self.sep_id]

        if len(token_ids) < max_length:
            padding_len = max_length - len(token_ids)
            token_ids += [self.pad_id] * padding_len
        else:
            token_ids = token_ids[:max_length]

        attention_mask = [1 if idx != self.pad_id else 0 for idx in token_ids]
        return token_ids, attention_mask

    def save(self, save_path: str | Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump({"vocab": self.vocab}, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str | Path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(vocab=data["vocab"])
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


import jieba
jieba.setLogLevel(logging.INFO)
class WordTokenizer(CharTokenizer):
    def __init__(self, vocab: dict[str, int] | None = None):
        super().__init__(vocab)
        
    def build_vocab(
        self,
        texts: list[str],
        min_freq: int = 1,
        max_vocab_size: int | None = 12000,
    ):
        counter = Counter()
        for text in texts:
            text = clean_text(text)
            words = jieba.lcut(text)
            counter.update(words)

        tokens = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
        tokens.sort(key=lambda x: (-x[1], x[0]))

        base_size = 4
        if max_vocab_size is not None:
            limit = max(max_vocab_size - base_size, 0)
            tokens = tokens[:limit]

        self.vocab = {
            self.PAD: 0,
            self.UNK: 1,
            self.CLS: 2,
            self.SEP: 3,
        }
        for token, _ in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.id2token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str, max_length: int):
        text = clean_text(text)
        words = jieba.lcut(text)

        content_limit = max(max_length - 2, 0)

        token_ids = [
            self.vocab.get(w, self.unk_id)
            for w in words[:content_limit]
        ]

        token_ids = [self.cls_id] + token_ids + [self.sep_id]

        if len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids += [self.pad_id] * pad_len
        else:
            token_ids = token_ids[:max_length]

        attention_mask = [1 if t != self.pad_id else 0 for t in token_ids]

        return token_ids, attention_mask
