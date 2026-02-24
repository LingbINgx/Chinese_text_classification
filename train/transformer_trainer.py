
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from loguru import logger
from tqdm import tqdm

import sys
sys.path.append("..")
from models.transformer_classifier import TransformerClassifier
from utils.train_config import get_config
from utils.data_loader import *
from utils.evaluate import evaluate
from utils.device import to_device


@logger.catch
def train(config_path: str | Path = "params/params.yaml"):
    root_dir = Path(__file__).resolve().parents[1]
    config = get_config(root_dir / config_path, "transformer")

    train_df = load_dataframe(root_dir / config.train_file_path)
    val_df = load_dataframe(root_dir / config.val_file_path)
    test_df = load_dataframe(root_dir / config.test_file_path)

    label2id, id2label = build_label_mapping(train_df)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader = prepare_dataloader(
        train_df,
        label2id,
        tokenizer,
        config.max_length,
        config.train_batch_size,
        shuffle=True,
    )
    val_loader = prepare_dataloader(
        val_df,
        label2id,
        tokenizer,
        config.max_length,
        config.eval_batch_size,
    )
    test_loader = prepare_dataloader(
        test_df,
        label2id,
        tokenizer,
        config.max_length,
        config.eval_batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(
        model_name=config.model_name,
        num_labels=len(label2id),
        dropout_prob=config.dropout_prob,
        freeze_encoder=config.freeze_encoder,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = root_dir / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"

    best_val_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = to_device(batch, device)
            optimizer.zero_grad()
            loss, logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item() * batch["labels"].size(0)
            preds = torch.argmax(logits, dim=1)
            total_train_correct += (preds == batch["labels"]).sum().item()
            total_train_count += batch["labels"].size(0)

        train_loss = total_train_loss / max(total_train_count, 1)
        train_acc = total_train_correct / max(total_train_count, 1)

        val_loss, val_acc, val_recall, val_f1 = evaluate(model, val_loader, device, model_name="transformer")
        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_recall={val_recall:.4f}, val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_recall, test_f1 = evaluate(model, test_loader, device, model_name="transformer")
    logger.info(f"Test | loss={test_loss:.4f}, acc={test_acc:.4f}, recall={test_recall:.4f}, f1={test_f1:.4f}")

    tokenizer.save_pretrained(output_dir / "tokenizer")
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as file:
        json.dump({"label2id": label2id, "id2label": id2label}, file, ensure_ascii=False, indent=2)

    logger.info(f"最佳模型与映射已保存到: {output_dir}")


if __name__ == "__main__":
    train()
