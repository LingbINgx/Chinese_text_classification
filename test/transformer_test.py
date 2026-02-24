from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from models.transformer_classifier import TransformerClassifier
from utils.preprocessing import encode_text
from utils.train_config import load_config


def _load_label_mapping(output_dir: Path) -> dict[int, str]:
    mapping_path = output_dir / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"未找到标签映射文件: {mapping_path}，请先完成训练并确保输出目录正确。"
        )

    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    id2label_raw = mapping.get("id2label", {})
    return {int(key): str(value) for key, value in id2label_raw.items()}


def _build_runtime(config_path: str, checkpoint: str | None):
    config = load_config(ROOT_DIR / config_path)
    output_dir = ROOT_DIR / config.output_dir
    model_path = Path(checkpoint) if checkpoint else output_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}，请先训练模型。")

    id2label = _load_label_mapping(output_dir)
    num_labels = len(id2label)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer_dir.exists()
        else AutoTokenizer.from_pretrained(config.model_name)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(
        model_name=config.model_name,
        num_labels=num_labels,
        dropout_prob=config.dropout_prob,
        freeze_encoder=False,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device, id2label, config.max_length


def predict_one(
    text: str,
    model: TransformerClassifier,
    tokenizer,
    device: torch.device,
    id2label: dict[int, str],
    max_length: int,
):
    encoded = encode_text(text, tokenizer, max_length=max_length)
    batch = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        _, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=None,
        )
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probabilities).item())
    pred_label = id2label[pred_id]
    pred_conf = float(probabilities[pred_id].item())
    return pred_id, pred_label, pred_conf


def main(text: str | None = None):
    
    model, tokenizer, device, id2label, max_length = _build_runtime(config_path="params.yaml", checkpoint=None)

    if text is not None:
        pred_id, pred_label, conf = predict_one(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            id2label=id2label,
            max_length=max_length,
        )
        print(f"text: {text}")
        print(f"pred_id: {pred_id}, pred_label: {pred_label}, confidence: {conf:.4f}")


if __name__ == "__main__":
    text = "2006年第一季度，安徽省质量技术监督局组织蜂蜜产品省级监督抽查，共在合肥、亳州、淮北3个市抽查样品31组，经检验，合格23组，抽样合格率为74.2％。主要不合格项目为还原糖、淀粉酶活性、羟甲基糠醛。人为加入蔗糖，使得蔗糖含量超标。蜂蜜中的果糖和葡萄糖都具备还原性，统称为还原糖，它们属于单糖，能被人体直接吸收。蔗糖、麦芽糖同属双糖，蔗糖是非还原性双糖，它们不能被人体直接吸收。蜂蜜中含有的蔗糖是天然花粉中所含有未转化的部分，非人工加入的。此次抽查发现，部分蜂蜜被人为做了手脚，加入了大量蔗糖。淀粉酶值偏低。酶值高低是检测蜂蜜质量优劣的重要指标，酶值高，表明蜂蜜营养价值高。造成酶值低的主要原因有：一是蜂蜜未经充分酿制。未经充分酿制的蜂蜜，其内在质量明显低于成熟蜂蜜；二是经过高温加热处理，加工温度过高，时间过长，会损失蜂蜜营养成分，使蜂蜜色泽加深，羟甲基糠醛升高，酶值降低；三是储藏不当(或过久)的蜂蜜，其酶值也会受到破坏，甚至完全消失，一般气温下，酶含量在17个月内可能降低一半；阳光照射对酶值也有一定影响。另外，掺假掺杂也是酶值降低的一个主要原因，掺入蔗糖的蜂蜜中的酶就要对蔗糖起转化作用，把蔗糖转化成还原糖，从而消耗了蜂蜜中的转化酶，致使酶值下降。羟甲基糠醛值偏高。羟甲基糠醛是蜂蜜中一个重要的质量指标，主要来源于蜂蜜中的还原糖在高温条件下发生的美拉德反应，或者在酸性条件下发生的脱水反应。羟甲基糠醛值过高，说明蜂蜜可能经过了高温处理，或者储藏不当，甚至掺入了大量蔗糖。消费者在选购蜂蜜时，要注意查看产品标签上的生产日期和保质期，选择正规厂家生产的合格产品，并且尽量避免购买价格过低的蜂蜜，以免买到掺假产品。"
    main(text)
