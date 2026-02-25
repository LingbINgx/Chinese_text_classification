
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from models.transformer_classifier import TransformerClassifier
from utils.preprocessing import encode_text
from utils.train_config import get_config


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
    config = get_config(ROOT_DIR / config_path, "transformer")
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
    text,
    model,
    tokenizer,
    device,
    id2label,
    max_length,
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
    
    model, tokenizer, device, id2label, max_length = _build_runtime(config_path="params/params.yaml", checkpoint=None)

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
    text = "北京时间3月16日，NBA官方公布了对于灰熊球星贾-莫兰特直播中持枪事件的调查结果灰熊，由于无法确定枪支是否为莫兰特所有，也无法证明他曾持枪到过NBA场馆，因为对他处以禁赛八场的处罚，且此前已禁赛场次将算在禁赛八场的场次内，他最早将在下周复出。"
    main(text)
