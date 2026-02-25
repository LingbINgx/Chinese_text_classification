import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from models.textcnn_classifier import TextCNNClassifier
from utils.scratch_tokenizer import CharTokenizer
from train.textcnn_trainer import TrainConfig_textcnn



def _load_label_mapping(output_dir: Path) -> dict[int, str]:
    mapping_path = output_dir / "textcnn_label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"未找到标签映射文件: {mapping_path}，请先完成训练并确保输出目录正确。"
        )

    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    id2label_raw = mapping.get("id2label", {})
    return {int(key): str(value) for key, value in id2label_raw.items()}



def _build_runtime(config_path: str, checkpoint: str | None):
    config = TrainConfig_textcnn.from_yaml(ROOT_DIR / config_path)
    output_dir = ROOT_DIR / config.output_dir
    model_path = Path(checkpoint) if checkpoint else output_dir / "textcnn_best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}，请先训练模型。")

    id2label = _load_label_mapping(output_dir)
    num_labels = len(id2label)

    tokenizer_path = output_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"未找到分词器文件: {tokenizer_path}，请先训练模型。")
    tokenizer = CharTokenizer.load(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNNClassifier(
        vocab_size=tokenizer.vocab_size,
        num_labels=num_labels,
        embed_dim=config.embed_dim,
        num_filters=config.num_filters,
        kernel_sizes=tuple(config.kernel_sizes),
        dropout=config.dropout,
        padding_idx=tokenizer.pad_id,
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
    input_ids, attention_mask = tokenizer.encode(text, max_length=max_length)
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
    }

    with torch.no_grad():
        _, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
        )
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probabilities).item())
    pred_label = id2label[pred_id]
    pred_conf = float(probabilities[pred_id].item())
    return pred_id, pred_label, pred_conf



def main(text: str | None = None):
    model, tokenizer, device, id2label, max_length = _build_runtime(
        config_path="params/params_textcnn.yaml",
        checkpoint=None,
    )

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
