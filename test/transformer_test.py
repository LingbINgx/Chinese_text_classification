
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from models.transformer_classifier import TransformerClassifier
from train.transformer_trainer import TrainConfig_transformer
from utils.evaluate import evaluate
from utils.data_loader import load_dataframe, prepare_dataloader

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
    config = TrainConfig_transformer.from_json(ROOT_DIR / config_path)
    output_dir = ROOT_DIR / config.output_dir
    model_path = Path(checkpoint) if checkpoint else output_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}，请先训练模型。")

    id2label = _load_label_mapping(output_dir)
    label2id = {label: idx for idx, label in id2label.items()}
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

    return model, tokenizer, device, id2label, label2id, config


# def predict_one(
#     text,
#     model,
#     tokenizer,
#     device,
#     id2label,
#     max_length,
# ):
#     encoded = encode_text(text, tokenizer, max_length=max_length)
#     batch = {key: value.to(device) for key, value in encoded.items()}

#     with torch.no_grad():
#         _, logits = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             token_type_ids=batch.get("token_type_ids"),
#             labels=None,
#         )
#         probabilities = torch.softmax(logits, dim=-1).squeeze(0)

#     pred_id = int(torch.argmax(probabilities).item())
#     pred_label = id2label[pred_id]
#     pred_conf = float(probabilities[pred_id].item())
#     return pred_id, pred_label, pred_conf


def main(text: str | None = None):
    
    model, tokenizer, device, id2label, label2id, config = _build_runtime(config_path="checkpoints/transformer_cnews/config_snapshot.json", checkpoint=None)
    
    test_df = load_dataframe(ROOT_DIR / config.test_file_path)
    test_loader = prepare_dataloader(
        test_df,
        label2id,
        tokenizer,
        config.max_length,
        config.eval_batch_size,
        dataset_cls="transformer",
        shuffle=False,
    )
    
    evaluate(model, test_loader, device, model_name="transformer", plt_confusion_matrix=True, labels=list(id2label.values()))
    

    # if text is not None:
    #     pred_id, pred_label, conf = predict_one(
    #         text=text,
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device,
    #         id2label=id2label,
    #         max_length=max_length,
    #     )
    #     print(f"text: {text}")
    #     print(f"pred_id: {pred_id}, pred_label: {pred_label}, confidence: {conf:.4f}")


if __name__ == "__main__":
    text = ""
    main(text)
