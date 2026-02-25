import torch
from sklearn.metrics import recall_score, f1_score
from .device import to_device

#transformer with bert
def transformer(model, batch):
    with torch.no_grad():
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
        )
    return loss, logits
    

#scratch transformer
def scartch_transformer(model, batch):
    with torch.no_grad():
        loss, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    return loss, logits    



evaluate_function_map = {
    "scratch_transformer": scartch_transformer,
    "transformer": transformer,
}


def evaluate(model, data_loader, device, model_name=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = to_device(batch, device)
            
            if model_name not in evaluate_function_map:
                raise ValueError("model_name must be provided for evaluation.")
            
            loss, logits = evaluate_function_map[model_name](model, batch)
            
            total_loss += loss * batch["labels"].size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_count += batch["labels"].size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, recall, f1