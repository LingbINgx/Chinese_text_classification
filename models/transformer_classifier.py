import torch
from torch import nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
