import torch
import torch.nn as nn


class ScratchTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        max_length: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        src_key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = self.norm(pooled)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
