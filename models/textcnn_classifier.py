import torch
import torch.nn as nn


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 256,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(kernel_size, embed_dim),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            feature = torch.relu(conv(x)).squeeze(3)
            pooled = torch.max(feature, dim=2).values
            conv_outputs.append(pooled)

        features = torch.cat(conv_outputs, dim=1)
        logits = self.classifier(self.dropout(features))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
