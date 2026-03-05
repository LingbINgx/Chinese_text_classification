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
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            feature = torch.relu(conv(x))#.squeeze(3)
            #pooled = torch.max(feature, dim=2).values
            pooled = self.max_pool(feature).squeeze(3).squeeze(2)
            conv_outputs.append(pooled)

        features = torch.cat(conv_outputs, dim=1)
        logits = self.classifier(self.dropout(features))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
