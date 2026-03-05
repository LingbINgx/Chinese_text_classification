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
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, seq_len = input_ids.size()
        
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=attention_mask.device)
        
        x = self.embed_layer_norm(x)
        x = self.dropout(x)

        #src_key_padding_mask = attention_mask == 0
        src_key_padding_mask = torch.cat([cls_mask, (attention_mask == 0)], dim=1)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask = attention_mask.unsqueeze(-1).float()
        #pooled = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = x[:, 0]
        '''1. 为什么使用 CLS Token 而不是平均池化？
            平均池化会平摊所有 token 的权重，包含那些意义较小的词。
            而 CLS Token 是一个可学习的参数向量，
            它在多层 Self-Attention 中通过注意力机制主动吸纳全句中最具分类代表性的信息。
        '''
        pooled = self.norm(pooled)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits
