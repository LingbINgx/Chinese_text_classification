import torch
import torch.nn as nn

class TextRNNGRUClassifier(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 num_labels, 
                 embed_dim, 
                 hidden_dim, 
                 num_layers=1, 
                 bidirectional=False, 
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)    
    

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return loss, logits
    
    