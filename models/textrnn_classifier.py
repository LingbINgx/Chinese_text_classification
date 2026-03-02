import torch
import torch.nn as nn

class TextRNNClassifier(nn.Module):
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
        self.rnn = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return loss, logits