import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

class LSTMmodel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocabSize,dropout):
        super(LSTMmodel, self).__init__()
        self.vocab_size = vocabSize
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, device=device)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size, batch_first=True, device=device)
        self.dropLayer = nn.Dropout(p=self.dropout)
        self.output = nn.Linear(
            self.hidden_size, self.vocab_size, bias=False, device=device)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, xContext):
        xembed = self.dropLayer(self.embedding(xContext))
        out, hidden = self.lstm(xembed)
        out = self.log_softmax(self.output(out[:,-1]))
        return out, hidden

