# This script was modified from
# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py
import torch
from torch import nn


class LSTM_shakespeare_1L(torch.nn.Module):
    def __init__(self, n_vocab=80, embedding_dim=64, hidden_dim_1=256, nb_layers_1=1):
        super(LSTM_shakespeare_1L, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1

        self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
        self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1)
        self.hidden1out = torch.nn.Linear(hidden_dim_1, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.hidden2out = torch.nn.Linear(128, n_vocab)

    def forward(self, seq_in, state=None):
        if state is not None:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings, state["h1"])
        else:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings)
        ht = lstm_out[-1]
        out = self.hidden1out(ht)
        out = self.hidden2out(self.drop1(out))

        return out, {"h1": (h_state1[0].clone().detach(), h_state1[1].clone().detach())}

    def zero_state(self, batch_size, device=torch.device('cpu')):
        zero_state = {
            "h1": (torch.zeros(1, batch_size, self.hidden_dim_1).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim_1).to(device))
        }
        return zero_state