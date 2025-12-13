
import torch
import torch.nn as nn

class LSTMSeq2Seq(nn.Module):

    def __init__(self, n_features, hidden_size=128, num_layers=2, horizon=24, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, horizon * 2)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        h_last = out[:, -1, :]       
        out = self.head(h_last)       
        return out
