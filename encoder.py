import torch
import torch.nn as nn
from utils import positional_encoding
from encoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1, n_features=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Conv1d(n_features, d_model, kernel_size=1)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate)
                                         for _ in range(num_layers)])

        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, n_features)
        seq_len = x.size(1)

        # Conv1d expects (batch, channels, length), so we need to transpose
        x = x.transpose(1, 2)  # (batch_size, n_features, seq_len)
        x = self.embedding(x)  # (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)

        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # Add positional encoding
        device = x.device
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(device)
        x = x + pos_encoding

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, seq_len, d_model)