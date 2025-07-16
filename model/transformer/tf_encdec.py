import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, batch-first version.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # create positional encodings [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # reshape to [1, max_len, d_model] for batch-first
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        # add positional encodings
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerEncoderModel(nn.Module):
    """
    Transformer-based encoder for ISAC autoencoder, batch-first.
    Input: m of shape [B, K] (bits).
    Output: c of shape [B, 2*N].
    """
    def __init__(self, K, N, d_model=128, nhead=4, num_layers=3, dim_feedforward=512):
        super().__init__()
        self.K = K
        self.N = N
        self.d_model = d_model
        # Embed each bit (scalar) to d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=K)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Project flattened features to codeword
        self.output_proj = nn.Linear(K * d_model, 2 * N)

    def forward(self, m):
        # m: [B, K]
        x = m.unsqueeze(-1)               # [B, K, 1]
        x = self.input_proj(x)           # [B, K, d_model]
        x = self.pos_encoder(x)          # [B, K, d_model]
        x = self.transformer(x)          # [B, K, d_model]
        x_flat = x.reshape(x.size(0), self.K * self.d_model)
        c = self.output_proj(x_flat)     # [B, 2*N]
        return c


class TransformerDecoderModel(nn.Module):
    """
    Transformer-based decoder for ISAC autoencoder, batch-first.
    """
    def __init__(self, K, N, d_model=128, nhead=4, num_layers=3, dim_feedforward=512):
        super().__init__()
        self.K = K
        self.N = N
        self.d_model = d_model
        # Embed channel-use scalar to d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=2 * N)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        # Final projection to K*2 logits
        self.output_proj = nn.Linear((2 * N) * d_model, K * 2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, c_noisy):
        """
        Decode noisy codeword into per-bit P(bit=1).
        Used by isac_loss for BCE term.
        """
        # get raw logits [B, K, 2]
        B = c_noisy.size(0)
        x = c_noisy.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.reshape(B, (2 * self.N) * self.d_model)
        logits = self.output_proj(x)  # [B, K*2]
        logits = logits.view(B, self.K, 2)  # [B, K, 2]
        # softmax over the last dim → [P(0), P(1)]
        probs = F.softmax(logits, dim=-1)
        # return P(bit=1)  → shape [B, K]
        return probs[:, :, 1]