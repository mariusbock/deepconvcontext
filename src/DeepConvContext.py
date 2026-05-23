# ------------------------------------------------------------------------
# Proposed DeepConvContext model (with LSTM, self-attention, or transformer variants)
# ------------------------------------------------------------------------

from torch import nn
import torch

from src.utils.attention import PositionalEncoding


class DeepConvContext(nn.Module):
    """
    DeepConvContext model with LSTM, self-attention, or transformer variants.
    
    Args:
        batch_size: int
            Size of the input batch.
        channels: int
            Number of channels in the input data.
        classes: int
            Number of classes for classification.
        window_size: int
            Size of the input window.
        conv_kernels: int
            Number of convolutional kernels.
        conv_kernel_size: int
            Size of the convolutional kernels.
        lstm_units: int
            Number of LSTM units.
        lstm_layers: int
            Number of LSTM layers.
        dropout: float
            Dropout rate.
        bidirectional: bool
            Whether to use a bidirectional LSTM.
        type: str
            Type of context model ('lstm', 'self-attention', or 'transformer').
        attention_num_heads: int
            Number of attention heads for self-attention and transformer models.
        transformer_depth: int
            Depth of the transformer model.
    """
    def __init__(self, batch_size, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, 
                 bidirectional=False, type='lstm', attention_num_heads=4, transformer_depth=6, dim_feedforward=768):
        super(DeepConvContext, self).__init__()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.window_lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers, batch_first=True)
        self.fc_project = nn.Linear(lstm_units * self.final_seq_len, lstm_units)
        if type == 'lstm':
            self.context_lstm = nn.LSTM(lstm_units, lstm_units, num_layers=lstm_layers, batch_first=False, bidirectional=bidirectional)
        elif type == 'transformer':
            self.pos_enc = PositionalEncoding(lstm_units, max_len=batch_size)
            self.is_causal = not bidirectional
            encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_units, nhead=attention_num_heads, dim_feedforward=dim_feedforward, batch_first=False, dropout=0.0)
            self.context_lstm = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.dropout = nn.Dropout(dropout)
        if bidirectional and type == 'lstm':
            self.classifier = nn.Linear(2*lstm_units, classes)
        else:
            self.classifier = nn.Linear(lstm_units, classes)
        self.bidirectional = bidirectional
        self.activation = nn.ReLU()
        self.lstm_units = lstm_units
        self.classes = classes

    def forward(self, x, hidden=None, window_hidden=None):
        B = x.shape[0]
        x = x.unsqueeze(1) # batch, 1, sequence, axes
        x = self.activation(self.conv1(x)) # batch, kernels, sequence, axes
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 3, 1) # batch, sequence, axes, kernels
        T_prime = x.shape[1] # sequence length after conv layers
        x = x.reshape(x.shape[0], T_prime, -1) # batch, sequence, axes*kernels

        if self.window_lstm.batch_first:
            x, _ = self.window_lstm(x)
        else:
            # if batch_first is False, we need to reshape for LSTM and then reshape back
            x = x.reshape(B * T_prime, -1).unsqueeze(1) # (B*T_prime, 1, axes*kernels)
            x, window_hidden = self.window_lstm(x, window_hidden)
            x = x.reshape(B, T_prime, self.lstm_units) # batch, sequence, lstm_units -> batch, lstm_units
        #x = x[:, -1, :] # batch, lstm_units
        x = x.reshape(x.shape[0], -1) # batch, sequence*lstm_units
        x = self.fc_project(x) # batch, lstm_units
        x = x.unsqueeze(1) # batch, 1, lstm_units
        if isinstance(self.context_lstm, torch.nn.TransformerEncoder):
            x = self.pos_enc(x)
            if self.is_causal:
                seq_len = x.size(0)
                attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            else:
                attn_mask = None
            x = self.context_lstm(x, is_causal=self.is_causal, mask=attn_mask)
        else:
            x, hidden = self.context_lstm(x, hidden)

        if self.bidirectional and isinstance(self.context_lstm, torch.nn.LSTM):
            x = x.view(-1, 2 * self.lstm_units)
        else:
            x = x.view(-1, self.lstm_units)

        x = self.dropout(x)
        return self.classifier(x), hidden, window_hidden
    

    def set_batch_first(self, batch_first):
        self.context_lstm.batch_first = batch_first
        self.window_lstm.batch_first = batch_first