# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------

from torch import nn


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM model as described in "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition" (https://doi.org/10.1145/3460421.3480419).
    Args:
    
    Args:
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
        causal: bool
            Whether to use causal padding.
    """
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes

    def forward(self, x, hidden=None):
        B = x.shape[0]
        x = x.unsqueeze(1) # batch, 1, sequence, axes
        x = self.activation(self.conv1(x))# batch, kernels, sequence, axes
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 3, 1) # batch, sequence, axes, kernels
        T_prime = x.shape[1] # sequence length after conv layers
        x = x.reshape(B, T_prime, -1) # batch, sequence, axes*kernels

        if self.lstm.batch_first:
            x, hidden = self.lstm(x, hidden)
        else:
            # if batch_first is False, we need to reshape for LSTM and then reshape back
            x = x.reshape(B * T_prime, -1).unsqueeze(1) # (B*T_prime, 1, axes*kernels)
            x, hidden = self.lstm(x, hidden)
            x = x.reshape(B, T_prime, self.lstm_units) # batch, sequence, lstm_units -> batch, lstm_units
        x = x[:, -1, :]

        x = self.dropout(x)
        return self.classifier(x), hidden
    
    def set_batch_first(self, batch_first):
        self.lstm.batch_first = batch_first
        