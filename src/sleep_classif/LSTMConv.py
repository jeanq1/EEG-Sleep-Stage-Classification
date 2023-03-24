### Implementation inspired from https://www.kaggle.com/prith189/starter-code-for-3rd-place-solution

import torch
import torch.nn as nn
import torch.nn.functional as func

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
       # assert drop is None or (0.0 < drop < 1.0)
        layers = [nn.Conv1d(ni, no, kernel, stride, padding=pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
    
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)
    
class LSTM_Conv(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()
        
        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 128, 8, 1, 1),
            nn.MaxPool1d(8, stride=8),
            nn.Dropout(drop),
            SepConv1d(128,  128, 8, 1, 1),
            SepConv1d(128, 128, 8, 1, 1),
            SepConv1d(128, 128, 8, 1, 1),
            nn.MaxPool1d(4, stride=4),
            nn.Dropout(drop)
        )
              
        self.model_lstm = nn.LSTM(42, 128, bidirectional=False)
        
        self.out = nn.Sequential(
            Flatten([2,3]), 
            nn.Dropout(0.5),
            nn.Linear(16384, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        


    def forward(self, input_data):
        
        t_raw, t_raw_position = input_data
        raw_out = self.raw(t_raw)
        output, states = self.model_lstm(raw_out)
        out = self.out(output)
        return out
    
    
