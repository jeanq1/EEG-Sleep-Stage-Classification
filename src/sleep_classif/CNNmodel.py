import torch
import torch.nn as nn
import torch.nn.functional as func


import torch
import torch.nn as nn
import torch.nn.functional as func


class SimpleCNN(torch.nn.Module): 
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.eeg_in = nn.Sequential(
          nn.Conv1d(5, 16,15),
          nn.MaxPool1d(2,stride = 2),
          nn.BatchNorm1d(16),
          nn.ReLU())

        self.acc_in = nn.Sequential(
          nn.Conv1d(3, 16,15),
          nn.MaxPool1d(2,stride = 2),
          nn.BatchNorm1d(16),
          nn.ReLU()) # 143 * 16

        self.conv_eeg = nn.Sequential(
          nn.Conv1d(16, 4,1),
          nn.BatchNorm1d(4),
          nn.ReLU(),
          nn.Conv1d(4, 4,3),
          nn.BatchNorm1d(4),
          nn.ReLU(),
          nn.Conv1d(4, 16,1),
          nn.BatchNorm1d(16),
          nn.ReLU(),
          nn.Flatten()
        ) # 741*16


        self.conv_acc = nn.Sequential(
          nn.Conv1d(16, 4,1),
          nn.BatchNorm1d(4),
          nn.ReLU(),
          nn.Conv1d(4, 4,3),
          nn.BatchNorm1d(4),
          nn.ReLU(),
          nn.Conv1d(4, 16,1),
          nn.BatchNorm1d(16),
          nn.ReLU(),
          nn.Flatten()
        ) # 141*16


        self.dense = nn.Sequential(
          nn.Linear(882*16, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(),
          nn.Linear(1000,5)
        ) 


    def forward(self, x):

        x_eeg = self.eeg_in(x[0])
        x_eeg = self.conv_eeg(x_eeg)

        x_acc = self.acc_in(x[1])
        x_acc = self.conv_acc(x_acc)

        x_tot = torch.cat((x_eeg, x_acc), 1)

        x_tot = self.dense(x_tot)
        out = func.softmax(x_tot, dim=1)
        return out
