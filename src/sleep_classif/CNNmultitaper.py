import torch
import torch.nn as nn
import torch.nn.functional as func


class ConvNetMultitaper(torch.nn.Module): 
    def __init__(self):
        super(ConvNetMultitaper, self).__init__()

        self.model_cnn_egg = nn.Sequential(
          nn.Conv2d(5, 40, (3,3)),
          nn.BatchNorm2d(40),
          nn.ReLU(),
          nn.Conv2d(40, 60, (3,3)),
          nn.BatchNorm2d(60),
          nn.ReLU(),
          torch.nn.MaxPool2d((2,5)),
          nn.Flatten(),
          nn.Linear(17400, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(),
          nn.Linear(1000,200),
          nn.BatchNorm1d(200),
          nn.ReLU()
        ) 


        self.model_cnn_position = nn.Sequential(
          nn.Conv2d(3, 40, (1,3)),
          nn.BatchNorm2d(40),
          nn.ReLU(),
          nn.Conv2d(40, 60, (1,3)),
          nn.BatchNorm2d(60),
          nn.ReLU(),
          torch.nn.MaxPool2d((1,5)),
          nn.Flatten(),
          nn.Linear(6960, 1000),
          nn.BatchNorm1d(1000),
          nn.ReLU(),
          nn.Linear(1000,200),
          nn.BatchNorm1d(200),
          nn.ReLU()
        ) 
            
        self.dense = nn.Sequential(
          nn.Linear(400, 100),
          nn.BatchNorm1d(100),
          nn.ReLU(),
          nn.Linear(100, 5)
        )


    def forward(self, x):

        x_eeg = self.model_cnn_egg(x[0])
        x_position = self.model_cnn_position(x[1])

        x_tot = torch.cat((x_eeg, x_position), 1)

        x_tot = self.dense(x_tot)
        out = func.softmax(x_tot, dim=1)
        return out




