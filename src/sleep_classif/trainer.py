
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, data_loader, device,
                 valid_data_loader=None, class_weights=None, requires_softmax = False):

        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.optimizer = optimizer
        self.class_weights = torch.tensor(class_weights, dtype=torch.double)
        self.device = device
        self.requires_softmax = requires_softmax



    def train_epoch(self):
        """
        Training logic for an epoch
        """
        batch_id = 0
        losses = []

        for (features, target) in tqdm(self.data_loader):    
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, target)
            losses.append(float(loss))
            loss.backward()
            self.optimizer.step()
            batch_id+=1

            
        self.model.eval()
        if self.do_validation:
            with torch.no_grad():
                val_correct_preds = 0
                val_losses = []
                total = 0
                for (features, target) in (self.valid_data_loader):
                    predictions = self.model(features)
                    #import pdb; pdb.set_trace()
                    if self.requires_softmax:
                        print("hello requires softmax")
                        predictions = nn.functional.log_softmax(predictions, dim=1)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    val_correct_preds += int(torch.sum(predicted_classes==target))
                    total += predictions.shape[0]
                    val_losses.append(float(self.criterion(predictions, target)))
                
                accuracy = val_correct_preds/total
                loss = np.mean(val_losses)
                print('average train loss: ', np.mean(losses))
                print('validation loss: ', loss)
                print('validation accuracy: ', accuracy)

        return(loss, accuracy)