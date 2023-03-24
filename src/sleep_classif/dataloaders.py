import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


from scipy import fftpack
from scipy.signal import butter, lfilter, sosfilt, sosfreqz


def return_fft_DSP(row):
    row_fft = fftpack.fft(row)
    dsp = np.vectorize(lambda x: np.abs(x)**2)
    return dsp(row_fft)
    

class MultiTaperSet(Dataset):
    def __init__(self,
                 device,
                 features_eeg_path = './data/pre_processed_data/Multitaper_eeg_train.npy',
                 features_position_path = './data/pre_processed_data/Multitaper_position_train.npy',
                 target_path = './data/raw_data/y_train.csv'
                 ):
      
        # read features (ie multitaper)
        self.features_eeg = torch.tensor(np.abs(np.load(features_eeg_path)))
        self.features_position = torch.tensor(np.abs(np.load(features_position_path)))
        self.device = device

        # read target
        if target_path:
          self.target = list(pd.read_csv(target_path, index_col = "index")['sleep_stage'])
          
        self.target_path = target_path

    def __len__(self):
        return self.features_eeg.shape[0]


    def __getitem__(self, idx):
        features_eeg = self.features_eeg[idx].to(self.device, dtype=torch.float)
        features_position = self.features_position[idx].to(self.device, dtype=torch.float)
           
        if self.target_path is not None:
          target = torch.tensor(int(self.target[idx])).to(self.device)
          return (features_eeg, features_position), target
        return (features_eeg, features_position)
    
    
    
class RawDataSet(Dataset):
    def __init__(self,
                 device,
                 data_path = './data/raw_data/X_train.h5',
                 target_path = './data/raw_data/y_train.csv'
                 ):
      
        # read features (ie multitaper)
        self.data_path = data_path
    
        with h5py.File(data_path, 'r') as f:
            eeg_1 = torch.tensor(f['eeg_1'])
            eeg_2 = torch.tensor(f['eeg_2'])
            eeg_4 = torch.tensor(f['eeg_4'])
            eeg_5 = torch.tensor(f['eeg_5'])
            eeg_6 = torch.tensor(f['eeg_6'])
            features_eeg = torch.stack((eeg_1,eeg_2,eeg_4,eeg_5,eeg_6), dim=1)

            position_x = torch.tensor(f['x'])
            position_y = torch.tensor(f['y'])
            position_z = torch.tensor(f['z'])
            features_position = torch.stack((position_x,position_y,position_z), dim=1)
                    
                
        self.features_eeg = features_eeg
        self.features_position = features_position
        self.device = device

        # read target
        if target_path:
            self.target = list(pd.read_csv(target_path, index_col = "index")['sleep_stage'])
          
        self.target_path = target_path

    def __len__(self):
        return self.features_eeg.shape[0]

    def feature_shape(self):
        return self.features_eeg.shape[1]


    def __getitem__(self, idx):
        features_eeg = self.features_eeg[idx].to(self.device, dtype=torch.float)
        features_position = self.features_position[idx].to(self.device, dtype=torch.float)
           
        if self.target_path is not None:
            target = torch.tensor(int(self.target[idx])).to(self.device)
            return (features_eeg, features_position), target
        
        return (features_eeg, features_position)

    

class FFT_Raw_DataSet(Dataset):
    def __init__(self,
                 device,
                 data_path = './data/raw_data/X_train.h5',
                 target_path = './data/raw_data/y_train.csv'
                 ):
      
        self.data_path = data_path
    
        with h5py.File(data_path, 'r') as f:
            
                print("1")
                eeg_1 = torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['eeg_1'])))
                print("2")
                eeg_2 = torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['eeg_2'])))
                print("4")
                eeg_4 = torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['eeg_4'])))
                print("5")
                eeg_5 = torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['eeg_5'])))
                print("6")
                eeg_6 = torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['eeg_6'])))

                position_x =  torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['x'])))
                position_y =  torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['y'])))
                position_z =  torch.tensor(np.apply_along_axis(return_fft_DSP, 1, np.array(f['z'])))
                print("over")

                features_eeg_fft = torch.stack((eeg_1,eeg_2,eeg_4,eeg_5,eeg_6), dim=1)
                features_position_fft = torch.stack((position_x,position_y,position_z), dim=1)
        
        
        with h5py.File(data_path, 'r') as f:
            
                eeg_1 = torch.tensor(f['eeg_1'])
                eeg_2 = torch.tensor(f['eeg_2'])
                eeg_4 = torch.tensor(f['eeg_4'])
                eeg_5 = torch.tensor(f['eeg_5'])
                eeg_6 = torch.tensor(f['eeg_6'])
                features_eeg = torch.stack((eeg_1,eeg_2,eeg_4,eeg_5,eeg_6), dim=1)

                position_x = torch.tensor(f['x'])
                position_y = torch.tensor(f['y'])
                position_z = torch.tensor(f['z'])
                features_position = torch.stack((position_x,position_y,position_z), dim=1)

                
        self.features_eeg = features_eeg
        self.features_position = features_position

        self.features_eeg_fft = features_eeg_fft
        self.features_position_fft = features_position_fft
        
        self.device = device

        # read target
        if target_path:
            self.target = list(pd.read_csv(target_path, index_col = "index")['sleep_stage'])
          
        self.target_path = target_path

    def __len__(self):
        return self.features_eeg.shape[0]

    def feature_shape(self):
        raw_feat = self.features_eeg.shape[1]
        fft_feat = self.features_position.shape[1]

        raw_pos_feat = self.features_eeg_fft.shape[1]
        fft_pos_feat = self.features_position_fft.shape[1]

        return (raw_feat, fft_feat, raw_pos_feat, fft_pos_feat)


    def __getitem__(self, idx):
        
        features_eeg = self.features_eeg[idx].to(self.device, dtype=torch.float)
        features_position = self.features_position[idx].to(self.device, dtype=torch.float)

        features_eeg_fft = self.features_eeg_fft[idx].to(self.device, dtype=torch.float)
        features_position_fft = self.features_position_fft[idx].to(self.device, dtype=torch.float)
           
        if self.target_path is not None:
            target = torch.tensor(int(self.target[idx])).to(self.device)
            return (features_eeg, features_eeg_fft, features_position, features_position_fft), target
        
        return (features_eeg, features_eeg_fft, features_position, features_position_fft)
