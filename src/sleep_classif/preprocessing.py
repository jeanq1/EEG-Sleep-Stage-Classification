import numpy as np
import h5py
import mne
import os
from tqdm import tqdm




def load_data():
    '''
    loads data from the raw_data folder and returns position and egg dataframes from test and train set
    '''

    with h5py.File('./data/raw_data/X_train.h5', 'r') as f:
            eeg_1 = np.expand_dims(np.array(f['eeg_1']),axis=1)
            eeg_2 = np.expand_dims(np.array(f['eeg_2']),axis=1)
            eeg_4 = np.expand_dims(np.array(f['eeg_4']),axis=1)
            eeg_5 = np.expand_dims(np.array(f['eeg_5']),axis=1)
            eeg_6 = np.expand_dims(np.array(f['eeg_6']),axis=1)
            features_eeg = np.concatenate((eeg_1,eeg_2,eeg_4,eeg_5,eeg_6), axis=1)

            position_x = np.expand_dims(np.array(f['x']),axis=1)
            position_y = np.expand_dims(np.array(f['y']),axis=1)
            position_z = np.expand_dims(np.array(f['z']),axis=1)
            features_position = np.concatenate((position_x,position_y,position_z), axis=1)

    with h5py.File('./data/raw_data/X_test.h5', 'r') as f:
            eeg_1 = np.expand_dims(np.array(f['eeg_1']),axis=1)
            eeg_2 = np.expand_dims(np.array(f['eeg_2']),axis=1)
            eeg_4 = np.expand_dims(np.array(f['eeg_4']),axis=1)
            eeg_5 = np.expand_dims(np.array(f['eeg_5']),axis=1)
            eeg_6 = np.expand_dims(np.array(f['eeg_6']),axis=1)
            features_eeg_test = np.concatenate((eeg_1,eeg_2,eeg_4,eeg_5,eeg_6), axis=1)

            position_x = np.expand_dims(np.array(f['x']),axis=1)
            position_y = np.expand_dims(np.array(f['y']),axis=1)
            position_z = np.expand_dims(np.array(f['z']),axis=1)
            features_position_test = np.concatenate((position_x,position_y,position_z), axis=1)

    return(features_eeg,features_position,features_eeg_test,features_position_test)



def create_and_save_multitapers(features_df, saving_path, position = False):
  """
  position is a boolean which is true if the features given to the function are accelerometers features
  since the accelerometers data and the egg data are not sampled at the same frenquencies, we need to distinguish them
  during the pre-processing
  The multitaper files are quiet heavy, so we apply a sub-sampling factor (decim) to them before the saving, we use a decim factor of 2 for the accelerometers data and a decim
  factor of 10 for the eeg data 
  """

  size = features_df.shape[0]
  features = np.array(features_df)
  total_multi_taper = None
  if position == True:
    sampling_freq = 10
    decim = 2
  else:
    sampling_freq = 50
    decim = 10

  mutlitaper_freq_range = int(sampling_freq/2)

  for i in tqdm(range(int(size/100)), position = 0):
    features_to_process = features[i*100:(i+1)*100]

    multitaper = mne.time_frequency.tfr_array_multitaper(features_to_process, sampling_freq, np.array([i for i in range(1, mutlitaper_freq_range, 1)]), decim =  decim)

    if total_multi_taper is None:
      total_multi_taper = multitaper
    else:
      total_multi_taper = np.concatenate((total_multi_taper, multitaper), axis = 0)


  print(f"saving mutlitaper to: {saving_path}")
  np.save(saving_path, total_multi_taper)
  return()



def compute_tapers():
  print("load data")
  features_eeg_train,features_position_train,features_eeg_test,features_position_test = load_data()

  if len(os.listdir('./data/pre_processed_data')) == 0:
    print('computing tapers')
    create_and_save_multitapers(features_position_test, './data/pre_processed_data/Multitaper_position_test', position = True)
    create_and_save_multitapers(features_position_train, './data/pre_processed_data/Multitaper_position_train', position = True)
    create_and_save_multitapers(features_eeg_test, './data/pre_processed_data/Multitaper_eeg_test', position = False)
    create_and_save_multitapers(features_eeg_train, './data/pre_processed_data/Multitaper_eeg_train', position = False)
  else:
    print("tapers already computed")
  return()