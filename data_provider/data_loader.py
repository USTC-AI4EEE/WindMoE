import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class Dataset_Windpower(Dataset):

    def __init__(self, data_path: str, dataset_name: str, weather: str, station: int, use: str, 
                 input_length: int, output_length: int, index_col: str = 'DateTime'):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.weather = weather
        self.use = use
        self.samples = []

        valid_uses = [
            'train', 'valid', 'test',
            'train_extreame', 'valid_extreame', 'test_extreame'
        ]

        if use not in valid_uses:
            raise ValueError(f"Invalid 'use' flag: '{use}'. Valid options are: {valid_uses}")

        if 'extreame' in use:
            self.is_extreame = True

        if self.dataset_name == 'fujian':
            station_str = f"{station:03d}"
        elif self.dataset_name == 'jilin':
            station_str = f"{station:03d}"
        elif self.dataset_name == 'goldwind':
            station_str = str(station)
        else:
            print(f"Warning: Unhandled dataset '{self.dataset_name}' for scaler loading. "
                  f"Using station ID without padding.")
            station_str = str(station)

        filename = f'{self.dataset_name}_{weather}_{station_str}_{use}.csv'
        data_file = os.path.join(data_path, filename)
        
        try:
            df = pd.read_csv(data_file, index_col=index_col)
            self.data = df.values.astype(np.float32)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file}")
            print("Please ensure you have run the data preprocessing script to generate CSV files.")
            self.data = np.array([]) 

    def __len__(self):
        if self.is_extreame:
            return len(self.samples)
        
        if self.data.size == 0:
            return 0
        
        return len(self.data) - self.input_length - self.output_length + 1
    
    
    def __getitem__(self, index):
        if self.is_extreame:
            sample_slice = self.samples[index]
        else:
            s_begin = index
            s_end = s_begin + self.input_length + self.output_length
            sample_slice = self.data[s_begin:s_end]

        history_slice = sample_slice[:self.input_length]
        future_slice = sample_slice[self.input_length:]
        history_power = history_slice[:, -1:]
        history_nwp = history_slice[:, :-1]
        future_power = future_slice[:, -1:]
        future_nwp = future_slice[:, :-1]
            
        return (
            torch.from_numpy(history_power).float(), 
            torch.from_numpy(history_nwp).float(),
            torch.from_numpy(future_power).float(),
            torch.from_numpy(future_nwp).float()
        )