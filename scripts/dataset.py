import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class RecoveryRateDataset(Dataset):

    def __init__(self, file_path, features_to_drop, sequence_length=5, transform=None, fill_with=None):

        self.dataframe = pd.read_pickle(file_path)
        self.transform = transform
        self.window = sequence_length
        self.feature_names = [feature for feature in self.dataframe.columns.tolist()
                              if feature not in features_to_drop]
        self.apt_count = len(self.dataframe.reset_index()['apt'].unique())
        self.fill_with = fill_with

        try:
            if isinstance(float(self.fill_with), float):
                self.dataframe = self._fill_recovery_rate_nans(self.dataframe, self.fill_with)
        except:
            pass


    def _fill_recovery_rate_nans(self, dataframe, fill_with):
        available_methods = ["backfill", "bfill", "pad", "ffill"]
        if fill_with in available_methods:
            dataframe = dataframe.fillna(method=fill_with)
        elif isinstance(float(fill_with), float):
            dataframe = dataframe.fillna(value=dict(recovery_rate=fill_with))
        else:
            raise ValueError('fill_with must be one of the strings in {"backfill", "bfill", "pad", "ffill"} or an integer.')

        return dataframe

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.window))

    def __getitem__(self, item):

        last_index = item + self.apt_count*(self.window + 1)

        slice_with_target_row = self.dataframe.iloc[item : last_index : self.apt_count, :].loc[:, self.feature_names]

        #NaN value handling. Apply filling methods until there is no NaN value left in the sample.
        fill_methods = [self.fill_with, 'ffill', 0]
        for method in fill_methods:
            slice_with_target_row = self._fill_recovery_rate_nans(slice_with_target_row, fill_with=method)
            if not slice_with_target_row.isna().any().any():
                break

        target = slice_with_target_row.iloc[-1, :]['recovery_rate']
        sample = slice_with_target_row.iloc[0:-1, :]


        sample = torch.tensor(sample.to_numpy(dtype=np.float32))
        target = torch.tensor(target, dtype=torch.float32)

        return sample, target

