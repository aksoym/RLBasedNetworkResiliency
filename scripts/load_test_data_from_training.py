import torch
from torch.utils.data import BatchSampler, SequentialSampler
import pandas as pd
import numpy as np

from dataset import RecoveryRateDataset

features_to_drop = ["reg_cause", "reg_bool_type", "weather_prediction", "reg_type"]

rr_dataset = RecoveryRateDataset("../data/airport_state_weather_prediction_added.pickle", features_to_drop=features_to_drop,
                                 sequence_length=3, fill_with="backfill")

train_val_test_ratios = (0.6, 0.2, 0.2)

lengths = [int(len(rr_dataset)*ratio) for ratio in train_val_test_ratios]

#If there is a mismatch between split sizes and the total size, 
#calculate the offset and trim the training data accoringly.
if sum(lengths) != len(rr_dataset):
    offset = sum(lengths) - len(rr_dataset)
    lengths[0] = lengths[0] - offset
                                                                      
train, val, test = torch.utils.data.random_split(rr_dataset, lengths=lengths)

#TODO Save the test data subset in training then use it for test cases.
