from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl

import wandb


class LSTMEstimator(pl.LightningModule):

    def __init__(self, feature_size=15, initial_dense_layer_size=16, dense_parameter_multiplier=1, dense_layer_count=2,
                 lstm_layer_count=1, lstm_hidden_units=16, sequence_length=3, dropout=0.5, lr=0.001, loss='huber'):
        super(LSTMEstimator, self).__init__()
        self.save_hyperparameters()
        self.lstm_depth = lstm_layer_count
        self.dense_multiplier = dense_parameter_multiplier
        self.feature_size = feature_size
        self.dense_size = initial_dense_layer_size
        self.dense_count = dense_layer_count
        self.lstm_hidden_count = lstm_hidden_units
        self.window = sequence_length
        self.loss_functions = {'huber': F.huber_loss, 'mse': F.mse_loss}
        self.loss = self.loss_functions[loss]
        self.lr = lr
        self.dropout = dropout



        layer_list = []
        for layer_count in range(self.dense_count):
            if layer_count == 0:
                input_size = self.feature_size
                self.dense_output_size = self.dense_size

            layers_to_add = [('linear' + str(layer_count), nn.Linear(input_size, self.dense_output_size)),
                             ('relu' + str(layer_count), nn.LeakyReLU()),
                             ('dropout' + str(layer_count), nn.Dropout(self.dropout))]
            layer_list.extend(layers_to_add)

            input_size = self.dense_output_size
            self.dense_output_size = int(input_size * self.dense_multiplier)

        layer_list.extend([('linear' + str(layer_count+1), nn.Linear(self.dense_output_size,
                                                                    self.dense_output_size*self.dense_multiplier))])

        self.feature_extracting_layers = nn.Sequential(
            OrderedDict(layer_list)
        )

        self.lstm_layer = nn.LSTM(
            int(self.dense_output_size / self.dense_multiplier), self.lstm_hidden_count, dropout=self.dropout,
            batch_first=True, num_layers=self.lstm_depth
        )

        estimator_layers = []
        for layer_count in range(self.dense_count):
            if layer_count == self.dense_count - 1:
                layer = [("output_layer", nn.Linear(self.dense_size, 1, bias=False))]
            elif layer_count == 0:
                layer = [("linear_" + str(layer_count), nn.Linear(self.lstm_hidden_count * self.window, self.dense_size)),
                         ("relu_" + str(layer_count), nn.LeakyReLU()),
                         ("dropout_" + str(layer_count), nn.Dropout(self.dropout))]
            else:
                layer = [("linear_" + str(layer_count), nn.Linear(self.dense_size, self.dense_size)),
                         ("relu_" + str(layer_count), nn.LeakyReLU()),
                         ("dropout_" + str(layer_count), nn.Dropout(self.dropout))]

            estimator_layers.extend(layer)

        self.estimator_layers = nn.Sequential(
            OrderedDict(estimator_layers)
        )

    def forward(self, x):
        #Apply feature extraction to every vector in the sequence.
        #Input is in the form x = (batch, sequence, features)

        flattened_features = x.reshape(-1, x.shape[-1])
        flattened_extracted_features = self.feature_extracting_layers(flattened_features)
        sequenced_features = flattened_extracted_features.reshape(x.shape[0], self.window, -1)

        lstm_output, _ = self.lstm_layer(sequenced_features)

        #Squeeze sequence and feature dimensions together for estimating layers.
        flattened_lstm_output = lstm_output.reshape(x.shape[0], -1)
        estimator_output = self.estimator_layers(flattened_lstm_output)

        return estimator_output


    def training_step(self, batch, batch_idx):
        feature_sequence, target = batch
        estimation = self(feature_sequence)
        loss = self.loss(estimation, target.reshape(-1, 1))
        self.log('training_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature_sequence, target = batch
        estimation = self(feature_sequence)
        loss = self.loss(estimation, target.reshape(-1, 1))
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            plot_data = []
            for idx in range(target.shape[0]):
                plot_data.append([idx, estimation.reshape(-1)[idx], 'prediction'])
                plot_data.append([idx, target.reshape(-1)[idx], 'actual'])

            log_table = wandb.Table(data=plot_data, columns=['sample', 'recovery_rate', 'label'])
            wandb.log({"snapshot": wandb.plot.scatter(log_table,'sample', 'recovery_rate', 'label')})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler1 = StepLR(optimizer, step_size=1, gamma=0.99)
        scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, cooldown=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler2, "monitor": "val_loss"}