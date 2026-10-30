import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from team.models.abstract_model import AbstractModel
from team.utils_circular import LoggingWrapper, ReducedCriterion
import os
import wandb
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out = self.bn4(self.conv4(out))
        if self.downsample:
            identity = self.downsample(x)
        final = self.relu(out + identity)
        return final

class ResNet(pl.LightningModule, AbstractModel):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv1d(12, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = self._create_layer(64, 64, 2, 3)
        self.layer2 = self._create_layer(64, 128, 2, 3, 2)
        self.layer3 = self._create_layer(128, 256, 2, 3, 2)
        self.layer4 = self._create_layer(256, 512, 2, 3, 2)
        self.fc = None  
        self.final_function = torch.nn.Sigmoid()
        self.criterion = ReducedCriterion(nn.BCELoss(reduction='none'))
        self.config = config
        self.downstream = None
        # logging
        self.min_val_loss = float("inf")  

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))  # TODO: this is not efficient i.e. do this one when creating dataset
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=-1)
        x = self.fc(x)
        p = self.final_function(x)

        return p

    def _create_layer(self, in_c, out_c, blocks, k, stride=1):
        downsample = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride), nn.BatchNorm1d(out_c)) if stride != 1 else None
        layers = [ResBlock(in_c, out_c, k, stride, downsample)] + [ResBlock(out_c, out_c, k) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def on_fit_start(self):
        # unfortunately this can't be done in constructor since 
        # at that point in time the lightning logger hasn't been created yet
        self.log_wrapper = LoggingWrapper(self.logger, ["training", "validation"], self.config.evaluation.binarization_threshold)

    def training_step(self, batch, batch_idx):
        x, y, src, _ = batch
        p = self(x)
        loss = self.criterion.compute(p, y, self.downstream)
        self.log_wrapper.store_logging_data(0, src, p, y, loss, self.downstream)
        reduced_loss = loss.mean()
        return reduced_loss

    def validation_step(self, batch, batch_idx):
        x, y, src, _ = batch
        p = self(x)
        loss = self.criterion.compute(p, y, self.downstream)
        self.log_wrapper.store_logging_data(1, src, p, y, loss, self.downstream)
        reduced_loss = loss.mean()
        return reduced_loss

    def on_train_epoch_end(self):
        avg_loss = self.log_wrapper.log(0, self.downstream)
        self.log_wrapper.reset_logging_data(0)

    def on_validation_epoch_end(self): 
        avg_loss = self.log_wrapper.log(1, self.downstream)
        self.log_wrapper.reset_logging_data(1)
        # remember minimum validation loss (only in case we are fine-tuning)
        if self.downstream: self.min_val_loss = min(self.min_val_loss, avg_loss)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.model.lr)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.fc = self._get_downstream_fc_layer()  # self.fc is None after construction
        model_path = os.path.join(path, 'tuned.pth')
        self.load_state_dict(torch.load(model_path, map_location=device))

    # model created by constructor has one ouptut neuron (which is not what we need for fine tuning)
    def prepare_pretraining(self):
        self.downstream = False
        # replace last layer with classification layer that contains appropriate amount of output neurons
        self.fc = nn.Linear(512, 6)

    # replace last layer by fresh classification layer with single class output and disable grads for all pre-trained layers
    def prepare_tuning(self):
        self.downstream = True
        # disable grads in entire model
        for param in self.parameters():
            param.requires_grad = False

        # replace last layer with fresh classification layer
        self.fc = self._get_downstream_fc_layer()

        # enable grads for the fresh final layer
        for param in self.fc.parameters():
            param.requires_grad = True
        
        # reset logger step
        self.log_wrapper.reset_step()

    def _get_downstream_fc_layer(self):
        return nn.Linear(512, 1)
