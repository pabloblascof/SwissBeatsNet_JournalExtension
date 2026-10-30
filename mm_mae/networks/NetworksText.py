import numpy as np
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from transformers import BertForMaskedLM, AutoConfig, AutoModel
from typing import List, Optional
import torch.nn as nn
import torch


class BiomedBert_Encoder(pl.LightningModule):
    """
    Bert style encoder - it accepts BertForMaskedLM pretrained models.
    """

    def __init__(self, pretrained_model_name: str, random_init: bool, hidden_state_index:int, freeze_all: bool, emb_dim: int, compute_rec: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.pretrained_model_name = pretrained_model_name
        self.random_init = random_init
        
        if self.random_init:
            config = AutoConfig.from_pretrained(self.pretrained_model_name)
            self.model = BertForMaskedLM(config)
        else:
            self.model = BertForMaskedLM.from_pretrained(self.pretrained_model_name)
        
        print(self.model)

        self.compute_rec = compute_rec
        self.hidden_state_index = hidden_state_index
        self.freeze_all = freeze_all

        if self.freeze_all:
            for param in self.model.bert.parameters():
                param.requires_grad = False
            
            for param in self.model.cls.parameters():
                param.requires_grad = False
    
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Trainable:", name)

    def forward(self, inputs):
        inputs_data = inputs["data"]
        model_inputs = {
            "input_ids": inputs_data["input_ids"],
            "attention_mask": inputs_data["attention_mask"],
            "output_hidden_states": True,
        }
        if self.compute_rec:
            model_inputs["labels"] = inputs_data["labels"]
        else:
            model_inputs["labels"] = None
        outputs = self.model(**model_inputs)

        loss = outputs["loss"] if self.compute_rec else None
    
        cls_embedding = outputs["hidden_states"][self.hidden_state_index][:, 0, :]
        return {"loss": loss, "cls_embedding": cls_embedding}

    def set_mask_ratio(self, ratio):
        # DUMMY MASKED RATIO SETUP
        pass


class BiomedBert_Decoder(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        print("TEXT DECODER INIT")

    def forward(self, inputs):
        return None
