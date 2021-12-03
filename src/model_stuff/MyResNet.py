import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
#from pl_bolts.callbacks import ORTCallback
import torchmetrics

from ..data_stuff import dataset_tools
import re
from collections import defaultdict

class MyResNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
#             torch.nn.Sigmoid(),
        )
        self.criteria = torch.nn.BCEWithLogitsLoss()
        
        self.train_patient_eval_dict = defaultdict(list)
        self.val_patient_eval_dict = defaultdict(list)
        self.all_patient_targets = {}
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, num_classes=2).float())
        acc = torchmetrics.functional.accuracy(out, y)
        
        self.patient_eval(path, out, y, 'train')
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        
        val_loss = self.criteria(out, torch.nn.functional.one_hot(y, num_classes=2).float())
        val_acc = torchmetrics.functional.accuracy(out, y)
        
        self.patient_eval(path, out, y, 'val')
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc}
    
    def on_validation_epoch_end(self):
        """ 
        Calculate Error on patient level and Clear the patient level eval dict(s),
        So that it can fill up for next epoch
        """        
        # eval and record results
        # need to loop over dicts to make this to ensure order is correct
        # dicts dont necessarily enforce order
        if len(self.train_patient_eval_dict) > 0:
            train_patient_scores = []
            train_patient_targets = []
            for patient in self.train_patient_eval_dict.keys():
                train_patient_score = sum(self.train_patient_eval_dict[patient])/len(self.train_patient_eval_dict[patient])
                train_patient_score = torch.tensor(train_patient_score)
                train_patient_target = torch.tensor(self.all_patient_targets[patient])
                train_patient_scores.append(train_patient_score)
                train_patient_targets.append(train_patient_target)
            
            train_patient_scores = torch.stack(train_patient_scores)
            train_patient_targets = torch.stack(train_patient_targets)
            train_loss = self.criteria(train_patient_scores, torch.nn.functional.one_hot(train_patient_targets, num_classes=2).float())
            train_acc = torchmetrics.functional.accuracy(train_patient_scores, train_patient_targets)
            self.log('train_patientlvl_loss', train_loss)
            self.log('train_patientlvl_acc', train_acc)

        if len(self.val_patient_eval_dict) > 0:
            val_patient_scores = []
            val_patient_targets = []
            for patient in self.val_patient_eval_dict.keys():
                val_patient_score = sum(self.val_patient_eval_dict[patient])/len(self.val_patient_eval_dict[patient])
                val_patient_score = torch.tensor(val_patient_score)
                val_patient_target = torch.tensor(self.all_patient_targets[patient])
                val_patient_scores.append(val_patient_score)
                val_patient_targets.append(val_patient_target)

            val_patient_scores = torch.stack(val_patient_scores)
            val_patient_targets = torch.stack(val_patient_targets)
            val_loss = self.criteria(val_patient_scores, torch.nn.functional.one_hot(val_patient_targets, num_classes=2).float())
            val_acc = torchmetrics.functional.accuracy(val_patient_scores, val_patient_targets)
            self.log('val_patientlvl_loss', val_loss)
            self.log('val_patientlvl_acc', val_acc)

        self.train_patient_eval_dict = defaultdict(list)
        self.val_patient_eval_dict = defaultdict(list)
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def patient_eval(self, batch_paths, batch_scores, batch_targets, train_or_val: str):
        """
        Fill the patient eval dicts with patients & scores of current batch.
        Please specify train_or_val with "train" or "val" batch.
        """        
        path_pattern = r'TCGA-\w{2}-\w{4}|/MSIMUT/|/MSS/'
        
        # matches look like:
        # [['/MSS/', 'TCGA-AZ-5403'],... ['/MSIMUT/', 'TCGA-CM-6674']]
        status_patient_regex = [re.findall(path_pattern, path) for path in batch_paths]
        
        with torch.no_grad():
            for i in zip(status_patient_regex, batch_scores, batch_targets):
                status_patient, score, target = i
                score = score.softmax(dim=-1)
                status = status_patient[0]
                patient = status_patient[1]
                
                # check if all_patient_targets is consistent
                if self.current_epoch == 0:
                    if patient in self.all_patient_targets:
                        assert self.all_patient_targets[patient] == target, f"targets not consistent for patient: {patient}"
                    else:
                        self.all_patient_targets[patient] = target
                if train_or_val == 'train':
                    self.train_patient_eval_dict[patient].append(score)
                elif train_or_val == 'val':
                    self.val_patient_eval_dict[patient].append(score)
                else:
                    raise Exception('train_or_val can be either train or val yo!')
