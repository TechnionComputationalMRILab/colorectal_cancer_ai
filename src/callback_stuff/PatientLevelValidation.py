import pandas as pd
from collections import defaultdict
import pytorch_lightning as pl
import re
import torch
import torchmetrics

class PatientLevelValidation(pl.Callback):
    def __init__(self) -> None:

        print("Patient Level Eval initialized")

        self.train_patient_eval_dict = defaultdict(list)
        self.val_patient_eval_dict = defaultdict(list)
        self.all_patient_targets = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        paths, x, y = batch
        batch_outputs = outputs["batch_outputs"]
        self.patient_eval(paths, batch_outputs, y, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        paths, x, y = batch
        batch_outputs = outputs["batch_outputs"]
        self.patient_eval(paths, batch_outputs, y, 'val')

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
                # if self.current_epoch == 0:
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

    def on_validation_epoch_end(self, trainer, pl_module):
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
                train_patient_score = train_patient_score.clone().detach()
                train_patient_target = self.all_patient_targets[patient].clone().detach()
                train_patient_scores.append(train_patient_score)
                train_patient_targets.append(train_patient_target)
            
            train_patient_scores = torch.stack(train_patient_scores)
            train_patient_targets = torch.stack(train_patient_targets)
            train_loss = pl_module.criteria(train_patient_scores, torch.nn.functional.one_hot(train_patient_targets, num_classes=2).float())
            train_acc = torchmetrics.functional.accuracy(torch.argmax(train_patient_scores, dim=1), train_patient_targets)
            self.log('train_patientlvl_loss', train_loss)
            self.log('train_patientlvl_acc', train_acc)

        if len(self.val_patient_eval_dict) > 0:
            val_patient_scores = []
            val_patient_targets = []
            for patient in self.val_patient_eval_dict.keys():
                val_patient_score = sum(self.val_patient_eval_dict[patient])/len(self.val_patient_eval_dict[patient])
                val_patient_score = val_patient_score.clone().detach()
                val_patient_target = self.all_patient_targets[patient].clone().detach()
                val_patient_scores.append(val_patient_score)
                val_patient_targets.append(val_patient_target)

            val_patient_scores = torch.stack(val_patient_scores)
            val_patient_targets = torch.stack(val_patient_targets)
            val_loss = pl_module.criteria(val_patient_scores, torch.nn.functional.one_hot(val_patient_targets, num_classes=2).float())
            val_acc = torchmetrics.functional.accuracy(torch.argmax(val_patient_scores, dim=1), val_patient_targets)
            self.log('val_patientlvl_loss', val_loss)
            self.log('val_patientlvl_acc', val_acc)

        self.train_patient_eval_dict = defaultdict(list)
        self.val_patient_eval_dict = defaultdict(list)
