from collections import defaultdict
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
import pandas as pd
import wandb
from torchvision.utils import save_image

class PatientLevelValidation(pl.Callback):
    def __init__(self, group_size: int, debug_mode = False) -> None:

        print(f"Patient Level Eval initialized with group size {group_size}")
        # self.train_eval_dict = defaultdict(list)
        # self.val_eval_dict = defaultdict(list)
        self.all_patient_targets = {}
        self.group_size = group_size
        self.debug_mode = debug_mode
        self.MSIMUT_label = None

    def setup(self, trainer, pl_module, stage=None):
        # we need the following dicts to check for label corectness
        self.train_samples_dict = trainer.datamodule.train_ds.get_samples_dict()
        self.val_samples_dict = trainer.datamodule.val_ds.get_samples_dict()

        try:
            self.MSIMUT_label = trainer.datamodule.train_ds.class_to_idx["MSIMUT"]
        except:
            pass

        # we need these dicts to fill with patch scores
        self.train_img_samples_score_dict = trainer.datamodule.train_ds.get_img_samples_score_dict()
        self.val_img_samples_score_dict = trainer.datamodule.val_ds.get_img_samples_score_dict()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        img_id, img_paths, y, x = batch
        batch_outputs = outputs["batch_outputs"]

        # separate patch groups
        if self.group_size > 1:
            img_paths_lol = [p.split(",") for p in img_paths]
            img_paths = [item for sublist in img_paths_lol for item in sublist]
            y = y.repeat_interleave(self.group_size)
            batch_outputs = batch_outputs.repeat_interleave(self.group_size, axis=0)
            img_id = tuple(np.repeat(np.array(img_id), self.group_size))
            if self.debug_mode:
                for p in img_paths:
                    if p == "/tcmldrive/tcga_data_formatted/train/MSS/TCGA-AF-6655/blk-YYDEFDMRGTTP-TCGA-AF-6655-01Z-00-DX1.png":
                        # if p == "/tcmldrive/tcga_data_formatted/train/MSS/TCGA-SS-A7HO/blk-YFWIFFLIFPTE-TCGA-SS-A7HO-01Z-00-DX1.png":
                        p_idx = img_paths.index(p)
                        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
                        p_img = x[p_idx]
                        # save_image(p_img, f"/home/shats/stupid_test/img_idx{p_idx}_r{np.random.randint(50)}.png")
                        print("--- img saved@@@ ---")
        elif self.group_size==1:
            img_paths = list(img_paths[0])

        self.update_dicts(img_id, img_paths, batch_outputs, y, self.train_img_samples_score_dict)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        img_id, img_paths, y, x = batch
        batch_outputs = outputs["batch_outputs"]

        # separate patch groups
        if self.group_size > 1:
            img_paths_lol = [p.split(",") for p in img_paths]
            img_paths = [item for sublist in img_paths_lol for item in sublist]
            y = y.repeat_interleave(self.group_size)
            batch_outputs = batch_outputs.repeat_interleave(self.group_size, axis=0)
            img_id = tuple(np.repeat(np.array(img_id), self.group_size))
        elif self.group_size==1:
            img_paths = list(img_paths[0])

        self.update_dicts(img_id, img_paths, batch_outputs, y, self.val_img_samples_score_dict)


    def update_dicts(self, batch_img_ids, batch_paths, batch_scores, batch_targets, img_samples_score_dict):
        """
        Fill the patient eval dicts with patients & scores of current batch.
        curr_dict is either the trianing dict that we want to fill or the validation dict (the one to update)
        """
        # ensure lengths are correct:
        assert len(batch_paths) == len(batch_scores) and len(batch_scores) == len(batch_targets) and len(batch_img_ids) == len(batch_paths), (
                f"\nError. lengths are not the same. lens: batch_paths-{len(batch_paths)}, batch_scores-{len(batch_scores)}, batch_targets-{len(batch_targets)}, batch_img_ids-{len(batch_img_ids)}\n")

        # fill dict
        with torch.no_grad():
            for img_id, patch_path, patch_score, patch_target in zip(batch_img_ids, batch_paths, batch_scores, batch_targets):
                if img_samples_score_dict[img_id][patch_path] is not None:
                    # import pdb; pdb.set_trace()
                    if self.debug_mode:
                        print(f"-- collision on patch {patch_path}")
                img_samples_score_dict[img_id][patch_path] = patch_score


    def on_validation_epoch_end(self, trainer, pl_module):
        """ 
        Calculate Error on patient level and Clear the patient level eval dict(s),
        So that it can fill up for next epoch
        """
        # eval and record results
        if not trainer.sanity_checking:
            train_rawsum_acc, train_majority_vote_acc, train_percent_class_1 = self.score_dict(self.train_img_samples_score_dict, self.train_samples_dict)
            val_rawsum_acc, val_majority_vote_acc, val_percent_class_1 = self.score_dict(self.val_img_samples_score_dict, self.val_samples_dict)

            self.log('train_rawsum_acc', train_rawsum_acc, on_step=False, on_epoch=True)
            self.log('train_majority_vote_acc', train_majority_vote_acc, on_step=False, on_epoch=True)
            self.log('train_percent_class_MSIMUT', train_percent_class_1, on_step=False, on_epoch=True)

            self.log('val_rawsum_acc', val_rawsum_acc, on_step=False, on_epoch=True)
            self.log('val_majority_vote_acc', val_majority_vote_acc, on_step=False, on_epoch=True)
            self.log('val_percent_class_MSIMUT', val_percent_class_1, on_step=False, on_epoch=True)

            # log dicts
            # import pdb; pdb.set_trace()
            # wandb.Table(dataframe=pd.DataFrame(self.train_samples_dict))

        # refresh dicts
        self.train_samples_dict = trainer.datamodule.train_ds.get_samples_dict()
        self.val_samples_dict = trainer.datamodule.val_ds.get_samples_dict()
        self.train_img_samples_score_dict = trainer.datamodule.train_ds.get_img_samples_score_dict()
        self.val_img_samples_score_dict = trainer.datamodule.val_ds.get_img_samples_score_dict()


    def score_dict(self, img_samples_score_dict, samples_dict, mode=None):
        if self.debug_mode:
            print(f"\n----------------- Debugging Patient-level Validation --------------------")
        y = []
        y_hat_rawsum = []
        y_hat_majority_vote = []
        for img_id in samples_dict.keys():
            img_y = samples_dict[img_id][1]
            patch_yhats = img_samples_score_dict[img_id]
            img_yhat = []
            img_yhat_none_count = 0 # just for error checking
            for patch_path in patch_yhats:
                if patch_yhats[patch_path] is not None:
                    img_yhat.append(patch_yhats[patch_path])
                else:
                    img_yhat_none_count += 1
            try:
                img_yhat = torch.stack(img_yhat)
            except:
                import pdb; pdb.set_trace()
            # img_yhat_rawsum_argmax = torch.argmax(img_yhat_rawsum_logits)
            if img_yhat.shape[1] == 2:
                img_yhat_rawsum_logits = torch.sum(img_yhat, dim=0)
                img_yhat_majority_vote = torch.mode(torch.argmax(img_yhat, dim=1)).values
            elif img_yhat.shape[1] == 1: ### REGRESSION CASE
                img_yhat_rawsum_logits = torch.tensor(0) ### FIX THIS FOR REGRESSION CASE
                # for the above, intuition says to sigmoid them, and then just check if average is greater or less than 0.5
                img_yhat_majority_vote = torch.mode((torch.nn.functional.sigmoid(img_yhat)>0.5).type(torch.uint8).squeeze(-1)).values
            else:
                print("ERROR!!! NO fucking idea what is going on!")
            y.append(img_y)
            y_hat_rawsum.append(img_yhat_rawsum_logits)
            y_hat_majority_vote.append(img_yhat_majority_vote)
            if self.debug_mode:
                print(f"Amount of nones for {img_id}: {img_yhat_none_count}")
        y = torch.stack(y)
        y_hat_rawsum = torch.stack(y_hat_rawsum)
        y_hat_majority_vote = torch.stack(y_hat_majority_vote)

        rawsum_acc = torchmetrics.functional.accuracy(y_hat_rawsum.cpu(), y)
        majority_vote_acc = torchmetrics.functional.accuracy(y_hat_majority_vote.cpu(), y)
        
        percent_class_MSIMUT = sum(y_hat_majority_vote==self.MSIMUT_label)/len(y_hat_majority_vote)
        # percent_class_MSIMUT = 1-sum(y_hat_majority_vote)/len(y_hat_majority_vote)

        return rawsum_acc, majority_vote_acc, percent_class_MSIMUT


