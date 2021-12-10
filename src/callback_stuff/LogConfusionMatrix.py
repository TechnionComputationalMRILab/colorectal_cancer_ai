import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics

# for making conf matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import cv2

class LogConfusionMatrix(pl.Callback):
    def __init__(self, num_classes=2) -> None:
        print("Logging Confusion Mat initialized")
        self.num_classes = num_classes
        self.train_dict = {"preds": [], "gt": []}
        self.val_dict = {"preds": [], "gt": []}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        paths, x, y = batch
        batch_outputs = outputs["batch_outputs"]
        self.train_dict["preds"].extend(batch_outputs.cpu().detach())
        self.train_dict["gt"].extend(y.cpu().detach())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        paths, x, y = batch
        batch_outputs = outputs["batch_outputs"]
        self.val_dict["preds"].extend(batch_outputs.cpu().detach())
        self.val_dict["gt"].extend(y.cpu().detach())

    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.sanity_checking:
            train_conf_mat, val_conf_mat = self.compute_conf_matrices()

            train_conf_mat_image = self.make_conf_matrix_image(train_conf_mat)
            val_conf_mat_image = self.make_conf_matrix_image(val_conf_mat)
            trainer.logger.experiment.add_image("train_conf_mat_img", train_conf_mat_image, global_step=trainer.global_step)
            trainer.logger.experiment.add_image("val_conf_mat_img", val_conf_mat_image, global_step=trainer.global_step)

        # empty the dicts
        self.train_dict = {"preds": [], "gt": []}
        self.val_dict = {"preds": [], "gt": []}

    def compute_conf_matrix(self, preds, gt):
        return torchmetrics.ConfusionMatrix(num_classes=self.num_classes)(preds, gt)

    def compute_conf_matrices(self):

        train_conf_mat = None
        val_conf_mat = None

        # first need to make them torch vectors
        if len(self.train_dict["preds"])>0:
            train_preds = torch.stack(self.train_dict["preds"])
            train_gt = torch.stack(self.train_dict["gt"])
            train_conf_mat = self.compute_conf_matrix(train_preds, train_gt)
        if len(self.val_dict["preds"])>0:
            val_preds = torch.stack(self.val_dict["preds"])
            val_gt = torch.stack(self.val_dict["gt"])
            val_conf_mat = self.compute_conf_matrix(val_preds, val_gt)

        return train_conf_mat, val_conf_mat

    def make_conf_matrix_image(self, conf_mat):
        df_cm = pd.DataFrame(conf_mat.tolist(), 
                index = [i for i in range(self.num_classes)], 
                columns = [i for i in range(self.num_classes)])
        fig, ax = plt.subplots()
        # fig.canvas.draw()
        conf_mat_image = sns.heatmap(df_cm, annot=True, ax=ax);

        # need to convert conf_mat_image (AxesSubplot object) to np array (image) for logging
        conf_mat_image = self.get_img_from_fig(fig)
        
        # close so that we dont see the plot in a jupyter notebook
        plt.close()

        return conf_mat_image


    def get_img_from_fig(self, fig, dpi=180):
        """ from here: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array """

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # did a few extra things before returning because the link wasnt enough
        return torch.tensor(img).permute(2, 0, 1)





