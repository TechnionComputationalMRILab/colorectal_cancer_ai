import torch
import pytorch_lightning as pl
from collections import defaultdict
from tqdm import tqdm

from ..data_stuff.data_tools import get_patient_name_from_path

class DownstreamMLP(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, self.hparams.num_classes),
            torch.nn.Sigmoid(),
        )
        # self.criteria = torch.nn.BCEWithLogitsLoss()
        self.criteria = torch.nn.BCELoss()

        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc, "batch_outputs": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        
        val_loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        val_acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc, "batch_outputs": out.clone().detach()}
    
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
 

class DownstreamTrainer(pl.Callback):
    def __init__(self) -> None:
        print("Downsteam Evaluation initialized")
        """
        Need to build a dict of {patient: [p1e, p2e, ... pne]} where pne is the embedding for patch #n of the patient
        Since n changes, I will just take the first (or random) j embeddings
        Build a picture (matrix) of these embeddings and train on a patient level this way.
        """

    # # Was going to use this in init but realized I dont have pl_module there... so I have to do it in on_validation_epoch_end on the fly
    # def make_patient_label_dict(self, dl):
    #     """ makes train and val dicts like {patient: label} """

    #     train_patient_label_dict = {}
    #     val_patient_label_dict = {}

    #     for paths, x, y in dl:
    #         patient_ids = [get_patient_name_from_path(path) for path in paths]
    #         for p, l in zip(patient_ids, y):
    #             # if patient is already in dict, just check that the label is right
    #             if p in patient_label_dict:
    #                 assert (patient_label_dict[p] == l), "🛑 labels for patches not consistent"
    #             else:
    #                 patient_label_dict[p] = l


    def on_validation_epoch_end(self, trainer, pl_module):
        # print("IN VAL EPOCH END from downstream trainer 🤫 \n")
        train_dl = trainer.datamodule.train_dataloader()
        val_dl = trainer.datamodule.val_dataloader()
        logger = trainer.logger

        # get all embeddings
        patient_embedding_dict = defaultdict(list)
        patient_label_dict = defaultdict(list)
        with torch.no_grad():
            for paths, x, y in tqdm(train_dl, desc="downstream training..." , leave=True):
                x = x.to(pl_module.device)
                batch_embeddings = pl_module.extract_features(x).cpu().detach()
                del x
                patient_ids = [get_patient_name_from_path(path) for path in paths]
                assert(len(patient_ids) == len(batch_embeddings) and len(batch_embeddings == len(y))), "🛑 lengths bad"
                for p, e, l in zip(patient_ids, batch_embeddings, y):
                    patient_embedding_dict[p].append(e) # e is an individual patch embedding
                    if p in patient_label_dict:
                        assert(patient_label_dict[p] == l), "🛑 labels for patches not consistent"
                    else:
                        patient_label_dict[p] = l

        num_patients = len(patient_embedding_dict)
        all_embeddings = list(patient_embedding_dict.values())
        embedding_lens = [len(e_list) for e_list in all_embeddings]
        min_num_embeddings = min(embedding_lens)
        max_num_embeddings = max(embedding_lens)
        avg_num_embeddings = sum(embedding_lens)/len(embedding_lens)
        median_num_embeddings = embedding_lens[int(len(embedding_lens)/2)]

        f_str =(f"\n---\n"
        f"Num Patients        : {num_patients}\n"
        f"Embedding ex shape  : {all_embeddings[0][0].shape}\n"
        # f"embedding ex        : {all_embeddings[0]}"
        f"First 15 emb lens   : {sorted(embedding_lens)[:20]}\n"
        f"Min num embeddings  : {min_num_embeddings}\n"
        f"Max num embeddings  : {max_num_embeddings}\n"
        f"Avg num embeddings  : {avg_num_embeddings}\n"
        f"Med num embeddings  : {median_num_embeddings}\n"
        f"---\n")
        print(f_str)

        torch.cuda.empty_cache()
        del patient_embedding_dict
        del patient_label_dict
        del train_dl
        del val_dl

        # now need to make very custom dataloaders from these dicts and train on them

        # trainer = pl.Trainer(gpus=1, max_epochs=3, logger=logger)
        
        
