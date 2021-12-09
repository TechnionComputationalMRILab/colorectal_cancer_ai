from pytorch_lightning import Callback
import pandas as pd

class PatientLevelEval(Callback):
    """
    Runs Patient level evaluation on dataset.
    """
    def __init__(self) -> None:
        self.train_df = pd.DataFrame({''}
