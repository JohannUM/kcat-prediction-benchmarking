import sys
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder

CATAPRO_CODE_DIR = MODELS_DIR / "CataPro"
CATAPRO_DATA_DIR = DATA_DIR / "CataPro"

if str(CATAPRO_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CATAPRO_CODE_DIR))

import pandas as pd
import numpy as np
import torch as th
from huggingface_hub import snapshot_download
from kcatbench.model_wrapper.base_model import BaseModel

from inference.utils import *
from inference.model import *
from inference.act_model import KcatModel # as _KcatModel
from torch.utils.data import DataLoader, Dataset

class CataProWrapper(BaseModel):
    name = "CataPro"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(CATAPRO_DATA_DIR)

        prot_t5_dir = CATAPRO_DATA_DIR / "prot_t5_xl_uniref50"
        molt5_dir = CATAPRO_DATA_DIR / "molt5-base-smiles2caption"
        
        if not prot_t5_dir.exists():
            snapshot_download(repo_id="Rostlab/prot_t5_xl_uniref50", local_dir=prot_t5_dir)
        
        if not molt5_dir.exists():
            snapshot_download(repo_id="laituan245/molt5-base-smiles2caption", local_dir=molt5_dir)

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:

        model_dpath = str(CATAPRO_CODE_DIR / "models")
        batch_size = 64
        device = DEVICE

        kcat_model_dpath = f"{model_dpath}/kcat_models"
        ProtT5_model = str(CATAPRO_DATA_DIR / "prot_t5_xl_uniref50")
        MolT5_model = str(CATAPRO_DATA_DIR / "molt5-base-smiles2caption")

        clean_data = self._prepare_data(input_data)

        dataloader = self._get_datasets(clean_data, ProtT5_model, MolT5_model)

        pred_kcat_list = []
        for fold in range(10):    
            kcat_model = KcatModel(device=device)
            kcat_model.load_state_dict(th.load(f"{kcat_model_dpath}/{fold}_bestmodel.pth", map_location=device))

            pred_score = self._inference(kcat_model, dataloader, device)
            pred_kcat_list.append(pred_score[:, :1])
        
        pred_kcat = np.mean(np.concatenate(pred_kcat_list, axis=1), axis=1, keepdims=True)
        pred_kcat_linear = np.power(10, pred_kcat)

        output = input_data.copy()
        output['catapro_kcat'] = pd.NA
        output.loc[clean_data["valid_indices"], 'catapro_kcat'] = pred_kcat_linear

        return output
    
    def _get_datasets(self, input_data, ProtT5_model, MolT5_model):
        sequences = input_data["sequence"]
        smiles = input_data["substrates"]
        
        seq_ProtT5 = Seq_to_vec(sequences, ProtT5_model)
        smi_molT5 = get_molT5_embed(smiles, MolT5_model)
        smi_macc = GetMACCSKeys(smiles)
        
        feats = th.from_numpy(np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)).to(th.float32)
        datasets = EnzymeDatasets(feats)
        dataloader = DataLoader(datasets)
        
        return dataloader
    
    def _inference(self, kcat_model, dataloader, device="cuda:0"):
        kcat_model.eval()
        with th.no_grad():
            pred_list = []
            for step, data in enumerate(dataloader):
                data = data.to(device)
                ezy_feats = data[:, :1024]
                sbt_feats = data[:, 1024:]
                pred_kcat = kcat_model(ezy_feats, sbt_feats)[0].cpu().numpy()
                pred_list.append(np.concatenate([pred_kcat], axis=1))
            
            return np.concatenate(pred_list, axis=0)

class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)