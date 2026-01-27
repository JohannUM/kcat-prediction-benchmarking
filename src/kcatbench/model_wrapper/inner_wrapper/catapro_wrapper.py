import sys
from kcatbench.util import MODELS_DIR, DATA_DIR

CATAPRO_CODE_DIR = MODELS_DIR / "CataPro"
CATAPRO_DATA_DIR = DATA_DIR / "CataPro"

if str(CATAPRO_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CATAPRO_CODE_DIR))

import pandas as pd
import numpy as np
import torch as th
from huggingface_hub import snapshot_download
from kcatbench.model_wrapper.base import BaseModel

from utils import *
from model import *
from act_model import KcatModel # as _KcatModel
from act_model import KmModel # as _KmModel
from act_model import ActivityModel
from torch.utils.data import DataLoader, Dataset

class CataProWrapper(BaseModel):
    name = "CataPro"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        print("START DOWNLOAD")
        tmp1 = snapshot_download(repo_id="Rostlab/prot_t5_xl_uniref50", local_dir=(CATAPRO_DATA_DIR / "prot_t5_xl_uniref50"))
        print(f"DOWNLOAD 1 COMPLETE:{tmp1}")
        tmp2 = snapshot_download(repo_id="laituan245/molt5-base-smiles2caption", local_dir=(CATAPRO_DATA_DIR / "molt5-base-smiles2caption"))
        print(f"DOWNLOAD 2 COMPLETE:{tmp2}")

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:

        model_dpath = str(CATAPRO_CODE_DIR / "models")
        batch_size = 64
        device = "cuda:0"

        kcat_model_dpath = f"{model_dpath}/kcat_models"
        Km_model_dpath = f"{model_dpath}/Km_models"
        act_model_dpath = f"{model_dpath}/act_models"
        ProtT5_model = str(CATAPRO_DATA_DIR / "prot_t5_xl_uniref50")
        MolT5_model = str(CATAPRO_DATA_DIR / "molt5-base-smiles2caption")

        ezy_ids, smiles_list, dataloader = self._get_datasets(input_data, ProtT5_model, MolT5_model)

        pred_kcat_list = []
        pred_Km_list = []
        pred_act_list = []
        for fold in range(10):    
            kcat_model = KcatModel(device=device)
            kcat_model.load_state_dict(th.load(f"{kcat_model_dpath}/{fold}_bestmodel.pth", map_location=device))
            Km_model = KmModel(device=device)
            Km_model.load_state_dict(th.load(f"{Km_model_dpath}/{fold}_bestmodel.pth", map_location=device))
            act_model = ActivityModel(device=device)
            act_model.load_state_dict(th.load(f"{act_model_dpath}/{fold}_bestmodel.pth", map_location=device))

            pred_score = self._inference(kcat_model, Km_model, act_model, dataloader, device)
            pred_kcat_list.append(pred_score[:, :1])
            pred_Km_list.append(pred_score[:, 1:2])
            pred_act_list.append(pred_score[:, -1:])
        
        pred_kcat = np.mean(np.concatenate(pred_kcat_list, axis=1), axis=1, keepdims=True)
        pred_Km = np.mean(np.concatenate(pred_Km_list, axis=1), axis=1, keepdims=True)
        pred_act = np.mean(np.concatenate(pred_act_list, axis=1), axis=1, keepdims=True)
        
        final_score = np.concatenate([np.array(ezy_ids).reshape(-1, 1), np.array(smiles_list).reshape(-1, 1), pred_kcat, pred_Km, pred_act], axis=1)
        final_df = pd.DataFrame(final_score, columns=["fasta_id", "smiles", "pred_log10[kcat(s^-1)]", "pred_log10[Km(mM)]", "pred_log10[kcat/Km(s^-1mM^-1)]"])
        
        return final_df
    
    def _get_datasets(input_data: pd.DataFrame, ProtT5_model, MolT5_model):
        # inp_df = pd.read_csv(inp_fpath, index_col=0)
        ezy_ids = input_data["Enzyme_id"].values
        ezy_type = input_data["type"].values
        ezy_keys = [f"{_id}_{t}" for _id, t in zip(ezy_ids, ezy_type)]
        sequences = input_data["sequence"].values 
        smiles = input_data["smiles"].values
        
        seq_ProtT5 = Seq_to_vec(sequences, ProtT5_model)
        smi_molT5 = get_molT5_embed(smiles, MolT5_model)
        smi_macc = GetMACCSKeys(smiles)
        
        feats = th.from_numpy(np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)).to(th.float32)
        datasets = EnzymeDatasets(feats)
        dataloader = DataLoader(datasets)
        
        return ezy_keys, smiles, dataloader
    
    def _inference(kcat_model, Km_model, act_model, dataloader, device="cuda:0"):
        kcat_model.eval()
        Km_model.eval()
        act_model.eval()
        with th.no_grad():
            pred_list = []
            for step, data in enumerate(dataloader):
                data = data.to(device)
                ezy_feats = data[:, :1024]
                sbt_feats = data[:, 1024:]
                pred_kcat = kcat_model(ezy_feats, sbt_feats).cpu().numpy()
                pred_Km = Km_model(ezy_feats, sbt_feats).cpu().numpy()
                pred_act = act_model(ezy_feats, sbt_feats)[-1].cpu().numpy()
                pred_list.append(np.concatenate([pred_kcat, pred_Km, pred_act], axis=1))
            
            return np.concatenate(pred_list, axis=0)

class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)