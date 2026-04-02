import sys
import re
import gc
import math
import torch
import pickle
import numpy as np
import pandas as pd
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder, wget_download, work_in_dir
from kcatbench.model_wrapper.base_model import BaseModel

UNIKP_CODE_DIR = MODELS_DIR / "UniKP"
UNIKP_DATA_DIR = DATA_DIR / "UniKP"

if str(UNIKP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(UNIKP_CODE_DIR))

class UniKPWrapper(BaseModel):
    name = "UniKP"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(UNIKP_DATA_DIR)

        success_marker = UNIKP_DATA_DIR / ".setup_complete"
        if success_marker.exists():
            return
        
        archive_path = UNIKP_DATA_DIR / "UniKP for kcat.pkl"

        result = wget_download(url="https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat.pkl", output_path=archive_path)
        if not result['success']:
            raise RuntimeError(result['message'])

        expected_data_link = UNIKP_DATA_DIR / "prot_t5_xl_uniref50"

        if not expected_data_link.exists():
            expected_data_link.symlink_to((DATA_DIR / "CataPro" / "prot_t5_xl_uniref50"), target_is_directory=True)
            print(f"Created symlink: {expected_data_link} -> {(DATA_DIR / 'CataPro' / 'prot_t5_xl_uniref50')}")

        success_marker.touch()


    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:

        output = input_data.copy()

        clean_data = self._prepare_data(input_data)
        
        if not clean_data["valid_indices"]:
            output['unikp_kcat'] = pd.NA
            return output
        
        with work_in_dir(UNIKP_CODE_DIR):
            from build_vocab import WordVocab
            from pretrain_trfm import TrfmSeq2seq
            from utils import split
            from transformers import T5EncoderModel, T5Tokenizer
            
            smiles_vec = self._smiles_to_vec(clean_data["substrates"], WordVocab, TrfmSeq2seq, split)
            seq_vec = self._seq_to_vec(clean_data["sequence"], T5Tokenizer, T5EncoderModel)
            
        fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)
        
        model_path = UNIKP_DATA_DIR / "UniKP for kcat.pkl"
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
            
        pre_label = xgb_model.predict(fused_vector)
        pre_label_pow = [math.pow(10, p) for p in pre_label]
        
        output['unikp_kcat'] = pd.NA
        output.loc[clean_data["valid_indices"], 'unikp_kcat'] = pre_label_pow
        
        return output
    

    def _prepare_unikp_data(self, df: pd.DataFrame):
        valid_indices = []
        smiles_list = []
        sequences_list = []

        for idx, row in df.iterrows():
            subs = row.get('substrates')
            seq = row.get('sequence')

            if pd.isna(seq) or not isinstance(seq, str) or not seq.strip():
                continue

            if not isinstance(subs, list) or len(subs) == 0:
                continue
                
            primary_sub = subs[0]

            if pd.isna(primary_sub) or not isinstance(primary_sub, str) or not primary_sub.strip():
                continue

            valid_indices.append(idx)
            smiles_list.append(primary_sub.strip())
            sequences_list.append(seq.strip())

        return smiles_list, sequences_list, valid_indices

    
    def _smiles_to_vec(self, smiles_list, WordVocab, TrfmSeq2seq, split_fn):
        pad_index, unk_index, eos_index, sos_index = 0, 1, 2, 3

        import __main__
        __main__.WordVocab = WordVocab

        vocab = WordVocab.load_vocab('vocab.pkl')
        
        def get_inputs(sm):
            seq_len = 220
            sm = sm.split()
            if len(sm) > 218:
                sm = sm[:109] + sm[-109:]
            ids = [vocab.stoi.get(token, unk_index) for token in sm]
            ids = [sos_index] + ids + [eos_index]
            seg = [1] * len(ids)
            padding = [pad_index] * (seq_len - len(ids))
            ids.extend(padding)
            seg.extend(padding)
            return ids, seg

        x_id, x_seg = [], []
        x_split = [split_fn(sm) for sm in smiles_list]
        for sm in x_split:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
            
        x_id = torch.tensor(x_id)
        x_seg = torch.tensor(x_seg)

        trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
        cpu_device = torch.device('cpu')
        trfm.load_state_dict(torch.load('trfm_12_23000.pkl', map_location=cpu_device))
        trfm = trfm.to(cpu_device)
        trfm.eval()
        
        with torch.no_grad():
            X = trfm.encode(torch.t(x_id).to(cpu_device))
            
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
            
        return X

    def _seq_to_vec(self, sequence_list, T5Tokenizer, T5EncoderModel):
        sequences_Example = []
        for seq in sequence_list:
            if len(seq) > 1000:
                seq = seq[:500] + seq[-500:]
            spaced_seq = " ".join(list(seq))
            sequences_Example.append(spaced_seq)

        prottrans_dir = str(UNIKP_DATA_DIR / "prot_t5_xl_uniref50")
        tokenizer = T5Tokenizer.from_pretrained(prottrans_dir, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(prottrans_dir)
        
        model = model.to(DEVICE)
        model.eval()
        
        features = []
        for i, seq in enumerate(sequences_Example):
            seq = re.sub(r"[UZOB]", "X", seq)
            ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True)
            
            input_ids = torch.tensor(ids['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(ids['attention_mask']).to(DEVICE)
            
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
                
            embedding = embedding.last_hidden_state.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum().item()
            seq_emd = embedding[0][:seq_len - 1]
            features.append(seq_emd)
            
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
        for i in range(len(features)):
            for k in range(len(features[0][0])):
                for j in range(len(features[i])):
                    features_normalize[i][k] += features[i][j][k]
                features_normalize[i][k] /= len(features[i])
                
        return features_normalize
