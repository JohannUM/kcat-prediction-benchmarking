import sys

from kcatbench.util import MODELS_DIR, DATA_DIR, extract_tar_gz, wget_download

CATPRED_CODE_DIR = MODELS_DIR / "CatPred"
CATPRED_DATA_DIR = DATA_DIR / "CatPred"

from kcatbench.model_wrapper.base import BaseModel
import pandas as pd
import numpy as np
from rdkit import Chem

class CatPredWrapper(BaseModel):
    name = "CatPred"

    def __init__(self):
        super.__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        archive_path = CATPRED_DATA_DIR / "capsule_data_update.tar.gz"

        print(f"Downloading data to {archive_path}...")
        result = wget_download(url="https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz", output_path=archive_path)
        if not result['success']:
            raise RuntimeError(result['message'])
        
        print(f"Extracting archive to {CATPRED_DATA_DIR}...")
        result = extract_tar_gz(archive_path, CATPRED_DATA_DIR)
        if not result['success']:
            raise RuntimeError(result['message'])
        
        try:
            archive_path.unlink()
            print("Cleaned up archive file.")
        except OSError as e:
            print(f"Warning: Could not remove archive file: {e}")

    def predict(self, input_data: pd.DataFrame):
        return super().predict(input_data)
    
    def _create_csv_sh(self, parameter, input_file_path, checkpoint_dir):
        df = pd.read_csv(input_file_path)
        smiles_list = df.SMILES
        seq_list = df.sequence
        smiles_list_new = []

        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                smi = Chem.MolToSmiles(mol)
                if parameter == 'kcat' and '.' in smi:
                    smi = '.'.join(sorted(smi.split('.')))
                smiles_list_new.append(smi)
            except:
                print(f'Invalid SMILES input in input row {i}')
                print('Correct your input! Exiting..')
                return None

        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        for i, seq in enumerate(seq_list):
            if not set(seq).issubset(valid_aas):
                print(f'Invalid Enzyme sequence input in row {i}!')
                print('Correct your input! Exiting..')
                return None

        input_file_new_path = f'{input_file_path[:-4]}_input.csv'
        df['SMILES'] = smiles_list_new
        df.to_csv(input_file_new_path)

        with open('predict.sh', 'w') as f:
            f.write(f'''
            TEST_FILE_PREFIX={input_file_new_path[:-4]}
            RECORDS_FILE=${{TEST_FILE_PREFIX}}.json
            CHECKPOINT_DIR={checkpoint_dir}
            
            python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
            python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path ${{TEST_FILE_PREFIX}}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE
            ''')

        return input_file_new_path[:-4]+'_output.csv'
    
    def _get_predictions(parameter, outfile):
        """
        Process prediction results and add additional metrics.

        Args:
            parameter (str): The kinetics parameter that was predicted.
            outfile (str): Path to the output CSV file from the prediction.

        Returns:
            pandas.DataFrame: Processed predictions with additional metrics.
        """
        df = pd.read_csv(outfile)
        pred_col, pred_logcol, pred_sd_totcol, pred_sd_aleacol, pred_sd_epicol = [], [], [], [], []

        unit = 'mM'
        if parameter == 'kcat':
            target_col = 'log10kcat_max'
            unit = 's^(-1)'
        elif parameter == 'km':
            target_col = 'log10km_mean'
        else:
            target_col = 'log10ki_mean'

        unc_col = f'{target_col}_mve_uncal_var'

        for _, row in df.iterrows():
            model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]

            unc = row[unc_col]
            prediction = row[target_col]
            prediction_linear = np.power(10, prediction)

            model_outs = np.array([row[col] for col in model_cols])
            epi_unc = np.var(model_outs)
            alea_unc = unc - epi_unc
            epi_unc = np.sqrt(epi_unc)
            alea_unc = np.sqrt(alea_unc)
            unc = np.sqrt(unc)

            pred_col.append(prediction_linear)
            pred_logcol.append(prediction)
            pred_sd_totcol.append(unc)
            pred_sd_aleacol.append(alea_unc)
            pred_sd_epicol.append(epi_unc)

        df[f'Prediction_({unit})'] = pred_col
        df['Prediction_log10'] = pred_logcol
        df['SD_total'] = pred_sd_totcol
        df['SD_aleatoric'] = pred_sd_aleacol
        df['SD_epistemic'] = pred_sd_epicol

        return df
    