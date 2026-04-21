import os
import logging
import subprocess
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, extract_tar_gz, wget_download, work_in_dir, ensure_data_subfolder
from kcatbench.model_wrapper.wrapper_progress import progress_completed, progress_started

CATPRED_CODE_DIR = MODELS_DIR / "CatPred"
CATPRED_DATA_DIR = DATA_DIR / "CatPred"

from kcatbench.model_wrapper.base_model import BaseModel
import pandas as pd
import numpy as np
from rdkit import Chem


LOGGER = logging.getLogger(__name__)
ORIG_IDX_COL = "_orig_idx"

class CatPredWrapper(BaseModel):
    name = "CatPred"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(CATPRED_DATA_DIR)

        success_marker = CATPRED_DATA_DIR / ".setup_complete"
        if success_marker.exists():
            return

        archive_path = CATPRED_DATA_DIR / "capsule_data_update.tar.gz"

        result = wget_download(url="https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz", output_path=archive_path)
        if not result['success']:
            raise RuntimeError(result['message'])
        
        result = extract_tar_gz(archive_path, CATPRED_DATA_DIR)
        if not result['success']:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(result['message'])
        
        try:
            archive_path.unlink()
            success_marker.touch()
        except OSError as e:
            LOGGER.warning("Could not remove CatPred archive file: %s", e)

    def predict(self, input_data: pd.DataFrame):
        progress_started(LOGGER, "catpred.predict", "CatPred prediction started rows=%s.", len(input_data.index))

        output = input_data.copy()
        output['catpred_kcat'] = pd.NA

        with work_in_dir(CATPRED_CODE_DIR):
            progress_started(LOGGER, "catpred.validation", "CatPred input validation started.")
            outfile, clean_data = self._create_csv_sh("kcat", input_data, str(CATPRED_DATA_DIR / "data" / "pretrained" / "production" / "kcat"))
            if outfile is None or not clean_data["valid_indices"]:
                progress_completed(LOGGER, "catpred.validation", "CatPred validation completed valid_rows=0.")
                message = "CatPred validation failed: no valid rows after validation."
                LOGGER.error(message)
                raise RuntimeError(message)

            progress_completed(
                LOGGER,
                "catpred.validation",
                "CatPred validation completed valid_rows=%s.",
                len(clean_data["valid_indices"]),
            )

            progress_started(LOGGER, "catpred.inference", "CatPred subprocess prediction started.")
            run_result = self._run_prediction_script()
            if run_result.returncode != 0:
                LOGGER.error(
                    "CatPred predict.sh failed with exit code %s. stdout=%s stderr=%s",
                    run_result.returncode,
                    run_result.stdout,
                    run_result.stderr,
                )
                raise RuntimeError(
                    f"CatPred inference failed with exit code {run_result.returncode}."
                )

            progress_completed(LOGGER, "catpred.inference", "CatPred subprocess prediction completed.")

            progress_started(LOGGER, "catpred.parse", "CatPred output parsing started.")
            try:
                output_catpred = self._get_predictions("kcat", outfile)
            except Exception as exc:
                LOGGER.exception("CatPred failed to parse prediction output file: %s", outfile)
                raise RuntimeError(
                    f"CatPred parse failed while reading prediction output: {outfile}"
                ) from exc

            progress_completed(LOGGER, "catpred.parse", "CatPred output parsing completed rows=%s.", len(output_catpred.index))

        prediction_col = 'Prediction_(s^(-1))'
        if prediction_col not in output_catpred.columns:
            message = f"CatPred output is missing expected column: {prediction_col}"
            LOGGER.error(message)
            raise RuntimeError(message)

        target_indices = clean_data["valid_indices"]
        target_groups = clean_data.get("index_groups", [[idx] for idx in target_indices])
        representative_to_group = {
            rep_index: group_indices
            for rep_index, group_indices in zip(target_indices, target_groups)
        }

        mapped_representatives = set()
        mapped_row_indices = set()
        assign_count = 0

        if ORIG_IDX_COL in output_catpred.columns:
            missing_id_count = 0
            out_of_range_count = 0

            for raw_idx, prediction in output_catpred[[ORIG_IDX_COL, prediction_col]].itertuples(index=False, name=None):
                if pd.isna(raw_idx):
                    missing_id_count += 1
                    continue

                matched_index = None
                if raw_idx in output.index:
                    matched_index = raw_idx
                else:
                    try:
                        converted_idx = int(raw_idx)
                    except (TypeError, ValueError):
                        converted_idx = None

                    if converted_idx is not None and converted_idx in output.index:
                        matched_index = converted_idx

                if matched_index is None:
                    out_of_range_count += 1
                    continue

                group_indices = representative_to_group.get(matched_index)
                if group_indices is None:
                    out_of_range_count += 1
                    continue

                output.loc[group_indices, 'catpred_kcat'] = prediction
                mapped_representatives.add(matched_index)
                mapped_row_indices.update(group_indices)

            assign_count = len(mapped_representatives)
            if (
                assign_count != len(target_indices)
                or len(output_catpred.index) != len(target_indices)
                or missing_id_count > 0
                or out_of_range_count > 0
            ):
                LOGGER.warning(
                    "CatPred prediction mapping mismatch: expected_valid=%s output_rows=%s mapped=%s missing_ids=%s out_of_range_ids=%s",
                    len(target_indices),
                    len(output_catpred.index),
                    assign_count,
                    missing_id_count,
                    out_of_range_count,
                )
        else:
            LOGGER.warning(
                "CatPred output missing '%s'; using positional fallback assignment. This may misalign rows when CatPred drops entries.",
                ORIG_IDX_COL,
            )
            predictions = output_catpred[prediction_col].tolist()
            assign_count = min(len(target_indices), len(predictions))
            if assign_count > 0:
                partial_clean_data = {
                    "valid_indices": target_indices[:assign_count],
                    "index_groups": target_groups[:assign_count],
                }
                assign_indices, assign_values = self._expand_predictions(
                    partial_clean_data,
                    predictions[:assign_count],
                )
                output.loc[assign_indices, 'catpred_kcat'] = assign_values
                mapped_representatives.update(target_indices[:assign_count])
                mapped_row_indices.update(assign_indices)

            if assign_count != len(target_indices) or len(predictions) != len(target_indices):
                LOGGER.warning(
                    "CatPred prediction count mismatch (fallback mode): expected=%s predicted=%s assigned=%s",
                    len(target_indices),
                    len(predictions),
                    assign_count,
                )

        if assign_count == 0:
            message = "CatPred assignment failed: no assignable predictions after filtering."
            LOGGER.error(message)
            raise RuntimeError(message)

        assigned_row_indices = list(mapped_row_indices)
        assigned_non_null = int(output.loc[assigned_row_indices, 'catpred_kcat'].notna().sum()) if assigned_row_indices else 0
        if assigned_non_null == 0:
            message = "CatPred assignment failed: assigned predictions are all NA/NaN."
            LOGGER.error(message)
            raise RuntimeError(message)

        progress_completed(LOGGER, "catpred.predict", "CatPred predictions assigned rows=%s.", len(assigned_row_indices))

        return output

    def _empty_clean_data(self):
        return {
            "valid_indices": [],
            "index_groups": [],
            "substrates": [],
            "sequence": [],
        }

    def _run_prediction_script(self):
        env = os.environ.copy()
        env.pop("PROTEIN_EMBED_USE_CPU", None)
        env["CLEAR_CACHE"] = "1"
        env.setdefault("CATPRED_CACHE_PATH", "/mnt/burning_scratch/jlotter/.cache.esm2_embeddings")
        return subprocess.run(
            ["bash", "./predict.sh"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _canonicalize_smiles(self, smiles: str, parameter: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return None

        if mol is None:
            return None

        try:
            canonical = Chem.MolToSmiles(mol)
        except Exception:
            return None

        if parameter == 'kcat' and '.' in canonical:
            canonical = '.'.join(sorted(canonical.split('.')))
        return canonical

    def _is_valid_sequence(self, sequence: str, valid_aas: set):
        return isinstance(sequence, str) and bool(sequence) and set(sequence).issubset(valid_aas)

    def _prepare_substrate_smiles(self, substrates):
        if isinstance(substrates, str):
            substrate = substrates.strip()
            return substrate if substrate else None

        if isinstance(substrates, list):
            cleaned_substrates = []
            for substrate in substrates:
                if not isinstance(substrate, str):
                    continue
                value = substrate.strip()
                if value:
                    cleaned_substrates.append(value)

            if not cleaned_substrates:
                return None

            if len(cleaned_substrates) == 1:
                return cleaned_substrates[0]

            return '.'.join(cleaned_substrates)

        return None
    
    def _create_csv_sh(self, parameter, input_data:pd.DataFrame, checkpoint_dir):

        clean_data = self._prepare_data(input_data, multiple_smiles=True)
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        filtered_clean_data = self._empty_clean_data()
        smiles_list_new = []
        sequence_list_new = []
        invalid_smiles_indices = []
        invalid_sequence_indices = []

        for row_index, index_group, smi, seq in zip(
            clean_data["valid_indices"],
            clean_data["index_groups"],
            clean_data["substrates"],
            clean_data["sequence"],
        ):
            prepared_smiles = self._prepare_substrate_smiles(smi)
            if prepared_smiles is None:
                invalid_smiles_indices.append(row_index)
                continue

            canonical_smiles = self._canonicalize_smiles(prepared_smiles, parameter)
            if canonical_smiles is None:
                invalid_smiles_indices.append(row_index)
                continue

            if not self._is_valid_sequence(seq, valid_aas):
                invalid_sequence_indices.append(row_index)
                continue

            filtered_clean_data["valid_indices"].append(row_index)
            filtered_clean_data["index_groups"].append(index_group)
            filtered_clean_data["substrates"].append(canonical_smiles)
            filtered_clean_data["sequence"].append(seq)
            smiles_list_new.append(canonical_smiles)
            sequence_list_new.append(seq)

        if invalid_smiles_indices:
            LOGGER.warning("CatPred skipped invalid SMILES rows: %s", invalid_smiles_indices)
        if invalid_sequence_indices:
            LOGGER.warning("CatPred skipped invalid sequence rows: %s", invalid_sequence_indices)
        if not filtered_clean_data["valid_indices"]:
            LOGGER.warning("CatPred has no valid rows after row-level validation.")
            return None, filtered_clean_data

        input_file_new_path = str(CATPRED_DATA_DIR / "kcat_prediction_input.csv")
        df = pd.DataFrame()
        df['SMILES'] = smiles_list_new
        df['sequence'] = sequence_list_new
        df['pdbpath'] = [f"sequence_{i}" for i in range(len(df))]
        df[ORIG_IDX_COL] = filtered_clean_data["valid_indices"]
        df.to_csv(input_file_new_path, index=False)

        gpu_id = DEVICE.split(":")[1] if ":" in DEVICE else "0"

        with open('predict.sh', 'w') as f:
            f.write(f'''
            TEST_FILE_PREFIX={input_file_new_path[:-4]}
            RECORDS_FILE=${{TEST_FILE_PREFIX}}.json
            CHECKPOINT_DIR={checkpoint_dir}
            
            python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
            python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path ${{TEST_FILE_PREFIX}}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE --gpu {gpu_id}
            ''')

        return input_file_new_path[:-4]+'_output.csv', filtered_clean_data
    
    def _get_predictions(self, parameter, outfile):
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
    