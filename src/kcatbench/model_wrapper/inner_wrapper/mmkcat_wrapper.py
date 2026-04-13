import gc
import importlib
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import torch

from kcatbench.model_wrapper.base_model import BaseModel
from kcatbench.model_wrapper.wrapper_progress import progress_completed, progress_started
from kcatbench.util import (
	DATA_DIR,
	DEVICE,
	MODELS_DIR,
	ensure_data_subfolder,
	force_torch_load_device,
	gdrive_download_file_from_folder,
	work_in_dir,
)


LOGGER = logging.getLogger(__name__)

MMKCAT_CODE_DIR = MODELS_DIR / "MMKcat"
MMKCAT_MODEL_DIR = MMKCAT_CODE_DIR / "model"
MMKCAT_UTIL_DIR = MMKCAT_CODE_DIR / "util"
MMKCAT_DATA_DIR = DATA_DIR / "MMKcat"
MMKCAT_TORCH_CACHE_DIR = MMKCAT_DATA_DIR / "torch_cache"
MMKCAT_TMP_DIR = MMKCAT_DATA_DIR / "tmp"

MMKCAT_CHECKPOINT_FILENAME = "concat_best_checkpoint.pth"
MMKCAT_CHECKPOINT_FOLDER_URL = "https://drive.google.com/drive/folders/1sVg9gfi_wQxZwbnylLrpmek15_aEbNs8?usp=drive_link"
MMKCAT_CHECKPOINT_PATH = MMKCAT_DATA_DIR / MMKCAT_CHECKPOINT_FILENAME
MMKCAT_MEAN_ATTR_PATH = MMKCAT_UTIL_DIR / "mean_attr.pt"

PRIMARY_MASK = np.array([True, True, True, True])
FALLBACK_MASKS = (
	np.array([True, True, True, False]),
	np.array([True, True, False, True]),
	np.array([True, True, False, False]),
)


class MMKcatWrapper(BaseModel):
	name = "MMKcat"

	def __init__(self) -> None:
		super().__init__()
		self._device = self._resolve_torch_device()
		self._mmkcat_model = None
		self._esm2_model = None
		self._esmfold_model = None
		self._alphabet = None
		self._batch_converter = None
		self._pdb2graph = None
		self._prepare_resources()

	def _prepare_resources(self) -> None:
		ensure_data_subfolder(MMKCAT_DATA_DIR)
		MMKCAT_TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
		MMKCAT_TMP_DIR.mkdir(parents=True, exist_ok=True)

		if MMKCAT_CHECKPOINT_PATH.exists() and MMKCAT_CHECKPOINT_PATH.stat().st_size == 0:
			LOGGER.warning("MMKcat checkpoint is empty at %s. Removing and re-downloading.", MMKCAT_CHECKPOINT_PATH)
			MMKCAT_CHECKPOINT_PATH.unlink(missing_ok=True)

		if not MMKCAT_CHECKPOINT_PATH.exists():
			LOGGER.info("MMKcat checkpoint not found at %s. Downloading from Google Drive.", MMKCAT_CHECKPOINT_PATH)
			result = gdrive_download_file_from_folder(
				folder_url=MMKCAT_CHECKPOINT_FOLDER_URL,
				expected_filename=MMKCAT_CHECKPOINT_FILENAME,
				output_path=MMKCAT_CHECKPOINT_PATH,
			)
			if not result["success"]:
				raise RuntimeError(
					"MMKcat checkpoint download failed. "
					+ result["message"]
					+ f" Manually download '{MMKCAT_CHECKPOINT_FILENAME}' from "
					+ f"{MMKCAT_CHECKPOINT_FOLDER_URL} and place it at {MMKCAT_CHECKPOINT_PATH}."
				)
			LOGGER.info("MMKcat checkpoint downloaded successfully to %s.", MMKCAT_CHECKPOINT_PATH)

		missing_paths = [
			str(path)
			for path in (
				MMKCAT_CODE_DIR,
				MMKCAT_MODEL_DIR,
				MMKCAT_UTIL_DIR,
				MMKCAT_CHECKPOINT_PATH,
				MMKCAT_MEAN_ATTR_PATH,
			)
			if not path.exists()
		]
		if missing_paths:
			joined_paths = "\n  - ".join([""] + missing_paths)
			raise FileNotFoundError(
				"MMKcat resources are missing. Expected paths:" + joined_paths
			)

		if shutil.which("mkdssp") is None:
			raise RuntimeError(
				"MMKcat requires 'mkdssp' on PATH for protein graph generation. "
				"Install DSSP and ensure the mkdssp binary is available in the active environment."
			)

		os.environ.setdefault("TORCH_HOME", str(MMKCAT_TORCH_CACHE_DIR))

	def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
		progress_started(LOGGER, "mmkcat.predict", "MMKcat prediction started rows=%s.", len(input_data.index))

		output = input_data.copy()
		output["mmkcat_kcat"] = pd.NA

		progress_started(LOGGER, "mmkcat.validation", "MMKcat input validation started.")
		clean_data = self._prepare_data(
			input_data,
			products_required=False,
			multiple_smiles=True,
		)
		if not clean_data["valid_indices"]:
			progress_completed(LOGGER, "mmkcat.validation", "MMKcat input validation completed valid_rows=0.")
			LOGGER.warning("MMKcat found no valid rows after preprocessing.")
			return output

		progress_completed(
			LOGGER,
			"mmkcat.validation",
			"MMKcat input validation completed valid_rows=%s.",
			len(clean_data["valid_indices"]),
		)

		progress_started(LOGGER, "mmkcat.runtime", "MMKcat runtime initialization started.")
		self._ensure_models_loaded()
		progress_completed(LOGGER, "mmkcat.runtime", "MMKcat runtime initialization completed.")

		predictions: list[Any] = []
		success_count = 0
		progress_started(
			LOGGER,
			"mmkcat.inference",
			"MMKcat inference started valid_rows=%s.",
			len(clean_data["valid_indices"]),
		)

		for pos, row_index in enumerate(clean_data["valid_indices"]):
			sequence = clean_data["sequence"][pos]
			substrate_smiles = clean_data["substrates"][pos]
			product_smiles = self._extract_products(input_data, row_index)

			try:
				predicted_log10 = self._predict_single_log10(
					substrate_smiles=substrate_smiles,
					protein_sequence=sequence,
					product_smiles=product_smiles,
				)
				predictions.append(float(10 ** predicted_log10))
				success_count += 1
			except Exception:
				LOGGER.exception("MMKcat failed for row index %s.", row_index)
				predictions.append(pd.NA)

		output.loc[clean_data["valid_indices"], "mmkcat_kcat"] = predictions
		LOGGER.info(
			"MMKcat prediction summary: total_valid=%s success=%s failed=%s",
			len(clean_data["valid_indices"]),
			success_count,
			len(clean_data["valid_indices"]) - success_count,
		)
		progress_completed(
			LOGGER,
			"mmkcat.inference",
			"MMKcat inference completed success=%s failed=%s.",
			success_count,
			len(clean_data["valid_indices"]) - success_count,
		)

		self._cleanup_gpu()
		progress_completed(LOGGER, "mmkcat.cleanup", "MMKcat cleanup completed.")
		progress_completed(
			LOGGER,
			"mmkcat.predict",
			"MMKcat predictions assigned rows=%s.",
			len(clean_data["valid_indices"]),
		)
		return output

	def _predict_single_log10(
		self,
		substrate_smiles: list[str],
		protein_sequence: str,
		product_smiles: list[Optional[str]],
	) -> float:
		if not substrate_smiles:
			raise ValueError("Substrate SMILES are required for MMKcat prediction.")
		if not protein_sequence:
			raise ValueError("Protein sequence is required for MMKcat prediction.")

		protein_sequence_rep = self._get_protein_sequence_rep(protein_sequence)
		protein_graph = self._get_protein_graph(protein_sequence)

		data = [
			[substrate_smiles],
			[protein_sequence_rep],
			[(protein_graph.x, protein_graph.edge_index)],
			[product_smiles],
			[torch.tensor(0.0)],
		]

		ordered_masks = (PRIMARY_MASK,) + FALLBACK_MASKS
		last_error = None
		for mask_idx, mask in enumerate(ordered_masks):
			try:
				self._mmkcat_model.test_mask = mask
				with torch.no_grad():
					result = self._mmkcat_model(data)
				predicted_x5 = result[-1]
				value = float(predicted_x5.reshape(-1)[0].item())
				if mask_idx > 0:
					LOGGER.warning(
						"MMKcat row used fallback mask %s after primary mask failure.",
						mask.tolist(),
					)
				return value
			except Exception as exc:
				last_error = exc
				if mask_idx < len(ordered_masks) - 1:
					LOGGER.warning(
						"MMKcat mask %s failed; trying next fallback mask.",
						mask.tolist(),
					)

		raise RuntimeError("All MMKcat masks failed for current row.") from last_error

	def _get_protein_sequence_rep(self, sequence: str) -> torch.Tensor:
		data = [("protein", sequence)]
		_, _, batch_tokens = self._batch_converter(data)
		batch_tokens = batch_tokens.to(self._device)
		batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

		with torch.no_grad():
			results = self._esm2_model(batch_tokens, repr_layers=[33], return_contacts=False)

		token_representations = results["representations"][33]
		tokens_len = int(batch_lens[0].item())
		sequence_rep = token_representations[0, 1 : tokens_len - 1].mean(0)
		return sequence_rep.detach().cpu().unsqueeze(0)

	def _get_protein_graph(self, sequence: str):
		with torch.no_grad():
			pdb_output = self._esmfold_model.infer_pdb(sequence)

		handle, tmp_path = tempfile.mkstemp(prefix="mmkcat_", suffix=".pdb", dir=MMKCAT_TMP_DIR)
		os.close(handle)
		pdb_path = Path(tmp_path)
		try:
			pdb_path.write_text(pdb_output)
			with self._mmkcat_runtime_context():
				graph = self._pdb2graph(str(pdb_path), str(MMKCAT_MEAN_ATTR_PATH))
		finally:
			pdb_path.unlink(missing_ok=True)

		if graph is None:
			raise RuntimeError("MMKcat graph generation returned no graph.")
		return graph

	def _ensure_models_loaded(self) -> None:
		if (
			self._mmkcat_model is not None
			and self._esm2_model is not None
			and self._esmfold_model is not None
			and self._batch_converter is not None
			and self._pdb2graph is not None
		):
			return

		try:
			with self._mmkcat_runtime_context(), force_torch_load_device(str(self._device)):
				esm_module = importlib.import_module("esm")

				build_vocab_module = importlib.import_module("build_vocab")
				word_vocab_cls = getattr(build_vocab_module, "WordVocab")

				import __main__
				setattr(__main__, "WordVocab", word_vocab_cls)

				worker_module = sys.modules.get("kcatbench.model_wrapper.model_worker")
				if worker_module is not None:
					setattr(worker_module, "WordVocab", word_vocab_cls)
					
				model_module = importlib.import_module("basic_model_mm")
				graph_module = importlib.import_module("util.generate_graph")

				if self._mmkcat_model is None:
					mmkcat_cls = getattr(model_module, "mmKcatPrediction")
					mmkcat_model = mmkcat_cls(
						device=self._device,
						batch_size=1,
						nhead=4,
						nhid=1024,
						nlayers=4,
						gcn_hidden=512,
						dropout=0.2,
						lambda_1=0.8,
						lambda_2=0.2,
						mode="test",
					).to(self._device)
					mmkcat_model.load_state_dict(torch.load(MMKCAT_CHECKPOINT_PATH))
					self._mmkcat_model = mmkcat_model.eval()

				if self._esm2_model is None or self._batch_converter is None:
					esm2_model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
					self._esm2_model = esm2_model.to(self._device).eval()
					self._alphabet = alphabet
					self._batch_converter = alphabet.get_batch_converter()

				if self._esmfold_model is None:
					esmfold_model = esm_module.pretrained.esmfold_v1()
					self._esmfold_model = esmfold_model.to(self._device).eval()

				if self._pdb2graph is None:
					self._pdb2graph = getattr(graph_module, "pdb2graph")

		except Exception as exc:
			raise RuntimeError(
				"Failed to initialize MMKcat runtime dependencies. "
				"Check MMKcat assets, ESM installation, checkpoint availability, and device configuration."
			) from exc

	def _extract_products(self, input_data: pd.DataFrame, row_index: Any) -> list[Optional[str]]:
		if "products" not in input_data.columns:
			return [None]
		return self._normalize_products(input_data.at[row_index, "products"])

	def _normalize_products(self, raw_products: Any) -> list[Optional[str]]:
		if not isinstance(raw_products, list):
			return [None]

		cleaned = [
			str(token).strip()
			for token in raw_products
			if pd.notna(token) and str(token).strip() and str(token).strip() != "[H+]" and str(token).strip().lower() != "none"
		]
		return cleaned if cleaned else [None]

	def _resolve_torch_device(self) -> torch.device:
		requested_device = str(DEVICE)
		if requested_device.startswith("cuda") and not torch.cuda.is_available():
			LOGGER.warning(
				"MMKcat requested device '%s' but CUDA is unavailable. Falling back to cpu.",
				requested_device,
			)
			return torch.device("cpu")

		try:
			return torch.device(requested_device)
		except Exception:
			LOGGER.warning(
				"MMKcat could not parse device '%s'. Falling back to cpu.",
				requested_device,
			)
			return torch.device("cpu")

	@contextmanager
	def _mmkcat_runtime_context(self) -> Iterator[None]:
		inserted_paths = []
		for target_path in (MMKCAT_UTIL_DIR, MMKCAT_CODE_DIR, MMKCAT_MODEL_DIR):
			path_str = str(target_path)
			if path_str not in sys.path:
				sys.path.insert(0, path_str)
				inserted_paths.append(path_str)

		try:
			with work_in_dir(MMKCAT_MODEL_DIR):
				yield
		finally:
			for path_str in reversed(inserted_paths):
				with suppress(ValueError):
					sys.path.remove(path_str)

	def _cleanup_gpu(self) -> None:
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
