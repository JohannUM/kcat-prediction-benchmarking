from __future__ import annotations

import importlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from kcatbench.util import DATA_DIR, ensure_data_subfolder, load_chemeo_api_key


logger = logging.getLogger(__name__)

_DEFAULT_PROCESSED_BASENAME = "brenda_processed"
_DEFAULT_BRENDA_SUBDIR = "brenda"
_DEFAULT_SEQUENCE_CACHE_FILENAME = "uniprot_sequence_cache.json"
_DEFAULT_LIGAND_INFO_FILENAME = "brenda_ligand_info.csv"
_DEFAULT_LIGAND_LOOKUP_CACHE_FILENAME = "brenda_ligand_lookup_cache.json"
_MOLECULE_RESOLVER_IDENTIFIER_NAME = "name"
_MOLECULE_RESOLVER_IDENTIFIER_INCHI = "inchi"
_MUTANT_KEYWORDS = {"mutant", "mutated", "variant", "engineered"}
_LIGAND_INFO_COLUMNS = (
    "ligand",
    "ec_number",
    "role",
    "structure",
    "inchi",
    "chebi",
    "references",
)
_LIGAND_LOOKUP_COLUMNS = (
    "substrate_or_product_name",
    "inchi",
    "chebi",
    "smiles",
)
_DEFAULT_COLUMNS = [
    "source",
    "ec_number",
    "protein_id",
    "organism",
    "UniProt_ID",
    "sequence",
    "kcat_substrate_name",
    "substrates",
    "products",
    "substrates_names",
    "products_names",
    "reaction_equation",
    "reaction_source",
    "reaction_reversibility",
    "reaction_annotation",
    "experimental_kcat",
    "temperature",
    "pH",
    "references",
    "tn_comment",
    "tn_raw",
]
_REACTION_PLUS_SPLIT_PATTERN = re.compile(r"\s+\+\s+|(?<![A-Za-z0-9])\+(?![A-Za-z0-9])")
_UNIPROT_ACCESSION_CORE_PATTERN = (
    r"(?:"
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r")"
)
_UNIPROT_ACCESSION_SEARCH_PATTERN = re.compile(
    rf"\b{_UNIPROT_ACCESSION_CORE_PATTERN}\b",
    re.IGNORECASE,
)
_UNIPROT_ACCESSION_FULL_PATTERN = re.compile(
    rf"^{_UNIPROT_ACCESSION_CORE_PATTERN}$",
    re.IGNORECASE,
)
_UNIPROT_FASTA_URL_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


def _bump_counter(stats: Optional[dict[str, int]], key: str, amount: int = 1) -> None:
    """Increment a named counter in an optional stats dictionary."""
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + amount


def _log_counter_event(enable_logging: bool, event: str, **payload: Any) -> None:
    """Emit one structured JSON log event when logging is enabled."""
    if not enable_logging:
        return
    logger.info("brenda_build_db.%s %s", event, json.dumps(payload, sort_keys=True, default=str))


def parse_brenda_flatfile(filepath: str | Path, stats: Optional[dict[str, int]] = None):
    """Parse a BRENDA flat text file into EC-level records.

    The parser follows BRENDA text conventions where each data line starts with a
    2-3 character key, values begin after the first tab, continuation lines start
    with whitespace, and `///` marks the end of an EC record.

    Args:
        filepath: Absolute or relative path to the BRENDA flat text file.
        stats: Optional counter dictionary used for structured build diagnostics.

    Yields:
        EC records where each record maps field keys (for example ID, PR, SP,
        TN) to a list of string entries.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find the BRENDA file at: {path}")

    current_record: defaultdict[str, list[str]] = defaultdict(list)
    current_key: Optional[str] = None
    current_buffer: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_num, line in enumerate(handle, 1):
            _bump_counter(stats, "flatfile_lines_total")
            line_text = line.rstrip("\n")

            if line_text == "///":
                if current_key and current_buffer:
                    current_record[current_key].append(" ".join(current_buffer))
                if current_record:
                    _bump_counter(stats, "flatfile_records_yielded")
                    yield dict(current_record)
                current_record = defaultdict(list)
                current_key = None
                current_buffer = []
                continue

            if not line_text:
                _bump_counter(stats, "flatfile_empty_lines")
                continue

            if line.startswith((" ", "\t")):
                if current_key:
                    current_buffer.append(line_text.strip())
                else:
                    _bump_counter(stats, "flatfile_orphan_continuation_lines")
                    logger.debug("Line %s has continuation data without an active key.", line_num)
                continue

            if current_key and current_buffer:
                current_record[current_key].append(" ".join(current_buffer))
                current_buffer = []

            parts = line_text.split("\t", 1)
            if len(parts) != 2:
                _bump_counter(stats, "flatfile_malformed_key_lines")
                logger.debug("Line %s does not include a tab-delimited value section.", line_num)
                current_key = None
                current_buffer = []
                continue

            current_key = parts[0].strip()
            current_buffer = [parts[1].strip()]

    if current_key and current_buffer:
        current_record[current_key].append(" ".join(current_buffer))
    if current_record:
        _bump_counter(stats, "flatfile_records_yielded")
        yield dict(current_record)


def _resolve_flatfile_path(target_dir: Path, flatfile_path: str | Path | None) -> Path:
    """Resolve and validate the BRENDA flatfile path.

    Args:
        target_dir: Directory used for default path discovery.
        flatfile_path: Optional explicit path to the flatfile.

    Returns:
        Absolute path to the selected flatfile.
    """
    if flatfile_path is not None:
        candidate = Path(flatfile_path)
        if not candidate.is_absolute():
            if candidate.is_file():
                candidate = candidate.resolve()
            else:
                candidate = (target_dir / candidate).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"BRENDA flatfile was not found at: {candidate}")
        return candidate

    candidate_files = [
        path
        for path in target_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ""}
        and "readme" not in path.name.lower()
        and not path.name.lower().endswith(".json")
        and not path.name.lower().endswith(".csv")
        and not path.name.lower().endswith(".pkl")
        and not path.name.lower().endswith(".pickle")
        and not path.name.lower().endswith(".gz")
        and not path.name.lower().endswith(".tar")
    ]

    candidate_files.sort()
    if not candidate_files:
        raise FileNotFoundError(
            "No BRENDA flatfile candidate found in "
            f"{target_dir}. Provide flatfile_path explicitly."
        )
    return candidate_files[0]


def _is_mutant_text(text: str) -> bool:
    """Determine whether a text fragment indicates a mutant enzyme context."""
    clean = text.lower()
    return any(keyword in clean for keyword in _MUTANT_KEYWORDS)


def _normalize_compound_name(value: str) -> str:
    """Normalize a compound name for case-insensitive matching."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _ensure_kcat_substrate_first(
    substrates: list[str],
    kcat_substrate_name: Optional[str],
    stats: Optional[dict[str, int]] = None,
) -> list[str]:
    """Move kcat_substrate_name to index 0 when it already exists in substrates.

    Matching is case-insensitive via _normalize_compound_name. If the target is
    missing, empty, or the list is empty, the original list order is returned.
    """
    _bump_counter(stats, "substrate_order_rows_checked")

    if not substrates:
        _bump_counter(stats, "substrate_order_skipped_empty_substrates")
        return list(substrates)

    if kcat_substrate_name is None or kcat_substrate_name is pd.NA:
        _bump_counter(stats, "substrate_order_skipped_missing_target")
        return list(substrates)

    try:
        if bool(pd.isna(kcat_substrate_name)):
            _bump_counter(stats, "substrate_order_skipped_missing_target")
            return list(substrates)
    except TypeError:
        pass

    target_text = str(kcat_substrate_name).strip()
    if not target_text:
        _bump_counter(stats, "substrate_order_skipped_missing_target")
        return list(substrates)

    normalized_target = _normalize_compound_name(target_text)
    match_index: Optional[int] = None
    for index, substrate in enumerate(substrates):
        if _normalize_compound_name(str(substrate)) == normalized_target:
            match_index = index
            break

    if match_index is None:
        _bump_counter(stats, "substrate_order_target_absent")
        return list(substrates)

    if match_index == 0:
        _bump_counter(stats, "substrate_order_already_first")
        return list(substrates)

    reordered = [substrates[match_index], *substrates[:match_index], *substrates[match_index + 1 :]]
    _bump_counter(stats, "substrate_order_rows_reordered")
    return reordered


def _deduplicate_values(values: list[str]) -> list[str]:
    """Deduplicate a list of strings while preserving original order."""
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _parse_protein_id_group(group_text: str) -> list[str]:
    """Parse one protein ID group token from a BRENDA #...# section."""
    ids = [token.strip() for token in group_text.split(",") if token.strip()]
    numeric_ids = [token for token in ids if token.isdigit()]
    return _deduplicate_values(numeric_ids)


def _extract_all_protein_ids(text: str) -> list[str]:
    """Extract all protein IDs found in any #...# occurrence in a text line."""
    all_ids: list[str] = []
    for group in re.findall(r"#([\d,\s]+)#", text):
        all_ids.extend(_parse_protein_id_group(group))
    return _deduplicate_values(all_ids)


def _strip_leading_protein_tag(text: str) -> tuple[list[str], str]:
    """Strip a leading #...# protein section and return IDs plus remaining text."""
    match = re.match(r"^\s*#([\d,\s]+)#\s*(.*)$", text)
    if not match:
        return [], text.strip()
    leading_ids = _parse_protein_id_group(match.group(1))
    remainder = match.group(2).strip()
    return leading_ids, remainder


def _extract_uniprot_id(text: str) -> Optional[str]:
    """Extract a UniProt accession candidate from a protein entry line."""
    match = _UNIPROT_ACCESSION_SEARCH_PATTERN.search(text)
    return match.group(0).upper() if match else None


def _normalize_uniprot_accession(value: Any) -> Optional[str]:
    """Normalize and validate a UniProt accession value."""
    if value is None or value is pd.NA:
        return None

    try:
        if bool(pd.isna(value)):
            return None
    except TypeError:
        pass

    normalized = str(value).strip().upper()
    if not normalized:
        return None
    if not _UNIPROT_ACCESSION_FULL_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _resolve_sequence_cache_path(target_dir: Path, sequence_cache_path: str | Path | None) -> Path:
    """Resolve the UniProt sequence cache path."""
    if sequence_cache_path is None:
        return (target_dir / _DEFAULT_SEQUENCE_CACHE_FILENAME).resolve()

    candidate = Path(sequence_cache_path)
    if candidate.is_absolute():
        return candidate
    return (target_dir / candidate).resolve()


def _resolve_ligand_info_path(target_dir: Path, ligand_info_path: str | Path | None) -> Path:
    """Resolve and validate the BRENDA ligand info source path."""
    if ligand_info_path is None:
        candidate = (target_dir / _DEFAULT_LIGAND_INFO_FILENAME).resolve()
    else:
        candidate = Path(ligand_info_path)
        if not candidate.is_absolute():
            candidate = (target_dir / candidate).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(f"BRENDA ligand info file was not found at: {candidate}")
    return candidate


def _resolve_ligand_lookup_cache_path(
    target_dir: Path,
    ligand_lookup_cache_path: str | Path | None,
) -> Path:
    """Resolve the output path for the ligand lookup JSON cache."""
    if ligand_lookup_cache_path is None:
        return (target_dir / _DEFAULT_LIGAND_LOOKUP_CACHE_FILENAME).resolve()

    candidate = Path(ligand_lookup_cache_path)
    if candidate.is_absolute():
        return candidate
    return (target_dir / candidate).resolve()


def _clean_ligand_mapping_value(value: Any) -> Optional[str]:
    """Normalize ligand mapping values and convert blank markers to None."""
    if value is None or value is pd.NA:
        return None

    try:
        if bool(pd.isna(value)):
            return None
    except TypeError:
        pass

    text = str(value).strip()
    if not text:
        return None
    if text == "-":
        return None
    if text.lower() == "nan":
        return None
    return text


def _load_brenda_ligand_info(
    ligand_info_path: Path,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Load and normalize the BRENDA ligand info table for name-based mapping."""
    ligand_info_df: Optional[pd.DataFrame] = None
    decode_error: Optional[UnicodeDecodeError] = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            ligand_info_df = pd.read_csv(
                ligand_info_path,
                sep="\t",
                names=list(_LIGAND_INFO_COLUMNS),
                header=None,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                encoding=encoding,
                usecols=list(range(len(_LIGAND_INFO_COLUMNS))),
            )
            _bump_counter(stats, f"ligand_info_encoding_{encoding}")
            break
        except UnicodeDecodeError as exc:
            decode_error = exc

    if ligand_info_df is None:
        assert decode_error is not None
        raise decode_error

    _bump_counter(stats, "ligand_info_rows_total", len(ligand_info_df.index))
    if ligand_info_df.empty:
        _bump_counter(stats, "ligand_info_rows_empty")
        return pd.DataFrame(columns=["ligand", "_normalized_ligand", "inchi", "chebi"])

    first_row_values = [str(ligand_info_df.iloc[0][column]).strip().lower() for column in _LIGAND_INFO_COLUMNS]
    if first_row_values == list(_LIGAND_INFO_COLUMNS):
        ligand_info_df = ligand_info_df.iloc[1:].reset_index(drop=True)
        _bump_counter(stats, "ligand_info_header_row_dropped")

    ligand_info_df["ligand"] = ligand_info_df["ligand"].astype(str).str.strip()
    rows_before_ligand_filter = len(ligand_info_df.index)
    ligand_info_df = ligand_info_df[ligand_info_df["ligand"] != ""].copy()
    _bump_counter(
        stats,
        "ligand_info_rows_without_ligand",
        rows_before_ligand_filter - len(ligand_info_df.index),
    )

    if ligand_info_df.empty:
        _bump_counter(stats, "ligand_info_rows_empty_after_ligand_filter")
        return pd.DataFrame(columns=["ligand", "_normalized_ligand", "inchi", "chebi"])

    ligand_info_df["_normalized_ligand"] = ligand_info_df["ligand"].map(_normalize_compound_name)
    ligand_info_df["inchi"] = ligand_info_df["inchi"].map(_clean_ligand_mapping_value)
    ligand_info_df["chebi"] = ligand_info_df["chebi"].map(_clean_ligand_mapping_value)
    _bump_counter(stats, "ligand_info_rows_with_ligand", len(ligand_info_df.index))

    return ligand_info_df[["ligand", "_normalized_ligand", "inchi", "chebi"]].reset_index(drop=True)


def _extract_unique_ligand_names_from_curated_df(
    curated_df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Extract unique substrate/product names from the curated BRENDA DataFrame."""
    unique_names: dict[str, str] = {}
    for column in ("substrates", "products"):
        if column not in curated_df.columns:
            continue

        for values in curated_df[column]:
            if not isinstance(values, list):
                continue

            for value in values:
                text = str(value).strip()
                if not text:
                    continue
                if text in unique_names:
                    continue
                unique_names[text] = _normalize_compound_name(text)

    if not unique_names:
        _bump_counter(stats, "ligand_lookup_names_total", 0)
        return pd.DataFrame(columns=["substrate_or_product_name", "normalized_name"])

    rows = [
        {
            "substrate_or_product_name": name,
            "normalized_name": normalized_name,
        }
        for name, normalized_name in unique_names.items()
    ]
    rows.sort(key=lambda row: (row["normalized_name"], row["substrate_or_product_name"]))

    names_df = pd.DataFrame(rows).reset_index(drop=True)
    _bump_counter(stats, "ligand_lookup_names_total", len(names_df.index))
    return names_df


def _select_most_frequent_ligand_value(
    values: pd.Series,
    *,
    ambiguity_counter_key: str,
    tie_counter_key: str,
    stats: Optional[dict[str, int]] = None,
) -> Any:
    """Select the most frequent non-null mapping value with ambiguity tracking."""
    frequencies: defaultdict[str, int] = defaultdict(int)
    for value in values:
        clean_value = _clean_ligand_mapping_value(value)
        if clean_value is None:
            continue
        frequencies[clean_value] += 1

    if not frequencies:
        return pd.NA

    if len(frequencies) > 1:
        _bump_counter(stats, ambiguity_counter_key)

    ranked_values = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    top_frequency = ranked_values[0][1]
    top_values = [value for value, frequency in ranked_values if frequency == top_frequency]
    if len(top_values) > 1:
        _bump_counter(stats, tie_counter_key)
        return pd.NA

    return ranked_values[0][0]


def _build_ligand_lookup_dataframe(
    curated_df: pd.DataFrame,
    ligand_info_df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Build the substrate/product lookup DataFrame with inchi/chebi placeholders."""
    names_df = _extract_unique_ligand_names_from_curated_df(curated_df, stats=stats)
    if names_df.empty:
        _bump_counter(stats, "ligand_lookup_rows_total", 0)
        return pd.DataFrame(columns=list(_LIGAND_LOOKUP_COLUMNS))

    ligand_groups: dict[str, pd.DataFrame] = {
        str(group_key): group.copy()
        for group_key, group in ligand_info_df.groupby("_normalized_ligand", sort=False)
    }

    lookup_rows: list[dict[str, Any]] = []
    for row in names_df.itertuples(index=False):
        substrate_or_product_name = str(row.substrate_or_product_name)
        normalized_name = str(row.normalized_name)
        matches = ligand_groups.get(normalized_name)

        inchi_value: Any = pd.NA
        chebi_value: Any = pd.NA
        if matches is not None and not matches.empty:
            inchi_value = _select_most_frequent_ligand_value(
                matches["inchi"],
                ambiguity_counter_key="ligand_lookup_inchi_ambiguity",
                tie_counter_key="ligand_lookup_inchi_tie",
                stats=stats,
            )
            chebi_value = _select_most_frequent_ligand_value(
                matches["chebi"],
                ambiguity_counter_key="ligand_lookup_chebi_ambiguity",
                tie_counter_key="ligand_lookup_chebi_tie",
                stats=stats,
            )

        inchi_missing = _is_missing_scalar(inchi_value)
        chebi_missing = _is_missing_scalar(chebi_value)
        if inchi_missing:
            _bump_counter(stats, "ligand_lookup_unresolved_inchi")
        else:
            _bump_counter(stats, "ligand_lookup_mapped_inchi")

        if chebi_missing:
            _bump_counter(stats, "ligand_lookup_unresolved_chebi")
        else:
            _bump_counter(stats, "ligand_lookup_mapped_chebi")

        if inchi_missing and chebi_missing:
            _bump_counter(stats, "ligand_lookup_unresolved_both")

        lookup_rows.append(
            {
                "substrate_or_product_name": substrate_or_product_name,
                "inchi": pd.NA if inchi_missing else inchi_value,
                "chebi": pd.NA if chebi_missing else chebi_value,
                "smiles": pd.NA,
            }
        )

    lookup_df = pd.DataFrame(lookup_rows, columns=list(_LIGAND_LOOKUP_COLUMNS)).reset_index(drop=True)
    _bump_counter(stats, "ligand_lookup_rows_total", len(lookup_df.index))
    return lookup_df


def _write_ligand_lookup_cache(
    cache_path: Path,
    ligand_lookup_df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> None:
    """Persist the ligand lookup DataFrame as an atomic JSON cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")

    payload: list[dict[str, Any]] = []
    for row in ligand_lookup_df.itertuples(index=False):
        payload.append(
            {
                "substrate_or_product_name": str(row.substrate_or_product_name),
                "inchi": None if _is_missing_scalar(row.inchi) else str(row.inchi),
                "chebi": None if _is_missing_scalar(row.chebi) else str(row.chebi),
                "smiles": None if _is_missing_scalar(row.smiles) else str(row.smiles),
            }
        )

    payload.sort(key=lambda record: _normalize_compound_name(record["substrate_or_product_name"]))
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(cache_path)
        _bump_counter(stats, "ligand_lookup_cache_write_success")
        _bump_counter(stats, "ligand_lookup_cache_rows_written", len(payload))
    except OSError:
        _bump_counter(stats, "ligand_lookup_cache_write_error")
        logger.warning("Could not write ligand lookup cache to %s.", cache_path)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _clean_scalar_text(value: Any) -> Optional[str]:
    """Normalize a scalar-like value into a stripped string or None."""
    if _is_missing_scalar(value):
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_chebi_identifier(chebi_value: Any) -> Optional[str]:
    """Normalize CheBI IDs to CHEBI:<digits> format."""
    text = _clean_scalar_text(chebi_value)
    if text is None:
        return None

    match = re.search(r"(?:CHEBI[:_\s]*)?(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return f"CHEBI:{match.group(1)}"


def _canonicalize_smiles(smiles_value: Any, stats: Optional[dict[str, int]] = None) -> Optional[str]:
    """Canonicalize one SMILES string via RDKit when available."""
    text = _clean_scalar_text(smiles_value)
    if text is None:
        return None

    try:
        from rdkit import Chem
    except ImportError:
        _bump_counter(stats, "smiles_rdkit_import_missing")
        return text

    try:
        molecule = Chem.MolFromSmiles(text)
    except Exception:
        molecule = None

    if molecule is None:
        _bump_counter(stats, "smiles_canonicalization_invalid")
        return None

    try:
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception:
        _bump_counter(stats, "smiles_canonicalization_failed")
        return None


def _smiles_values_match(
    left_smiles: Any,
    right_smiles: Any,
    stats: Optional[dict[str, int]] = None,
) -> bool:
    """Return True if two SMILES values represent the same canonical structure."""
    left_canonical = _canonicalize_smiles(left_smiles, stats=stats)
    right_canonical = _canonicalize_smiles(right_smiles, stats=stats)
    if left_canonical is None or right_canonical is None:
        return False
    return left_canonical == right_canonical


def _extract_smiles_from_chebi_entity(entity: Any) -> Optional[str]:
    """Extract a SMILES value from a bioservices ChEBI entity response."""
    if entity is None:
        return None

    if isinstance(entity, dict):
        for key in ("smiles", "SMILES"):
            value = entity.get(key)
            text = _clean_scalar_text(value)
            if text:
                return text

    for attr_name in ("smiles", "SMILES"):
        value = getattr(entity, attr_name, None)
        text = _clean_scalar_text(value)
        if text:
            return text

    return None


def _resolve_smiles_from_chebi(
    chebi_service: Any,
    chebi_value: Any,
    stats: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Resolve SMILES via ChEBI using bioservices."""
    normalized_chebi = _normalize_chebi_identifier(chebi_value)
    if normalized_chebi is None:
        _bump_counter(stats, "smiles_resolution_chebi_missing_id")
        return None

    _bump_counter(stats, "smiles_resolution_chebi_attempts")
    try:
        entity = chebi_service.getCompleteEntity(normalized_chebi)
    except Exception:
        _bump_counter(stats, "smiles_resolution_chebi_errors")
        logger.debug("ChEBI query failed for %s.", normalized_chebi, exc_info=True)
        return None

    smiles = _canonicalize_smiles(_extract_smiles_from_chebi_entity(entity), stats=stats)
    if smiles is None:
        _bump_counter(stats, "smiles_resolution_chebi_not_found")
        return None

    _bump_counter(stats, "smiles_resolution_chebi_success")
    return smiles


def _extract_smiles_from_molecule_result(
    molecule_result: Any,
    stats: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Extract and canonicalize SMILES from one MoleculeResolver result object."""
    if molecule_result is None:
        return None

    if isinstance(molecule_result, dict):
        smiles_candidate = molecule_result.get("SMILES") or molecule_result.get("smiles")
    else:
        smiles_candidate = getattr(molecule_result, "SMILES", None)
        if smiles_candidate is None:
            smiles_candidate = getattr(molecule_result, "smiles", None)

    return _canonicalize_smiles(smiles_candidate, stats=stats)


def _resolve_smiles_from_molecule_resolver_query(
    molecule_resolver: Any,
    identifier_value: Any,
    identifier_type: str,
    stats: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Resolve one identifier through MoleculeResolver and return canonical SMILES."""
    identifier_text = _clean_scalar_text(identifier_value)
    if identifier_text is None:
        _bump_counter(stats, f"smiles_resolution_resolver_{identifier_type}_missing")
        return None

    _bump_counter(stats, f"smiles_resolution_resolver_{identifier_type}_attempts")
    try:
        molecule = molecule_resolver.find_single_molecule([identifier_text], [identifier_type])
    except Exception:
        _bump_counter(stats, f"smiles_resolution_resolver_{identifier_type}_errors")
        logger.debug(
            "MoleculeResolver query failed for type=%s value=%s.",
            identifier_type,
            identifier_text,
            exc_info=True,
        )
        return None

    smiles = _extract_smiles_from_molecule_result(molecule, stats=stats)
    if smiles is None:
        _bump_counter(stats, f"smiles_resolution_resolver_{identifier_type}_not_found")
        return None

    _bump_counter(stats, f"smiles_resolution_resolver_{identifier_type}_success")
    return smiles


def _resolve_smiles_from_name_and_inchi(
    molecule_resolver: Any,
    molecule_name: Any,
    molecule_inchi: Any,
    stats: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Resolve SMILES through MoleculeResolver name/InChI with name-priority conflicts."""
    name_smiles = _resolve_smiles_from_molecule_resolver_query(
        molecule_resolver,
        identifier_value=molecule_name,
        identifier_type=_MOLECULE_RESOLVER_IDENTIFIER_NAME,
        stats=stats,
    )
    inchi_smiles = _resolve_smiles_from_molecule_resolver_query(
        molecule_resolver,
        identifier_value=molecule_inchi,
        identifier_type=_MOLECULE_RESOLVER_IDENTIFIER_INCHI,
        stats=stats,
    )

    if name_smiles and inchi_smiles:
        if _smiles_values_match(name_smiles, inchi_smiles, stats=stats):
            _bump_counter(stats, "smiles_resolution_resolver_name_inchi_agree")
        else:
            _bump_counter(stats, "smiles_resolution_resolver_name_inchi_conflict")
            _bump_counter(stats, "smiles_resolution_resolver_name_preferred")
        _bump_counter(stats, "smiles_resolution_resolver_success")
        return name_smiles

    if name_smiles:
        _bump_counter(stats, "smiles_resolution_resolver_name_only")
        _bump_counter(stats, "smiles_resolution_resolver_success")
        return name_smiles

    if inchi_smiles:
        _bump_counter(stats, "smiles_resolution_resolver_inchi_only")
        _bump_counter(stats, "smiles_resolution_resolver_success")
        return inchi_smiles

    _bump_counter(stats, "smiles_resolution_resolver_unresolved")
    return None


def _enrich_ligand_lookup_with_smiles(
    ligand_lookup_df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Populate lookup smiles with ChEBI-first and MoleculeResolver fallback logic."""
    if ligand_lookup_df.empty:
        _bump_counter(stats, "smiles_resolution_rows_total", 0)
        return ligand_lookup_df.copy()

    try:
        chebi_module = importlib.import_module("bioservices")
        ChEBI = getattr(chebi_module, "ChEBI")
    except Exception as exc:
        raise RuntimeError(
            "bioservices is required for ChEBI SMILES resolution. "
            "Install dependencies in the kcatbench environment."
        ) from exc

    enriched_df = ligand_lookup_df.copy()
    _bump_counter(stats, "smiles_resolution_rows_total", len(enriched_df.index))
    chebi_service = ChEBI(verbose=False)

    unresolved_indices: list[int] = []
    for row in enriched_df.itertuples(index=True):
        existing_smiles = _canonicalize_smiles(row.smiles, stats=stats)
        if existing_smiles:
            enriched_df.at[row.Index, "smiles"] = existing_smiles
            _bump_counter(stats, "smiles_resolution_preexisting")
            continue

        chebi_smiles = _resolve_smiles_from_chebi(
            chebi_service=chebi_service,
            chebi_value=row.chebi,
            stats=stats,
        )
        if chebi_smiles:
            enriched_df.at[row.Index, "smiles"] = chebi_smiles
            _bump_counter(stats, "smiles_resolution_source_chebi")
            _bump_counter(stats, "smiles_resolution_total_resolved")
            continue

        unresolved_indices.append(int(row.Index))

    if unresolved_indices:
        try:
            resolver_module = importlib.import_module("moleculeresolver")
            MoleculeResolver = getattr(resolver_module, "MoleculeResolver")
        except Exception as exc:
            raise RuntimeError(
                "molecule-resolver is required for fallback SMILES resolution. "
                "Install dependencies in the kcatbench environment."
            ) from exc

        chemeo_api_key = load_chemeo_api_key(required=True)
        with MoleculeResolver(available_service_API_keys={"chemeo": chemeo_api_key}) as molecule_resolver:
            for row_index in unresolved_indices:
                row = enriched_df.loc[row_index]
                resolved_smiles = _resolve_smiles_from_name_and_inchi(
                    molecule_resolver=molecule_resolver,
                    molecule_name=row.get("substrate_or_product_name"),
                    molecule_inchi=row.get("inchi"),
                    stats=stats,
                )
                if resolved_smiles:
                    enriched_df.at[row_index, "smiles"] = resolved_smiles
                    _bump_counter(stats, "smiles_resolution_source_molecule_resolver")
                    _bump_counter(stats, "smiles_resolution_total_resolved")
                else:
                    _bump_counter(stats, "smiles_resolution_total_unresolved")

    resolved_total = int((~enriched_df["smiles"].apply(_is_missing_scalar)).sum())
    unresolved_total = int(len(enriched_df.index) - resolved_total)
    _bump_counter(stats, "smiles_resolution_rows_resolved_final", resolved_total)
    _bump_counter(stats, "smiles_resolution_rows_unresolved_final", unresolved_total)
    return enriched_df


def _build_ligand_smiles_maps(
    ligand_lookup_df: pd.DataFrame,
) -> tuple[dict[str, Optional[str]], dict[str, Optional[str]]]:
    """Build exact and normalized name-to-smiles maps from lookup rows."""
    exact_map: dict[str, Optional[str]] = {}
    normalized_map: dict[str, Optional[str]] = {}

    for row in ligand_lookup_df.itertuples(index=False):
        name = _clean_scalar_text(row.substrate_or_product_name)
        if name is None:
            continue

        smiles = _canonicalize_smiles(row.smiles)
        exact_map[name] = smiles

        normalized_name = _normalize_compound_name(name)
        existing = normalized_map.get(normalized_name)
        if existing is None and smiles is not None:
            normalized_map[normalized_name] = smiles
        elif normalized_name not in normalized_map:
            normalized_map[normalized_name] = smiles

    return exact_map, normalized_map


def _map_names_to_smiles(
    names: list[Any],
    exact_map: dict[str, Optional[str]],
    normalized_map: dict[str, Optional[str]],
    *,
    stats: Optional[dict[str, int]] = None,
    counter_prefix: str,
) -> list[Optional[str]]:
    """Map a list of compound names to SMILES while preserving list length."""
    smiles_values: list[Optional[str]] = []
    for name in names:
        clean_name = _clean_scalar_text(name)
        if clean_name is None:
            smiles_values.append(None)
            _bump_counter(stats, f"{counter_prefix}_missing_name")
            continue

        smiles = exact_map.get(clean_name)
        if smiles is None:
            smiles = normalized_map.get(_normalize_compound_name(clean_name))

        smiles_values.append(smiles)
        if smiles is None:
            _bump_counter(stats, f"{counter_prefix}_unresolved")
        else:
            _bump_counter(stats, f"{counter_prefix}_resolved")

    _bump_counter(stats, f"{counter_prefix}_total", len(smiles_values))
    return smiles_values


def _apply_ligand_lookup_smiles_to_dataset(
    df: pd.DataFrame,
    ligand_lookup_df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Replace substrates/products with SMILES and preserve names in *_names columns."""
    transformed_df = df.copy()
    exact_map, normalized_map = _build_ligand_smiles_maps(ligand_lookup_df)

    transformed_df["substrates_names"] = transformed_df["substrates"].apply(
        lambda values: list(values) if isinstance(values, list) else []
    )
    transformed_df["products_names"] = transformed_df["products"].apply(
        lambda values: list(values) if isinstance(values, list) else []
    )

    transformed_df["substrates"] = transformed_df["substrates_names"].apply(
        lambda names: _map_names_to_smiles(
            names,
            exact_map,
            normalized_map,
            stats=stats,
            counter_prefix="smiles_dataset_substrates",
        )
    )
    transformed_df["products"] = transformed_df["products_names"].apply(
        lambda names: _map_names_to_smiles(
            names,
            exact_map,
            normalized_map,
            stats=stats,
            counter_prefix="smiles_dataset_products",
        )
    )
    _bump_counter(stats, "smiles_dataset_rows_transformed", len(transformed_df.index))
    return transformed_df


def _sanitize_amino_acid_sequence(sequence_text: str) -> Optional[str]:
    """Normalize a sequence string and validate it as amino-acid content."""
    sequence = re.sub(r"\s+", "", sequence_text).upper().strip()
    if not sequence:
        return None
    if not re.fullmatch(r"[A-Z*]+", sequence):
        return None
    return sequence


def _parse_fasta_sequence(fasta_payload: str) -> Optional[str]:
    """Parse FASTA text and return the normalized sequence string."""
    sequence_lines: list[str] = []
    for line in fasta_payload.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        sequence_lines.append(line)

    if not sequence_lines:
        return None

    return _sanitize_amino_acid_sequence("".join(sequence_lines))


def _load_uniprot_sequence_cache(
    cache_path: Path,
    stats: Optional[dict[str, int]] = None,
) -> dict[str, Optional[str]]:
    """Load a local UniProt sequence cache file if it exists."""
    if not cache_path.is_file():
        _bump_counter(stats, "sequence_cache_file_missing")
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        _bump_counter(stats, "sequence_cache_load_error")
        logger.warning("Could not load UniProt sequence cache at %s. Using empty cache.", cache_path)
        return {}

    if not isinstance(payload, dict):
        _bump_counter(stats, "sequence_cache_invalid_payload")
        logger.warning("UniProt sequence cache payload is not a JSON object at %s.", cache_path)
        return {}

    cache: dict[str, Optional[str]] = {}
    for raw_accession, raw_sequence in payload.items():
        accession = _normalize_uniprot_accession(raw_accession)
        if accession is None:
            _bump_counter(stats, "sequence_cache_invalid_accession_keys")
            continue

        if isinstance(raw_sequence, dict):
            raw_sequence = raw_sequence.get("sequence")

        if raw_sequence is None:
            cache[accession] = None
            continue

        sequence = _sanitize_amino_acid_sequence(str(raw_sequence))
        if sequence is None:
            _bump_counter(stats, "sequence_cache_invalid_sequence_values")
            continue

        cache[accession] = sequence

    _bump_counter(stats, "sequence_cache_entries_loaded", len(cache))
    return cache


def _write_uniprot_sequence_cache(
    cache_path: Path,
    cache: dict[str, Optional[str]],
    stats: Optional[dict[str, int]] = None,
) -> None:
    """Persist the UniProt sequence cache with an atomic file replace."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    ordered_payload = {accession: cache[accession] for accession in sorted(cache)}

    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(ordered_payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(cache_path)
        _bump_counter(stats, "sequence_cache_write_success")
        _bump_counter(stats, "sequence_cache_entries_written", len(ordered_payload))
    except OSError:
        _bump_counter(stats, "sequence_cache_write_error")
        logger.warning("Could not write UniProt sequence cache to %s.", cache_path)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _fetch_uniprot_sequence(
    accession: str,
    request_timeout_seconds: float = 20.0,
    request_retries: int = 2,
    stats: Optional[dict[str, int]] = None,
) -> tuple[Optional[str], bool]:
    """Fetch one UniProt sequence and return (sequence, cacheable_result)."""
    normalized_accession = _normalize_uniprot_accession(accession)
    if normalized_accession is None:
        _bump_counter(stats, "sequence_lookup_invalid_accession")
        return None, False

    max_retries = max(int(request_retries), 0)
    request_timeout = float(request_timeout_seconds)
    lookup_url = _UNIPROT_FASTA_URL_TEMPLATE.format(accession=quote(normalized_accession))

    for attempt in range(max_retries + 1):
        _bump_counter(stats, "sequence_api_attempts")
        request = Request(
            lookup_url,
            headers={
                "Accept": "text/x-fasta",
                "User-Agent": "kcatbench/0.1",
            },
        )

        try:
            with urlopen(request, timeout=request_timeout) as response:
                fasta_payload = response.read().decode("utf-8", errors="replace")

            sequence = _parse_fasta_sequence(fasta_payload)
            if sequence is None:
                _bump_counter(stats, "sequence_api_invalid_fasta")
                return None, False

            _bump_counter(stats, "sequence_api_success")
            return sequence, True
        except HTTPError as exc:
            if exc.code == 404:
                _bump_counter(stats, "sequence_api_not_found")
                return None, True
            if exc.code in {408, 425, 429, 500, 502, 503, 504} and attempt < max_retries:
                _bump_counter(stats, "sequence_api_retryable_http")
                continue
            _bump_counter(stats, "sequence_api_http_error")
            return None, False
        except (TimeoutError, URLError):
            if attempt < max_retries:
                _bump_counter(stats, "sequence_api_retryable_network")
                continue
            _bump_counter(stats, "sequence_api_network_error")
            return None, False
        except ValueError:
            _bump_counter(stats, "sequence_api_value_error")
            return None, False

    _bump_counter(stats, "sequence_api_retry_exhausted")
    return None, False


def _extract_protein_map(
    ec_record: dict[str, list[str]],
    stats: Optional[dict[str, int]] = None,
) -> dict[str, dict[str, Any]]:
    """Build a protein map from PR entries keyed by BRENDA protein IDs."""
    protein_map: dict[str, dict[str, Any]] = {}
    for pr_line in ec_record.get("PR", []):
        _bump_counter(stats, "pr_entries_total")
        protein_ids, remainder = _strip_leading_protein_tag(pr_line)
        if not protein_ids:
            _bump_counter(stats, "pr_entries_missing_protein_tag")
            continue
        if len(protein_ids) > 1:
            _bump_counter(stats, "pr_entries_multi_protein_tag")

        uniprot_id = _extract_uniprot_id(remainder)

        organism = re.sub(r"<[^>]+>", "", remainder)
        organism = re.sub(r"\{[^}]*\}", "", organism)
        organism = re.sub(r"\([^)]+\)", "", organism)
        if uniprot_id:
            organism = organism.replace(uniprot_id, "")
        organism = re.sub(r"\s+", " ", organism).strip()

        for protein_id in protein_ids:
            if protein_id in protein_map:
                _bump_counter(stats, "pr_duplicate_protein_id_overwrite")
            protein_map[protein_id] = {
                "protein_id": protein_id,
                "organism": organism,
                "UniProt_ID": uniprot_id,
                "is_mutant": _is_mutant_text(remainder),
            }

    return protein_map


def _split_reaction_equation(equation_text: str) -> tuple[list[str], list[str]]:
    """Split a reaction equation text into substrate and product name lists."""
    sides = re.split(r"\s*(?:<=>|=>|<=|=|->|<-)\s*", equation_text, maxsplit=1)
    if len(sides) != 2:
        return [], []

    substrates = [item.strip() for item in _REACTION_PLUS_SPLIT_PATTERN.split(sides[0]) if item.strip()]
    products = [item.strip() for item in _REACTION_PLUS_SPLIT_PATTERN.split(sides[1]) if item.strip()]
    return substrates, products


def _normalize_reaction_equation_text(
    equation_payload: str,
    stats: Optional[dict[str, int]] = None,
) -> tuple[str, Optional[str], list[str]]:
    """Normalize a reaction equation and extract non-compound annotation metadata."""
    equation_text = re.sub(r"\(\s*#[\d,\s]+#[^)]*\)", "", equation_payload)
    equation_text = re.sub(r"<[^>]+>", "", equation_text)

    reversibility_markers = [
        marker.strip().lower()
        for marker in re.findall(r"\{([^{}]*)\}", equation_text)
        if marker.strip()
    ]
    reversibility_markers = _deduplicate_values(reversibility_markers)
    if reversibility_markers:
        _bump_counter(stats, "reaction_entries_with_reversibility_marker")
    equation_text = re.sub(r"\{[^{}]*\}", "", equation_text)

    pipe_comments = [comment.strip() for comment in re.findall(r"\|([^|]+)\|", equation_text) if comment.strip()]
    pipe_comments = _deduplicate_values(pipe_comments)
    if pipe_comments:
        _bump_counter(stats, "reaction_entries_with_pipe_comment")
    equation_text = re.sub(r"\|[^|]*\|", "", equation_text)
    if "|" in equation_text:
        _bump_counter(stats, "reaction_entries_with_unbalanced_pipe")
        equation_text = equation_text.replace("|", " ")

    equation_text = re.sub(r"\s+", " ", equation_text).strip(" ;,")
    reversibility = ",".join(reversibility_markers) if reversibility_markers else None
    return equation_text, reversibility, pipe_comments


def _clean_reaction_token(raw_token: str) -> tuple[Optional[str], Optional[str]]:
    """Clean one reaction-side token and optionally extract an inline note."""
    token = raw_token.strip()
    if not token:
        return None, None

    token = re.sub(r"<[^>]+>", "", token)
    token = re.sub(r"#[\d,\s]+#", "", token)
    token = token.replace("|", " ")

    inline_note: Optional[str] = None
    if " ; " in token:
        left, right = token.split(" ; ", 1)
        token = left.strip()
        right = right.strip(" ;,")
        if right:
            inline_note = right

    unknown_with_dash = re.match(r"^\?\s*-(.+)$", token)
    if unknown_with_dash:
        token = "?"
        trailing_note = unknown_with_dash.group(1).strip()
        if trailing_note:
            inline_note = f"{inline_note} | {trailing_note}" if inline_note else trailing_note

    if " -" in token and not token.startswith("-"):
        token_head, token_tail = token.split(" -", 1)
        token_head = token_head.strip()
        token_tail = token_tail.strip()
        if token_head and token_tail:
            token = token_head
            inline_note = f"{inline_note} | {token_tail}" if inline_note else token_tail

    if token.startswith("-"):
        token = token[1:].strip()

    token = re.sub(r"\s+", " ", token).strip(" ;,")
    if token.endswith(")") and token.count("(") < token.count(")"):
        token = token[:-1].strip()
    if token.startswith("(") and token.count("(") > token.count(")"):
        token = token[1:].strip()

    token = token.strip(" ;,")
    if token in {"", "(", ")"}:
        return None, inline_note

    return token, inline_note


def _clean_reaction_side_tokens(
    raw_tokens: list[str],
    *,
    drop_unknown_token: bool,
    stats: Optional[dict[str, int]] = None,
) -> tuple[list[str], list[str]]:
    """Clean a reaction side token list and collect inline annotation notes."""
    cleaned_tokens: list[str] = []
    inline_notes: list[str] = []

    for raw_token in raw_tokens:
        cleaned_token, inline_note = _clean_reaction_token(raw_token)

        if inline_note:
            inline_notes.append(inline_note)
            _bump_counter(stats, "reaction_tokens_with_inline_note")

        if cleaned_token is None:
            if raw_token.strip():
                _bump_counter(stats, "reaction_tokens_dropped_debris")
            continue

        if drop_unknown_token and cleaned_token == "?":
            _bump_counter(stats, "reaction_unknown_product_tokens_removed")
            continue

        if cleaned_token.endswith("+"):
            _bump_counter(stats, "reaction_tokens_with_charge_suffix")

        cleaned_tokens.append(cleaned_token)

    return cleaned_tokens, _deduplicate_values(inline_notes)


def _extract_reaction_map(
    ec_record: dict[str, list[str]],
    stats: Optional[dict[str, int]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a protein-indexed reaction map from SP and NSP entries."""
    reaction_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_key in ("SP", "NSP"):
        for source_line in ec_record.get(source_key, []):
            _bump_counter(stats, "reaction_entries_total")
            _bump_counter(stats, f"reaction_entries_{source_key.lower()}")

            leading_protein_ids, equation_payload = _strip_leading_protein_tag(source_line)
            if leading_protein_ids:
                _bump_counter(stats, "reaction_entries_with_leading_protein_tag")
            else:
                _bump_counter(stats, "reaction_entries_without_leading_protein_tag")

            inline_protein_ids = _extract_all_protein_ids(source_line)
            protein_ids = leading_protein_ids if leading_protein_ids else inline_protein_ids
            protein_ids = _deduplicate_values(protein_ids)
            if not protein_ids:
                _bump_counter(stats, "reaction_entries_missing_protein_ids")
                continue
            if len(protein_ids) > 1:
                _bump_counter(stats, "reaction_entries_multi_protein_ids")

            equation_text, reversibility, pipe_comments = _normalize_reaction_equation_text(
                equation_payload,
                stats=stats,
            )
            if not equation_text:
                _bump_counter(stats, "reaction_entries_missing_equation_text")
                continue

            substrates_raw, products_raw = _split_reaction_equation(equation_text)
            substrates, substrate_notes = _clean_reaction_side_tokens(
                substrates_raw,
                drop_unknown_token=False,
                stats=stats,
            )
            products, product_notes = _clean_reaction_side_tokens(
                products_raw,
                drop_unknown_token=True,
                stats=stats,
            )

            reaction_annotation_parts = _deduplicate_values(pipe_comments + substrate_notes + product_notes)
            reaction_annotation = " | ".join(reaction_annotation_parts)
            if reaction_annotation:
                _bump_counter(stats, "reaction_entries_with_annotation_metadata")

            if any(token.strip() == "?" for token in products_raw):
                _bump_counter(stats, "reaction_entries_with_unknown_product_token")
                if not products:
                    _bump_counter(stats, "reaction_entries_unknown_products_only")

            if not substrates and not products:
                _bump_counter(stats, "reaction_entries_unparsed_equation")
                continue
            if not substrates:
                _bump_counter(stats, "reaction_entries_missing_substrates")
            if not products:
                _bump_counter(stats, "reaction_entries_missing_products")

            reaction_record = {
                "substrates": substrates,
                "products": products,
                "reaction_equation": equation_text,
                "reaction_source": source_key,
                "reaction_reversibility": reversibility,
                "reaction_annotation": reaction_annotation,
            }
            for protein_id in protein_ids:
                reaction_map[protein_id].append(reaction_record)
            _bump_counter(stats, "reaction_records_created", len(protein_ids))

    return dict(reaction_map)


def _extract_tn_context(tn_line: str, stats: Optional[dict[str, int]] = None) -> Optional[dict[str, Any]]:
    """Parse one TN entry into value, substrate, protein links, and metadata."""
    _bump_counter(stats, "tn_entries_total")

    leading_protein_ids, tn_payload = _strip_leading_protein_tag(tn_line)
    if leading_protein_ids:
        _bump_counter(stats, "tn_with_leading_protein_tag")
    else:
        _bump_counter(stats, "tn_without_leading_protein_tag")

    value_match = re.search(r"^\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", tn_payload)
    if not value_match:
        _bump_counter(stats, "tn_missing_numeric_value")
        return None

    kcat_value = float(value_match.group(1))
    if kcat_value <= 0:
        _bump_counter(stats, "tn_non_positive_kcat")
        return None

    substrate_match = re.search(r"\{([^{}]+)\}", tn_payload)
    kcat_substrate = substrate_match.group(1).strip() if substrate_match else None
    if not kcat_substrate and "{" in tn_payload:
        _bump_counter(stats, "tn_unclosed_or_nested_substrate_brace")
        inferred_substrate = tn_payload.split("{", 1)[1]
        inferred_substrate = inferred_substrate.split(" (#", 1)[0]
        inferred_substrate = inferred_substrate.split(" <", 1)[0]
        inferred_substrate = inferred_substrate.strip()
        if inferred_substrate:
            kcat_substrate = inferred_substrate
            _bump_counter(stats, "tn_substrate_inferred_without_closing_brace")

    protein_ids = _extract_all_protein_ids(tn_line)
    if not protein_ids and leading_protein_ids:
        protein_ids = leading_protein_ids
    if not protein_ids:
        _bump_counter(stats, "tn_missing_protein_ids")
    if len(protein_ids) > 1:
        _bump_counter(stats, "tn_multi_protein_ids")

    comment_blocks: list[str] = []
    for raw_block in re.findall(r"\(([^)]*)\)", tn_line):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        clean_block = re.sub(r"#[\d,\s]+#", "", raw_block)
        clean_block = re.sub(r"<[^>]+>", "", clean_block)
        clean_block = re.sub(r"\s+", " ", clean_block).strip(" ;,")
        if clean_block:
            comment_blocks.append(clean_block)
    comment_text = " | ".join(comment_blocks)
    if not comment_text:
        _bump_counter(stats, "tn_without_clean_comment")

    references = [ref.strip() for ref in re.findall(r"<([^>]+)>", tn_line) if ref.strip()]
    references = sorted(set(references))

    ph_match = re.search(r"\bpH\s*([0-9]+(?:\.[0-9]+)?)", comment_text, re.IGNORECASE)
    pH_value = float(ph_match.group(1)) if ph_match else None

    temp_match = re.search(
        r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:°|º)?\s*C",
        comment_text,
        re.IGNORECASE,
    )
    temperature = float(temp_match.group(1)) if temp_match else None

    return {
        "experimental_kcat": kcat_value,
        "kcat_substrate_name": kcat_substrate,
        "protein_ids": protein_ids,
        "tn_comment": comment_text,
        "references": references,
        "pH": pH_value,
        "temperature": temperature,
        "tn_raw": tn_line,
    }


def _select_reactions_for_tn(
    protein_reactions: list[dict[str, Any]],
    kcat_substrate_name: Optional[str],
) -> list[dict[str, Any]]:
    """Select reaction candidates for a TN entry and protein ID pair."""
    if not protein_reactions:
        return []

    if not kcat_substrate_name:
        return protein_reactions if len(protein_reactions) > 1 else [protein_reactions[0]]

    normalized_target = _normalize_compound_name(kcat_substrate_name)
    matched = [
        reaction
        for reaction in protein_reactions
        if any(
            _normalize_compound_name(substrate) == normalized_target
            for substrate in reaction.get("substrates", [])
        )
    ]
    if matched:
        return matched

    if len(protein_reactions) == 1:
        return [protein_reactions[0]]

    return []


def extract_benchmark_data_from_record(
    ec_record: dict[str, list[str]],
    require_uniprot: bool = False,
    stats: Optional[dict[str, int]] = None,
) -> list[dict[str, Any]]:
    """Extract row-wise, non-aggregated benchmark entries from one EC record.

    Each output row represents a single TN observation linked to one protein ID.
    If a TN entry matches multiple reactions for the same protein, one row is
    generated per matched reaction.

    Args:
        ec_record: Parsed EC record.
        require_uniprot: Whether rows without UniProt IDs should be excluded.

    Returns:
        List of row dictionaries for DataFrame construction.
    """
    ec_number = (ec_record.get("ID") or [""])[0].strip()
    if not ec_number:
        _bump_counter(stats, "ec_missing_id")
        return []
    _bump_counter(stats, "ec_with_id")

    protein_map = _extract_protein_map(ec_record, stats=stats)
    reaction_map = _extract_reaction_map(ec_record, stats=stats)
    _bump_counter(stats, "pr_proteins_indexed", len(protein_map))
    rows: list[dict[str, Any]] = []

    for tn_line in ec_record.get("TN", []):
        tn_context = _extract_tn_context(tn_line, stats=stats)
        if not tn_context:
            continue
        if _is_mutant_text(tn_context["tn_comment"]):
            _bump_counter(stats, "tn_dropped_mutant_comment")
            continue
        if not tn_context["protein_ids"]:
            _bump_counter(stats, "tn_dropped_no_protein_links")
            continue

        for protein_id in tn_context["protein_ids"]:
            _bump_counter(stats, "tn_protein_links_total")
            protein_info = protein_map.get(protein_id)
            if not protein_info:
                _bump_counter(stats, "drop_missing_pr_protein_info")
                continue
            if protein_info["is_mutant"]:
                _bump_counter(stats, "drop_pr_marked_mutant")
                continue
            if require_uniprot and not protein_info["UniProt_ID"]:
                _bump_counter(stats, "drop_missing_uniprot_required")
                continue

            selected_reactions = _select_reactions_for_tn(
                reaction_map.get(protein_id, []),
                tn_context["kcat_substrate_name"],
            )

            if not selected_reactions:
                fallback_substrates = [tn_context["kcat_substrate_name"]] if tn_context["kcat_substrate_name"] else []
                selected_reactions = [
                    {
                        "substrates": fallback_substrates,
                        "products": [],
                        "reaction_equation": "",
                    }
                ]
                _bump_counter(stats, "fallback_reaction_assignments")

            for reaction in selected_reactions:
                ordered_substrates = _ensure_kcat_substrate_first(
                    list(reaction.get("substrates", [])),
                    tn_context["kcat_substrate_name"],
                    stats=stats,
                )
                if not ordered_substrates:
                    _bump_counter(stats, "rows_with_empty_substrates")
                if not reaction.get("products", []):
                    _bump_counter(stats, "rows_with_empty_products")
                rows.append(
                    {
                        "source": "brenda_txt",
                        "ec_number": ec_number,
                        "protein_id": protein_id,
                        "organism": protein_info["organism"],
                        "UniProt_ID": protein_info["UniProt_ID"],
                        "sequence": pd.NA,
                        "kcat_substrate_name": tn_context["kcat_substrate_name"],
                        "substrates": ordered_substrates,
                        "products": list(reaction.get("products", [])),
                        "reaction_equation": reaction.get("reaction_equation", ""),
                        "reaction_source": reaction.get("reaction_source", "fallback"),
                        "reaction_reversibility": reaction.get("reaction_reversibility", pd.NA),
                        "reaction_annotation": reaction.get("reaction_annotation", pd.NA),
                        "experimental_kcat": tn_context["experimental_kcat"],
                        "temperature": tn_context["temperature"],
                        "pH": tn_context["pH"],
                        "references": list(tn_context["references"]),
                        "tn_comment": tn_context["tn_comment"],
                        "tn_raw": tn_context["tn_raw"],
                    }
                )
                _bump_counter(stats, "rows_created")
                _bump_counter(
                    stats,
                    f"rows_created_from_{str(reaction.get('reaction_source', 'fallback')).lower()}",
                )

    return rows


def _normalize_output_dataframe(
    rows: list[dict[str, Any]],
    require_uniprot: bool,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Normalize extracted rows into the final BRENDA benchmark DataFrame."""
    _bump_counter(stats, "rows_input_total", len(rows))

    if not rows:
        _bump_counter(stats, "rows_output_total", 0)
        return pd.DataFrame(columns=_DEFAULT_COLUMNS)

    df = pd.DataFrame(rows)

    if require_uniprot:
        rows_before_uniprot_filter = len(df.index)
        df = df[df["UniProt_ID"].fillna("").astype(str).str.strip() != ""].copy()
        _bump_counter(
            stats,
            "rows_dropped_post_uniprot_filter",
            rows_before_uniprot_filter - len(df.index),
        )

    if df.empty:
        _bump_counter(stats, "rows_output_total", 0)
        return pd.DataFrame(columns=_DEFAULT_COLUMNS)

    for list_column in (
        "substrates",
        "products",
        "substrates_names",
        "products_names",
        "references",
    ):
        if list_column not in df.columns:
            continue
        df[list_column] = df[list_column].apply(
            lambda value: value if isinstance(value, list) else []
        )

    if "sequence" not in df.columns:
        df["sequence"] = pd.NA

    for column in _DEFAULT_COLUMNS:
        if column not in df.columns:
            if column in {"substrates", "products", "substrates_names", "products_names", "references"}:
                df[column] = [[] for _ in range(len(df.index))]
            elif column == "sequence":
                df[column] = pd.NA
            else:
                df[column] = pd.NA

    df = df[_DEFAULT_COLUMNS].reset_index(drop=True)
    _bump_counter(stats, "rows_output_total", len(df.index))
    return df


def _is_missing_scalar(value: Any) -> bool:
    """Return True when value should be treated as a missing scalar."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _enrich_sequences_from_uniprot(
    df: pd.DataFrame,
    *,
    sequence_cache_path: Path,
    enrich_sequences: bool = True,
    sequence_request_timeout_seconds: float = 20.0,
    sequence_request_retries: int = 2,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Populate missing sequence values from UniProt IDs using cache and REST calls."""
    _bump_counter(stats, "sequence_enrichment_input_rows", len(df.index))

    if not enrich_sequences:
        _bump_counter(stats, "sequence_enrichment_skipped_disabled")
        _bump_counter(stats, "sequence_enrichment_output_rows", len(df.index))
        return df

    if df.empty:
        _bump_counter(stats, "sequence_enrichment_skipped_empty_dataframe")
        _bump_counter(stats, "sequence_enrichment_output_rows", len(df.index))
        return df

    if "UniProt_ID" not in df.columns:
        _bump_counter(stats, "sequence_enrichment_skipped_missing_uniprot_column")
        _bump_counter(stats, "sequence_enrichment_output_rows", len(df.index))
        return df

    if "sequence" not in df.columns:
        df["sequence"] = pd.NA

    normalized_uniprot_ids = df["UniProt_ID"].apply(_normalize_uniprot_accession)
    _bump_counter(stats, "sequence_rows_with_uniprot", int(normalized_uniprot_ids.notna().sum()))

    sequence_missing_mask = df["sequence"].apply(_is_missing_scalar)
    candidate_mask = normalized_uniprot_ids.notna() & sequence_missing_mask
    _bump_counter(stats, "sequence_rows_missing_before_enrichment", int(candidate_mask.sum()))

    candidate_ids = _deduplicate_values(
        [accession for accession in normalized_uniprot_ids[candidate_mask].tolist() if accession]
    )
    _bump_counter(stats, "sequence_uniprot_ids_candidates", len(candidate_ids))

    if not candidate_ids:
        _bump_counter(stats, "sequence_rows_missing_after_enrichment", int(sequence_missing_mask.sum()))
        _bump_counter(stats, "sequence_enrichment_output_rows", len(df.index))
        return df

    cache = _load_uniprot_sequence_cache(sequence_cache_path, stats=stats)

    cache_updated = False
    resolved_sequences: dict[str, Optional[str]] = {}

    for accession in candidate_ids:
        if accession in cache:
            _bump_counter(stats, "sequence_cache_hits")
            resolved_sequences[accession] = cache[accession]
            continue

        _bump_counter(stats, "sequence_cache_misses")
        sequence, cacheable = _fetch_uniprot_sequence(
            accession,
            request_timeout_seconds=sequence_request_timeout_seconds,
            request_retries=sequence_request_retries,
            stats=stats,
        )
        resolved_sequences[accession] = sequence

        if cacheable:
            cache[accession] = sequence
            cache_updated = True

    if cache_updated:
        _write_uniprot_sequence_cache(sequence_cache_path, cache, stats=stats)

    resolved_id_count = sum(1 for accession in candidate_ids if resolved_sequences.get(accession))
    _bump_counter(stats, "sequence_uniprot_ids_resolved", resolved_id_count)
    _bump_counter(stats, "sequence_uniprot_ids_unresolved", len(candidate_ids) - resolved_id_count)

    rows_populated = 0
    for row_index, accession in normalized_uniprot_ids[candidate_mask].items():
        sequence = resolved_sequences.get(accession)
        if not sequence:
            continue
        df.at[row_index, "sequence"] = sequence
        rows_populated += 1

    _bump_counter(stats, "sequence_rows_populated", rows_populated)
    missing_after_mask = df["sequence"].apply(_is_missing_scalar)
    _bump_counter(stats, "sequence_rows_missing_after_enrichment", int(missing_after_mask.sum()))
    _bump_counter(stats, "sequence_enrichment_output_rows", len(df.index))
    return df


def _make_hashable_value(value: Any) -> Any:
    """Convert values into stable hashable keys for strict row comparisons."""
    if isinstance(value, list):
        return tuple(_make_hashable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_make_hashable_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _make_hashable_value(val)) for key, val in value.items()))
    if _is_missing_scalar(value):
        return ("__MISSING__",)
    return value


def _normalize_reversibility_value(value: Any) -> Optional[str]:
    """Normalize reaction reversibility values for strict dedup decisions."""
    if _is_missing_scalar(value):
        return None
    text = str(value).strip()
    return text if text else None


def _apply_strict_reversibility_dedup(
    df: pd.DataFrame,
    stats: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Merge rows only when they differ exclusively by empty vs non-empty reversibility.

    The merge rule is intentionally strict:
    - all columns except reaction_reversibility must match exactly,
    - at least one row must have empty/None reversibility,
    - at least one row must have a non-empty reversibility value,

    Rows are merged by dropping only the empty/None-reversibility members.
    """
    _bump_counter(stats, "reversibility_dedup_input_rows", len(df.index))
    if df.empty or "reaction_reversibility" not in df.columns:
        _bump_counter(stats, "reversibility_dedup_output_rows", len(df.index))
        return df

    key_columns = [column for column in df.columns if column != "reaction_reversibility"]
    grouped_indices: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)

    for row_index, row in df.iterrows():
        key = tuple(_make_hashable_value(row[column]) for column in key_columns)
        grouped_indices[key].append(int(row_index))

    _bump_counter(stats, "reversibility_dedup_groups_total", len(grouped_indices))
    keep_rows = pd.Series(True, index=df.index)

    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue

        _bump_counter(stats, "reversibility_dedup_groups_with_duplicates")
        normalized_values = [
            _normalize_reversibility_value(df.at[row_index, "reaction_reversibility"])
            for row_index in indices
        ]
        empty_indices = [
            row_index
            for row_index, value in zip(indices, normalized_values)
            if value is None
        ]
        non_empty_indices = [
            row_index
            for row_index, value in zip(indices, normalized_values)
            if value is not None
        ]
        if not empty_indices:
            _bump_counter(stats, "reversibility_dedup_skipped_no_empty")
            continue
        if not non_empty_indices:
            _bump_counter(stats, "reversibility_dedup_skipped_no_non_empty")
            continue

        distinct_non_empty_values = {
            value
            for value in normalized_values
            if value is not None
        }
        if len(distinct_non_empty_values) > 1:
            _bump_counter(stats, "reversibility_dedup_groups_with_conflicting_non_empty")

        for row_index in empty_indices:
            keep_rows.at[row_index] = False

        _bump_counter(stats, "reversibility_dedup_groups_merged")
        _bump_counter(stats, "reversibility_dedup_rows_removed", len(empty_indices))

    deduped_df = df[keep_rows].reset_index(drop=True)
    _bump_counter(stats, "reversibility_dedup_output_rows", len(deduped_df.index))
    return deduped_df


def _write_brenda_artifacts(df: pd.DataFrame, target_dir: Path, output_basename: str) -> tuple[Path, Path]:
    """Write DataFrame artifacts to CSV and pickle files under target_dir."""
    csv_path = target_dir / f"{output_basename}.csv"
    pkl_path = target_dir / f"{output_basename}.pkl"

    df.to_pickle(pkl_path)

    csv_df = df.copy()
    csv_df["substrates"] = csv_df["substrates"].apply(json.dumps)
    csv_df["products"] = csv_df["products"].apply(json.dumps)
    csv_df["substrates_names"] = csv_df["substrates_names"].apply(json.dumps)
    csv_df["products_names"] = csv_df["products_names"].apply(json.dumps)
    csv_df["references"] = csv_df["references"].apply(json.dumps)
    csv_df.to_csv(csv_path, index=False)

    return csv_path, pkl_path


def brenda_build_db(
    path: Path = (DATA_DIR / _DEFAULT_BRENDA_SUBDIR),
    flatfile_path: str | Path | None = None,
    output_basename: str = _DEFAULT_PROCESSED_BASENAME,
    require_uniprot: bool = False,
    enrich_sequences: bool = True,
    sequence_cache_path: str | Path | None = None,
    sequence_request_timeout_seconds: float = 20.0,
    sequence_request_retries: int = 2,
    build_ligand_lookup: bool = True,
    resolve_ligand_smiles: bool = True,
    ligand_info_path: str | Path | None = None,
    ligand_lookup_cache_path: str | Path | None = None,
    write_artifacts: bool = True,
    enable_logging: bool = False,
    log_every_n_records: int = 0,
) -> pd.DataFrame:
    """Build the BRENDA benchmark DataFrame from an existing local TXT flatfile.

    Args:
        path: BRENDA data directory under DATA_DIR.
        flatfile_path: Optional flatfile path. If omitted, the builder attempts to
            discover a suitable flatfile in `path`.
        output_basename: Output file basename when writing artifacts.
        require_uniprot: Whether to enforce non-empty UniProt IDs.
        enrich_sequences: Whether to fetch amino-acid sequences from UniProt for
            rows with a valid UniProt accession.
        sequence_cache_path: Optional path to a JSON sequence cache file. Relative
            paths are resolved against `path`.
        sequence_request_timeout_seconds: Timeout used for UniProt REST lookups.
        sequence_request_retries: Retry count for retryable UniProt request errors.
        build_ligand_lookup: Whether to create the substrate/product ligand lookup
            DataFrame and write its JSON cache artifact.
        resolve_ligand_smiles: Whether to populate ligand lookup SMILES using
            ChEBI-first and MoleculeResolver fallback and then replace dataset
            substrates/products with SMILES lists.
        ligand_info_path: Optional path to the BRENDA ligand info table. If omitted,
            defaults to `brenda_ligand_info.csv` in `path`.
        ligand_lookup_cache_path: Optional path to the ligand lookup JSON cache.
            Relative paths are resolved against `path`.
        write_artifacts: Whether to write CSV and PKL artifacts.
        enable_logging: Whether to emit structured info-level counter logs.
        log_every_n_records: Interval for structured progress logs while parsing.
            Use 0 to disable intermediate progress logs.

    Returns:
        DataFrame with one kcat observation per row where `substrates` and
        `products` are SMILES lists and original names are preserved in
        `substrates_names` and `products_names`.
    """
    target_dir = Path(path)
    ensure_data_subfolder(target_dir)

    if sequence_request_timeout_seconds <= 0:
        raise ValueError("sequence_request_timeout_seconds must be greater than 0.")
    if sequence_request_retries < 0:
        raise ValueError("sequence_request_retries must be non-negative.")
    if resolve_ligand_smiles and not build_ligand_lookup:
        raise ValueError("resolve_ligand_smiles requires build_ligand_lookup=True.")

    stats: dict[str, int] = defaultdict(int)

    resolved_flatfile = _resolve_flatfile_path(target_dir, flatfile_path)
    resolved_sequence_cache_path = _resolve_sequence_cache_path(target_dir, sequence_cache_path)

    resolved_ligand_info_path: Optional[Path] = None
    resolved_ligand_lookup_cache_path: Optional[Path] = None
    if build_ligand_lookup:
        resolved_ligand_info_path = _resolve_ligand_info_path(target_dir, ligand_info_path)
        resolved_ligand_lookup_cache_path = _resolve_ligand_lookup_cache_path(
            target_dir,
            ligand_lookup_cache_path,
        )

    _log_counter_event(
        enable_logging,
        "start",
        data_dir=str(target_dir),
        flatfile=str(resolved_flatfile),
        require_uniprot=require_uniprot,
        enrich_sequences=enrich_sequences,
        sequence_cache_path=str(resolved_sequence_cache_path),
        sequence_request_timeout_seconds=sequence_request_timeout_seconds,
        sequence_request_retries=sequence_request_retries,
        build_ligand_lookup=build_ligand_lookup,
        resolve_ligand_smiles=resolve_ligand_smiles,
        ligand_info_path=str(resolved_ligand_info_path) if resolved_ligand_info_path else None,
        ligand_lookup_cache_path=(
            str(resolved_ligand_lookup_cache_path) if resolved_ligand_lookup_cache_path else None
        ),
        write_artifacts=write_artifacts,
        log_every_n_records=log_every_n_records,
    )

    extracted_rows: list[dict[str, Any]] = []
    for record_index, ec_record in enumerate(
        parse_brenda_flatfile(resolved_flatfile, stats=stats),
        start=1,
    ):
        _bump_counter(stats, "ec_records_total")
        record_rows = extract_benchmark_data_from_record(
            ec_record,
            require_uniprot=require_uniprot,
            stats=stats,
        )
        extracted_rows.extend(record_rows)
        _bump_counter(stats, "rows_from_records_total", len(record_rows))

        if enable_logging and log_every_n_records > 0 and record_index % log_every_n_records == 0:
            _log_counter_event(
                enable_logging,
                "progress",
                ec_records_processed=record_index,
                rows_collected=len(extracted_rows),
            )

    df = _normalize_output_dataframe(
        extracted_rows,
        require_uniprot=require_uniprot,
        stats=stats,
    )
    df = _apply_strict_reversibility_dedup(df, stats=stats)
    df = _enrich_sequences_from_uniprot(
        df,
        sequence_cache_path=resolved_sequence_cache_path,
        enrich_sequences=enrich_sequences,
        sequence_request_timeout_seconds=sequence_request_timeout_seconds,
        sequence_request_retries=sequence_request_retries,
        stats=stats,
    )

    ligand_lookup_df = pd.DataFrame(columns=list(_LIGAND_LOOKUP_COLUMNS))
    if build_ligand_lookup:
        assert resolved_ligand_info_path is not None
        assert resolved_ligand_lookup_cache_path is not None
        ligand_info_df = _load_brenda_ligand_info(resolved_ligand_info_path, stats=stats)
        ligand_lookup_df = _build_ligand_lookup_dataframe(df, ligand_info_df, stats=stats)
        if resolve_ligand_smiles:
            ligand_lookup_df = _enrich_ligand_lookup_with_smiles(ligand_lookup_df, stats=stats)
        else:
            _bump_counter(stats, "smiles_resolution_skipped_disabled")

        _write_ligand_lookup_cache(resolved_ligand_lookup_cache_path, ligand_lookup_df, stats=stats)
        df.attrs["ligand_lookup_cache_path"] = str(resolved_ligand_lookup_cache_path)
        df.attrs["ligand_lookup_rows"] = int(len(ligand_lookup_df.index))
        df.attrs["ligand_lookup_smiles_resolved"] = int(
            (~ligand_lookup_df["smiles"].apply(_is_missing_scalar)).sum()
        )
        df.attrs["ligand_lookup_df"] = ligand_lookup_df

    if resolve_ligand_smiles:
        df = _apply_ligand_lookup_smiles_to_dataset(df, ligand_lookup_df, stats=stats)
        df.attrs["smiles_dataset_transformed"] = True
    else:
        _bump_counter(stats, "smiles_dataset_transform_skipped_disabled")
        df.attrs["smiles_dataset_transformed"] = False

    if write_artifacts:
        csv_path, pkl_path = _write_brenda_artifacts(df, target_dir, output_basename)
        df.attrs["csv_path"] = str(csv_path)
        df.attrs["pkl_path"] = str(pkl_path)

    summary_stats = {key: int(value) for key, value in sorted(stats.items())}
    _log_counter_event(
        enable_logging,
        "summary",
        output_rows=len(df.index),
        stats=summary_stats,
    )

    if enable_logging and df.empty:
        logger.warning(
            "brenda_build_db.empty_output %s",
            json.dumps(
                {
                    "flatfile": str(resolved_flatfile),
                    "require_uniprot": require_uniprot,
                    "stats": summary_stats,
                },
                sort_keys=True,
                default=str,
            ),
        )

    return df