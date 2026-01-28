#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  CONDA_BASE=""
fi

if [ -z "${CONDA_BASE}" ]; then
  for candidate in \
    "/opt/conda" \
    "$HOME/miniconda3" \
    "$HOME/mambaforge" \
    "$HOME/anaconda3" \
    "/usr/local/anaconda3" \
    "/usr/local/miniconda3"
  do
    if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="${candidate}"
      break
    fi
  done
fi

if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Error: Could not find conda.sh." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAMES=(
  "dlkcat_env"
  "catapro_env"
  "catpred_env"
  # "mmkcat_env"
)

ENV_FILES=(
  "${ROOT_DIR}/environments/dlkcat_environment.yml"
  # "${ROOT_DIR}/environments/mmkcat_environment.yml"
  "${ROOT_DIR}/environments/catapro_environment.yml"
  "${ROOT_DIR}/environments/catpred_environment.yml"
)

if [ "${#ENV_NAMES[@]}" -ne "${#ENV_FILES[@]}" ]; then
  echo "Error: ENV_NAMES and ENV_FILES must have the same number of entries." >&2
  exit 1
fi

echo "Repo root: ${ROOT_DIR}"
echo

for i in "${!ENV_FILES[@]}"; do
  env_name="${ENV_NAMES[$i]}"
  env_file="${ENV_FILES[$i]}"

  echo "==> Creating/updating env '${env_name}' from: ${env_file}"
  conda env update -n "${env_name}" -f "${env_file}" --prune

  echo "==> Installing this repo into env: ${env_name}"
  conda run -n "${env_name}" python -m pip install -e "${ROOT_DIR}"

  case "${env_name}" in
    "catpred_env")
      echo "==> [CatPred] Installing submodule dependencies..."
      if [ -d "${ROOT_DIR}/models/CatPred" ]; then
          conda run -n "${env_name}" python -m pip install -e "${ROOT_DIR}/models/CatPred"
      else
          echo "Warning: models/CatPred directory not found. Skipping submodule install."
      fi
      ;;
      
    "catapro_env")
      ;;

    *)
      # Default case
  esac

  echo
done

echo "All model environments are ready."