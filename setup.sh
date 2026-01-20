#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAMES=(
  "dlkcat_env"
  # "othermodel_env"
)

ENV_FILES=(
  "${ROOT_DIR}/environments/dlkcat_environment.yml"
  # "${ROOT_DIR}/envs/othermodel.yml"
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

  echo
done

echo "All model environments are ready."