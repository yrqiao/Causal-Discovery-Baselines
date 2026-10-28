#!/usr/bin/env bash
# Create the shared conda env and install everything needed by all methods.
#   bash scripts/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-causal_baselines}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Creating conda env '${ENV_NAME}' from environment.yml"
conda env create -n "${ENV_NAME}" -f "${ROOT}/environment.yml"

PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"

echo "[2/4] Unsetting R_HOME inside the env (a system-wide R_HOME would point at another R)"
mkdir -p "${PREFIX}/etc/conda/activate.d" "${PREFIX}/etc/conda/deactivate.d"
cat > "${PREFIX}/etc/conda/activate.d/r_home.sh" <<'EOF'
if [ -n "${R_HOME+x}" ]; then export _CB_OLD_R_HOME="${R_HOME}"; fi
unset R_HOME
EOF
cat > "${PREFIX}/etc/conda/deactivate.d/r_home.sh" <<'EOF'
if [ -n "${_CB_OLD_R_HOME+x}" ]; then export R_HOME="${_CB_OLD_R_HOME}"; unset _CB_OLD_R_HOME; fi
EOF

echo "[3/4] Installing R packages not available on conda (mboost, pcalg, SID, CAM)"
conda run --no-capture-output -n "${ENV_NAME}" Rscript "${ROOT}/scripts/install_r_packages.R"

echo "[4/4] Checking the environment"
conda run --no-capture-output -n "${ENV_NAME}" python "${ROOT}/scripts/check_env.py"

echo "Done. Activate with:  conda activate ${ENV_NAME}"
