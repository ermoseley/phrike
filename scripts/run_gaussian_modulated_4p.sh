#!/usr/bin/env bash
set -euo pipefail

# Run gaussian_modulated example with parallel FFT/threaded math.
# Usage: scripts/run_gaussian_modulated_4p.sh [WORKERS]
# Default WORKERS=4

WORKERS="${1:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CFG_IN="${REPO_ROOT}/configs/gaussian_modulated.yaml"
TMP_CFG="$(mktemp /tmp/gm_cfg_XXXX.yaml)"

cleanup() {
  rm -f "$TMP_CFG" || true
}
trap cleanup EXIT

# Create a temporary config that injects grid.fft_workers
python - "$CFG_IN" "$TMP_CFG" "$WORKERS" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
if not isinstance(cfg, dict):
    cfg = {}
if 'grid' not in cfg or cfg['grid'] is None:
    cfg['grid'] = {}
cfg['grid']['fft_workers'] = workers
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f)
PY

# Threading environment for underlying libs
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"
export OPENBLAS_NUM_THREADS="$WORKERS"
export NUMEXPR_NUM_THREADS="$WORKERS"
export NUMBA_NUM_THREADS="$WORKERS"
export FFTW_NUM_THREADS="$WORKERS"

echo "Running gaussian_modulated with WORKERS=${WORKERS}"
python "${REPO_ROOT}/examples/run_gaussian_modulated.py" --config "$TMP_CFG"


