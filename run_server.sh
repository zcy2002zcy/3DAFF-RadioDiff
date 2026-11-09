#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${CONDA_DEFAULT_ENV:-}" != "radiodiffzcy" ]]; then
  echo 'Activate radiodiffzcy first: conda activate radiodiffzcy' >&2
  exit 2
fi
exec python -B -u run.py "$@"
