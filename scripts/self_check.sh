#!/usr/bin/env bash
set -euo pipefail

VENV_BIN="${VENV_BIN:-.venv/bin}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_BIN/python}"
RUFF_BIN="${RUFF_BIN:-$VENV_BIN/ruff}"
MYPY_BIN="${MYPY_BIN:-$VENV_BIN/mypy}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -x "$RUFF_BIN" ]]; then
  RUFF_BIN="ruff"
fi
if [[ ! -x "$MYPY_BIN" ]]; then
  MYPY_BIN="mypy"
fi

echo "[1/7] Ruff"
"$RUFF_BIN" check .

echo "[2/7] Mypy"
"$MYPY_BIN" business_policy_env/

echo "[3/7] Pytest"
"$PYTHON_BIN" -m pytest tests/ -v

echo "[4/7] Rule baseline"
"$PYTHON_BIN" baseline.py --agent rule

echo "[5/7] OpenEnv contract validation"
"$PYTHON_BIN" scripts/validate_openenv_contract.py

echo "[6/7] OpenEnv validate (if installed)"
if command -v openenv >/dev/null 2>&1; then
  openenv validate openenv.yaml
else
  echo "SKIP: openenv CLI not found."
fi

echo "[7/7] Docker smoke (if docker is installed)"
if command -v docker >/dev/null 2>&1; then
  bash scripts/docker_smoke.sh
else
  echo "SKIP: docker not found."
fi

echo "Self-check completed."
