#!/usr/bin/env bash
# Installs tools via Homebrew only, creates isolated test directories, runs
# pre-flight checks. Never modifies your production Ollama install or its
# model store.
set -euo pipefail

echo "== MLX vs Ollama benchmark: setup =="

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it first: https://brew.sh"
  exit 1
fi

echo "-- mlx-lm (brew formula; brings mlx as a dependency; no pip install, no venv) --"
brew list mlx-lm >/dev/null 2>&1 || brew install mlx-lm

echo "-- asitop (GPU/power monitoring, optional but recommended) --"
brew list asitop >/dev/null 2>&1 || brew install asitop || echo "asitop install failed/skipped — not required, continuing"

echo "-- Confirming Ollama is present (NOT reinstalling — production depends on your existing install) --"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found on PATH. This script won't install it — confirm your existing production setup first."
  exit 1
fi
ollama --version

echo "-- Confirming mlx_lm CLI is on PATH --"
mlx_lm.server --help >/dev/null 2>&1 && echo "OK: mlx_lm CLI available"

echo "-- Creating isolated test directories (production model store untouched) --"
mkdir -p results
mkdir -p ollama-test-models

echo "-- Pre-flight: production Ollama health check (read-only, port 11434) --"
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "OK: production Ollama responding on 11434"
else
  echo "NOTE: production Ollama not responding on 11434 right now."
  echo "      Benchmarks below don't depend on it, but you won't have a baseline health check."
fi

echo ""
echo "Setup complete."
echo "Next: fill in config.json (model tags + HF repo), then run: python3 run_matrix.py"
