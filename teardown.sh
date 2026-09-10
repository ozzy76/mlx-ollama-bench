#!/usr/bin/env bash
# Confirms the isolated test Ollama instance is not running, confirms
# production is healthy, and optionally reclaims disk from the test
# model store. Does not touch production Ollama or ~/.ollama.
set -euo pipefail

echo "== Cleanup =="

TEST_PORT=11435
if lsof -i ":${TEST_PORT}" >/dev/null 2>&1; then
  echo "NOTE: something is still listening on ${TEST_PORT}."
  echo "      run_matrix.py should have stopped it — if it's still up, find and stop it manually:"
  lsof -i ":${TEST_PORT}" || true
else
  echo "OK: isolated test Ollama instance (port ${TEST_PORT}) is not running."
fi

read -r -p "Remove isolated test model store (./ollama-test-models) to reclaim disk? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  rm -rf ./ollama-test-models
  echo "Removed ./ollama-test-models"
else
  echo "Left ./ollama-test-models in place."
fi

echo ""
echo "-- Production Ollama health check (port 11434) --"
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "OK: production Ollama responding."
else
  echo "WARNING: production Ollama not responding on 11434 — check it directly, though this "
  echo "         toolkit never sent it any load-bearing requests."
fi
