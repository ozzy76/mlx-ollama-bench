#!/usr/bin/env python3
"""
Quick sanity check before trusting the full run_matrix.py matrix.

Starts the isolated test Ollama instance, pulls both model tags, fires ONE
short-prompt request at each of configs A/B (and C, if enabled) using the
same num_ctx and cache-busting nonce logic as the real run, and writes
results/dry_run.json. Does not touch production Ollama except the same
before/after health check run_matrix.py does.

Usage: python3 dry_run.py
"""
import json
import time
from pathlib import Path

import run_matrix as rm

ROOT = Path(__file__).parent
OUT_PATH = ROOT / rm.CFG["paths"]["results_dir"] / "dry_run.json"

SHORT_PROMPT = "Say hello in exactly one short sentence."


def main():
    rm.assert_safe_config()
    result = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    result["production_health_before"] = rm.check_production_health()
    print(f"Production Ollama health before: {result['production_health_before']}")

    num_ctx = rm.resolve_num_ctx(rm.bucket_token_targets())
    dry_run_prompt = rm.with_rep_nonce(SHORT_PROMPT, "dryrun", 0)

    proc = rm.start_test_ollama()
    try:
        rm.pull_test_model(rm.CFG["model"]["ollama_tag"], rm.CFG["ollama"]["test_port"])
        rm.pull_test_model(rm.CFG["model"]["ollama_mlx_tag"], rm.CFG["ollama"]["test_port"])
        host = f"http://127.0.0.1:{rm.CFG['ollama']['test_port']}"

        print("Running config A (Ollama GGUF)...")
        result["A_ollama_gguf"] = rm.run_ollama_generate(
            host, rm.CFG["model"]["ollama_tag"], dry_run_prompt, max_tokens=20, num_ctx=num_ctx)
        print(" ->", result["A_ollama_gguf"])

        print("Running config B (Ollama MLX)...")
        result["B_ollama_mlx"] = rm.run_ollama_generate(
            host, rm.CFG["model"]["ollama_mlx_tag"], dry_run_prompt, max_tokens=20, num_ctx=num_ctx)
        print(" ->", result["B_ollama_mlx"])

        if rm.CFG["run"].get("include_mlx_raw", True):
            print("Running config C (raw mlx_lm.benchmark, tiny)...")
            result["C_mlx_raw"] = rm.run_mlx_benchmark(
                rm.CFG["model"]["hf_repo"], prompt_tokens=50, generation_tokens=20, num_trials=1)
            print(" ->", result["C_mlx_raw"])
        else:
            print("Config C skipped (include_mlx_raw=false in config.json)")
    finally:
        rm.stop_test_ollama(proc)
        result["production_health_after"] = rm.check_production_health()
        print(f"Production Ollama health after: {result['production_health_after']}")

    required_configs = ["A_ollama_gguf", "B_ollama_mlx"]
    if rm.CFG["run"].get("include_mlx_raw", True):
        required_configs.append("C_mlx_raw")
    result["all_required_produced_decode_tok_s"] = all(
        result.get(cfg, {}).get("decode_tok_s") is not None
        for cfg in required_configs
    )

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print("DRY RUN OK" if result["all_required_produced_decode_tok_s"] else "DRY RUN INCOMPLETE — check output above")


if __name__ == "__main__":
    main()
