#!/usr/bin/env python3
"""
Orchestrates the MLX vs Ollama benchmark matrix.

Safety design:
  - Starts a SECOND, isolated Ollama instance (different port, different
    OLLAMA_MODELS dir) for all benchmark traffic — configs A and B.
  - Production Ollama (default port/model dir) is only ever hit with a
    read-only GET /api/tags health check, before and after the run.
  - Config C (raw mlx-lm) shells out to the brew-installed `mlx_lm.benchmark`
    CLI — mlx-lm's own purpose-built benchmarking tool (prompt tok/s,
    generation tok/s, and peak memory, with internal multi-trial medians) —
    no server, no shared state with anything Ollama-related.
  - Stdlib only. No pip install required to run this script.

  NOTE: don't run this alongside `brew services start mlx-lm` — that starts
  mlx_lm.server as a persistent background launchd service, which is the
  opposite of what this script wants (nothing left running once it exits).
  If you've ever used `brew services` for mlx-lm, run
  `brew services stop mlx-lm` before benchmarking.

Data-quality fixes (v2, applied after the first full run surfaced them):
  1. Prompt-cache busting: Ollama caches/snapshots repeated prompt prefixes,
     so identical prompts across reps 2-5 of a bucket returned near-zero
     prompt_eval_duration, which blew up prefill_tok_s into physically
     impossible numbers (millions of tok/s) once reps 1..4 hit the cache.
     Every request now gets a short unique nonce prepended (see
     `with_rep_nonce()`), which changes the prompt's leading tokens and
     defeats prefix-based caching, so every rep does a genuine prefill.
  2. Explicit num_ctx: the first full run never set options.num_ctx, so
     Ollama silently truncated the "long" bucket's prompt to whatever the
     GGUF backend's default context window is (~2K tokens observed) while
     the MLX backend used a larger default — invalidating that comparison.
     num_ctx is now read from config.json (run.num_ctx) and sent on every
     request, sized above the longest bucket you configure.
  Both are what make this script safe to reuse as a standing regression
  test rather than a one-off — see README.md.

Usage:
  python3 run_matrix.py
"""
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text())

RESULTS_DIR = ROOT / CFG["paths"]["results_dir"]
RESULTS_DIR.mkdir(exist_ok=True)

# One id for this run, reused for the CSV filename and the per-rep cache-
# busting nonce below, so a nonce can always be traced back to the run
# that produced it.
RUN_ID = int(time.time())
CSV_PATH = RESULTS_DIR / f"results_{RUN_ID}.csv"

FIELDS = [
    "config", "prompt_bucket", "rep", "ttft_ms", "prefill_tok_s", "decode_tok_s",
    "load_ms", "total_wall_ms", "prompt_tokens", "completion_tokens", "peak_mem_gb",
]


# ---------- safety guards ----------

def assert_safe_config():
    test_port = CFG["ollama"]["test_port"]
    test_models_dir = CFG["paths"]["test_ollama_models"]
    assert test_port != 11434, (
        "Refusing to run: config.json test_port is 11434, the production Ollama port."
    )
    assert "test" in test_models_dir.lower(), (
        "Refusing to run: test_ollama_models path doesn't look like a test directory "
        f"({test_models_dir!r}). Rename it to something with 'test' in it as a safety check."
    )
    assert "FILL_ME_IN" not in json.dumps(CFG["model"]), (
        "config.json still has placeholder model values. Fill in model.ollama_tag, "
        "model.ollama_mlx_tag, and model.hf_repo before running."
    )


def check_production_health():
    try:
        req = urllib.request.Request(CFG["ollama"]["production_host"] + "/api/tags")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


# ---------- isolated test Ollama instance ----------

def _wait_for_port(port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Nothing listening on 127.0.0.1:{port} after {timeout}s")


def start_test_ollama():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{CFG['ollama']['test_port']}"
    env["OLLAMA_MODELS"] = str((ROOT / CFG["paths"]["test_ollama_models"]).resolve())
    print(f"Starting isolated Ollama test instance on port {CFG['ollama']['test_port']} "
          f"(models dir: {env['OLLAMA_MODELS']})")
    proc = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _wait_for_port(CFG["ollama"]["test_port"])
    return proc


def stop_test_ollama(proc):
    print("Stopping isolated Ollama test instance...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def pull_test_model(tag, test_port):
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{test_port}"
    print(f"Pulling {tag} into isolated test instance...")
    subprocess.run(["ollama", "pull", tag], env=env, check=True)


# ---------- benchmark calls ----------

def run_ollama_generate(host, model, prompt, max_tokens, num_ctx=None):
    options = {"num_predict": max_tokens}
    if num_ctx:
        # Without this, Ollama falls back to a backend-specific default
        # context window and silently truncates any prompt longer than
        # that — different backends can default differently, which is
        # what corrupted the first run's "long" bucket comparison.
        options["num_ctx"] = num_ctx
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        host + "/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    wall_ms = (time.time() - t0) * 1000

    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration = data.get("prompt_eval_duration", 0)
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    load_duration = data.get("load_duration", 0)

    prefill_tok_s = (prompt_eval_count / (prompt_eval_duration / 1e9)) if prompt_eval_duration else None
    decode_tok_s = (eval_count / (eval_duration / 1e9)) if eval_duration else None

    return {
        "ttft_ms": (load_duration + prompt_eval_duration) / 1e6,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "load_ms": load_duration / 1e6,
        "total_wall_ms": wall_ms,
        "prompt_tokens": prompt_eval_count,
        "completion_tokens": eval_count,
    }


def run_mlx_benchmark(hf_repo, prompt_tokens, generation_tokens, num_trials):
    """
    Runs mlx-lm's own benchmarking CLI (`mlx_lm.benchmark`) instead of
    hand-timing `mlx_lm.generate`. It's purpose-built for exactly this:
    synthetic prompts sized by token count, internal multi-trial runs,
    and it reports peak memory alongside prompt/generation tok/s — which
    a hand-rolled harness around `generate` doesn't give you for free.

    Trade-off worth knowing: this uses mlx_lm.benchmark's own synthetic
    token sequence, not the English filler text used for configs A/B.
    That's fine for a pure throughput comparison (and matches how public
    MLX benchmarks are usually run) but means config C's "prompt" isn't
    literally the same text as A/B's — only the same length.

    NOTE: mlx_lm.benchmark's exact output format wasn't hand-verified
    against a live run — the parser below is defensive (regex over
    numbers near keywords) and will print raw output + a warning if it
    can't find what it expects. Check `mlx_lm.benchmark --help` on your
    machine on the first run and adjust the parsing below if the flag
    names or output text differ from what's assumed here.
    """
    t0 = time.time()
    result = subprocess.run(
        ["mlx_lm.benchmark", "--model", hf_repo,
         "--prompt-tokens", str(prompt_tokens),
         "--generation-tokens", str(generation_tokens),
         "--num-trials", str(num_trials)],
        capture_output=True, text=True, timeout=1800,
    )
    wall_ms = (time.time() - t0) * 1000
    output = result.stdout + result.stderr

    # Always keep the raw output on disk (not just printed) so it can be
    # read back from the results/ folder without needing terminal scrollback.
    debug_dir = RESULTS_DIR / "mlx_benchmark_raw"
    debug_dir.mkdir(exist_ok=True)
    debug_path = debug_dir / f"{int(t0)}_{prompt_tokens}tok.txt"
    debug_path.write_text(
        f"cmd: mlx_lm.benchmark --model {hf_repo} --prompt-tokens {prompt_tokens} "
        f"--generation-tokens {generation_tokens} --num-trials {num_trials}\n"
        f"returncode: {result.returncode}\n"
        f"--- output ---\n{output}"
    )

    prefill_tok_s = decode_tok_s = peak_mem_gb = None
    for line in output.splitlines():
        low = line.lower()
        nums = [tok for tok in line.replace(",", "").split() if _is_number(tok)]
        if "prompt" in low and ("tok" in low or "sec" in low) and nums:
            prefill_tok_s = float(nums[-1])
        if "generat" in low and ("tok" in low or "sec" in low) and nums:
            decode_tok_s = float(nums[-1])
        if "peak" in low and "mem" in low and nums:
            peak_mem_gb = float(nums[-1])

    if decode_tok_s is None:
        print("  WARNING: could not parse mlx_lm.benchmark output "
              f"(returncode={result.returncode}). Raw output saved to {debug_path}")

    return {
        "ttft_ms": None,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "load_ms": None,
        "total_wall_ms": wall_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generation_tokens,
        "peak_mem_gb": peak_mem_gb,
    }


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------- prompts ----------

def synthetic_prompt(word_count):
    filler = "Analyze the following control mapping context and continue the assessment. "
    text = (filler * ((word_count // len(filler.split())) + 1)).split()
    return " ".join(text[:word_count])


def build_prompts():
    p = CFG["prompts"]
    return {
        "short": synthetic_prompt(p["short_words"]),
        "medium": synthetic_prompt(p["medium_words"]),
        "long": synthetic_prompt(p["long_words"]),
    }
    # Swap these for real captured ADK agent turns (tool output + instruction)
    # once you want higher-fidelity numbers than synthetic filler gives you.
    # (Config C doesn't use this text — see run_mlx_benchmark.)


def bucket_token_targets():
    p = CFG["prompts"]
    return {"short": p["short_tokens"], "medium": p["medium_tokens"], "long": p["long_tokens"]}


def with_rep_nonce(prompt, bucket, rep):
    """
    Prepends a short, unique marker to the prompt so Ollama's prefix-based
    prompt cache can never match this request against a previous one.
    The cache keys off the leading tokens, so changing token 0 is enough
    to force a full, genuine prefill on every single rep — the fix for
    the prefill_tok_s corruption seen in the first full run (see the
    module docstring). Prepending rather than appending matters: a
    change buried at the end of a 12,000-word prompt would still share
    the whole prefix and could still hit the cache.

    Adds a small, constant number of tokens (roughly a dozen) to every
    request across every config/bucket/rep — negligible next to the
    500-16,000 token buckets, and uniform so it doesn't bias any one
    config over another.
    """
    nonce = f"[[bench run={RUN_ID} bucket={bucket} rep={rep}]]\n\n"
    return nonce + prompt


def resolve_num_ctx(token_targets):
    """
    num_ctx to send with every Ollama request. Prefers an explicit
    run.num_ctx in config.json; otherwise derives one from the largest
    configured prompt bucket plus max_new_tokens plus headroom, so a
    future edit to the bucket sizes doesn't silently reintroduce the
    truncation bug.
    """
    explicit = CFG["run"].get("num_ctx")
    if explicit:
        return explicit
    largest_bucket = max(token_targets.values())
    return largest_bucket + CFG["prompts"]["max_new_tokens"] + 1024


# ---------- main ----------

def main():
    assert_safe_config()

    before = check_production_health()
    print(f"Production Ollama health before run: {'OK' if before else 'UNREACHABLE (continuing anyway)'}")

    prompts = build_prompts()
    token_targets = bucket_token_targets()
    max_tokens = CFG["prompts"]["max_new_tokens"]
    reps = CFG["run"]["reps_per_cell"]
    num_ctx = resolve_num_ctx(token_targets)
    print(f"Using num_ctx={num_ctx} for every Ollama request "
          f"(largest configured bucket: {max(token_targets.values())} tokens)")
    row_count = 0

    # Opened once, up front, and flushed after every row — a long matrix
    # (the 3x5x2 default runs ~an hour) shouldn't lose everything if you
    # need to Ctrl+C partway through. Ctrl+C still leaves the isolated test
    # Ollama instance running until the `finally` block below stops it.
    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()
    csv_file.flush()

    def write_row(row):
        nonlocal row_count
        writer.writerow(row)
        csv_file.flush()
        row_count += 1

    proc = start_test_ollama()
    try:
        pull_test_model(CFG["model"]["ollama_tag"], CFG["ollama"]["test_port"])
        pull_test_model(CFG["model"]["ollama_mlx_tag"], CFG["ollama"]["test_port"])
        host = f"http://127.0.0.1:{CFG['ollama']['test_port']}"

        for bucket, prompt in prompts.items():
            for rep in range(reps):
                for cfg_name, model_tag in [
                    ("A_ollama_gguf", CFG["model"]["ollama_tag"]),
                    ("B_ollama_mlx", CFG["model"]["ollama_mlx_tag"]),
                ]:
                    rep_prompt = with_rep_nonce(prompt, bucket, rep)
                    metrics = run_ollama_generate(host, model_tag, rep_prompt, max_tokens, num_ctx)
                    write_row({"config": cfg_name, "prompt_bucket": bucket, "rep": rep,
                               "peak_mem_gb": None, **metrics})
                    print(f"{cfg_name:15s} {bucket:8s} rep={rep} decode_tok_s={metrics['decode_tok_s']}")

            # Config C: mlx_lm.benchmark handles its own trials internally
            # (--num-trials) and reports a median, so this runs once per
            # bucket rather than once per rep like A/B above.
            # Gated by include_mlx_raw — see config.json's note. As of the
            # dry run, mlx-community/gemma-4-12B-it-4bit fails to load under
            # Homebrew's mlx-lm 0.31.3 (ValueError: Model type gemma4_unified
            # not supported — ml-explore/mlx-lm#1481, open upstream). Flip
            # this back on once that's fixed or hf_repo points elsewhere.
            if CFG["run"].get("include_mlx_raw", True):
                metrics = run_mlx_benchmark(CFG["model"]["hf_repo"], token_targets[bucket], max_tokens, reps)
                write_row({"config": "C_mlx_raw", "prompt_bucket": bucket, "rep": "median_of_trials", **metrics})
                print(f"{'C_mlx_raw':15s} {bucket:8s} (internal {reps} trials) decode_tok_s={metrics['decode_tok_s']}")
            else:
                print(f"{'C_mlx_raw':15s} {bucket:8s} skipped (include_mlx_raw=false in config.json)")
    finally:
        csv_file.close()
        stop_test_ollama(proc)
        after = check_production_health()
        print(f"Production Ollama health after run: {'OK' if after else 'UNREACHABLE — check it'}")

    print(f"\nWrote {row_count} rows to {CSV_PATH}")
    print("Next: python3 generate_report.py")


if __name__ == "__main__":
    main()
