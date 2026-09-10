#!/usr/bin/env python3
"""
Reads the latest results/results_*.csv and prints a parity verdict per
prompt bucket, matching the thresholds in the test plan:
  - decode throughput within +/-10% of config A counts as parity
  - prefill is noisier; treat under 20% as inconclusive

run_matrix.py (v2) prepends a unique nonce to every request so Ollama's
prompt cache can't shortcut a repeated prefill — see that script's
docstring. That means prefill_tok_s should now be trustworthy across
every rep, not just rep 0. As a safety net in case a cache still gets
hit (e.g. the nonce logic gets edited out later), PREFILL_SANITY_MAX_TOK_S
below flags any group whose median prefill is implausibly high rather
than silently reporting it — real Apple Silicon prefill on a model this
size tops out in the low thousands of tok/s, not millions.

Writes results/summary.md alongside the printed output.
"""
import csv
import statistics
import sys
from pathlib import Path

PREFILL_SANITY_MAX_TOK_S = 3000

RESULTS_DIR = Path(__file__).parent / "results"
candidates = sorted(RESULTS_DIR.glob("results_*.csv"))
if not candidates:
    sys.exit("No results_*.csv found in results/. Run run_matrix.py first.")
latest = candidates[-1]
print(f"Reading {latest}\n")

rows = list(csv.DictReader(open(latest)))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


groups = {}
for r in rows:
    key = (r["config"], r["prompt_bucket"])
    groups.setdefault(key, []).append(r)

summary = {}
flagged = []
for (cfg, bucket), rs in groups.items():
    # Config C (mlx_lm.benchmark) already reports one median-of-N-trials row
    # per bucket — nothing to discard. Configs A/B log one row per rep, so
    # sort numerically and drop the first as a cold-start rep (first-load
    # overhead, not the prompt-cache issue — that's handled by run_matrix.py's
    # per-rep nonce now, see module docstring above).
    if cfg == "C_mlx_raw":
        rs_use = rs
    else:
        rs_sorted = sorted(rs, key=lambda r: int(r["rep"]))
        rs_use = rs_sorted[1:] if len(rs_sorted) > 1 else rs_sorted
    decode_vals = [to_float(r["decode_tok_s"]) for r in rs_use]
    prefill_vals = [to_float(r["prefill_tok_s"]) for r in rs_use]
    mem_vals = [to_float(r.get("peak_mem_gb")) for r in rs_use]
    decode_vals = [v for v in decode_vals if v is not None]
    prefill_vals = [v for v in prefill_vals if v is not None]
    mem_vals = [v for v in mem_vals if v is not None]
    prefill_median = statistics.median(prefill_vals) if prefill_vals else None
    if prefill_median is not None and prefill_median > PREFILL_SANITY_MAX_TOK_S:
        flagged.append((cfg, bucket, prefill_median))
    summary[(cfg, bucket)] = {
        "decode_tok_s": statistics.median(decode_vals) if decode_vals else None,
        "prefill_tok_s": prefill_median,
        "peak_mem_gb": statistics.median(mem_vals) if mem_vals else None,
        "n": len(rs_use),
    }

lines = ["# MLX vs Ollama — results summary", "",
         f"Source: `{latest.name}`", "",
         "| Config | Bucket | Decode tok/s (median) | Prefill tok/s (median) | Peak mem (GB) | n |",
         "|---|---|---|---|---|---|"]
for (cfg, bucket), s in sorted(summary.items()):
    d = f"{s['decode_tok_s']:.1f}" if s["decode_tok_s"] is not None else "n/a"
    p = f"{s['prefill_tok_s']:.1f}" if s["prefill_tok_s"] is not None else "n/a"
    m = f"{s['peak_mem_gb']:.1f}" if s["peak_mem_gb"] is not None else "n/a"
    lines.append(f"| {cfg} | {bucket} | {d} | {p} | {m} | {s['n']} |")

lines += ["", "## Parity vs. A_ollama_gguf (decode, +/-10% = parity)", ""]
buckets = sorted({b for _, b in summary})
for bucket in buckets:
    a = summary.get(("A_ollama_gguf", bucket), {}).get("decode_tok_s")
    if not a:
        lines.append(f"- **{bucket}**: no config A data")
        continue
    for cfg in ("B_ollama_mlx", "C_mlx_raw"):
        v = summary.get((cfg, bucket), {}).get("decode_tok_s")
        if v is None:
            lines.append(f"- **{bucket}** / {cfg}: no data")
            continue
        pct = (v - a) / a * 100
        verdict = "PARITY" if abs(pct) <= 10 else ("FASTER" if pct > 0 else "SLOWER")
        lines.append(f"- **{bucket}** / {cfg}: {pct:+.1f}% vs. A -> **{verdict}**")

if flagged:
    lines += ["", "## ⚠ Prefill sanity check failed", "",
              f"The following groups reported a median prefill above "
              f"{PREFILL_SANITY_MAX_TOK_S} tok/s, which isn't physically "
              f"plausible on Apple Silicon — treat these as corrupted "
              f"(likely a prompt cache hit) and don't trust them:", ""]
    for cfg, bucket, val in flagged:
        lines.append(f"- **{cfg} / {bucket}**: {val:,.1f} tok/s")

report = "\n".join(lines)
print(report)

out_path = RESULTS_DIR / "summary.md"
out_path.write_text(report)
print(f"\nWrote {out_path}")
