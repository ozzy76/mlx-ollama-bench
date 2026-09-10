# MLX vs. Ollama — Automation Toolkit

## What is this?

A regression-testing harness that answers one repeatable question: **for a given model, is Ollama's native MLX backend actually faster than its standard GGUF backend on the Mac, and by how much?**

It was originally built to evaluate `gemma4:12b` for local-first macOS agent team (SwiftUI frontend, Python/Google ADK agents, an MLX-served fine-tuned Gemma on the Secure Controls Framework), but nothing about the harness is specific to that model or product — see "Testing a new model" below.

**What it measures:** raw inference speed only — decode tok/s (how fast the model writes its response), prefill tok/s (how fast it reads the prompt), time-to-first-token, and load time. It does **not** measure whether an agent completes its task correctly.

To align with responsible AI practices run it to verify model swaps meet performance expectations. Use scenearios: After Ollama updates, before deciding whether a new model is even worth trying on-device. It spins up an isolated Ollama instance, pulls both backend variants of a model, runs a matched prompt matrix against each, and tears it down without breaking your "prod" instance functionality. Edit `config.json` to change targets.


## Folder layout

Drop this whole folder into your testing/artifacts folder as-is:

```
mlx-ollama-bench/
├── README.md          (this file)
├── LICENSE             MIT
├── .gitignore           excludes results/, ollama-test-models/, __pycache__
├── setup.sh            installs tools, creates isolated dirs, pre-flight checks
├── config.json         fill in your model tags/paths before running
├── run_matrix.py        the benchmark orchestrator — isolated Ollama instance + raw mlx-lm
├── dry_run.py           one short-prompt sanity check before trusting the full matrix
├── generate_report.py  turns results/*.csv into a parity verdict
├── teardown.sh          stops the test instance, optional cleanup, confirms prod is healthy
└── results/             CSV output + summary.md land here (gitignored — regenerated per run)
```

Nothing here writes outside this folder except: (1) Homebrew installs (`mlx-lm`, `asitop` — shared tools, same as any other `brew install`), and (2) the isolated Ollama test instance's model store, which you point at a subfolder of this same directory in `config.json` — never at `~/.ollama`.

## Run order

```bash
cd /path/to/your/testing-artifacts/mlx-ollama-bench

chmod +x setup.sh teardown.sh
./setup.sh                      # brew installs, isolated dirs, pre-flight health check

$EDITOR config.json             # fill in your Ollama model tag, the -mlx tag, and the
                                 # matching mlx-community Hugging Face repo (see below)

python3 dry_run.py              # ONE short prompt per config, ~seconds not minutes — confirms
                                 # both model tags pull and respond before you trust the full run

python3 run_matrix.py           # runs the full matrix, writes results/results_<ts>.csv
python3 generate_report.py      # prints + writes results/summary.md with a parity verdict

./teardown.sh                   # stops the isolated test instance, confirms prod is healthy,
                                 # optionally clears the test model store to reclaim disk
```

## Testing a new model

This is the workflow the harness is meant to be reused for. To re-run the same comparison against a different model:

1. Confirm Ollama has (or can pull) both a standard tag and an `-mlx` tag for the model you want to test — run `ollama list` or check the model's page on ollama.com. If no `-mlx` tag exists yet, there's nothing to compare — Ollama hasn't shipped an MLX build of it.
2. Update `config.json`: `model.ollama_tag`, `model.ollama_mlx_tag`, and (only if you also want config C) `model.hf_repo` to the matching `mlx-community/...` Hugging Face repo.
3. Optionally set `include_mlx_raw: true` if you want raw `mlx-lm` in the matrix too — check first whether `mlx-lm` supports that model's architecture (`mlx_lm.generate --model <repo> --max-tokens 5` is a fast way to find out; a `ValueError: Model type ... not supported` means not yet).
4. Re-run `python3 dry_run.py` first, then `python3 run_matrix.py` and `python3 generate_report.py` as above.

No code changes needed for a same-architecture swap — that's the point of the harness.

Expect the full matrix (3 configs × 3 prompt buckets × 5 reps, 256 tokens generated per run) to take a while on a 16-24GB machine, mostly dominated by the long-context (~16K token) prefill reps. Run it when the agent team is idle — not because it can interfere with production (it can't, by design), but because your Mac only has so much unified memory and GPU to give a benchmark and everything else at once.

## Prompts

Two different prompt mechanisms now, matched to what each tool actually wants:

- **Configs A/B** (Ollama, both GGUF and MLX-format) use English filler text sized to roughly 500/4,000/16,000 tokens by word count — the `build_prompts()` function in `run_matrix.py`. Swap in real captured ADK agent turns (tool output + instruction) later for higher-fidelity numbers.
- **Config C** (`mlx_lm.benchmark`) uses its own internal synthetic token sequence, sized by an exact token count via `--prompt-tokens` — the `short_tokens`/`medium_tokens`/`long_tokens` fields in `config.json`. This means config C's prompt content isn't literally the same text as A/B's, only the same length — a disclosed limitation, not a bug, and it's how public MLX benchmarks are usually run anyway.

## Reading the output

`generate_report.py` computes median decode tok/s, prefill tok/s, and peak memory per config/bucket, and prints a parity verdict against config A at ±10% decode, matching the threshold in the original test plan. Configs A/B contribute one row per rep (first rep discarded as cold-start); config C contributes one row per bucket, since `mlx_lm.benchmark` already medians its own internal trials. Any group whose median prefill comes back implausibly high gets its own flagged section at the bottom of the report rather than being silently included. `results/summary.md` keeps a copy.

The **decode tok/s** numbers are the reliable, actionable metric here — that's the rate the model actually writes its response at, and it's what the parity verdict is built on. **Prefill tok/s** is how fast the model reads the prompt before writing anything; trust it too now that v2's per-rep nonce stops the cache from corrupting it, but the sanity-check flag above is there as a backstop.

## Safety checklist this automation enforces

- Refuses to run if the test Ollama port is 11434 (the production port) or the test model directory doesn't look like a test directory — both are asserted in `run_matrix.py` before anything starts.
- Never sends a generate/pull request to the production host — only `GET /api/tags` health checks, before and after.
- Isolated Ollama instance is started and stopped entirely within the script run; nothing is left running in the background after `teardown.sh`.
- All tool installs are Homebrew formulas, not `pip install`s into any project environment.

## Known limitations

- **Speed only, not correctness.** This measures how fast a model responds, not whether an agent using it actually completes tasks correctly. If you're choosing between two different models (not two backends of the same model), pair this with an actual task-completion eval before deciding.
- **Ollama-API-specific.** Configs A/B are built around Ollama's `/api/generate`. Testing a backend outside Ollama (vLLM, LM Studio, a raw MLX server) would need a new config, not just a `config.json` edit.
- **Config C depends on mlx-lm's architecture support.** Raw `mlx-lm` benchmarking only works for architectures mlx-lm's Homebrew build currently supports — check before enabling `include_mlx_raw` for a new model.
- **No peak-memory tracking for configs A/B.** `peak_mem_gb` is only populated for config C (via `mlx_lm.benchmark`). Worth adding via `asitop`/`powermetrics` sampling around the A/B calls if memory pressure becomes a concern.
- **Synthetic prompts.** `build_prompts()` generates filler text sized by word count, not real captured agent traffic — swap in real ADK agent turns for higher-fidelity numbers (see the comment in `build_prompts()`).

## License

MIT — see [LICENSE](LICENSE).
