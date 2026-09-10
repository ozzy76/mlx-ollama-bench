# MLX vs. Ollama — Automation Toolkit

## What this is

A standing regression-testing harness that answers one repeatable question: **for a given model, is Ollama's native MLX backend actually faster than its standard GGUF backend on this Mac, and by how much?**

It exists because that question doesn't have a one-time answer — it needs re-asking every time you swap the model backing your agent team (new Gemma release, different size, different quant), every time you update Ollama itself, or every time you're deciding whether a new model is even worth trying on-device. This toolkit automates the whole cycle — spin up an isolated Ollama instance, pull both backend variants of a model, run a matched prompt matrix against each, tear down, report — so re-running that question later is a `config.json` edit and one command, not a rebuild.

It was originally built to evaluate `gemma4:12b` for [Bedrock Intelligence](https://bedrockintel.com)'s local-first macOS agent team (SwiftUI frontend, Python/Google ADK agents, an MLX-served fine-tuned Gemma on the Secure Controls Framework), but nothing about the harness is specific to that model or product — see "Testing a new model" below.

**What it measures:** raw inference speed only — decode tok/s (how fast the model writes its response), prefill tok/s (how fast it reads the prompt), time-to-first-token, and load time. It does **not** measure whether an agent actually completes its task correctly — that's a separate, harder evaluation this harness deliberately doesn't attempt (see "Known limitations" below).

Automates the test plan from `mlx-vs-ollama-test-plan.md` (the earlier deliverable), with one change driven by your production constraint: **no benchmark traffic ever touches your production Ollama instance.**

## What changed from the manual plan, and why

The original plan had config A (current setup) query your live Ollama directly, since generate calls are read-only. That's fine in isolation, but repeated benchmark load — 3-5 reps × 3 prompt lengths × long-context runs — competing for memory and GPU time with an agent team you called "technically production" isn't a risk worth taking for a same-day speed test.

So this automation spins up a **second, isolated Ollama instance** — different port, different model directory, started and stopped only by these scripts — and pulls fresh copies of your model into it. Every benchmark call (configs A and B) goes to that isolated instance. Your production Ollama (port 11434, default model store) is touched exactly once per run, twice total: a health-check ping before and after, nothing else. If your production instance is busy or briefly unreachable, the benchmark still runs — it doesn't depend on it.

Second constraint you named — Homebrew for tool installs — also solves the "don't break production" problem a second way: `mlx-lm` installs via `brew install mlx-lm` as standalone CLI binaries (`mlx_lm.server`, `mlx_lm.generate`, etc.), not as a `pip install` into any Python environment. It never touches the virtualenv your ADK agent code runs in. The orchestrator script below (`run_matrix.py`) is stdlib-only Python — no `pip install` required for it either.

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

## Homebrew mlx-lm vs. the Python library, and where mlxserver.com fits

You asked whether the Homebrew install of `mlx-lm` has parity with the Python library, and how `mlxserver.com` compares. Checked both directly:

**Homebrew `mlx-lm` has full parity with `pip install mlx-lm`.** The formula (currently pinned to 0.31.3) installs the same package from PyPI via `virtualenv_install_with_resources`, with all 33 sub-dependencies vendored and pinned — it's not a stripped-down build. Every CLI entry point ships: `mlx_lm.server`, `.generate`, `.chat`, `.convert`, `.cache_prompt`, `.lora`, `.evaluate`, `.benchmark`, `.perplexity`, `.fuse`, plus the quantization tools (`.awq`, `.gptq`, `.dwq`, `.dynamic_quant`) and repo management (`.manage`, `.share`, `.upload`). No caveat in the formula excludes any of them. The only Homebrew-specific wrinkles are macOS build patches for two dependencies (`hf-xet`, `sentencepiece`) — packaging details, not missing functionality. **One thing to watch:** the formula also defines a `brew services` launchd entry for `mlx_lm.server`. Don't start it — see the warning at the top of `run_matrix.py`.

**`mlxserver.com` (`mustafaaljadery/mlxserver`) is a different, smaller project — not a Homebrew vs. pip question at all.** It's an unofficial two-person community wrapper around raw MLX (`pip install mlxserver` only, no brew formula), offering seven actions — Chat, Convert, Delete, Generate, List, Pull, Show — an Ollama-like veneer for quick prototyping. It doesn't add capability beyond `mlx-lm`: no LoRA fine-tuning, no evaluation harness, no dedicated benchmark tool, no AWQ/GPTQ/DWQ quantization. Whatever "more functionality in the Python libraries" signal you were picking up, it's likely `mlx-lm` (the full CLI toolkit above) compared against `mlxserver`'s much narrower feature set — not `mlx-lm` itself varying by install method. **Verdict: no gap, and no need to add mlxserver to the test matrix.** It would also reintroduce the exact risk the Homebrew-only approach avoids — a `pip install` with no formula, into whatever Python environment happens to be active.

**One real gap this surfaced, now fixed:** `mlx-lm` ships `mlx_lm.benchmark` — a purpose-built benchmarking CLI (synthetic prompts sized by exact token count, internal multi-trial runs with median reporting, and peak memory alongside prompt/generation tok/s) that the original config-C implementation wasn't using. `run_matrix.py` now shells out to `mlx_lm.benchmark` instead of hand-timing `mlx_lm.generate --verbose` — it's more precise, and it closes the "MLX-side peak memory" gap the original plan left to `asitop`/`powermetrics` as an external observer.

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

## `config.json` — pre-filled, confirm before running

Model fields are already filled in based on what's actually published for your stack (see the earlier model-selection message):

- `model.ollama_tag`: `gemma4:12b` — confirm against your own `ollama list`/`ollama pull gemma4:12b` if you run a different size
- `model.ollama_mlx_tag`: `gemma4:12b-mlx`
- `model.hf_repo`: `mlx-community/gemma-4-12B-it-4bit` — the plain 4-bit conversion, deliberately not `-qat-OptiQ-4bit` (a mixed-precision community requant that would confound the engine comparison with a quantization-format difference)

All three are matched at plain 4-bit. If you re-run at 8-bit for the sanity check the plan recommends, the 8-bit sibling repos exist too: `mlx-community/gemma-4-12B-it-8bit`.

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

## Changelog

**v2 — data-quality fixes, applied after the first full run:**
- Fixed: `prefill_tok_s` could report physically impossible values (millions of tok/s) because Ollama's prompt cache returned near-instant `prompt_eval_duration` on repeated identical prompts across reps. Every request now gets a unique per-rep nonce prepended (`with_rep_nonce()` in `run_matrix.py`), which defeats the cache so every rep does a genuine prefill.
- Fixed: the long-prompt bucket wasn't a valid comparison because `options.num_ctx` was never set, so the GGUF backend silently truncated the prompt to a smaller default context window than the MLX backend used. `num_ctx` is now set explicitly from `config.json` (or derived automatically — see `resolve_num_ctx()`).
- Added: `generate_report.py` now flags (rather than silently reports) any group whose median prefill exceeds a sanity threshold, as a safety net if a cache hit ever slips through again.
- Added: `dry_run.py` is now a first-class script in the repo (previously ad hoc) and uses the same num_ctx/nonce logic as the full run, so a passing dry run is actually representative of what the full run will do.

**v1 — initial harness:** isolated test Ollama instance, config A/B/C matrix, Homebrew-only tooling, `mlx_lm.benchmark` for config C.

## Known limitations

- **Speed only, not correctness.** This measures how fast a model responds, not whether an agent using it actually completes tasks correctly. If you're choosing between two different models (not two backends of the same model), pair this with an actual task-completion eval before deciding.
- **Ollama-API-specific.** Configs A/B are built around Ollama's `/api/generate`. Testing a backend outside Ollama (vLLM, LM Studio, a raw MLX server) would need a new config, not just a `config.json` edit.
- **Config C depends on mlx-lm's architecture support.** Raw `mlx-lm` benchmarking only works for architectures mlx-lm's Homebrew build currently supports — check before enabling `include_mlx_raw` for a new model.
- **No peak-memory tracking for configs A/B.** `peak_mem_gb` is only populated for config C (via `mlx_lm.benchmark`). Worth adding via `asitop`/`powermetrics` sampling around the A/B calls if memory pressure becomes a concern.
- **Synthetic prompts.** `build_prompts()` generates filler text sized by word count, not real captured agent traffic — swap in real ADK agent turns for higher-fidelity numbers (see the comment in `build_prompts()`).

## License

MIT — see [LICENSE](LICENSE).
