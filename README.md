# TaskBench SFT Target-Format Comparison

A reproducible experiment harness that compares **two supervised fine-tuning
(SFT) target formats** on Microsoft JARVIS / **TaskBench** data, holding
everything else fixed (same base LLM, same data, same split, same
hyper-parameters):

| Mode | Assistant target | What the model emits |
| --- | --- | --- |
| **A – Full JSON** | `{"task_steps", "task_nodes", "task_links"}` | the complete plan |
| **B – Tool Trajectory** | `["Tool_A", "Tool_B", ...]` | only the ordered tool IDs |

The **only** intended difference between the two SFT runs is the prompt's output
instruction and the assistant target. Catalog, user request, system prompt, and
the one-shot example are identical across modes.

> The official TaskBench **gold schema and evaluation logic are reused**, not
> reimplemented. See [`taskbench_sft/official/`](taskbench_sft/official/): the
> upstream `evaluate.py` / `format_data.py` / `inference.py` are vendored
> verbatim for provenance, and the pure metric functions (`flatten`, `prfs`
> wrappers, `ratio_levenshtein`, link-F1, …) are reused so the numbers match.

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core + training deps
# or, minimal (data + eval + tests only, no GPU):
pip install numpy scipy scikit-learn Levenshtein pyyaml pydantic networkx rouge-score pytest
```

- Core data/eval/tests run **CPU-only**.
- Training/inference need `torch`, `transformers`, `peft` (+ `bitsandbytes` on
  Linux/CUDA for `training.method: qlora`).

## 2. Download the official data

```bash
bash scripts/download_data.sh data/raw
```

Fetches the official files for the three domains into `data/raw/<domain>/`
(`data.json`, `tool_desc.json`, `graph_desc.json`, `user_requests.json`) and
writes `SHA256SUMS.txt`. Sample counts are **never hard-coded** — every report is
computed from the data on disk.

## 3. Full pipeline (the main result)

```bash
# 0) one tiny model build is not needed for the real run; just use the default model.
python -m taskbench_sft.cli --config configs/default.yaml stats         # dataset_stats.json
python -m taskbench_sft.cli --config configs/default.yaml split         # artifacts/splits/*
python -m taskbench_sft.cli --config configs/default.yaml token-report  # token_length_report.json
python -m taskbench_sft.cli --config configs/default.yaml run-matrix    # 4 runs + comparison.md
```

or simply:

```bash
bash scripts/run_experiment_matrix.sh      # needs a GPU for the 1.5B default model
```

The main results table is written to `outputs/comparison.md`.

### Run the steps individually

```bash
CFG=configs/default.yaml
# Train one SFT run (writes outputs/<run-name>/ with 3 checkpoints + manifest)
python -m taskbench_sft.cli --config $CFG train --mode full_json  --run-name SFT-Full-JSON
python -m taskbench_sft.cli --config $CFG train --mode trajectory --run-name SFT-Trajectory

# Inference (omit --adapter for the no-SFT baseline; point it at a checkpoint for SFT)
python -m taskbench_sft.cli --config $CFG infer --mode full_json --run-name Base-Full-JSON --split test_all
python -m taskbench_sft.cli --config $CFG infer --mode full_json --run-name SFT-Full-JSON \
    --adapter outputs/SFT-Full-JSON/best_by_common_score --split test_all

# Evaluate a predictions file -> grouped metrics
python -m taskbench_sft.cli --config $CFG evaluate --mode full_json \
    --predictions outputs/SFT-Full-JSON/predictions_test_all.jsonl

# Build the comparison table from metric reports
python -m taskbench_sft.cli --config $CFG compare \
    --reports Base-Full-JSON=outputs/Base-Full-JSON/metrics.json \
              SFT-Full-JSON=outputs/SFT-Full-JSON/metrics.json \
    --out outputs/comparison.md
```

## 4. Smoke test (32 samples, seconds on CPU)

```bash
bash scripts/smoke_test.sh
```

Builds a tiny local Qwen2 model (`scripts/make_tiny_model.py`), then runs the
whole `prepare → train Full-JSON → train Trajectory → infer → evaluate` path in
smoke mode. Output: `outputs_smoke/comparison.md`. (Metrics will be ~0 — the
point is to exercise every code path, not to learn.)

## 5. Unit tests

```bash
pytest -q
```

Covers the 10 required behaviors: node target generation, chain recovery from
unordered links, DAG/disconnected exclusion, repeated-tool-name handling (no
dedup), full-JSON & trajectory round-trip parsing, prompt-token masking,
target-token supervision, and Node/Edge-F1/NED hand-cases.

---

## Project structure

```
taskbench_sft/
  config.py            Pydantic config (everything is configurable via YAML)
  schema.py            Typed gold schema (ToolCatalog, GoldSample, TaskNode, ...)
  targets.py           Canonical Mode-A / Mode-B target serialization
  tokenization.py      Supervised encoding + loss masking (prompt masked, target supervised)
  manifest.py          Reproducibility manifest
  cli.py               Command-line entry point
  experiment.py        4-run matrix orchestration
  official/            Vendored upstream code (verbatim) + reused metric functions
  data/                loader (normalize official data) · topology · split · prepare
  prompts/             catalog serialization · templates + one-shot · builder
  reports/             dataset_stats · token_length · compare
  train/               dataset/collator · model (full/LoRA/QLoRA) · trainer · checkpoint_select
  infer/               deterministic, resumable generation
  eval/                parse · metrics_common · metrics_fulljson · score · evaluator
configs/   default.yaml · smoke.yaml
scripts/   download_data.sh · make_tiny_model.py · smoke_test.sh · run_experiment_matrix.sh
tests/     test_topology · test_targets · test_parse · test_tokenization · test_metrics
```

---

## Data normalization decisions (assumptions, all configurable)

The shipped `data.json` is JSONL where several fields are JSON-encoded strings.
We parse them into the canonical TaskBench schema that the official `evaluate.py`
consumes (`user_request` / `task_steps` / `task_nodes` / `task_links`):

| shipped field | canonical field |
| --- | --- |
| `instruction` | `user_request` |
| `tool_steps` (string) | `task_steps: List[str]` |
| `tool_nodes` (string) | `task_nodes: List[{task, arguments}]` |
| `tool_links` (string) | `task_links: List[{source, target}]` |
| `type` | topology: `single` \| `chain` \| `dag` |

Decisions made (each is a logged exclusion or a config flag, never a silent fix):

1. **Scope = `single` + `chain`.** DAG samples are excluded
   (`data.include_topologies`). Topology is taken from the official `type` field.
2. **Trajectory recovery uses the explicit gold `task_links`.** The shipped data
   encodes resource dependencies inconsistently in arguments
   (`<node-j>` *and* `<output_of_ToolName>`), but provides clean explicit
   `task_links` for every domain. We topologically sort the links into the
   execution order; a `chain` must be a single connected simple path or it is
   excluded (`not_simple_connected_path`). Repeated tool names that make a link
   endpoint ambiguous are excluded (`ambiguous_repeated_names`) rather than
   guessed. Recovery works on node **indices**, so legitimately-repeated tools
   are never de-duplicated.
3. **Catalog-faithful gold** (`data.require_catalog_faithful_gold`, default on).
   The official `tool_nodes` label drifts off-catalog in ~5–12% of samples
   (the sampler ground truth `sampled_nodes` is always catalog-faithful). Such
   samples are excluded (`off_catalog_gold`) so that hallucination rate is
   well-defined and train tool coverage is satisfiable. Malformed gold records
   (bad links/nodes) are excluded as `unparseable_gold`.
4. **`max_seq_length`** is set from the token-length report so it covers ≥99% of
   full-JSON samples (measured p99 ≈ 2481, max ≈ 2654 for Qwen2.5 → default
   **3072**). Samples whose target would be truncated in *either* mode are added
   to a **shared exclusion set** so both modes train on identical IDs. Targets
   are never silently truncated.

## Split

- 80/10/10, stratified by `domain × topology × chain_length_bucket`
  (buckets: `node`, `chain_length_2`, `chain_length_3`, `chain_length_4_plus`).
- Default seed **42**; override with `--seed`.
- **Train tool coverage**: every tool appearing in validation/test must appear in
  train; otherwise the split is re-drawn with a new sub-seed (up to
  `split.max_resamples`). If still impossible, it errors and lists the rare
  tools — it never silently moves samples.
- Outputs: `train.jsonl`, `validation.jsonl`, `test_node.jsonl`,
  `test_chain.jsonl`, `test_all.jsonl`, `split_manifest.json`. Both SFT modes
  read the **same** manifest.

## Training

- Any HF causal LM (`model.name`, default `Qwen/Qwen2.5-1.5B-Instruct`).
- `training.method`: `full` | `lora` | `qlora` (default `qlora`; falls back to
  LoRA if `bitsandbytes` is unavailable). All hyper-parameters are config-driven.
- The two SFT runs share base checkpoint, train/val IDs, seed, optimizer, LoRA
  rank, epochs, and batch strategy.

### Checkpoint selection

Not by validation loss alone. At each eval we generate on (a capped subset of)
the validation set and compute Node F1 / Edge F1 / Sequence Exact Match / NED /
parse validity, and a configurable

```
validation_common_score = 0.4·node_f1 + 0.3·edge_f1 + 0.2·sequence_exact_match + 0.1·parse_valid_rate
```

(`checkpoint_selection` weights). We persist `best_by_loss`,
`best_by_common_score`, and `last_checkpoint`.

## Inference

Deterministic decoding (`do_sample=false`, `num_beams=1`), **no
grammar-constrained decoding** (so each format's intrinsic difficulty is what we
measure). Mode A and Mode B use different `max_new_tokens`, set from the
validation target-length distribution. Predictions are JSONL and **resumable**;
each record stores prompt, raw response, gold, token counts, latency, checkpoint.

## Metrics

**Common (both modes — the core comparison):** Node F1 (official set-based **and**
multiset-aware), Edge F1 (link-based + adjacent-edge from recovered trajectories),
NED, Trajectory Exact Match, hallucinated-tool rate, parse/schema validity,
tool-count accuracy/MAE, over/under-selection, prefix accuracy.

**Full-JSON-specific (Mode A only, never a substitute for the common metrics):**
task-step ROUGE-1/L, parameter-name F1, parameter-value F1, exact-JSON match,
JSON/schema/step-node-alignment/link validity.

All metrics are reported **overall** and grouped by **domain**, **topology**, and
**chain length** (2 / 3 / 4+).

## Compute fairness

Because Full JSON has many more target tokens, we never claim equal cost. Every
run reports: train examples, optimization steps, input tokens, assistant-target
tokens, total processed tokens, wall-clock time, and peak GPU memory
(`train_summary.json`). An optional `training.budget_mode: equal_target_tokens`
is reserved for matching budgets by assistant-target tokens (not the default).

## Reproducibility

Each run writes `outputs/<run_name>/run_manifest.json` with base model
name/revision, tokenizer revision, dataset file hashes, split-manifest hash,
sample-ID counts, git commit, Python/CUDA/torch/transformers/peft versions,
seed, training config, GPU name, training time, and peak memory.

## Implementation principles honored

Reuse the official evaluator; don't modify test gold; don't tune the prompt or
select checkpoints on test; both modes share the exact split; no extra
demonstrations or constrained decoding for either mode; no silent
fixing/mapping/truncation — every exclusion, repair, and parse failure is logged;
type hints throughout; `logging` instead of bare prints; dataclasses/Pydantic for
schemas.
