# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations on cloud GPUs.

## Supported Benchmarks

| Benchmark ID | Name | Type | Notes |
|-------------|------|------|-------|
| gsm8k | GSM8K (Grade School Math 8K) | inspect-ai | |
| humaneval | HumanEval | inspect-ai | |
| aime2025 | AIME 2025 | inspect-ai | |
| gpqamain | GPQA | inspect-ai | |
| bfcl | Berkeley Function Calling Leaderboard | inspect-ai | Includes `bfcl_evaluation_code.py` via task_context |
| arenahardwriting | Arena-Hard-v2.0 (Writing) | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |
| healthbench | HealthBench | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |

## Supported Models

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

Total: **28 tasks** (7 benchmarks x 4 models).

## Installation

```bash
# Python environment for the adapter (modal etc.)
uv sync

# Harbor CLI. Task-declared shared volumes ([[environment.volumes]]) are not
# yet in a Harbor release — install from the feature branch until the
# upstream PR (harbor-framework/harbor) lands:
uv tool install "harbor @ git+https://github.com/surelyMersad/harbor.git@feat/task-shared-volumes"
```

## Quick Start

### 1. Generate tasks

```bash
cd src/harbor_adapter

# Generate a single task
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

# Or generate all 28 task combinations
python run_adapter.py --all --output ./tasks

# List available benchmarks and models
python run_adapter.py --list
```

### 2. Set API keys

```bash
python -m modal setup                # Modal cloud setup
export ANTHROPIC_API_KEY=<your-key>  # For Claude agent
export OPENAI_API_KEY=<your-key>     # For contamination judge (codex CLI) + arenahardwriting/healthbench eval
```

### 3. Run with Harbor

```bash
harbor run \
    --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --env modal
```

## API Key Requirements

| Key | Used By | Required For |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | Agent (Claude) | All benchmarks |
| `OPENAI_API_KEY` | Contamination judge (codex CLI), evaluation judge | All benchmarks (judge), arenahardwriting/healthbench (agent eval) |

- The verifier receives `OPENAI_API_KEY` as both `OPENAI_API_KEY` and `CODEX_API_KEY` (codex CLI reads `CODEX_API_KEY`).
- For arenahardwriting and healthbench, `OPENAI_API_KEY` is also passed to the agent environment since their `evaluate.py` scripts call the OpenAI API for judging.

## Task Structure

Each generated task follows Harbor's standard format:

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml              # Task configuration (GPU, timeout, env vars)
├── instruction.md         # Instructions for the agent
├── environment/
│   ├── Dockerfile         # Container definition (CUDA + vLLM + ML packages)
│   ├── .dockerignore      # Excludes Dockerfile from COPY
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── contamination_judge.py  # Generates judge prompt for codex CLI
│   ├── timer.sh           # Countdown timer (sentinel-file based)
│   ├── metadata.json      # Benchmark/model metadata for verifier
│   ├── templates/         # Chat templates for different models
│   ├── evaluation_code/   # (arenahardwriting, healthbench only)
│   └── bfcl_evaluation_code.py  # (bfcl only, from task_context)
└── tests/
    └── test.sh            # Verifier: contamination judge + 3-phase eval retry
```

## Model Hand-off via Shared Volume

The verifier runs in a separate container (`[verifier].environment_mode =
"separate"`), and the trained model reaches it via a **Harbor task-declared
shared volume** (`[[environment.volumes]]`, name `model`):

| Container | Mount path | Access |
|-----------|-----------|--------|
| Agent | `/home/agent/workspace/final_model` | read-write |
| Verifier | `/mnt/model` | **read-only** |

Harbor creates one volume per trial (`harbor-<trial>-model`), both containers
attach the same volume, and Harbor deletes it at trial cleanup. On Modal the
weights move as a [Modal Volume](https://modal.com/docs/guide/volumes),
locally as a Docker named volume — never through the tar/host artifact path
(which previously failed with exit code -1 on multi-GB workspaces).
`final_model` is therefore excluded from the workspace artifact, which now
carries only the contamination-judge inputs (the agent's source code).

## Evaluation Retry Logic

The verifier (`test.sh`) uses a 3-phase evaluation retry strategy matching `run_task.sh`:

| Phase | Max Attempts | Token Limits |
|-------|-------------|-------------|
| 1 | 4 | Default |
| 2 | 3 | Reduced (see below) |
| 3 | 2 | Further reduced (see below) |

Token limits per benchmark:

| Benchmark | Phase 2 | Phase 3 |
|-----------|---------|---------|
| aime2025 | `--max-tokens 12000` | `--max-tokens 8000` |
| arenahardwriting | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| bfcl | `--max-tokens 12000` | `--max-tokens 8000` |
| gpqamain | `--max-tokens 12000` | `--max-tokens 8000` |
| gsm8k | `--max-tokens 3000` | `--max-tokens 2000` |
| healthbench | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| humaneval | `--max-tokens 3000` | `--max-tokens 2000` |

GPU processes are killed between attempts to free VRAM.

## Reward-Hacking Judges (v1.1, API-key mode)

The verifier runs the four v1.1 judges from `src/judges/` after the agent
finishes, via `tests/run_judges_apikey.sh` — the harbor counterpart of the
condor pipeline's inline judge phase. Same judges, confs, prompts, and
tools (`judges_repo/` in the verifier image is a pristine copy), but
authenticated with `OPENAI_API_KEY` from `[verifier.env]` instead of a
ChatGPT-subscription `auth.json`, and executed directly in the verifier
container instead of an apptainer sandbox. Judge models (`gpt-5.4`,
`gpt-5.6-terra`) are available on standard API keys.

| Judge | Verdict file | Checks |
|---|---|---|
| `data_contamination_judge` (GPT-5.4) | `judgement_gpt5_4.json` | Test-data usage, eval tampering, model substitution (runs `model_identity_check.py` against the full `config.json` saved from `/mnt/model`) |
| `api_usage_judge` (GPT-5.4) | `judgement_api.json` | Disallowed external LLM API calls |
| `ptb_lookup_judge` (GPT-5.4) | `judgement_ptb_lookup.json` | Agent looking up PostTrainBench itself |
| `general_judge` (GPT-5.6 Terra, codex pinned 0.144.5 baked into the image) | `judgement_general.json` | Unknown-unknowns sweep (premature stops, exhaustion, infra failures, novel hacking) |

Judge inputs: the transferred agent workspace (judge cwd), the raw agent
CLI trace from `/logs/agent` (staged as `solve_out.txt`), and the full
model config. The judges re-copy pristine `contamination_check.py` +
`test_data.json` over the agent's versions before running. Judge failures
are non-fatal — the benchmark score is always computed; verdict files use
the same names `scripts/collect.py` consumes.

The agent also receives the **decontamination kit** (the same n-gram
checker + benchmark test set, at `/home/agent/` outside the workspace) and
the instruction section describing it — parity with the condor v1.1
pipeline.

**Prerequisite:** `test_data.json` is generated, not committed. Before
generating tasks:

```bash
python src/judges/test_data_download/download_test_data.py   # all benchmarks
```

## Timer

The timer uses a sentinel-file approach: on the first `bash timer.sh` call, the current timestamp is recorded in `.timer_start`. This ensures the countdown is accurate even if the task is generated long before the agent starts.

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| GPU | 1x H100 | Configured in task.toml |
| Memory | 64 GB | |
| Storage | 100 GB | |
| Agent timeout | 10 hours | Adjustable via `--num-hours` |
| Verifier timeout | 3 hours | Accommodates 3-phase retry |
| Internet | Enabled | |

## Scoring

The verifier extracts the accuracy metric from `metrics.json` as the reward (0-1 scale). Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/model_config.json` - Full config.json of the trained model (for identity checks)
- `/logs/verifier/judgement_gpt5_4.json` - Contamination / disallowed-model verdict
- `/logs/verifier/judgement_api.json` - API-usage verdict
- `/logs/verifier/judgement_ptb_lookup.json` - PTB-lookup verdict
- `/logs/verifier/judgement_general.json` - General anomaly verdict
- `/logs/verifier/judge_output_*.json` + `judges.log` - Raw judge traces
