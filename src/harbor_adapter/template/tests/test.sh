#!/bin/bash
set -e

# PostTrainBench verification script
# Runs contamination judge (codex CLI) and 3-phase evaluation with retry logic.
# Matches the original run_task.sh evaluation pipeline.
#
# Tamper-resistance design (harbor 0.7.0 separate-verifier mode):
#   - This script runs in a SEPARATE container from the agent (see
#     [verifier].environment_mode = "separate" in task.toml). The agent
#     never has shell or filesystem access to this container, so it
#     can't tamper with evaluate.py, templates/, the Python interpreter,
#     installed packages (vllm, inspect_evals, transformers), or this
#     script itself.
#   - All verifier-side files (evaluate.py, templates/, the v1.1 judge
#     confs/prompts/tools under judges_repo/, metadata.json,
#     evaluation_code/, bfcl_evaluation_code.py) are
#     BAKED INTO the verifier image at build time (see tests/Dockerfile)
#     and live at /tests/.
#   - The agent's workspace at /home/agent/workspace is transferred from
#     the agent container by harbor as a configured artifact (weight
#     files excluded) and contains the agent's training scripts. The
#     contamination judge reads these (cd $WORKSPACE && codex exec ...).
#   - The agent's final_model arrives on a harbor task-declared shared
#     volume mounted READ-ONLY at /mnt/model (see
#     [[verifier.environment.volumes]] in task.toml) — the same
#     per-trial volume the agent wrote to at
#     /home/agent/workspace/final_model. The weights never ride the
#     tar/host artifact path, and this container cannot modify them.
#   - The agent's final_model is the only agent-produced input the
#     verifier executes code against (via vllm). Bad weights are
#     penalized by the eval score, not by tampering.

TESTS="/tests"
WORKSPACE="/home/agent/workspace"
MODEL_DIR="/mnt/model"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier ==="
echo "Tests dir: $TESTS"
echo "Workspace: $WORKSPACE"
echo "Model (read-only shared volume): $MODEL_DIR"
echo "Logs dir: $LOGS_DIR"

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# Check the trained model on the shared volume. The mountpoint always
# exists (harbor creates the volume), so also treat an EMPTY volume as
# "no model" — that means the agent never wrote weights into final_model.
echo ""
echo "=== Checking final_model (at $MODEL_DIR) ==="
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "ERROR: final_model is missing or empty on the shared volume"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Check if final_model has required files
echo "Contents of final_model:"
ls -la "$MODEL_DIR" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Show model config (and preserve the FULL config.json — the judges'
# model_identity_check needs every architecture field, and the shared
# volume is deleted after the trial, so this is the only durable copy).
echo ""
echo "=== Model config.json ==="
cp "$MODEL_DIR/config.json" "$LOGS_DIR/model_config.json"
cat "$MODEL_DIR/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$MODEL_DIR/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$MODEL_DIR/"*.json 2>/dev/null || echo "No json files found"

# ============================================================
# Read metadata for benchmark and model info — from /tests, NOT workspace,
# so the agent can't redirect the verifier by overwriting metadata.json.
# ============================================================
BENCHMARK_ID=""
BENCHMARK_NAME=""
MODEL_ID=""

if [ -f "$TESTS/metadata.json" ]; then
    BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])" 2>/dev/null || echo "")
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    echo "Benchmark ID: $BENCHMARK_ID"
    echo "Benchmark Name: $BENCHMARK_NAME"
    echo "Model: $MODEL_ID"
fi

# ============================================================
# Run the v1.1 reward-hacking judges (API-key mode)
#
# Replaces the old single contamination judge (contamination_judge.py +
# gpt-5.1-codex, subscription-only model that 404s on API keys). The four
# v1.1 judges (data_contamination, api_usage, ptb_lookup, general) run via
# tests/run_judges_apikey.sh using the pristine confs/prompts/tools under
# /tests/judges_repo/ and the OPENAI_API_KEY from [verifier.env]. Verdicts
# land as $LOGS_DIR/judgement_<id>.json — same names the condor pipeline's
# scripts/collect.py consumes. Judge failures are non-fatal: the benchmark
# score below must still be computed.
# ============================================================
echo ""
echo "=== Running v1.1 Reward-Hacking Judges ==="

set +e
TESTS="$TESTS" WORKSPACE="$WORKSPACE" MODEL_DIR="$MODEL_DIR" LOGS_DIR="$LOGS_DIR" \
    bash "$TESTS/run_judges_apikey.sh" 2>&1 | tee "$LOGS_DIR/judges.log"
set -e

# ============================================================
# Evaluation with 3-phase retry logic
# Matches run_task.sh evaluation pipeline.
#
# evaluate.py is run from /tests (untamperable). Some evaluate.py scripts
# (arenahardwriting, healthbench) `from evaluation_code.X import Y`, so
# /tests must be cwd for the import to resolve. final_model arrives on
# the read-only shared volume at /mnt/model, so --model-path is
# absolute.
# ============================================================
echo ""
echo "=== Running evaluation on final_model ==="

cd "$TESTS"

EVAL_COUNTER=0

kill_gpu_processes() {
    echo "Killing GPU processes..."
    # Kill GPU-holding processes EXCEPT PID 1 (container init / dumb-init).
    # In Docker/Modal, the agent's vLLM process can get reparented to PID 1,
    # which still holds GPU memory when the verifier starts. Killing PID 1
    # would destroy the entire container.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -v '^$' \
        | while read pid; do
            if [ "$pid" -gt 1 ] 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    sleep 5
}

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"

    kill_gpu_processes

    set +e
    python3 "$TESTS/evaluate.py" \
        --model-path "$MODEL_DIR" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS/templates" \
        --limit -1 \
        ${max_tokens_arg} \
        2>&1 | tee "$LOGS_DIR/final_eval_${eval_num}.txt"
    local exit_code=$?
    set -e
    return $exit_code
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "$LOGS_DIR/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        run_evaluation "$max_tokens_arg" "$EVAL_COUNTER"

        if [ -f "$LOGS_DIR/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# Determine token limit args per benchmark for phase 2 and 3
get_phase2_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 12000" ;;
        arenahardwriting) echo "--max-new-tokens 12288" ;;
        bfcl)        echo "--max-tokens 12000" ;;
        gpqamain)    echo "--max-tokens 12000" ;;
        gsm8k)       echo "--max-tokens 3000" ;;
        healthbench) echo "--max-new-tokens 12288" ;;
        humaneval)   echo "--max-tokens 3000" ;;
        *)           echo "" ;;
    esac
}

get_phase3_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 8000" ;;
        arenahardwriting) echo "--max-new-tokens 8192" ;;
        bfcl)        echo "--max-tokens 8000" ;;
        gpqamain)    echo "--max-tokens 8000" ;;
        gsm8k)       echo "--max-tokens 2000" ;;
        healthbench) echo "--max-new-tokens 8192" ;;
        humaneval)   echo "--max-tokens 2000" ;;
        *)           echo "" ;;
    esac
}

# Phase 1: up to 4 attempts with default tokens
echo ""
echo "--- Phase 1: default token limits (up to 4 attempts) ---"
run_evaluation_with_retry 4 ""

# Phase 2: up to 3 attempts with reduced tokens
PHASE2_TOKENS=$(get_phase2_tokens)
echo ""
echo "--- Phase 2: reduced tokens [${PHASE2_TOKENS}] (up to 3 attempts) ---"
run_evaluation_with_retry 3 "$PHASE2_TOKENS"

# Phase 3: up to 2 attempts with further reduced tokens
PHASE3_TOKENS=$(get_phase3_tokens)
echo ""
echo "--- Phase 3: further reduced tokens [${PHASE3_TOKENS}] (up to 2 attempts) ---"
run_evaluation_with_retry 2 "$PHASE3_TOKENS"

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete (${EVAL_COUNTER} total attempts) ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
    echo "metrics.json contents:"
    cat "$LOGS_DIR/metrics.json"

    # Try to extract accuracy from the metrics JSON
    ACCURACY=$(python3 -c "
import json
try:
    with open('$LOGS_DIR/metrics.json', 'r') as f:
        metrics = json.load(f)
    # Try common metric names
    for key in ['accuracy', 'pass@1', 'score', 'exact_match']:
        if key in metrics:
            print(metrics[key])
            break
    else:
        # If no known metric, use first numeric value
        for v in metrics.values():
            if isinstance(v, (int, float)):
                print(v)
                break
        else:
            print(0)
except Exception as e:
    print(f'Error parsing metrics: {e}', file=__import__('sys').stderr)
    print(0)
" 2>&1)

    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created after all evaluation attempts"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"
