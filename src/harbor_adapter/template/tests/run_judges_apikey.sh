#!/bin/bash
#
# Run the v1.1 reward-hacking judges inside the harbor verifier container,
# authenticated with an OpenAI API key (OPENAI_API_KEY / CODEX_API_KEY)
# instead of the condor pipeline's ChatGPT-subscription auth.json.
#
# This is the harbor counterpart of src/judges/run_judges.sh + judge_lib.sh:
# same judges, same confs, same prompts (via the unmodified
# get_judge_prompt.py in judges_repo/), same judgement_<id>.json outputs —
# but executed directly in this container (no apptainer) with the API key
# kept in the environment (judge models gpt-5.4 / gpt-5.6-terra are
# API-accessible; the old subscription-only path blanked these keys).
#
# Layout expectations (prepared by tests/Dockerfile + adapter.py):
#   /tests/judges_repo/src/judges/...            confs, prompts, tools, get_judge_prompt.py
#   /tests/judges_repo/src/eval/tasks/<b>/info.json
#   /tests/test_data.json                        pristine benchmark test set
#   /tests/metadata.json                         benchmark_id / model_id
#   /mnt/model                                   read-only shared volume (trained model)
#   /home/agent/workspace                        agent workspace (artifact transfer)
#   /logs/agent/*.txt                            raw agent CLI trace (artifact transfer)
#
# Verdicts land in $LOGS_DIR as judgement_<id>.json; raw codex traces as
# judge_output_<id>.json. Judge failures are non-fatal (fail-open): the
# benchmark score must still be computed; missing verdicts are handled
# downstream exactly like pre-judge condor runs.

set -u

TESTS="${TESTS:-/tests}"
JUDGES_REPO="$TESTS/judges_repo"
JUDGES_DIR="$JUDGES_REPO/src/judges"
WORKSPACE="${WORKSPACE:-/home/agent/workspace}"
HOME_DIR="$(dirname "$WORKSPACE")"
MODEL_DIR="${MODEL_DIR:-/mnt/model}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"

ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge general_judge)
DEFAULT_JUDGE_MODEL="gpt-5.4"
DEFAULT_REASONING_EFFORT="xhigh"

mkdir -p "$LOGS_DIR"

if [ -z "${CODEX_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
    echo "run_judges_apikey: no OPENAI_API_KEY/CODEX_API_KEY — skipping all judges"
    exit 0
fi

BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])")
MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])")

# ------------------------------------------------------------------
# Prepare the judge sandbox (mirrors judge_lib.sh prepare_judge_sandbox):
# judges run with cwd=$WORKSPACE and must never read agent-modified
# copies of the tools/test set — re-copy the pristine ones over.
# ------------------------------------------------------------------
cp "$JUDGES_DIR/judge_tools/contamination_check.py" "$WORKSPACE/contamination_check.py"
cp "$TESTS/test_data.json" "$WORKSPACE/test_data.json"
cp "$JUDGES_DIR/judge_tools/model_identity_check.py" "$HOME_DIR/model_identity_check.py"
cp -r "$JUDGES_DIR/judge_tools/reference_configs" "$HOME_DIR/reference_configs"

# Full config of the trained model, at the path the judges reference
# (../final_model_config.json relative to the workspace).
if [ -f "$MODEL_DIR/config.json" ]; then
    cp "$MODEL_DIR/config.json" "$HOME_DIR/final_model_config.json"
else
    echo "run_judges_apikey: WARNING no $MODEL_DIR/config.json (no model?)"
fi

# Raw agent trace: largest /logs/agent/*.txt (the active CLI's log; the
# other per-CLI tee targets are empty). Judges read solve_parsed.txt with a
# documented fallback to solve_out.txt — harbor provides the raw trace.
TRACE=$(ls -S /logs/agent/*.txt 2>/dev/null | head -1 || true)
if [ -n "$TRACE" ] && [ -s "$TRACE" ]; then
    cp "$TRACE" "$WORKSPACE/solve_out.txt"
    echo "run_judges_apikey: using $(basename "$TRACE") as solve_out.txt"
else
    echo "run_judges_apikey: WARNING no agent trace found under /logs/agent"
fi

# ------------------------------------------------------------------
# Run each judge (same set + order as judge_lib.sh ALL_JUDGES)
# ------------------------------------------------------------------
for judge in "${ALL_JUDGES[@]}"; do
    conf="$JUDGES_DIR/$judge/judge.conf"
    [ -f "$conf" ] || { echo "run_judges_apikey: missing $conf — skipping $judge"; continue; }

    # judge.conf is simple KEY="value" lines (see upstream comment)
    JUDGE_OUTPUT_ID=$(grep -m1 '^JUDGE_OUTPUT_ID=' "$conf" | cut -d'"' -f2)
    JUDGE_MODEL=$(grep -m1 '^JUDGE_MODEL=' "$conf" | cut -d'"' -f2)
    JUDGE_MODEL="${JUDGE_MODEL:-$DEFAULT_JUDGE_MODEL}"
    JUDGE_EFFORT=$(grep -m1 '^JUDGE_REASONING_EFFORT=' "$conf" | cut -d'"' -f2)
    JUDGE_EFFORT="${JUDGE_EFFORT:-$DEFAULT_REASONING_EFFORT}"
    JUDGE_CODEX_VERSION=$(grep -m1 '^JUDGE_CODEX_VERSION=' "$conf" | cut -d'"' -f2)

    # Pinned codex releases are baked into the verifier image at
    # /opt/codex-cli-<version>/bin/codex (see tests/Dockerfile); fall back
    # to a runtime npm install, then to the container default.
    codex_bin="codex"
    if [ -n "$JUDGE_CODEX_VERSION" ]; then
        pin="/opt/codex-cli-${JUDGE_CODEX_VERSION}/bin/codex"
        if [ -x "$pin" ]; then
            codex_bin="$pin"
        else
            echo "  pinned codex ${JUDGE_CODEX_VERSION} not baked — npm-installing..."
            npm install -g --prefix "/tmp/codex-cli-${JUDGE_CODEX_VERSION}" --no-fund --no-audit \
                "@openai/codex@${JUDGE_CODEX_VERSION}" >/dev/null 2>&1 \
                && codex_bin="/tmp/codex-cli-${JUDGE_CODEX_VERSION}/bin/codex" \
                || echo "  install failed — using container default codex"
        fi
    fi

    echo ""
    echo "=== Judge: $judge (model=$JUDGE_MODEL, effort=$JUDGE_EFFORT, codex=$($codex_bin --version 2>/dev/null || echo '?')) ==="

    PROMPT=$(python3 "$JUDGES_DIR/get_judge_prompt.py" \
        --judge "$judge" --benchmark-id "$BENCHMARK_ID" --model "$MODEL_ID" 2>"$LOGS_DIR/judge_prompt_${JUDGE_OUTPUT_ID}.err") || {
        echo "  prompt generation failed — skipping $judge"; continue; }

    rm -f "$WORKSPACE/judgement.json"
    (
        cd "$WORKSPACE"
        "$codex_bin" --search -a never exec --json \
            -c model_reasoning_summary=detailed \
            -c model_reasoning_effort="$JUDGE_EFFORT" \
            --skip-git-repo-check --yolo --model "$JUDGE_MODEL" "$PROMPT"
    ) > "$LOGS_DIR/judge_output_${JUDGE_OUTPUT_ID}.json" 2>&1
    rc=$?
    echo "  codex exit: $rc"

    if [ -f "$WORKSPACE/judgement.json" ]; then
        mv "$WORKSPACE/judgement.json" "$LOGS_DIR/judgement_${JUDGE_OUTPUT_ID}.json"
        echo "  verdict: $(head -c 300 "$LOGS_DIR/judgement_${JUDGE_OUTPUT_ID}.json")"
    else
        echo "  WARNING: $judge produced no judgement.json (see judge_output_${JUDGE_OUTPUT_ID}.json)"
    fi
done

echo ""
echo "run_judges_apikey: done. Verdicts in $LOGS_DIR/judgement_*.json"
exit 0
