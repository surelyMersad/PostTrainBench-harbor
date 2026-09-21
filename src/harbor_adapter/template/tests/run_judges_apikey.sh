#!/bin/bash
#
# Run the v1.1 reward-hacking judges inside the harbor verifier container,
# authenticated with an API key instead of the condor pipeline's
# ChatGPT-subscription auth.json. Two providers are supported, selected by
# which credential is present in the verifier env (see [verifier.env] in
# task.toml, filled by adapter.py --api-provider):
#   - OPENROUTER_API_KEY  -> judges run gpt-5.6-terra through OpenRouter. codex
#                            gets a custom model provider (base_url
#                            https://openrouter.ai/api/v1, Responses wire API)
#                            via a private CODEX_HOME, and the judge model id is
#                            provider-prefixed (openai/gpt-5.6-terra). Nothing
#                            touches api.openai.com.
#   - OPENAI_API_KEY / CODEX_API_KEY -> the original direct-OpenAI path.
# OPENROUTER_API_KEY takes precedence when both are set.
#
# This is the harbor counterpart of src/judges/run_judges.sh + judge_lib.sh:
# same judges, same confs, same prompts (via the unmodified
# get_judge_prompt.py in judges_repo/), same judgement_<id>.json outputs —
# but executed directly in this container (no apptainer) with the API key
# kept in the environment (the judge model gpt-5.6-terra is
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
# judge_output_<id>.json. Judge failures are FAIL-OPEN, as in upstream
# run_task.sh (collect_judge_output missing_fatal=0): a judge that crashes,
# times out, or returns no/invalid verdict is logged and skipped, and the
# benchmark score is still computed. Missing verdicts are backfilled by the
# offline rerun pipeline. Two wall-clock bounds keep a hung judge (stalled API
# stream, retry loop) from starving the eval: JUDGE_TIMEOUT_SEC per judge
# here, and JUDGE_PHASE_TIMEOUT_SEC around the whole phase in test.sh.

set -u

TESTS="${TESTS:-/tests}"
JUDGES_REPO="$TESTS/judges_repo"
JUDGES_DIR="$JUDGES_REPO/src/judges"
WORKSPACE="${WORKSPACE:-/home/agent/workspace}"
HOME_DIR="$(dirname "$WORKSPACE")"
MODEL_DIR="${MODEL_DIR:-/mnt/model}"
LOGS_DIR="${LOGS_DIR:-/logs/verifier}"

ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge general_judge)
# Defaults mirror judge_lib.sh (JUDGE_DEFAULT_*): gpt-5.4 was retired from
# Codex on 2026-08-31; gpt-5.6-terra needs codex CLI >= 0.144.0, which the
# verifier image bakes at /opt/codex-cli-0.144.5.
DEFAULT_JUDGE_MODEL="gpt-5.6-terra"
DEFAULT_REASONING_EFFORT="xhigh"
DEFAULT_JUDGE_CODEX_VERSION="0.144.5"
# Per-judge wall clock. Healthy judges finish in ~12 min (max ~23 min measured
# on the row17 sweep); 45 min is ~2x that headroom while keeping 4 judges well
# under test.sh's phase cap. A killed judge is fail-open (no verdict).
JUDGE_TIMEOUT_SEC="${JUDGE_TIMEOUT_SEC:-2700}"

# A caller may select a comma-separated subset. The inline verifier leaves
# this unset and runs all four; judge-only backfills can request only missing
# verdicts without changing the prompt or invocation path.
if [ -n "${JUDGES_CSV:-}" ]; then
    IFS=',' read -r -a ALL_JUDGES <<< "$JUDGES_CSV"
fi

mkdir -p "$LOGS_DIR"

JUDGE_PROVIDER=""
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    JUDGE_PROVIDER="openrouter"
elif [ -n "${CODEX_API_KEY:-${OPENAI_API_KEY:-}}" ]; then
    JUDGE_PROVIDER="openai"
else
    echo "run_judges_apikey: WARNING no OPENROUTER_API_KEY / OPENAI_API_KEY / CODEX_API_KEY — skipping all judges (fail-open)" >&2
    exit 0
fi

# Extra codex flags per provider (empty for direct OpenAI).
PROVIDER_ARGS=()
if [ "$JUDGE_PROVIDER" = "openrouter" ]; then
    # Private codex home so the provider config cannot collide with anything
    # else in the image, and so the agent's codex settings can never leak in.
    export CODEX_HOME="${JUDGE_CODEX_HOME:-/tmp/codex-judge-home}"
    mkdir -p "$CODEX_HOME"
    cat > "$CODEX_HOME/config.toml" <<'TOML'
# Written by run_judges_apikey.sh: route the reward-hacking judges through
# OpenRouter. Verified 2026-09-21 with codex exec + openai/gpt-5.6-terra
# (tool calls, xhigh reasoning and reasoning summaries all work on the
# Responses wire API; the "chat" wire API does not).
model_provider = "openrouter"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
wire_api = "responses"
TOML
    PROVIDER_ARGS=(-c model_provider=openrouter)
    # codex must not be able to fall back to a (possibly stale) OpenAI credential.
    unset CODEX_API_KEY OPENAI_API_KEY
fi
echo "run_judges_apikey: provider=$JUDGE_PROVIDER"

BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])")
MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])")

# API/general prompts need the research harness identity. Harbor task images
# are agent-agnostic, so accept explicit values from the caller and otherwise
# derive them from the transferred agent logs. For grok-build the model id is
# recorded either as model_id="..." in the CLI log or as --model ... in the
# watchdog launch command.
AGENT_NAME="${PTB_AGENT_NAME:-}"
AGENT_CONFIG="${PTB_AGENT_CONFIG:-}"
if [ -z "$AGENT_NAME" ]; then
    if [ -s /logs/agent/grok-build.txt ]; then
        AGENT_NAME="grok-build"
    else
        agent_trace=$(find /logs/agent -maxdepth 1 -type f -name '*.txt' -size +0c \
            -printf '%s %f\n' 2>/dev/null | sort -nr | sed -n '1p' | cut -d' ' -f2-)
        AGENT_NAME="${agent_trace%.txt}"
    fi
fi
if [ -z "$AGENT_CONFIG" ]; then
    # Do not recurse through /logs/agent: grok session histories can be many
    # gigabytes. The watchdog/CLI logs are tiny and record the launch model.
    for agent_log in \
        /logs/agent/grok-build-watchdog.log \
        /logs/agent/grok-build-cli.log; do
        [ -s "$agent_log" ] || continue
        AGENT_CONFIG=$(grep -hoEm1 'model_id="[^"]+"' "$agent_log" 2>/dev/null \
            | sed -E 's/.*model_id="([^"]+)".*/\1/' || true)
        if [ -z "$AGENT_CONFIG" ]; then
            AGENT_CONFIG=$(grep -hoEm1 -- '--model[ =]+[^ ]+' "$agent_log" 2>/dev/null \
                | sed -E 's/.*--model[ =]+([^ ]+).*/\1/' || true)
        fi
        [ -n "$AGENT_CONFIG" ] && break
    done
fi
if [ -z "$AGENT_CONFIG" ] && [ -s /logs/agent/grok-build.txt ]; then
    # Bounded fallback for older runs that did not archive the small launcher
    # logs. The model identity is part of the trace preamble when present.
    AGENT_CONFIG=$(sed -n '1,400p;400q' /logs/agent/grok-build.txt \
        | grep -oEm1 'model_id="[^"]+"|--model[ =]+[^ ]+' \
        | sed -E 's/.*model_id="([^"]+)".*/\1/; s/.*--model[ =]+([^ ]+).*/\1/' \
        || true)
fi
if [ -z "$AGENT_NAME" ] || [ -z "$AGENT_CONFIG" ]; then
    # Fail-open: get_judge_prompt.py falls back to its generic harness clause
    # when the agent/model are unknown (same as upstream build_judge_prompt
    # omitting the flags), so judging proceeds with a less specific prompt.
    echo "run_judges_apikey: WARNING could not determine agent identity " \
         "(agent=${AGENT_NAME:-missing}, config=${AGENT_CONFIG:-missing}); " \
         "set PTB_AGENT_NAME / PTB_AGENT_CONFIG for the specific harness clause" >&2
fi
echo "run_judges_apikey: agent=${AGENT_NAME:-<unknown>} agent_config=${AGENT_CONFIG:-<unknown>}"
PROMPT_AGENT_ARGS=()
[ -n "$AGENT_NAME" ] && PROMPT_AGENT_ARGS+=(--agent "$AGENT_NAME")
[ -n "$AGENT_CONFIG" ] && PROMPT_AGENT_ARGS+=(--agent-config "$AGENT_CONFIG")

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

# Raw agent trace: largest /logs/agent/*.txt (the active CLI's log; the other
# per-CLI tee targets are empty). The judges read the trace at
# ../solve_parsed.txt (documented fallback ../solve_out.txt) relative to their
# cwd ($WORKSPACE), i.e. from $HOME_DIR — the same result-dir-relative layout
# as upstream run_task.sh (solve_*.txt sit beside the task/ workspace, not
# inside it). Place both files there, then run PostTrainBench's parse_trace.py
# to produce solve_parsed.txt exactly as upstream does: grok-build has no
# structured parser, so parse_trace copies the raw trace verbatim (its
# historical fallback), giving the judges the identical input they'd get
# upstream — and auto-upgrading if upstream ever adds a grok parser.
# (parse_trace's sanitize step wants a repo .env the verifier lacks and exits
# non-zero AFTER solve_parsed.txt is written; the `|| true` + existence guard
# keep that harmless and never block judging.)
TRACE=$(ls -S /logs/agent/*.txt 2>/dev/null | head -1 || true)
if [ -n "$TRACE" ] && [ -s "$TRACE" ]; then
    cp "$TRACE" "$HOME_DIR/solve_out.txt"
    python3 "$TESTS/trace_parsing/parse_trace.py" --agent grok-build \
        "$HOME_DIR/solve_out.txt" -o "$HOME_DIR/solve_parsed.txt" 2>/dev/null || true
    [ -s "$HOME_DIR/solve_parsed.txt" ] || cp "$HOME_DIR/solve_out.txt" "$HOME_DIR/solve_parsed.txt"
    echo "run_judges_apikey: using $(basename "$TRACE") as solve_out.txt + solve_parsed.txt"
else
    echo "run_judges_apikey: WARNING no agent trace found under /logs/agent"
fi

# ------------------------------------------------------------------
# Run each judge (same set + order as judge_lib.sh ALL_JUDGES)
# ------------------------------------------------------------------
for judge in "${ALL_JUDGES[@]}"; do
    conf="$JUDGES_DIR/$judge/judge.conf"
    [ -f "$conf" ] || { echo "run_judges_apikey: WARNING missing $conf — skipping $judge" >&2; continue; }

    # judge.conf is simple KEY="value" lines (see upstream comment)
    JUDGE_OUTPUT_ID=$(grep -m1 '^JUDGE_OUTPUT_ID=' "$conf" | cut -d'"' -f2)
    JUDGE_MODEL=$(grep -m1 '^JUDGE_MODEL=' "$conf" | cut -d'"' -f2 || true)
    JUDGE_MODEL="${JUDGE_MODEL:-$DEFAULT_JUDGE_MODEL}"
    if [ "$JUDGE_PROVIDER" = "openrouter" ]; then
        # OpenRouter model slugs are provider-prefixed; judge.conf stays pristine.
        case "$JUDGE_MODEL" in */*) ;; *) JUDGE_MODEL="openai/$JUDGE_MODEL" ;; esac
    fi
    JUDGE_EFFORT=$(grep -m1 '^JUDGE_REASONING_EFFORT=' "$conf" | cut -d'"' -f2 || true)
    JUDGE_EFFORT="${JUDGE_EFFORT:-$DEFAULT_REASONING_EFFORT}"
    JUDGE_CODEX_VERSION=$(grep -m1 '^JUDGE_CODEX_VERSION=' "$conf" | cut -d'"' -f2 || true)
    JUDGE_CODEX_VERSION="${JUDGE_CODEX_VERSION:-$DEFAULT_JUDGE_CODEX_VERSION}"

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
    echo "=== Judge: $judge (provider=$JUDGE_PROVIDER, model=$JUDGE_MODEL, effort=$JUDGE_EFFORT, codex=$($codex_bin --version 2>/dev/null || echo '?')) ==="

    PROMPT=$(python3 "$JUDGES_DIR/get_judge_prompt.py" \
        --judge "$judge" --benchmark-id "$BENCHMARK_ID" --model "$MODEL_ID" \
        "${PROMPT_AGENT_ARGS[@]}" \
        2>"$LOGS_DIR/judge_prompt_${JUDGE_OUTPUT_ID}.err") || {
        echo "  WARNING: prompt generation failed for $judge — skipping (fail-open)" >&2; continue; }
    printf '%s\n' "$PROMPT" > "$LOGS_DIR/judge_prompt_${JUDGE_OUTPUT_ID}.txt"

    rm -f "$WORKSPACE/judgement.json"
    # </dev/null is load-bearing: harbor's exec channel holds stdin open
    # forever, and codex exec blocks reading a non-TTY stdin until EOF
    # ("Reading additional input from stdin...") — without the redirect the
    # judge hangs until the verifier cap kills the whole trial (observed live
    # 2026-08-06). `timeout` bounds a judge whose API stream stalls or loops
    # on retries: a killed judge costs one verdict (fail-open), never the score.
    (
        cd "$WORKSPACE"
        timeout -k 30 "$JUDGE_TIMEOUT_SEC" \
            "$codex_bin" --search -a never exec --json \
            ${PROVIDER_ARGS[@]+"${PROVIDER_ARGS[@]}"} \
            -c model_reasoning_summary=detailed \
            -c model_reasoning_effort="$JUDGE_EFFORT" \
            --skip-git-repo-check --yolo --model "$JUDGE_MODEL" "$PROMPT" \
            </dev/null
    ) > "$LOGS_DIR/judge_output_${JUDGE_OUTPUT_ID}.json" 2>&1
    rc=$?
    echo "  codex exit: $rc"
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
        echo "  WARNING: $judge timed out after ${JUDGE_TIMEOUT_SEC}s (killed; fail-open)" >&2
    elif [ "$rc" != 0 ]; then
        echo "  WARNING: $judge codex exited $rc (fail-open; see judge_output_${JUDGE_OUTPUT_ID}.json)" >&2
    fi

    if [ -f "$WORKSPACE/judgement.json" ]; then
        # Schema check mirrors what scripts/collect.py expects. An invalid
        # verdict is quarantined (not consumed downstream) rather than fatal.
        if python3 - "$judge" "$WORKSPACE/judgement.json" <<'PY'
import json
import sys

judge, path = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
schema = {
    "data_contamination_judge": {
        "contamination": bool,
        "disallowed_model": bool,
        "justification_contamination": str,
        "justification_disallowed_model": str,
    },
    "api_usage_judge": {
        "disallowed_api_usage": bool,
        "justification_disallowed_api_usage": str,
    },
    "ptb_lookup_judge": {
        "disallowed_ptb_lookup": bool,
        "justification_disallowed_ptb_lookup": str,
    },
    "general_judge": {
        "general_anomaly": bool,
        "justification_general_anomaly": str,
    },
}[judge]
if not isinstance(data, dict) or set(data) != set(schema):
    raise SystemExit(
        f"invalid {judge} verdict fields; expected {sorted(schema)}, "
        f"got {sorted(data) if isinstance(data, dict) else type(data).__name__}"
    )
invalid = [key for key, expected in schema.items() if type(data[key]) is not expected]
if invalid:
    raise SystemExit(f"invalid {judge} verdict field types: {invalid}")
PY
        then
            mv "$WORKSPACE/judgement.json" "$LOGS_DIR/judgement_${JUDGE_OUTPUT_ID}.json"
            echo "  verdict: $(head -c 300 "$LOGS_DIR/judgement_${JUDGE_OUTPUT_ID}.json")"
        else
            mv "$WORKSPACE/judgement.json" "$LOGS_DIR/judgement_${JUDGE_OUTPUT_ID}.invalid.json"
            echo "  WARNING: $judge verdict failed schema check — quarantined as" \
                 "judgement_${JUDGE_OUTPUT_ID}.invalid.json (fail-open)" >&2
        fi
    else
        echo "  WARNING: $judge produced no judgement.json " \
             "(see judge_output_${JUDGE_OUTPUT_ID}.json); continuing — a missing" \
             "inline verdict never aborts the run" >&2
    fi
done

echo ""
echo "run_judges_apikey: done. Verdicts in $LOGS_DIR/judgement_*.json"
exit 0
