import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# Which hosted API grades the LLM-judged benchmarks (healthbench,
# arenahardwriting) and powers the v1.1 codex judges in the verifier.
#   openai     -> evaluate.py + OPENAI_API_KEY/CODEX_API_KEY (direct api.openai.com)
#   openrouter -> evaluate_openrouter.py + OPENROUTER_API_KEY; the judge runner
#                 routes codex through OpenRouter (openai/gpt-5.6-terra).
API_PROVIDERS = ("openai", "openrouter")
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
GRADED_BENCHMARKS = ("arenahardwriting", "healthbench")
GRADER_API_ENV_MARKER = "# @@GRADER_API_ENV@@"

# PostTrainBench source directory (relative to repo root)
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent

# Claude-specific instruction clause (from original get_prompt.py)
CLAUDE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""


@dataclass
class BenchmarkInfo:
    task_id: str           # e.g., "gsm8k"
    benchmark_name: str    # e.g., "GSM8K (Grade School Math 8K)"
    setup_note: str = ""   # Additional setup instructions


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


BENCHMARKS = {
    "gsm8k": BenchmarkInfo(
        task_id="gsm8k",
        benchmark_name="GSM8K (Grade School Math 8K)",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai. "
                   "Also if you run into issues with the evaluate.py script, this is likely "
                   "due to memory constraints on the GPU. In this case please decrease "
                   "--max-connections or --max-tokens.\n"
    ),
    "humaneval": BenchmarkInfo(
        task_id="humaneval",
        benchmark_name="HumanEval",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "aime2025": BenchmarkInfo(
        task_id="aime2025",
        benchmark_name="AIME 2025",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "gpqamain": BenchmarkInfo(
        task_id="gpqamain",
        benchmark_name="GPQA",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "bfcl": BenchmarkInfo(
        task_id="bfcl",
        benchmark_name="Berkeley Function Calling Leaderboard",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "arenahardwriting": BenchmarkInfo(
        task_id="arenahardwriting",
        benchmark_name="Arena-Hard-v2.0 (Writing)",
        setup_note="",
    ),
    "healthbench": BenchmarkInfo(
        task_id="healthbench",
        benchmark_name="HealthBench",
        setup_note="",
    ),
}

MODELS = {
    "qwen3-1.7b": ModelInfo(
        model_id="Qwen/Qwen3-1.7B-Base",
        short_name="qwen3-1.7b"
    ),
    "qwen3-4b": ModelInfo(
        model_id="Qwen/Qwen3-4B-Base",
        short_name="qwen3-4b"
    ),
    "smollm3-3b": ModelInfo(
        model_id="HuggingFaceTB/SmolLM3-3B-Base",
        short_name="smollm3-3b"
    ),
    "gemma3-4b": ModelInfo(
        model_id="google/gemma-3-4b-pt",
        short_name="gemma3-4b"
    ),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 10,
        include_claude_clause: bool = True,
        api_provider: str = "openai",
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Number of hours for the training task (default: 10).
            include_claude_clause: Whether to include the Claude non-interactive clause.
            api_provider: "openai" or "openrouter" — see API_PROVIDERS.
        """
        if api_provider not in API_PROVIDERS:
            raise ValueError(f"api_provider must be one of {API_PROVIDERS}, got {api_provider!r}")
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.api_provider = api_provider
        self.api_key_env = API_KEY_ENV[api_provider]
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    def _grader_api_env_lines(self) -> str:
        """TOML lines that hand the verifier its grader/judge credential."""
        if self.api_provider == "openrouter":
            return 'OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}"'
        return 'OPENAI_API_KEY = "${OPENAI_API_KEY}"\nCODEX_API_KEY = "${OPENAI_API_KEY}"'

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        """Read the human-readable benchmark name from benchmark.txt."""
        bench_file = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        # Fallback to the dataclass info
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "") -> None:
        """Generate task.toml for the Harbor task."""
        # Copy template and adjust timeout based on num_hours
        template_path = TEMPLATE_DIR / "task.toml"
        target_path = task_dir / "task.toml"

        content = template_path.read_text()

        # Adjust agent timeout based on num_hours
        agent_timeout = self.num_hours * 3600  # Convert hours to seconds
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}"
        )

        # Verifier credential for graders + judges (provider-specific).
        if GRADER_API_ENV_MARKER not in content:
            raise ValueError(f"template/task.toml lacks the {GRADER_API_ENV_MARKER} marker")
        content = content.replace(GRADER_API_ENV_MARKER, self._grader_api_env_lines())

        # For arenahardwriting/healthbench, the agent needs the grader key during
        # its run so it can self-score checkpoints with evaluate.py's LLM judge
        # (mirrors upstream info.json required_api_keys). This MUST live under
        # [environment.env]: harbor injects that table into the agent sandbox
        # (environments/modal.py resolves task_env_config.env), whereas a bare
        # [agent.env] maps to AgentConfig, which has no env field and is silently
        # dropped — the cause of the "OPENAI_API_KEY is not set" failures that
        # invalidated the healthbench/arenahardwriting columns in the 2026-08-06 run.
        if benchmark_id in GRADED_BENCHMARKS:
            content += f'\n[environment.env]\n{self.api_key_env} = "${{{self.api_key_env}}}"\n'

        target_path.write_text(content)

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
        benchmark_id: str = "",
    ) -> None:
        """Generate instruction.md for the Harbor task."""
        template_path = TEMPLATE_DIR / "instruction.md"
        target_path = task_dir / "instruction.md"

        content = template_path.read_text()

        # Decontamination Tool section — always present (every PTB task ships a
        # test_data.json, so the checker + test-set copy are always in the agent
        # sandbox home). Mirrors upstream src/eval/general/get_prompt.py; the
        # benchmark name is interpolated here so the block does not depend on the
        # later {benchmark} pass.
        decontamination_tool = (
            "\n## Decontamination Tool\n"
            f"- A copy of the {benchmark_info.benchmark_name} test set is available at "
            "\\`../test_data.json\\`, together with the contamination checker "
            "\\`../contamination_check.py\\`\n"
            "- You can check data which you want to use for training, e.g.:\n"
            "  \\`python ../contamination_check.py --reference ../test_data.json --input train_data.jsonl\\`\n"
            "  The input is JSONL with one document per line (pass \\`--input-format text\\` for "
            "plain text lines, or pipe documents via stdin). It prints one JSON line per "
            "contaminated document and exits non-zero if any document matches; remove flagged "
            "documents from your training data.\n"
            "- Reading and running these two files in the home directory is an allowed exception "
            "to the working-directory rule below.\n"
            "- The test-set copy is provided ONLY as reference input for this checker. Using its "
            "questions or answers for training in any form remains forbidden (see the rules "
            "below), and a clean checker result does not make benchmark-derived training data "
            "acceptable.\n"
        )
        content = content.replace("{decontamination_tool}", decontamination_tool)

        # Rule 10 (eval_api_note): benchmarks whose grading needs a provider key
        # (healthbench/arenahardwriting) get an exception to rule 9, permitting
        # that key for evaluation only. Mirrors upstream get_prompt.py; the key
        # (OPENAI_API_KEY or OPENROUTER_API_KEY per --api-provider) is provisioned
        # into the agent env by generate_task_toml for the same two benchmarks.
        if benchmark_id in GRADED_BENCHMARKS:
            content = content.replace(
                "{eval_api_note}",
                f"10. The \\`{self.api_key_env}\\` in your environment is an exception to the "
                "previous rule: it is provided so that you can run this benchmark's grading "
                "via evaluate.py. Use it for that evaluation only, and never to generate "
                "training data or for any other purpose.\n"
            )
        else:
            content = content.replace("{eval_api_note}", "")

        # Fill in placeholders
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)

        if self.include_claude_clause:
            content += CLAUDE_CLAUSE

        target_path.write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        """Generate timer.sh script that tracks remaining time.

        Reads the start timestamp from the absolute path /timer_start, which
        is written by the task.toml healthcheck immediately before the agent
        launches. Using an absolute path makes the timer immune to the
        agent's `cd`s (the previous sentinel-file approach used
        `dirname "$0"` which resolved differently per cwd).
        """
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
START_FILE="/timer_start"

if [ ! -f "$START_FILE" ]; then
    echo "Timer not initialized (healthcheck has not run yet)."
    exit 1
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the environment/ directory: Dockerfile + agent runtime."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile template and .dockerignore
        shutil.copy(
            TEMPLATE_DIR / "environment" / "Dockerfile",
            env_dir / "Dockerfile"
        )
        dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        # Build-context support files (entrypoint, system monitor,
        # requirements-direct). Shared with tests/ — see _copy_build_context_support.
        self._copy_build_context_support(env_dir)

        # Eval files: evaluate.py, templates/, optional evaluation_code/
        # and task_context contents, plus contamination_judge.py and
        # metadata.json. The agent gets these in /home/agent/workspace
        # (via the Dockerfile's `COPY .`) for fast iteration during
        # training.
        self._copy_eval_files(env_dir, benchmark_id, model_info, benchmark_info)

        # Agent-side decontamination kit (v1.1): the same n-gram checker the
        # judges use + the benchmark test set, relocated by the Dockerfile to
        # /home/agent (outside the workspace). The judge phase re-copies
        # pristine versions, so agent edits cannot affect judging.
        shutil.copy(
            self.posttrainbench_root / "src" / "judges" / "judge_tools" / "contamination_check.py",
            env_dir / "contamination_check.py",
        )
        shutil.copy(self._require_test_data(benchmark_id), env_dir / "test_data.json")

        # timer.sh — agent reads it during the run. Verifier doesn't need it.
        self.generate_timer_sh(env_dir)

    def _require_test_data(self, benchmark_id: str) -> Path:
        """Path to the benchmark's test_data.json; hard error when absent.

        Mirrors upstream run_task.sh, which hard-errors when a benchmark
        lacks test_data.json (the decontamination tooling and the
        contamination judge both depend on it). The file is generated, not
        committed: run
            python src/judges/test_data_download/download_test_data.py --tasks <benchmark>
        """
        test_data = (
            self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "test_data.json"
        )
        if not test_data.is_file():
            raise FileNotFoundError(
                f"{test_data} not found — required for the v1.1 decontamination "
                "tooling and judges. Generate it first:\n"
                f"  python src/judges/test_data_download/download_test_data.py --tasks {benchmark_id}"
            )
        return test_data

    def _copy_judges_repo(self, tests_dir: Path, benchmark_id: str) -> None:
        """Bake the v1.1 judge system into the verifier build context.

        get_judge_prompt.py resolves REPO_ROOT as judges/../.., and reads
        src/eval/tasks/<benchmark>/info.json from there — so we replicate a
        minimal repo layout under tests/judges_repo/ and the script runs
        unmodified inside the verifier container.
        """
        judges_src = self.posttrainbench_root / "src" / "judges"
        repo_dst = tests_dir / "judges_repo"
        judges_dst = repo_dst / "src" / "judges"
        if judges_dst.exists():
            shutil.rmtree(repo_dst)
        judges_dst.parent.mkdir(parents=True, exist_ok=True)

        # Judge confs + prompts + prompt generator + deterministic tools.
        # The condor-only machinery (judge_lib.sh apptainer wrappers, rerun/,
        # test_data_download/) is intentionally excluded.
        judges_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(judges_src / "get_judge_prompt.py", judges_dst / "get_judge_prompt.py")
        shutil.copytree(judges_src / "judge_tools", judges_dst / "judge_tools")
        for judge_dir in judges_src.iterdir():
            if judge_dir.is_dir() and (judge_dir / "judge.conf").is_file():
                shutil.copytree(judge_dir, judges_dst / judge_dir.name)

        # info.json at the repo-relative path get_judge_prompt.py expects.
        info_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "info.json"
        if not info_src.is_file():
            raise FileNotFoundError(
                f"{info_src} not found — required by get_judge_prompt.py for the v1.1 judges."
            )
        info_dst = repo_dst / "src" / "eval" / "tasks" / benchmark_id / "info.json"
        info_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(info_src, info_dst)

    def _copy_build_context_support(self, target_dir: Path) -> None:
        """Copy entrypoint.sh + system_monitor.sh + requirements-direct.txt
        into a Dockerfile build context.

        Both environment/ (agent) and tests/ (verifier under harbor's
        separate-verifier mode) use the same Dockerfile structure and
        need these files at build time. The canonical sources live under
        template/environment/ and containers/.
        """
        # entrypoint.sh — Dockerfile installs it at /usr/local/bin/ and
        # sets it as ENTRYPOINT so its stdout becomes Modal's live log
        # stream (see template/environment/entrypoint.sh).
        entrypoint_src = TEMPLATE_DIR / "environment" / "entrypoint.sh"
        entrypoint_dst = target_dir / "entrypoint.sh"
        shutil.copy(entrypoint_src, entrypoint_dst)
        entrypoint_dst.chmod(0o755)

        # system_monitor.sh — kicked off by entrypoint.sh as a background
        # daemon; ports condor's src/utils/system_monitor.sh.
        monitor_src = TEMPLATE_DIR / "environment" / "system_monitor.sh"
        monitor_dst = target_dir / "system_monitor.sh"
        shutil.copy(monitor_src, monitor_dst)
        monitor_dst.chmod(0o755)

        # containers/requirements-direct.txt — the Dockerfile pins ML
        # deps from this file (mirrors the condor opus_4_6_1m.def
        # pipeline).
        reqs_src = self.posttrainbench_root / "containers" / "requirements-direct.txt"
        if not reqs_src.exists():
            raise FileNotFoundError(
                f"requirements-direct.txt not found at {reqs_src}; "
                f"the Dockerfile expects it in the build context."
            )
        shutil.copy(reqs_src, target_dir / "requirements-direct.txt")

    def _copy_eval_files(
        self,
        target_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Copy the evaluation pipeline files into target_dir.

        Used for both:
          - environment/ (so the agent has them in /home/agent/workspace
            for iterative testing during training)
          - tests/ (so the verifier runs against an untampered copy that
            Harbor uploads only after the agent process exits)

        Files copied:
          - evaluate.py            (benchmark-specific)
          - templates/             (chat templates for all model families)
          - evaluation_code/       (arenahardwriting, healthbench only)
          - task_context/<*>       (bfcl has bfcl_evaluation_code.py)
          - contamination_judge.py (judge prompt builder)
          - metadata.json          (benchmark + model info for verifier)
        """
        # evaluate.py — for the LLM-graded benchmarks the OpenRouter provider
        # uses upstream's evaluate_openrouter.py variant (same CLI, grader model
        # slug openai/gpt-5-mini, OPENROUTER_API_KEY). It is installed under the
        # canonical name so test.sh and the agent instructions need no change.
        task_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id
        eval_name = "evaluate.py"
        if self.api_provider == "openrouter" and benchmark_id in GRADED_BENCHMARKS:
            eval_name = "evaluate_openrouter.py"
        eval_src = task_src / eval_name
        if not eval_src.exists():
            raise FileNotFoundError(f"{eval_name} not found: {eval_src}")
        shutil.copy(eval_src, target_dir / "evaluate.py")

        # templates/
        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        if not templates_src.exists():
            raise FileNotFoundError(f"templates directory not found: {templates_src}")
        shutil.copytree(templates_src, target_dir / "templates", dirs_exist_ok=True)

        # evaluation_code/ (arenahardwriting, healthbench)
        eval_code_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, target_dir / "evaluation_code", dirs_exist_ok=True)

        # task_context/* (bfcl has bfcl_evaluation_code.py)
        task_context_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "task_context"
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)

        # contamination judge script (kept in template/environment/ as
        # the canonical source, copied into both env_dir and tests_dir)
        judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
        if judge_src.exists():
            shutil.copy(judge_src, target_dir / "contamination_judge.py")

        # metadata.json
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
            "api_provider": self.api_provider,
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_tests(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the tests/ directory.

        Under harbor 0.7.0's separate-verifier mode, tests/ doubles as
        the verifier image's build context: harbor builds it into a
        container the agent never touches, then transfers configured
        artifacts in at runtime. The image must self-contain test.sh and
        everything test.sh reads — harbor does not upload tests/ at
        runtime for separate verifier envs.

        Files placed here:
          - Dockerfile      builds the verifier image
          - test.sh         the verifier orchestrator (baked in via COPY .)
          - entrypoint.sh   PID-1 streamer (matches agent env)
          - system_monitor.sh  background system monitor
          - requirements-direct.txt  pinned ML deps for the Dockerfile
          - evaluate.py + templates/ + evaluation_code/ + task_context/*
            + contamination_judge.py + metadata.json — the eval pipeline
        """
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Verifier image Dockerfile (canonical source: template/tests/Dockerfile).
        shutil.copy(
            TEMPLATE_DIR / "tests" / "Dockerfile",
            tests_dir / "Dockerfile",
        )

        # Verifier orchestrator (test.sh)
        test_sh_src = TEMPLATE_DIR / "tests" / "test.sh"
        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(test_sh_src, test_sh_dst)
        test_sh_dst.chmod(0o755)

        # v1.1 judge runner (API-key mode) — invoked by test.sh.
        runner_dst = tests_dir / "run_judges_apikey.sh"
        shutil.copy(TEMPLATE_DIR / "tests" / "run_judges_apikey.sh", runner_dst)
        runner_dst.chmod(0o755)

        # PostTrainBench trace parser (vendored verbatim from upstream
        # src/trace_parsing/). run_judges_apikey.sh invokes
        # trace_parsing/parse_trace.py to produce solve_parsed.txt for the
        # judges, matching upstream run_task.sh. grok-build has no structured
        # parser upstream, so parse_trace copies the raw trace verbatim (its
        # historical fallback) — auto-upgrading if upstream adds a grok parser.
        shutil.copytree(
            TEMPLATE_DIR / "tests" / "trace_parsing",
            tests_dir / "trace_parsing",
            dirs_exist_ok=True,
        )

        # Build-context support files (same set the agent env needs).
        self._copy_build_context_support(tests_dir)

        # Eval pipeline (also baked into the agent workspace via
        # environment/, but the verifier reads from /tests/ where these
        # land via the verifier Dockerfile's `COPY .`).
        self._copy_eval_files(tests_dir, benchmark_id, model_info, benchmark_info)

        # v1.1 judge system: confs/prompts/tools under judges_repo/ (pristine
        # copies the judges run from), plus the benchmark test set the
        # contamination judge re-copies over the agent's version.
        self._copy_judges_repo(tests_dir, benchmark_id)
        shutil.copy(self._require_test_data(benchmark_id), tests_dir / "test_data.json")

    def generate_task(
        self,
        benchmark_id: str,
        model_key: str,
    ) -> Path:
        """
        Generate a complete Harbor task for a benchmark + model combination.

        Args:
            benchmark_id: The benchmark ID (e.g., "gsm8k").
            model_key: The model key (e.g., "qwen3-1.7b").

        Returns:
            Path to the generated task directory.
        """
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Try to get actual benchmark name from file
        try:
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )
        except FileNotFoundError:
            pass  # Use default from dataclass

        # Create task directory
        task_id = f"posttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        # Generate all components
        self.generate_task_toml(task_dir, benchmark_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id, model_info, benchmark_info)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        tasks = []
        for benchmark_id in BENCHMARKS:
            for model_key in MODELS:
                task_dir = self.generate_task(benchmark_id, model_key)
                tasks.append(task_dir)
        return tasks


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    tasks = []
    for benchmark_id in BENCHMARKS:
        for model_key in MODELS:
            task_id = f"posttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
            tasks.append(task_id)
    return tasks
