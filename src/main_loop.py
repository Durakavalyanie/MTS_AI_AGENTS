from __future__ import annotations

import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.config import AgentRuntimeConfig, load_runtime_config
from src.agents.factory import create_agents, AgentBundle
from src.chat_manager import parse_tool_calls, validate_tool_calls
from src.tools.chat_logger import TrajectoryLogger
from src.tools.kaggle_submitter import KaggleSubmitter
from src.tools.rag import load_best_trajectories

RETRY_DELAYS = [10, 30, 60]
DATA_FILES = ["train.csv", "test.csv", "sample_submission.csv"]
SEPARATOR_LINE = "=" * 50


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _next_run_dir(root: Path) -> Path:
    workspace_root = root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path.name for path in workspace_root.iterdir() if path.is_dir() and path.name.startswith("run_")
    )
    run_index = len(existing) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = workspace_root / f"run_{run_index:03d}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    _copy_data_files(root, run_dir)
    return run_dir


def _copy_data_files(root: Path, run_dir: Path) -> None:
    raw_dir = root / "data" / "raw"
    if not raw_dir.exists():
        return

    for file_name in DATA_FILES:
        src_file = raw_dir / file_name
        if src_file.exists():
            shutil.copy(src_file, run_dir / file_name)


def _record_benchmark(root: Path, run_name: str, best_score: float | None, rounds: int, stop_reason: str):
    benchmark_file = root / "logs" / "benchmark.jsonl"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": run_name,
        "best_score": best_score,
        "rounds": rounds,
        "stop_reason": stop_reason
    }
    
    with benchmark_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


def _build_initial_prompt(run_dir: Path) -> str:
    rag_context = load_best_trajectories(run_dir.parents[1] / "best_trajectories")
    return (
        "Solve the Kaggle competition task using strict role separation.\n"
        "You must produce submission.csv in workspace and call submit_to_kaggle.\n"
        f"{rag_context}"
    )


class WorkflowManager:
    def __init__(self, bundle: AgentBundle, logger: TrajectoryLogger, submitter: KaggleSubmitter, cfg: AgentRuntimeConfig, run_dir: Path):
        self.bundle = bundle
        self.logger = logger
        self.submitter = submitter
        self.cfg = cfg
        self.run_dir = run_dir
        
        self.shared_history: list[dict[str, Any]] = []
        self.best_score: float | None = None
        self.prev_error: str | None = None
        self.stop_condition_met = False
        self.round_count = 0

    def _generate_reply(self, agent: Any, messages: list[dict[str, Any]]) -> str:
        last_exception = None
        for delay in RETRY_DELAYS:
            try:
                reply = agent.generate_reply(messages=messages, sender=agent)
                if isinstance(reply, dict):
                    reply = reply.get("content", "")
                return str(reply)
            except Exception as e:
                last_exception = e
                print(f"\n\nERROR: {e}\nRetrying in {delay}s...", flush=True)
                time.sleep(delay)
        raise RuntimeError(f"Failed after {len(RETRY_DELAYS)} retries") from last_exception

    def _print_turn(self, speaker: str, content: str = "") -> None:
        print(f"\n{SEPARATOR_LINE}", flush=True)
        print(f"[{self.round_count}] TURN: {speaker}", flush=True)

        if speaker == "Orchestrator" and content:
            self._print_orchestrator_tools(content)

        print(f"{SEPARATOR_LINE}\n", flush=True)

    def _print_orchestrator_tools(self, content: str) -> None:
        calls = parse_tool_calls(content)
        for tool_name, args in calls:
            print(f"TOOL CALLED: {tool_name}", flush=True)
            if tool_name == "delegate":
                print(f"  -> Next Speaker: {args.get('next_speaker')}", flush=True)
                print(f"  -> Directive: {args.get('directive')}", flush=True)
            elif tool_name == "submit_to_kaggle":
                print(f"  -> Message: {args.get('message')}", flush=True)

    def _run_code_execution_loop(self, author_name: str, initial_code_msg: str) -> str:
        author_agent = self.bundle.get_agent_by_name(author_name)
        execution_history = list(self.shared_history)
        execution_history.append({"role": "assistant", "name": author_name, "content": initial_code_msg})

        self.logger.add_event("code_execution_loop_start", {"author": author_name, "initial_msg": initial_code_msg})

        for loop_round in range(1, self.cfg.max_loop_rounds + 1):
            last_msg = execution_history[-1]

            if last_msg["name"] == author_name:
                if self._handle_author_message(author_name, last_msg, execution_history):
                    return last_msg["content"]
            elif last_msg["name"] in ("CodeExecutor", "System"):
                self._handle_executor_response(author_name, author_agent, loop_round, execution_history)

        timeout_msg = '[TOOL_CALL]send_message\n{"thoughts": "Execution loop timed out.", "message": "Error: Code execution loop reached maximum attempts."}'
        self.logger.add_event("code_execution_loop_timeout", {"author": author_name, "msg": timeout_msg})
        return timeout_msg

    def _handle_author_message(self, author_name: str, last_msg: dict[str, Any], execution_history: list[dict[str, Any]]) -> bool:
        calls = parse_tool_calls(last_msg["content"])
        if any(t == "send_message" for t, _ in calls):
            self.logger.add_event("code_execution_loop_end", {"author": author_name, "final_msg": last_msg["content"]})
            return True

        exec_call = next((args for t, args in calls if t == "execute_code"), None)
        if exec_call:
            print(f"  [{author_name}] -> CodeExecutor (Executing code directly)", flush=True)
            code = exec_call.get("code", "")
            exec_result = self.bundle.execute_code(code)
            execution_history.append({"role": "user", "name": "CodeExecutor", "content": exec_result})
            self.logger.add_event("executor_result", {"author": author_name, "content": exec_result})
        else:
            error_msg = "Protocol Validation Error:\nError: You must use either 'execute_code' or 'send_message'."
            execution_history.append({"role": "user", "name": "System", "content": error_msg})
            self.logger.add_event("author_validation_error", {"author": author_name, "errors": [error_msg], "reply": last_msg["content"]})

        return False

    def _handle_executor_response(self, author_name: str, author_agent: Any, loop_round: int, execution_history: list[dict[str, Any]]) -> None:
        last_msg = execution_history[-1]
        print(f"  [{last_msg['name']}] -> {author_name} (Returning result/error)", flush=True)

        remaining_turns = (self.cfg.max_loop_rounds - loop_round) / 2
        execution_history.append({
            "role": "user",
            "name": "System",
            "content": f"You have {remaining_turns} turns left until you HAVE TO finish your work and response with send_message tool"
        })

        reply = self._generate_reply(author_agent, execution_history)
        errors = validate_tool_calls(reply, author_name)

        if errors:
            error_content = "Protocol Validation Error:\n" + "\n".join(errors) + "\nPlease fix your JSON format and try again."
            execution_history.append({"role": "user", "name": "System", "content": error_content})
            self.logger.add_event("author_validation_error", {"author": author_name, "errors": errors, "reply": reply})
        else:
            execution_history.append({"role": "assistant", "name": author_name, "content": reply})
            self.logger.add_event("author_reply", {"author": author_name, "content": reply})

    def _get_workspace_files(self) -> str:
        files = [f.name for f in self.run_dir.iterdir() if f.is_file() and f.suffix == '.csv']
        return f"\n\n[SYSTEM: Current files in workspace: {', '.join(files)}]"

    def _execute_orchestrator_turn(self) -> None:
        orch_reply = self._generate_reply(self.bundle.orchestrator, self.shared_history)
        self._print_turn("Orchestrator", orch_reply)
        self.logger.add_event("orchestrator_turn", {"content": orch_reply})

        errors = validate_tool_calls(orch_reply, "Orchestrator")
        if errors:
            self.shared_history.append({"role": "assistant", "name": "Orchestrator", "content": orch_reply})
            self.shared_history.append({"role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(errors)})
            return

        self.shared_history.append({"role": "assistant", "name": "Orchestrator", "content": orch_reply})

        calls = parse_tool_calls(orch_reply)
        for tool_name, args in calls:
            if tool_name == "submit_to_kaggle":
                self._handle_kaggle_submission(args)
            elif tool_name == "delegate":
                self._handle_delegation(args)

    def _handle_kaggle_submission(self, args: dict[str, Any]) -> None:
        submission = self.run_dir / "submission.csv"
        msg = args.get("message", "autogen submission")
        result = self.submitter.submit(submission_path=submission, message=msg)

        self.logger.add_event("kaggle_submission", {
            "status": result.status,
            "public_score": result.public_score,
            "message": result.message,
        })

        if result.public_score is None:
            self.shared_history.append({
                "role": "user",
                "name": "KaggleSystem",
                "content": f"Submission FAILED: {result.message}. Please fix the issue and try again."
            })
        else:
            self._process_kaggle_score(result.public_score)

    def _process_kaggle_score(self, score: float) -> None:
        self.best_score = score if self.best_score is None else min(self.best_score, score)

        if score <= self.cfg.target_mse:
            self.logger.add_event("stop_condition_met", {"best_score": score})
            self.stop_condition_met = True
            self.shared_history.append({
                "role": "user",
                "name": "KaggleSystem",
                "content": f"SUCCESS! Score {score} is better than target {self.cfg.target_mse}. Terminating."
            })
        else:
            self.shared_history.append({
                "role": "user",
                "name": "KaggleSystem",
                "content": f"Submission successful. Score: {score}. Target is {self.cfg.target_mse}. You must improve the score."
            })

    def _handle_delegation(self, args: dict[str, Any]) -> None:
        next_speaker = args.get("next_speaker")
        directive = args.get("directive")
        agent = self.bundle.get_agent_by_name(next_speaker)

        if not agent:
            self.shared_history.append({
                "role": "user",
                "name": "System",
                "content": f"Error: Agent {next_speaker} not found."
            })
            return

        self._print_turn(next_speaker)

        self.shared_history.append({
            "role": "user",
            "name": "Orchestrator",
            "content": f"Orchestrator directive for {next_speaker}:\n{directive}"
        })

        agent_reply = self._get_valid_agent_reply(agent, next_speaker)
        final_msg = self._execute_agent_task(next_speaker, agent_reply)

        self.shared_history.append({
            "role": "assistant",
            "name": next_speaker,
            "content": final_msg + self._get_workspace_files()
        })
        self.logger.add_event("agent_completed_task", {
            "agent": next_speaker,
            "content": final_msg + self._get_workspace_files()
        })

    def _get_valid_agent_reply(self, agent: Any, agent_name: str) -> str:
        agent_reply = self._generate_reply(agent, self.shared_history)
        agent_errors = validate_tool_calls(agent_reply, agent_name)

        while agent_errors:
            self.shared_history.append({"role": "assistant", "name": agent_name, "content": agent_reply})
            self.shared_history.append({"role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(agent_errors)})
            agent_reply = self._generate_reply(agent, self.shared_history)
            agent_errors = validate_tool_calls(agent_reply, agent_name)

        return agent_reply

    def _execute_agent_task(self, agent_name: str, agent_reply: str) -> str:
        agent_calls = parse_tool_calls(agent_reply)
        if any(t == "execute_code" for t, _ in agent_calls):
            return self._run_code_execution_loop(agent_name, agent_reply)
        return agent_reply

    def run(self):
        initial_prompt = _build_initial_prompt(self.run_dir)
        self.shared_history.append({"role": "user", "name": "System", "content": initial_prompt})
        
        while self.round_count < self.cfg.max_round and not self.stop_condition_met:
            self.round_count += 1
            self._execute_orchestrator_turn()



def run() -> None:
    root = _project_root()
    cfg = load_runtime_config(root)
    run_dir = _next_run_dir(root)
    logger = TrajectoryLogger(log_file=root / "logs" / f"{run_dir.name}.json")
    submitter = KaggleSubmitter(
        competition=cfg.kaggle_competition,
        username=cfg.kaggle_username,
        key=cfg.kaggle_key
    )

    try:
        manager = WorkflowManager(
            bundle=create_agents(cfg, run_dir),
            logger=logger,
            submitter=submitter,
            cfg=cfg,
            run_dir=run_dir
        )
        manager.run()
    except KeyboardInterrupt:
        logger.add_event("keyboard_interrupt", {"message": "Run interrupted by user"})
    finally:
        stop_reason = "interrupted" if "manager" not in locals() or not manager.stop_condition_met else "target_met"
        if "manager" in locals():
            if manager.round_count >= cfg.max_round:
                stop_reason = "limit_reached"
            _record_benchmark(root, run_dir.name, manager.best_score, manager.round_count, stop_reason)
        logger.flush()


if __name__ == "__main__":
    run()
