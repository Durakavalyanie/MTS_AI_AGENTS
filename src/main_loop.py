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
    
    # Copy data files to workspace
    raw_dir = root / "data" / "raw"
    if raw_dir.exists():
        for file_name in ["train.csv", "test.csv", "sample_submission.csv"]:
            src_file = raw_dir / file_name
            if src_file.exists():
                shutil.copy(src_file, run_dir / file_name)
                
    return run_dir


def _build_initial_prompt(cfg: AgentRuntimeConfig, run_dir: Path, prev_score: float | None, prev_error: str | None) -> str:
    return (
        "Solve the Kaggle competition task using strict role separation.\n"
        "You must produce submission.csv in workspace and call submit_to_kaggle."
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

    def _generate_reply(self, agent: Any, messages: list[dict[str, Any]], sender_name: str) -> str:
        delay = [10, 30, 60]
        for i in range(len(delay)):
            try:
                reply = agent.generate_reply(messages=messages, sender=agent)
                if isinstance(reply, dict):
                    reply = reply.get("content", "")
                return str(reply)
            except Exception as e:
                print("\n\nERROR:\n" + str(e))
                time.sleep(delay[i])
        raise

    def _print_turn(self, speaker: str, content: str = "") -> None:
        print(f"\n{'='*50}", flush=True)
        print(f"[{self.round_count}] TURN: {speaker}", flush=True)
        
        if speaker == "Orchestrator":
            calls = parse_tool_calls(content)
            for tool_name, args in calls:
                print(f"TOOL CALLED: {tool_name}", flush=True)
                if tool_name == "delegate":
                    print(f"  -> Next Speaker: {args.get('next_speaker')}", flush=True)
                    print(f"  -> Directive: {args.get('directive')}", flush=True)
                elif tool_name == "submit_to_kaggle":
                    print(f"  -> Message: {args.get('message')}", flush=True)
        print(f"{'='*50}\n", flush=True)

    def _run_code_execution_loop(self, author_name: str, initial_code_msg: str) -> str:
        """
        Runs the isolated execution loop: Author -> Executor -> Author.
        Returns the final message (send_message) from the Author to the Orchestrator.
        """
        author_agent = self.bundle.get_agent_by_name(author_name)
        
        # Initialize execution history with a copy of the shared history
        # so the agent remembers the context of the task during debugging
        execution_history = list(self.shared_history)
        execution_history.append({"role": "assistant", "name": author_name, "content": initial_code_msg})
        
        self.logger.add_event("code_execution_loop_start", {"author": author_name, "initial_msg": initial_code_msg})
        
        loop_rounds = 0
        while loop_rounds < self.cfg.max_loop_rounds:
            loop_rounds += 1
            last_msg = execution_history[-1]
            
            if last_msg["name"] == author_name:
                calls = parse_tool_calls(last_msg["content"])
                if any(t == "send_message" for t, _ in calls):
                    self.logger.add_event("code_execution_loop_end", {"author": author_name, "final_msg": last_msg["content"]})
                    return last_msg["content"]
                
                exec_call = next((args for t, args in calls if t == "execute_code"), None)
                if exec_call:
                    print(f"  [{author_name}] -> CodeExecutor (Executing code directly)", flush=True)
                    code = exec_call.get("code", "")
                    exec_result = self.bundle.execute_code(code)
                    execution_history.append({"role": "user", "name": "CodeExecutor", "content": exec_result})
                    self.logger.add_event("executor_result", {"content": exec_result})
                else:
                    # No recognizable tool call, should have been caught by validation, but just in case
                    error_msg = "Protocol Validation Error:\nError: You must use either 'execute_code' or 'send_message'."
                    execution_history.append({"role": "user", "name": "System", "content": error_msg})
                    self.logger.add_event("author_validation_error", {"errors": [error_msg], "reply": last_msg["content"]})
                        
            elif last_msg["name"] == "CodeExecutor" or last_msg["name"] == "System":
                print(f"  [{last_msg['name']}] -> {author_name} (Returning result/error)", flush=True)
                execution_history.append({"role": "user", "name": "System", "content": f"You have {(self.cfg.max_loop_rounds - loop_rounds - 1)/2} turns left until you HAVE TO finish your work and response with send_message tool"})
                reply = self._generate_reply(author_agent, execution_history, author_name)
                errors = validate_tool_calls(reply, author_name)
                if errors:
                    execution_history.append({
                        "role": "user", 
                        "name": "System", 
                        "content": "Protocol Validation Error:\n" + "\n".join(errors) + "\nPlease fix your JSON format and try again."
                    })
                    self.logger.add_event("author_validation_error", {"errors": errors, "reply": reply})
                else:
                    execution_history.append({"role": "assistant", "name": author_name, "content": reply})
                    self.logger.add_event("author_reply", {"content": reply})
                    
        timeout_msg = "[TOOL_CALL]send_message\n{\"thoughts\": \"Execution loop timed out.\", \"message\": \"Error: Code execution loop reached maximum attempts.\"}"
        self.logger.add_event("code_execution_loop_timeout", {"msg": timeout_msg})
        return timeout_msg

    def run(self):
        initial_prompt = _build_initial_prompt(self.cfg, self.run_dir, self.best_score, self.prev_error)
        self.shared_history.append({"role": "user", "name": "System", "content": initial_prompt})
        
        while self.round_count < self.cfg.max_round and not self.stop_condition_met:
            self.round_count += 1
            
            # 1. Orchestrator Turn
            orch_reply = self._generate_reply(self.bundle.orchestrator, self.shared_history, "Orchestrator")
            self._print_turn("Orchestrator", orch_reply)
            self.logger.add_event("orchestrator_turn", {"content": orch_reply})
            
            errors = validate_tool_calls(orch_reply, "Orchestrator")
            if errors:
                self.shared_history.append({
                    "role": "assistant", "name": "Orchestrator", "content": orch_reply
                })
                self.shared_history.append({
                    "role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(errors)
                })
                continue
                
            self.shared_history.append({"role": "assistant", "name": "Orchestrator", "content": orch_reply})
            
            calls = parse_tool_calls(orch_reply)
            for tool_name, args in calls:
                if tool_name == "submit_to_kaggle":
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
                            "role": "user", "name": "KaggleSystem", 
                            "content": f"Submission FAILED: {result.message}. Please fix the issue and try again."
                        })
                    else:
                        score = result.public_score
                        self.best_score = score if self.best_score is None else min(self.best_score, score)
                        
                        if score <= self.cfg.target_mse:
                            self.logger.add_event("stop_condition_met", {"best_score": score})
                            self.stop_condition_met = True
                            self.shared_history.append({
                                "role": "user", "name": "KaggleSystem", 
                                "content": f"SUCCESS! Score {score} is better than target {self.cfg.target_mse}. Terminating."
                            })
                        else:
                            self.shared_history.append({
                                "role": "user", "name": "KaggleSystem", 
                                "content": f"Submission successful. Score: {score}. Target is {self.cfg.target_mse}. You must improve the score."
                            })
                            
                elif tool_name == "delegate":
                    next_speaker = args.get("next_speaker")
                    directive = args.get("directive")
                    agent = self.bundle.get_agent_by_name(next_speaker)
                    
                    if not agent:
                        self.shared_history.append({
                            "role": "user", "name": "System", "content": f"Error: Agent {next_speaker} not found."
                        })
                        continue
                        
                    # 2. Agent Turn
                    self._print_turn(next_speaker)
                    
                    # Append the orchestrator's directive to the shared history so the agent knows they are being addressed
                    self.shared_history.append({
                        "role": "user", 
                        "name": "Orchestrator", 
                        "content": f"Orchestrator directive for {next_speaker}:\n{directive}"
                    })
                    
                    agent_reply = self._generate_reply(agent, self.shared_history, next_speaker)
                    agent_errors = validate_tool_calls(agent_reply, next_speaker)
                    
                    # If the agent messes up formatting immediately, handle it in the shared history
                    while agent_errors:
                        self.shared_history.append({"role": "assistant", "name": next_speaker, "content": agent_reply})
                        self.shared_history.append({"role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(agent_errors)})
                        agent_reply = self._generate_reply(agent, self.shared_history, next_speaker)
                        agent_errors = validate_tool_calls(agent_reply, next_speaker)
                        
                    agent_calls = parse_tool_calls(agent_reply)
                    
                    # If agent wants to execute code, enter the isolated loop
                    if any(t == "execute_code" for t, _ in agent_calls):
                        final_msg = self._run_code_execution_loop(next_speaker, agent_reply)
                    else:
                        # Agent just sent a message directly
                        final_msg = agent_reply
                        
                    # Return the final result to the shared history
                    # Add directory state to help Orchestrator verify pipeline rules
                    dir_files = [f.name for f in self.run_dir.iterdir() if f.is_file() and f.suffix == '.csv']
                    dir_state = f"\n\n[SYSTEM: Current files in workspace: {', '.join(dir_files)}]"
                    
                    self.shared_history.append({
                        "role": "assistant", "name": next_speaker, "content": final_msg + dir_state
                    })
                    self.logger.add_event("agent_completed_task", {"agent": next_speaker, "content": final_msg + dir_state})
                else:
                    self.shared_history.append({
                        "role": "assistant", "name": "System", "content": "Orchestrator has to use one of theese tools: delegate, submit_to_kaggle"
                    })
                    self.logger.add_event("agent_completed_task", {"agent": next_speaker, "content": final_msg})



def run() -> None:
    root = _project_root()
    cfg = load_runtime_config(root)
    run_dir = _next_run_dir(root)
    logger = TrajectoryLogger(log_file=root / "logs" / f"{run_dir.name}.json")
    submitter = KaggleSubmitter(competition=cfg.kaggle_competition)

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
        logger.flush()


if __name__ == "__main__":
    run()
