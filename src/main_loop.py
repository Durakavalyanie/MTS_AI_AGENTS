from __future__ import annotations

import json
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


def _next_run_dir(workspace_root: Path) -> Path:
    workspace_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path.name for path in workspace_root.iterdir() if path.is_dir() and path.name.startswith("run_")
    )
    run_index = len(existing) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = workspace_root / f"run_{run_index:03d}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_initial_prompt(cfg: AgentRuntimeConfig, run_dir: Path, prev_score: float | None, prev_error: str | None) -> str:
    if prev_error:
        score_line = f"Previous Kaggle submission FAILED with error: {prev_error}. Fix the submission format."
    elif prev_score is not None:
        score_line = f"Previous Kaggle public MSE: {prev_score}. Improve this score."
    else:
        score_line = "This is the first iteration. Build a baseline quickly."

    return (
        "Solve the Kaggle competition task using strict role separation.\n"
        f"Competition slug: {cfg.kaggle_competition}\n"
        f"Workspace: {run_dir}\n"
        f"Target MSE threshold: {cfg.target_mse} (lower is better)\n"
        f"{score_line}\n"
        "You must produce submission.csv in workspace and call submit_to_kaggle."
    )


class WorkflowManager:
    def __init__(self, bundle: AgentBundle, logger: TrajectoryLogger, submitter: KaggleSubmitter, cfg: AgentRuntimeConfig, run_dir: Path):
        self.bundle = bundle
        self.logger = logger
        self.submitter = submitter
        self.cfg = cfg
        self.run_dir = run_dir
        
        self.orchestrator_history: list[dict[str, Any]] = []
        self.best_score: float | None = None
        self.prev_error: str | None = None
        self.stop_condition_met = False
        self.round_count = 0

    def _generate_reply(self, agent: Any, messages: list[dict[str, Any]], sender_name: str) -> str:
        # We use autogen's generate_reply directly
        reply = agent.generate_reply(messages=messages, sender=agent)
        if isinstance(reply, dict):
            reply = reply.get("content", "")
        return str(reply)

    def _run_code_execution_loop(self, author_name: str, initial_code_msg: str) -> str:
        """
        Runs the isolated execution loop: Author -> Reviewer -> Executor -> Author.
        Returns the final message (send_message) from the Author to the Orchestrator.
        """
        author_agent = self.bundle.get_agent_by_name(author_name)
        reviewer = self.bundle.reviewer
        
        # This history is private to this specific execution loop
        execution_history: list[dict[str, Any]] = [
            {"role": "assistant", "name": author_name, "content": initial_code_msg}
        ]
        
        loop_rounds = 0
        while loop_rounds < 10: # Prevent infinite loops
            loop_rounds += 1
            last_msg = execution_history[-1]
            
            if last_msg["name"] == author_name:
                # Author just spoke. Did they send a message or execute code?
                calls = parse_tool_calls(last_msg["content"])
                if any(t == "send_message" for t, _ in calls):
                    # Author is done, returning to orchestrator
                    return last_msg["content"]
                
                # Author wants to execute code -> send to Reviewer
                # Reviewer only needs to see the code, not the whole history
                reviewer_prompt = f"Please review this code proposal:\n\n{last_msg['content']}"
                reply = self._generate_reply(reviewer, [{"role": "user", "content": reviewer_prompt}], "ReviewerDebugger")
                
                errors = validate_tool_calls(reply, "ReviewerDebugger")
                if errors:
                    execution_history.append({
                        "role": "user", 
                        "name": "System", 
                        "content": "Protocol Validation Error:\n" + "\n".join(errors)
                    })
                    continue
                    
                execution_history.append({"role": "user", "name": "ReviewerDebugger", "content": reply})
                
            elif last_msg["name"] == "ReviewerDebugger":
                calls = parse_tool_calls(last_msg["content"])
                review_call = next((args for t, args in calls if t == "review_code"), None)
                
                if review_call and review_call.get("decision") == "APPROVE":
                    # Execute code
                    code = review_call.get("code", "")
                    exec_result = self.bundle.execute_code(code)
                    execution_history.append({"role": "user", "name": "CodeExecutor", "content": exec_result})
                else:
                    # REJECT -> send back to author
                    reply = self._generate_reply(author_agent, execution_history, author_name)
                    errors = validate_tool_calls(reply, author_name)
                    if errors:
                        execution_history.append({
                            "role": "user", 
                            "name": "System", 
                            "content": "Protocol Validation Error:\n" + "\n".join(errors)
                        })
                    else:
                        execution_history.append({"role": "assistant", "name": author_name, "content": reply})
                        
            elif last_msg["name"] == "CodeExecutor":
                # Send execution result back to author
                reply = self._generate_reply(author_agent, execution_history, author_name)
                errors = validate_tool_calls(reply, author_name)
                if errors:
                    execution_history.append({
                        "role": "user", 
                        "name": "System", 
                        "content": "Protocol Validation Error:\n" + "\n".join(errors)
                    })
                else:
                    execution_history.append({"role": "assistant", "name": author_name, "content": reply})
                    
        return "[TOOL_CALL]send_message\n{\"thoughts\": \"Execution loop timed out.\", \"message\": \"Error: Code execution loop reached maximum attempts.\"}"

    def run(self):
        initial_prompt = _build_initial_prompt(self.cfg, self.run_dir, self.best_score, self.prev_error)
        self.orchestrator_history.append({"role": "user", "name": "System", "content": initial_prompt})
        
        while self.round_count < self.cfg.max_round and not self.stop_condition_met:
            self.round_count += 1
            
            # 1. Orchestrator Turn
            orch_reply = self._generate_reply(self.bundle.orchestrator, self.orchestrator_history, "Orchestrator")
            self.logger.add_event("orchestrator_turn", {"content": orch_reply})
            
            errors = validate_tool_calls(orch_reply, "Orchestrator")
            if errors:
                self.orchestrator_history.append({
                    "role": "assistant", "name": "Orchestrator", "content": orch_reply
                })
                self.orchestrator_history.append({
                    "role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(errors)
                })
                continue
                
            self.orchestrator_history.append({"role": "assistant", "name": "Orchestrator", "content": orch_reply})
            
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
                        self.orchestrator_history.append({
                            "role": "user", "name": "KaggleSystem", 
                            "content": f"Submission FAILED: {result.message}. Please fix the issue and try again."
                        })
                    else:
                        score = result.public_score
                        self.best_score = score if self.best_score is None else min(self.best_score, score)
                        
                        if score <= self.cfg.target_mse:
                            self.logger.add_event("stop_condition_met", {"best_score": score})
                            self.stop_condition_met = True
                            self.orchestrator_history.append({
                                "role": "user", "name": "KaggleSystem", 
                                "content": f"SUCCESS! Score {score} is better than target {self.cfg.target_mse}. Terminating."
                            })
                        else:
                            self.orchestrator_history.append({
                                "role": "user", "name": "KaggleSystem", 
                                "content": f"Submission successful. Score: {score}. Target is {self.cfg.target_mse}. You must improve the score."
                            })
                            
                elif tool_name == "delegate":
                    next_speaker = args.get("next_speaker")
                    directive = args.get("directive")
                    agent = self.bundle.get_agent_by_name(next_speaker)
                    
                    if not agent:
                        self.orchestrator_history.append({
                            "role": "user", "name": "System", "content": f"Error: Agent {next_speaker} not found."
                        })
                        continue
                        
                    # 2. Agent Turn
                    # Give the agent the directive from the orchestrator
                    agent_prompt = f"Orchestrator directive:\n{directive}"
                    agent_history = [{"role": "user", "name": "Orchestrator", "content": agent_prompt}]
                    
                    agent_reply = self._generate_reply(agent, agent_history, next_speaker)
                    agent_errors = validate_tool_calls(agent_reply, next_speaker)
                    
                    while agent_errors:
                        agent_history.append({"role": "assistant", "name": next_speaker, "content": agent_reply})
                        agent_history.append({"role": "user", "name": "System", "content": "Protocol Validation Error:\n" + "\n".join(agent_errors)})
                        agent_reply = self._generate_reply(agent, agent_history, next_speaker)
                        agent_errors = validate_tool_calls(agent_reply, next_speaker)
                        
                    agent_calls = parse_tool_calls(agent_reply)
                    
                    # If agent wants to execute code, enter the isolated loop
                    if any(t == "execute_code" for t, _ in agent_calls):
                        final_msg = self._run_code_execution_loop(next_speaker, agent_reply)
                    else:
                        # Agent just sent a message
                        final_msg = agent_reply
                        
                    # Return the final result to the orchestrator
                    self.orchestrator_history.append({
                        "role": "user", "name": next_speaker, "content": final_msg
                    })
                    self.logger.add_event("agent_completed_task", {"agent": next_speaker, "content": final_msg})


def run() -> None:
    root = _project_root()
    cfg = load_runtime_config(root)
    run_dir = _next_run_dir(root / "workspace")
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
