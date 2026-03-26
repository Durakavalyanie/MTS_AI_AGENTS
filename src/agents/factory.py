from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autogen import AssistantAgent
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor

from src.agents.config import AgentRuntimeConfig, llm_config_for_role
from src.agents.system_messages import (
    DATA_ANALYST_SYSTEM_MESSAGE,
    DATA_ENGINEER_SYSTEM_MESSAGE,
    ML_ENGINEER_SYSTEM_MESSAGE,
    ORCHESTRATOR_SYSTEM_MESSAGE,
)
from src.tools.code_policy import validate_python_code


@dataclass
class AgentBundle:
    orchestrator: AssistantAgent
    data_analyst: AssistantAgent
    data_engineer: AssistantAgent
    ml_engineer: AssistantAgent
    executor_backend: LocalCommandLineCodeExecutor
    work_dir: Path

    def get_agent_by_name(self, name: str) -> AssistantAgent | None:
        mapping = {
            "Orchestrator": self.orchestrator,
            "DataAnalyst": self.data_analyst,
            "DataEngineer": self.data_engineer,
            "MLEngineer": self.ml_engineer,
        }
        return mapping.get(name)

    def execute_code(self, code: str) -> str:
        is_allowed, reason_json = validate_python_code(code=code, work_dir=self.work_dir)
        if not is_allowed:
            return reason_json

        code_block = CodeBlock(code=code, language="python")
        result = self.executor_backend.execute_code_blocks([code_block])
        
        if result.exit_code == 0:
            resp = {
                "status": "SUCCESS",
                "summary": "Code executed without runtime errors.",
                "output": result.output
            }
        else:
            resp = {
                "status": "FAILED",
                "summary": "Runtime error occurred. Send traceback to ReviewerDebugger.",
                "output": result.output
            }
            
        return json.dumps(resp, ensure_ascii=False)


def create_agents(cfg: AgentRuntimeConfig, run_dir: Path) -> AgentBundle:
    executor_backend = LocalCommandLineCodeExecutor(work_dir=run_dir)

    orchestrator = AssistantAgent(
        name="Orchestrator",
        llm_config=llm_config_for_role(cfg, "orchestrator"),
        system_message=ORCHESTRATOR_SYSTEM_MESSAGE,
    )
    data_analyst = AssistantAgent(
        name="DataAnalyst",
        llm_config=llm_config_for_role(cfg, "data_analyst"),
        system_message=DATA_ANALYST_SYSTEM_MESSAGE,
    )
    data_engineer = AssistantAgent(
        name="DataEngineer",
        llm_config=llm_config_for_role(cfg, "data_engineer"),
        system_message=DATA_ENGINEER_SYSTEM_MESSAGE,
    )
    ml_engineer = AssistantAgent(
        name="MLEngineer",
        llm_config=llm_config_for_role(cfg, "ml_engineer"),
        system_message=ML_ENGINEER_SYSTEM_MESSAGE,
    )
    return AgentBundle(
        orchestrator=orchestrator,
        data_analyst=data_analyst,
        data_engineer=data_engineer,
        ml_engineer=ml_engineer,
        executor_backend=executor_backend,
        work_dir=run_dir,
    )
