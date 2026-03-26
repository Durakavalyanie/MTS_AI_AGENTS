from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentRuntimeConfig:
    openrouter_api_key: str
    openrouter_base_url: str
    kaggle_username: str
    kaggle_key: str
    kaggle_competition: str
    target_mse: float
    max_round: int
    max_loop_rounds: int
    max_tokens: int
    models: dict[str, str]


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _model_from_env(role_key: str) -> str:
    env_name = f"OPENROUTER_MODEL_{role_key}"
    return _require_env(env_name)


def load_runtime_config(project_root: Path) -> AgentRuntimeConfig:
    load_dotenv(project_root / ".env")

    models = {
        "orchestrator": _model_from_env("ORCHESTRATOR"),
        "data_analyst": _model_from_env("DATA_ANALYST"),
        "data_engineer": _model_from_env("DATA_ENGINEER"),
        "ml_engineer": _model_from_env("ML_ENGINEER"),
        "reviewer": _model_from_env("REVIEWER"),
    }

    return AgentRuntimeConfig(
        openrouter_api_key=_require_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_require_env("OPENROUTER_BASE_URL"),
        kaggle_username=_require_env("KAGGLE_USERNAME"),
        kaggle_key=_require_env("KAGGLE_API_TOKEN"),
        kaggle_competition=_require_env("KAGGLE_COMPETITION"),
        target_mse=float(_require_env("TARGET_MSE")),
        max_round=int(os.getenv("MAX_ROUND", "40")),
        max_loop_rounds=int(os.getenv("MAX_LOOP_ROUNDS", "10")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        models=models,
    )


def llm_config_for_role(cfg: AgentRuntimeConfig, role: str) -> dict[str, Any]:
    model_name = cfg.models[role]
    return {
        "config_list": [
            {
                "model": model_name,
                "api_key": cfg.openrouter_api_key,
                "base_url": cfg.openrouter_base_url,
                "price" : [0, 0]
            }
        ],
        "temperature": 0.2,
        "timeout": 180,
        "max_tokens": cfg.max_tokens,
    }
