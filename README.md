# Agentic Machine Learning Pipeline (MTS_AI_AGENTS)

Automated multi-agent system for Kaggle competitions based on AutoGen and OpenRouter.

## Overview

The system implements a collaborative environment where specialized agents (Orchestrator, Data Analyst, Data Engineer, ML Engineer, and Reviewer) work together to solve data science tasks. It features RAG-based context injection from past successful runs and automated benchmarking of results.

## Prerequisites

- Python 3.10+
- Virtual environment (recommended: `.venv`)
- Kaggle API credentials
- OpenRouter API key

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env` (refer to `.env.example` or the list below).

## Configuration (.env)

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter access |
| `OPENROUTER_BASE_URL` | Base URL for OpenRouter (https://openrouter.ai/api/v1) |
| `KAGGLE_USERNAME` | Kaggle account username |
| `KAGGLE_KEY` | Kaggle API key (32-character hex) |
| `KAGGLE_COMPETITION` | Slug of the Kaggle competition |
| `TARGET_MSE` | Target metric for pipeline termination |
| `OPENROUTER_MODEL_*` | Specific models for each agent role |

## Project Structure

- `src/` - Core logic including agent factory, tools, and main orchestration loop.
- `data/raw/` - Directory for input datasets (`train.csv`, `test.csv`, `sample_submission.csv`).
- `workspace/` - Isolated directories for each execution run.
- `logs/` - Trajectory logs (JSON) and benchmarking results (`benchmark.jsonl`).
- `best_trajectories/` - Knowledge base for RAG (successful past run logs).

## Execution

To start the pipeline, run the following command:
```bash
python3 -m src.main_loop
```

## Features

### 1. Retrieval-Augmented Generation (RAG)
The system automatically analyzes logs in `best_trajectories/` to extract successful strategies, code snippets, and hyperparameters from previous high-scoring runs. This context is injected into the initial prompt to guide current agents.

### 2. Automated Benchmarking
Every execution run is recorded in `logs/benchmark.jsonl`, including timestamps, best scores achieved, number of rounds, and termination reasons. This allows for systematic performance tracking across different model configurations.

### 3. Safety Guardrails
Code execution is restricted by a deterministic policy check (`src/tools/code_policy.py`) that prevents package installation, network access, and unsafe file operations.