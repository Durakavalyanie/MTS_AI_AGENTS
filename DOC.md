# DOC

## Architecture

The project implements a multi-agent AutoGen pipeline for Kaggle competitions.

- LLM access is routed through OpenRouter.
- Different agents can use different LLM models (configured in `.env`).
- Code execution is separated from reasoning and passed through policy checks.
- Kaggle submission is executed by a Python wrapper outside agents.

## Repository layout

- `data/raw/` - input datasets (read-only for agents).
- `workspace/run_XXX_timestamp/` - generated run workspace for one launch.
- `src/agents/` - agent prompts, config loading, factory.
- `src/tools/` - deterministic safety checks, trajectory logger, Kaggle submitter.
- `src/chat_manager.py` - group chat and speaker routing.
- `src/main_loop.py` - top-level orchestration loop.
- `logs/` - persisted trajectory events in JSON.

## Agent roles

1. `Orchestrator`  
   Coordinates process and decisions, does not write code.

2. `CodeExecutor`  
   Executes Python code only with `llm_config=False` and local executor in current run dir.

3. `DataAnalyst`  
   Produces textual EDA/hypotheses only (no plotting libraries or `.plot()`).

4. `DataEngineer`  
   Implements preprocessing/features, writes `X_train.csv`, `y_train.csv`, `X_test.csv`.

5. `MLEngineer`  
   Trains/evaluates model, reports MSE, creates `submission.csv`.

6. `ReviewerDebugger`  
   Analyzes tracebacks, checks leakage risks, blocks unsafe patterns.

## OpenRouter and model routing

Required environment variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`
- `OPENROUTER_MODEL_ORCHESTRATOR`
- `OPENROUTER_MODEL_DATA_ANALYST`
- `OPENROUTER_MODEL_DATA_ENGINEER`
- `OPENROUTER_MODEL_ML_ENGINEER`
- `OPENROUTER_MODEL_REVIEWER`

Each role receives its own model through `llm_config_for_role(...)`.

## Safety and code policy

Deterministic checks are implemented in `src/tools/code_policy.py` and enforced before code execution.

Blocked classes:
- package installation attempts
- network usage
- plotting libraries and `.plot()`
- obvious path traversal / absolute-path writes
- imports outside allowed ML stack (with small standard-library exceptions)

If policy check fails, execution is rejected and error is returned back into chat flow.

## Deterministic code interaction algorithm

Code execution routing is deterministic and does not depend on orchestrator decisions:

1. Any code proposal from `DataAnalyst`, `DataEngineer`, or `MLEngineer` is sent to `ReviewerDebugger`.
2. Reviewer returns:
   - `REVIEW_DECISION: REJECT` + `RETURN_TO: <author>` when code is not executable or does not match expected outcome.
   - `REVIEW_DECISION: APPROVE` + code block when code is executable and safe.
3. Approved code is sent to `CodeExecutor`.
4. `CodeExecutor` returns:
   - `EXECUTION_STATUS: FAILED|BLOCKED` -> routed to `ReviewerDebugger`.
   - `EXECUTION_STATUS: SUCCESS` -> routed back to author (`RETURN_RESULTS_TO`), fallback to latest code author.

To keep fixes grounded, code proposals and reviews carry `EXPECTED_OUTCOME`.

## Main loop

`src/main_loop.py` workflow:

1. Load env/config.
2. Create new `workspace/run_XXX_timestamp/`.
3. Start AutoGen session.
4. Collect and persist chat trajectory events.
5. Submit `submission.csv` with Kaggle API.
6. Compare score to `TARGET_MSE`.
7. Stop when target reached or iteration limit exceeded.
8. Handle `KeyboardInterrupt` gracefully and flush logs.

## Logging

`TrajectoryLogger` stores timestamped events and flushes to `logs/<run>.json`.
Saved events include:
- iteration starts
- chat messages snapshot
- Kaggle submission status and score
- stop conditions or interruptions
