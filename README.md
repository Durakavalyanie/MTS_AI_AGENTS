# MTS_AI_AGENTS

Quickstart for the AutoGen multi-agent Kaggle loop (OpenRouter-based).

## 1) Prepare environment

- Create/activate project virtual env (`.venv`).
- Install dependencies:
  - `pip install -r requirements.txt`
- Fill placeholders in `.env`:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_BASE_URL`
  - `OPENROUTER_MODEL_ORCHESTRATOR`
  - `OPENROUTER_MODEL_DATA_ANALYST`
  - `OPENROUTER_MODEL_DATA_ENGINEER`
  - `OPENROUTER_MODEL_ML_ENGINEER`
  - `OPENROUTER_MODEL_REVIEWER`
  - `KAGGLE_USERNAME`
  - `KAGGLE_KEY`
  - `KAGGLE_COMPETITION`
  - `TARGET_MSE`

## 2) Data layout

Put Kaggle files into `data/raw/`:
- `train.csv`
- `test.csv`
- `sample_submition.csv` (kept as-is based on current dataset filename)

## 3) Run loop

- Run: `python -m src.main_loop`
- The script creates `workspace/run_XXX_timestamp/` for each launch.
- Agents generate `submission.csv` in the run folder.
- Python wrapper submits to Kaggle and checks `TARGET_MSE`.
- Full trajectory is written into `logs/*.json`.

## 4) Notes

- OpenRouter is used for all LLM-based agents.
- Different roles can use different models via per-role env variables.
- Code execution goes through deterministic policy checks before running.