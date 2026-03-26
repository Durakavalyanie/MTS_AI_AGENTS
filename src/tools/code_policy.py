from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "category_encoders",
    "statsmodels",
}

BLOCKED_TOKENS = (
    "pip install",
    "subprocess",
    "requests.",
    "urllib.",
    "http://",
    "https://",
    "matplotlib",
    "seaborn",
    "plotly",
    ".plot(",
    "os.system(",
    "__import__",
    "importlib",
)


def _format_policy_feedback(reason: str, how_to_fix: str) -> str:
    return json.dumps({
        "status": "BLOCKED",
        "summary": "Deterministic safety policy rejected the code.",
        "output": f"POLICY_REASON: {reason}\nHOW_TO_FIX: {how_to_fix}"
    }, ensure_ascii=False)


def _extract_import_roots(code: str) -> set[str]:
    roots: set[str] = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    roots.add(node.module.split(".")[0])
    except SyntaxError:
        pass  # Syntax errors will be caught by the executor anyway
    return roots


def _has_path_escape(code: str) -> bool:
    # Deterministic guard: block relative escape and absolute writes.
    # Allow reading from ../../data/raw
    temp_code = re.sub(r"\.\./\.\./data/raw", "", code)
    escape_patterns = [
        r"\.\./",
        r"open\(\s*[\"']/",
        r"to_csv\(\s*[\"']/",
        r"to_pickle\(\s*[\"']/",
    ]
    return any(re.search(pattern, temp_code) for pattern in escape_patterns)


def validate_python_code(code: str, work_dir: Path) -> tuple[bool, str]:
    lowered = code.lower()
    for token in BLOCKED_TOKENS:
        if token in lowered:
            return (
                False,
                _format_policy_feedback(
                    reason=f"Blocked token detected: {token}",
                    how_to_fix="Remove blocked operations (package installs, web access, plotting) and retry.",
                ),
            )

    if _has_path_escape(code):
        return (
            False,
            _format_policy_feedback(
                reason="Path traversal or absolute-path write detected.",
                how_to_fix="Write only to files inside current workspace run directory.",
            ),
        )

    imports = _extract_import_roots(code)
    disallowed = sorted(
        module
        for module in imports
        if module not in ALLOWED_IMPORTS
        and module
        not in {"os", "pathlib", "json", "math", "time", "typing", "re", "warnings"}
    )
    if disallowed:
        return (
            False,
            _format_policy_feedback(
                reason="Disallowed imports: " + ", ".join(disallowed),
                how_to_fix=(
                    "Use only approved ML/data libraries and minimal stdlib helpers "
                    "(os/pathlib/json/math/time/typing/re/warnings)."
                ),
            ),
        )

    if not work_dir.exists():
        return (
            False,
            _format_policy_feedback(
                reason=f"work_dir does not exist: {work_dir}",
                how_to_fix="Recreate run workspace and resubmit code.",
            ),
        )

    return True, "ok"
