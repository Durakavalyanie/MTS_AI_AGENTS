from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


@dataclass(frozen=True)
class KaggleResult:
    status: str
    public_score: float | None
    message: str


class KaggleSubmitter:
    def __init__(self, competition: str) -> None:
        self.competition = competition
        self.api = KaggleApi()
        self.api.authenticate()

    def submit(self, submission_path: Path, message: str) -> KaggleResult:
        if not submission_path.exists():
            return KaggleResult(
                status="failed",
                public_score=None,
                message=f"Submission file not found: {submission_path}",
            )

        self.api.competition_submit(
            file_name=str(submission_path),
            message=message,
            competition=self.competition,
        )
        time.sleep(4)

        submissions = self.api.competition_submissions(self.competition)
        if not submissions:
            return KaggleResult(
                status="submitted",
                public_score=None,
                message="Submission sent, but submissions list is empty.",
            )

        latest = submissions[0]
        score_raw = str(getattr(latest, "publicScore", "") or "").strip()
        score = float(score_raw) if score_raw else None
        return KaggleResult(
            status=getattr(latest, "status", "submitted"),
            public_score=score,
            message=getattr(latest, "description", "No description"),
        )
