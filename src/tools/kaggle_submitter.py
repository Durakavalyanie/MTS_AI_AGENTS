import os
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
    def __init__(self, competition: str, username: str | None = None, key: str | None = None) -> None:
        self.competition = competition
        if username:
            os.environ["KAGGLE_USERNAME"] = username
        if key:
            os.environ["KAGGLE_KEY"] = key
            
        self.api = KaggleApi()
        self.api.authenticate()

    _POLL_INTERVALS = [5, 5, 10, 10, 15, 15, 20, 20, 30, 30]

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

        return self._poll_for_score()

    def _poll_for_score(self) -> KaggleResult:
        for wait in self._POLL_INTERVALS:
            time.sleep(wait)

            submissions = self.api.competition_submissions(self.competition)
            if not submissions:
                continue

            latest = submissions[0]
            status_str = str(getattr(latest, "status", "unknown"))

            if "pending" in status_str.lower():
                continue

            score_raw = str(getattr(latest, "publicScore", "") or "").strip()
            score = float(score_raw) if score_raw else None
            return KaggleResult(
                status=status_str,
                public_score=score,
                message=str(getattr(latest, "description", "No description")),
            )

        return KaggleResult(
            status="pending_timeout",
            public_score=None,
            message="Kaggle is still scoring the submission after all retries.",
        )
