"""SQLite-backed operator feedback, label workflow, and drift detection."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .util import format_timestamp, now_utc, parse_timestamp

ALLOWED_LABELS = {"useful", "not_useful", "incorrect", "unsafe", "needs_followup"}


class FeedbackError(ValueError):
    """Raised for invalid feedback records or store state."""


@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    request_id: str
    agent_version: str
    rating: int
    label: str
    comment: str
    site_id: str
    slice_name: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "agent_version": self.agent_version,
            "rating": self.rating,
            "label": self.label,
            "comment": self.comment,
            "site_id": self.site_id,
            "slice": self.slice_name,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DriftReport:
    status: str
    recent_n: int
    baseline_n: int
    recent_positive_rate: float
    baseline_positive_rate: float
    z_score: float
    recent_mean: float
    baseline_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recent_n": self.recent_n,
            "baseline_n": self.baseline_n,
            "recent_positive_rate": round(self.recent_positive_rate, 4),
            "baseline_positive_rate": round(self.baseline_positive_rate, 4),
            "z_score": round(self.z_score, 3),
            "recent_mean": round(self.recent_mean, 3),
            "baseline_mean": round(self.baseline_mean, 3),
        }


class FeedbackStore:
    """Small, transactional feedback store; one connection per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    agent_version TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    label TEXT NOT NULL DEFAULT '',
                    comment TEXT NOT NULL DEFAULT '',
                    site_id TEXT NOT NULL DEFAULT '',
                    slice_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_site ON feedback(site_id)"
            )

    def add(
        self,
        *,
        request_id: str,
        rating: int,
        agent_version: str = "1.0.0",
        label: str = "",
        comment: str = "",
        site_id: str = "",
        slice_name: str = "",
        created_at: datetime | str | None = None,
    ) -> FeedbackRecord:
        if not request_id:
            raise FeedbackError("request_id is required")
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise FeedbackError("rating must be an integer within [1, 5]")
        if label and label not in ALLOWED_LABELS:
            raise FeedbackError(f"label must be empty or one of {sorted(ALLOWED_LABELS)}")
        timestamp = (
            format_timestamp(parse_timestamp(created_at))
            if created_at is not None
            else format_timestamp(now_utc())
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback
                    (request_id, agent_version, rating, label, comment, site_id, slice_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    agent_version,
                    rating,
                    label,
                    comment,
                    site_id,
                    slice_name,
                    timestamp,
                ),
            )
            record_id = int(cursor.lastrowid)
        return self.get(record_id)

    def get(self, record_id: int) -> FeedbackRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM feedback WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise FeedbackError(f"feedback not found: {record_id}")
        return self._row(row)

    def list(
        self,
        *,
        limit: int = 100,
        site_id: str | None = None,
        label: str | None = None,
    ) -> list[FeedbackRecord]:
        if limit <= 0:
            raise FeedbackError("limit must be positive")
        query = "SELECT * FROM feedback"
        clauses: list[str] = []
        params: list[Any] = []
        if site_id is not None:
            clauses.append("site_id = ?")
            params.append(site_id)
        if label is not None:
            clauses.append("label = ?")
            params.append(label)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["id"],
            request_id=row["request_id"],
            agent_version=row["agent_version"],
            rating=row["rating"],
            label=row["label"],
            comment=row["comment"],
            site_id=row["site_id"],
            slice_name=row["slice_name"],
            created_at=row["created_at"],
        )

    def label(self, record_id: int, label: str) -> FeedbackRecord:
        if label not in ALLOWED_LABELS:
            raise FeedbackError(f"label must be one of {sorted(ALLOWED_LABELS)}")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE feedback SET label = ? WHERE id = ?", (label, record_id)
            )
            if cursor.rowcount == 0:
                raise FeedbackError(f"feedback not found: {record_id}")
        return self.get(record_id)

    def pending(self, *, limit: int = 50) -> list[FeedbackRecord]:
        if limit <= 0:
            raise FeedbackError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM feedback WHERE label = '' ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM feedback").fetchall()
        records = [self._row(row) for row in rows]
        if not records:
            return {
                "total": 0,
                "positive_rate": 0.0,
                "mean_rating": 0.0,
                "by_label": {},
                "by_slice": {},
                "by_version": {},
            }
        ratings = [record.rating for record in records]
        by_label: dict[str, int] = {}
        by_slice: dict[str, dict[str, Any]] = {}
        by_version: dict[str, int] = {}
        for record in records:
            key = record.label or "unlabelled"
            by_label[key] = by_label.get(key, 0) + 1
            by_version[record.agent_version] = by_version.get(record.agent_version, 0) + 1
            bucket = by_slice.setdefault(
                record.slice_name or "default", {"total": 0, "positive": 0}
            )
            bucket["total"] += 1
            bucket["positive"] += 1 if record.rating >= 4 else 0
        return {
            "total": len(records),
            "positive_rate": sum(1 for rating in ratings if rating >= 4) / len(ratings),
            "mean_rating": sum(ratings) / len(ratings),
            "by_label": dict(sorted(by_label.items())),
            "by_slice": {
                key: {
                    "total": value["total"],
                    "positive_rate": value["positive"] / value["total"],
                }
                for key, value in sorted(by_slice.items())
            },
            "by_version": dict(sorted(by_version.items())),
        }

    def drift(
        self,
        *,
        recent_n: int = 50,
        baseline_n: int = 200,
        positive_threshold: int = 4,
        z_threshold: float = 1.96,
    ) -> DriftReport:
        if recent_n <= 0 or baseline_n <= 0:
            raise FeedbackError("sample sizes must be positive")
        if not 1 <= positive_threshold <= 5:
            raise FeedbackError("positive_threshold must be within [1, 5]")
        if z_threshold <= 0:
            raise FeedbackError("z_threshold must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC, id DESC LIMIT ?",
                (recent_n + baseline_n,),
            ).fetchall()
        records = [self._row(row) for row in rows]
        if len(records) < recent_n + baseline_n:
            return DriftReport(
                "insufficient_data",
                recent_n=len(records),
                baseline_n=0,
                recent_positive_rate=0.0,
                baseline_positive_rate=0.0,
                z_score=0.0,
                recent_mean=0.0,
                baseline_mean=0.0,
            )
        recent = records[:recent_n]
        baseline = records[recent_n : recent_n + baseline_n]
        recent_positive = sum(1 for record in recent if record.rating >= positive_threshold)
        baseline_positive = sum(1 for record in baseline if record.rating >= positive_threshold)
        p1 = recent_positive / recent_n
        p0 = baseline_positive / baseline_n
        pooled = (recent_positive + baseline_positive) / (recent_n + baseline_n)
        denominator = math.sqrt(max(pooled * (1 - pooled) * (1 / recent_n + 1 / baseline_n), 1e-12))
        z_score = (p1 - p0) / denominator
        if abs(z_score) < z_threshold:
            status = "stable"
        elif z_score < 0:
            status = "regressed"
        else:
            status = "improved"
        return DriftReport(
            status=status,
            recent_n=recent_n,
            baseline_n=baseline_n,
            recent_positive_rate=p1,
            baseline_positive_rate=p0,
            z_score=z_score,
            recent_mean=sum(record.rating for record in recent) / recent_n,
            baseline_mean=sum(record.rating for record in baseline) / baseline_n,
        )

    def export_jsonl(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.list(limit=1_000_000)
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return len(records)

    def __len__(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM feedback").fetchone()
        return int(row["count"])
