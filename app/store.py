from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Artifact:
    name: str
    filename: str
    kind: str


@dataclass
class Job:
    id: str
    tool: str
    status: str
    created_at: str
    updated_at: str
    input_name: str
    message: str = ""
    artifacts: List[Artifact] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return payload


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create(self, job_id: str, tool: str, input_name: str, message: str = "") -> Job:
        job = Job(
            id=job_id,
            tool=tool,
            status="queued",
            created_at=utc_now(),
            updated_at=utc_now(),
            input_name=input_name,
            message=message,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job

        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        job = Job(
            id=data["id"],
            tool=data["tool"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            input_name=data["input_name"],
            message=data.get("message", ""),
            artifacts=[Artifact(**artifact) for artifact in data.get("artifacts", [])],
            error=data.get("error", ""),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        artifacts: Optional[List[Artifact]] = None,
    ) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if artifacts is not None:
                job.artifacts = artifacts
            job.updated_at = utc_now()
            self._persist(job)
            return job

    def _persist(self, job: Job) -> None:
        path = self.job_dir(job.id) / "job.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(job.to_dict(), handle, ensure_ascii=False, indent=2)

