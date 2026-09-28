"""
Guardrails for the public demo (enabled with DEMO_MODE=true).

Three layers keep a public link from running up a bill:
1. Per-session limits: questions per session, a cooldown between
   questions, question length, and uploads per session.
2. A global daily question cap shared by every visitor of this app process.
3. A hard monthly spend limit set in the Anthropic Console (outside the
   code, so it holds even if these counters reset on a restart).

Pure Python so it can be unit tested without Streamlit.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class DemoLimits:
    questions_per_session: int = field(default_factory=lambda: _int_env("DEMO_MAX_QUESTIONS_PER_SESSION", 10))
    questions_per_day: int = field(default_factory=lambda: _int_env("DEMO_MAX_QUESTIONS_PER_DAY", 150))
    cooldown_seconds: int = field(default_factory=lambda: _int_env("DEMO_COOLDOWN_SECONDS", 5))
    max_question_chars: int = field(default_factory=lambda: _int_env("DEMO_MAX_QUESTION_CHARS", 500))
    uploads_per_session: int = field(default_factory=lambda: _int_env("DEMO_MAX_UPLOADS_PER_SESSION", 2))
    max_upload_bytes: int = field(default_factory=lambda: _int_env("DEMO_MAX_UPLOAD_BYTES", 300_000))
    upload_ttl_seconds: int = field(default_factory=lambda: _int_env("DEMO_UPLOAD_TTL_SECONDS", 1800))


class DailyCounter:
    """Questions asked today across all sessions. Thread-safe, resets at midnight."""

    def __init__(self, today=date.today) -> None:
        self._today = today
        self._day = today()
        self._count = 0
        self._lock = threading.Lock()

    def try_take(self, cap: int) -> bool:
        with self._lock:
            if self._today() != self._day:
                self._day, self._count = self._today(), 0
            if self._count >= cap:
                return False
            self._count += 1
            return True

    @property
    def count(self) -> int:
        return self._count


@dataclass
class SessionUsage:
    questions: int = 0
    uploads: int = 0
    last_question_at: float = 0.0


def check_question(question: str, usage: SessionUsage, daily: DailyCounter, limits: DemoLimits,
                   now: float | None = None) -> str | None:
    """Return a user-facing reason to refuse, or None and record the question."""
    now = time.time() if now is None else now
    if len(question) > limits.max_question_chars:
        return f"Please keep questions under {limits.max_question_chars} characters."
    if usage.questions >= limits.questions_per_session:
        return (f"This demo allows {limits.questions_per_session} questions per session. "
                "The source code is linked in the sidebar if you want to run it yourself.")
    if now - usage.last_question_at < limits.cooldown_seconds:
        return "Please wait a few seconds between questions."
    if not daily.try_take(limits.questions_per_day):
        return "The demo has reached its daily question limit. Please try again tomorrow."
    usage.questions += 1
    usage.last_question_at = now
    return None


def check_upload(size_bytes: int, usage: SessionUsage, limits: DemoLimits) -> str | None:
    """Return a reason to refuse an upload, or None and record it."""
    if usage.uploads >= limits.uploads_per_session:
        return f"This demo allows {limits.uploads_per_session} uploads per session."
    if size_bytes > limits.max_upload_bytes:
        return f"Demo uploads are limited to {limits.max_upload_bytes // 1000} KB per file."
    usage.uploads += 1
    return None
