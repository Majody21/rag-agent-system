"""Demo guardrails: per-session limits, cooldown, daily cap, upload limits."""

from datetime import date

from app.demo_limits import DailyCounter, DemoLimits, SessionUsage, check_question, check_upload


def _limits(**kw):
    base = dict(questions_per_session=3, questions_per_day=5, cooldown_seconds=5, max_question_chars=50,
                uploads_per_session=1, max_upload_bytes=1000, upload_ttl_seconds=60)
    base.update(kw)
    return DemoLimits(**base)


def test_session_limit_and_cooldown():
    limits, usage, daily = _limits(), SessionUsage(), DailyCounter()
    assert check_question("q1", usage, daily, limits, now=100) is None
    assert "wait" in check_question("q2", usage, daily, limits, now=102)
    assert check_question("q2", usage, daily, limits, now=106) is None
    assert check_question("q3", usage, daily, limits, now=112) is None
    assert "3 questions per session" in check_question("q4", usage, daily, limits, now=120)


def test_long_question_rejected_without_using_quota():
    limits, usage, daily = _limits(), SessionUsage(), DailyCounter()
    assert "under 50" in check_question("x" * 51, usage, daily, limits, now=100)
    assert usage.questions == 0 and daily.count == 0


def test_daily_cap_is_shared_and_resets_next_day():
    today = [date(2026, 9, 28)]
    daily, limits = DailyCounter(today=lambda: today[0]), _limits(questions_per_session=100)
    sessions = [SessionUsage() for _ in range(6)]
    results = [check_question("q", s, daily, limits, now=100) for s in sessions]
    assert results[:5] == [None] * 5 and "daily question limit" in results[5]
    today[0] = date(2026, 9, 29)
    assert check_question("q", SessionUsage(), daily, limits, now=100) is None


def test_upload_limits():
    limits, usage = _limits(), SessionUsage()
    assert "1 KB" in check_upload(5000, usage, limits)
    assert check_upload(500, usage, limits) is None
    assert "1 uploads per session" in check_upload(500, usage, limits)
