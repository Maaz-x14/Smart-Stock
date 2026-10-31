# app/scheduler.py
"""
Daily expiry scan at 08:00 UTC (plan doc: alerts + scheduler).

A short-lived DB session is opened for each run rather than sharing
the app's request-scoped sessions - the scheduler runs outside any
request context, so it owns its own session lifecycle.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db import get_session_factory
from app.services.alerts import run_daily_expiry_scan

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _run_scan_job() -> None:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        created = run_daily_expiry_scan(db)
        logger.info(f"Daily expiry scan: created {created} alert(s)")
    except Exception:
        logger.exception("Daily expiry scan failed")
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _run_scan_job,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_expiry_scan",
        replace_existing=True,
    )
    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
