"""Engine and session management.

One engine is created per process and cached, because connection pools are
expensive to build and a pipeline run is long-lived relative to the work it
does.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from arbiter.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first call.

    Connections are checked before use because hosted PostgreSQL closes idle
    connections, and a scheduled run can sit idle between pipeline stages.
    """
    return create_engine(get_settings().database_url, pool_pre_ping=True, future=True)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Yield a session that commits on success and rolls back on failure.

    Raises:
        Exception: whatever the caller raised, re-raised after the rollback.
    """
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
