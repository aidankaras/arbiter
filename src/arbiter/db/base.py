"""Declarative base shared by every ORM model.

Kept separate from the models themselves so that migrations can import the
metadata without importing application code that opens connections.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all Arbiter tables."""
