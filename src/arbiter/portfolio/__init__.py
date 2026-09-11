"""Position sizing, risk limits, shadow books, and the broker adapter.

Deterministic throughout. Arms emit direction and conviction; this layer converts
them into orders under hard position and concentration caps, so no model output
reaches a broker unchecked.
"""
