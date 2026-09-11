"""Trigger evaluation and event normalization.

Decides which filings constitute events for each strategy, using form types,
item codes, and XBRL facts. Deterministic by design; this is where most of the
filing volume is filtered out before anything expensive runs.
"""
