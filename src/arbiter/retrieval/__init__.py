"""Comparable case lookup over relational and vector indexes.

Every query here takes the requesting event timestamp and excludes cases that had
not resolved by then. The filter lives in the query rather than in a caller
convention, because a comparable whose outcome postdates the prediction silently
invalidates every result derived from it.
"""
