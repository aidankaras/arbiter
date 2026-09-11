"""Evidence packet construction.

A packet is the immutable, content-hashed, timestamped view of an event that
every forecasting arm consumes. Nothing downstream of a packet fetches, which is
what makes the comparison between arms valid. See docs/methodology.md.
"""
