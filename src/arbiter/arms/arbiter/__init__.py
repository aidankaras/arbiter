"""Arbiter arm: a separate model family reviewing the other three arms.

Reads the other arms predictions and reasoning from the prediction store as data,
never by importing them. It is a fourth prediction rather than a gate, so the
other arms live records stay comparable.
"""
