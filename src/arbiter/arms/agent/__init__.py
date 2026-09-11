"""Agent arm: a LangGraph graph of per-strategy analysts.

Analyst nodes are constructed without tools. The packet is the only input they
can reach, which enforces the information boundary structurally rather than
through prompt instructions.
"""
