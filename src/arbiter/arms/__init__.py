"""The four forecasting approaches.

Each arm consumes an evidence packet and emits a prediction. Arms never import
from one another; `arbiter` receives the other arms' output as data through the
prediction store, which keeps their independence structural rather than
conventional.
"""
