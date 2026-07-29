"""Core data-profiling engine.

The package is split so that every expensive computation happens exactly once
and is then reused by every consumer:

``dataio``      loading, delimiter detection and lossless memory optimisation
``stats``       descriptive statistics, missing-value and correlation analysis
``summaries``   compact, pre-aggregated summaries of distributions/outliers
``plots``       interactive (Plotly) figures rendered from the summaries
``static``      static (Matplotlib) figures rendered from the same summaries
``report``      PDF and ZIP export built from the summaries

Because the interactive charts and the exported charts are both derived from
the same pre-aggregated summaries, the web app and the downloadable report can
never disagree with one another.
"""

from __future__ import annotations

__all__ = [
    "dataio",
    "stats",
    "summaries",
    "plots",
    "static",
    "report",
]

__version__ = "2.0.0"
