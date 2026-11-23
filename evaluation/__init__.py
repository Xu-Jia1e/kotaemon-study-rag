"""Offline-first evaluation harness for the study-material RAG project.

This package only *reads* the project's core libraries (``kotaemon`` / ``ktem``)
and never patches or duplicates them. See ``evaluation/README.md`` for scope and
``evaluation/DESIGN.md`` for the metric definitions and interface findings.
"""

__all__ = ["adapters", "matrix", "metrics", "reporting", "scripts"]
