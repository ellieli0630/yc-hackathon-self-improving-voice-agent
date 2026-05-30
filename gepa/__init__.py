"""Nomie voice-prompt optimization.

Whole-prompt evolution with dspy.GEPA (the published Genetic-Pareto optimizer —
reflective mutation + per-instance Pareto), judged by an OpenAI LLM-judge over a
golden set of real Nomie conversations, with Cekura as the independent held-out
evaluator (never seen during optimization). See dspy_gepa.py (optimize) and
report.py (held-out base-vs-winner on both judges).
"""
