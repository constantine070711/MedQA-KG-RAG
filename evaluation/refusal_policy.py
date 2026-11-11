"""Versioned evaluation policy for refusal labels.

Policy changes belong in the dataset annotation, not in hidden QID branches.
This module is kept as the single read path so evaluation runs can expose the
policy version alongside their results.
"""

from __future__ import annotations


POLICY_VERSION = "2026-11-09-three-layer-v1"


def expected_refusal_for(question: dict) -> bool:
    """Return the dataset refusal label under the current policy version."""
    return bool(question.get("expected_refusal", False))
