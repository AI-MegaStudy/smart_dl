from __future__ import annotations

from dataclasses import dataclass


GRADE_SCORES = {
    "A": 90.0,
    "B": 70.0,
    "C": 45.0,
}


@dataclass(frozen=True)
class FreshnessResult:
    fruit_type: str
    predicted_grade: str
    freshness_score: float
    color_score: float
    roundness_score: float
    bruise_probability: float
    shipping_decision: str
    model_confidence: float
    model_version: str


def calculate_freshness_score(
    predicted_grade: str,
    color_score: float,
    roundness_score: float,
    bruise_probability: float,
) -> float:
    grade_score = GRADE_SCORES.get(predicted_grade, GRADE_SCORES["C"])
    bruise_free_score = max(0.0, min(100.0, (1.0 - bruise_probability) * 100.0))
    score = (
        grade_score * 0.50
        + color_score * 0.25
        + roundness_score * 0.15
        + bruise_free_score * 0.10
    )
    return round(max(0.0, min(100.0, score)), 2)


def make_shipping_decision(freshness_score: float, bruise_probability: float) -> str:
    if freshness_score < 60.0 or bruise_probability >= 0.5:
        return "HOLD"
    if freshness_score >= 80.0 and bruise_probability < 0.2:
        return "PASS"
    return "REVIEW"

