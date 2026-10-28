"""
Cost model for the hard-hat compliance decision.

One case = one CCTV frame of a work area. Positive = the frame contains at
least one worker with a bare head (class `head`). The frame's score is the
highest `head` confidence the model gives it.

Three outcomes per frame, set by two thresholds t_low <= t_high:
    score <  t_low            -> PASS    (nothing happens)
    t_low <= score < t_high   -> REVIEW  (a person looks at the frame)
    score >= t_high           -> FAIL    (supervisor is radioed to intervene)

Every figure is defended in REPORT.md. Shared by train.py and app.py so the
sweep the pipeline saves and the numbers the dashboard shows cannot drift.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    fn: float = 120.0         # missed bare head: expected injury + sanction cost
    fp: float = 4.0           # false alert: supervisor walks over for nothing
    tp: float = 4.0           # correct alert: same intervention, risk averted
    tn: float = 0.0           # correctly ignored frame
    review: float = 0.15      # ~20 s of a safety assistant at ~27 EUR/h loaded
    human_accuracy: float = 0.97  # reviewer gets the frame right this often
    base_rate: float = 0.05   # share of production frames with a bare head
    frames_per_day: int = 3000    # 25 cameras x 1 frame / 5 min x 10 h
    review_seconds: float = 20.0
    reviewer_hours_per_day: float = 6.0  # productive review time per person

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_COSTS = CostModel()


def band_rates(scores: np.ndarray, labels: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of positives and negatives that land in each outcome band."""
    pos, neg = scores[labels == 1], scores[labels == 0]

    def split(s: np.ndarray) -> tuple[float, float, float]:
        if len(s) == 0:
            return 0.0, 0.0, 0.0
        passed = float(np.mean(s < t_low))
        failed = float(np.mean(s >= t_high))
        return passed, 1.0 - passed - failed, failed

    pos_pass, pos_review, pos_fail = split(pos)
    neg_pass, neg_review, neg_fail = split(neg)
    return dict(pos_pass=pos_pass, pos_review=pos_review, pos_fail=pos_fail,
                neg_pass=neg_pass, neg_review=neg_review, neg_fail=neg_fail)


def cost_per_case(rates: dict, c: CostModel) -> float:
    """Expected EUR per frame at the production base rate, not the test one.

    Rates are measured per true class on the test split, then reweighted by
    c.base_rate, because the test split (~20% positive) is far richer in
    violations than a real site.
    """
    h = c.human_accuracy
    reviewed_pos = c.review + h * c.tp + (1 - h) * c.fn
    reviewed_neg = c.review + h * c.tn + (1 - h) * c.fp
    pos = rates["pos_pass"] * c.fn + rates["pos_review"] * reviewed_pos + rates["pos_fail"] * c.tp
    neg = rates["neg_pass"] * c.tn + rates["neg_review"] * reviewed_neg + rates["neg_fail"] * c.fp
    return c.base_rate * pos + (1 - c.base_rate) * neg


def review_share(rates: dict, c: CostModel) -> float:
    return c.base_rate * rates["pos_review"] + (1 - c.base_rate) * rates["neg_review"]


def staffing(share: float, c: CostModel) -> dict:
    per_day = share * c.frames_per_day
    hours = per_day * c.review_seconds / 3600
    return dict(reviews_per_day=per_day, review_hours_per_day=hours,
                reviewers_needed=hours / c.reviewer_hours_per_day)


def evaluate_band(scores, labels, t_low, t_high, c: CostModel = DEFAULT_COSTS) -> dict:
    rates = band_rates(scores, labels, t_low, t_high)
    share = review_share(rates, c)
    return dict(t_low=t_low, t_high=t_high, **rates,
                cost_per_1000=1000 * cost_per_case(rates, c),
                review_share=share, **staffing(share, c))


def single_threshold_sweep(scores, labels, c: CostModel = DEFAULT_COSTS, grid=None) -> list[dict]:
    """Model decides alone: pass below t, fail at or above t, no review."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    return [evaluate_band(scores, labels, float(t), float(t), c) for t in grid]


def best_band(scores, labels, c: CostModel = DEFAULT_COSTS, grid=None,
              max_review_share: float | None = None) -> dict:
    """Cheapest (t_low, t_high) pair, optionally capped by review capacity."""
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            row = evaluate_band(scores, labels, float(lo), float(hi), c)
            if max_review_share is not None and row["review_share"] > max_review_share:
                continue
            if best is None or row["cost_per_1000"] < best["cost_per_1000"]:
                best = row
    return best


def alternatives(c: CostModel = DEFAULT_COSTS) -> dict:
    """Cost per 1,000 frames of the policies that use no model at all."""
    h = c.human_accuracy
    do_nothing = c.base_rate * c.fn + (1 - c.base_rate) * c.tn
    check_all = (c.review + c.base_rate * (h * c.tp + (1 - h) * c.fn)
                 + (1 - c.base_rate) * (h * c.tn + (1 - h) * c.fp))
    return dict(do_nothing=1000 * do_nothing, human_checks_everything=1000 * check_all,
                human_checks_everything_staff=staffing(1.0, c))
