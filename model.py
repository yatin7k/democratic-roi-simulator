from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import math


@dataclass(frozen=True)
class VoterGroup:
    """Stylized voter-group parameters."""
    name: str
    cost: float
    mobilization_effect: float
    support_probability: float
    persistence: float

    def validate(self) -> None:
        if self.cost <= 0:
            raise ValueError("cost must be > 0")
        for label, value in [
            ("mobilization_effect", self.mobilization_effect),
            ("support_probability", self.support_probability),
            ("persistence", self.persistence),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between 0 and 1")


def _validate_common(beta: float, horizon: int) -> None:
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")


def downstream_effect(group: VoterGroup, t: int) -> float:
    """
    Version 2 persistence structure:
        p_(g,t) = m_g * rho_g^t

    rho is best interpreted as a modeling persistence multiplier.
    """
    group.validate()
    if t < 1:
        raise ValueError("t must be >= 1")
    return group.mobilization_effect * (group.persistence ** t)


def current_election_roi(group: VoterGroup) -> float:
    """
    Candidate-specific current-election return:
        R^C_g = (s_g * m_g) / c_g
    """
    group.validate()
    return group.support_probability * group.mobilization_effect / group.cost


def democratic_roi(
    group: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> float:
    """
    Long-run democratic-participation return:
        R^D_g = [m_g + Σ(beta^t p_(g,t))] / c_g
    """
    group.validate()
    _validate_common(beta, horizon)

    future = sum(
        (beta ** t) * downstream_effect(group, t)
        for t in range(1, horizon + 1)
    )
    return (group.mobilization_effect + future) / group.cost


def organizational_roi(
    group: VoterGroup,
    lambda_internalization: float,
    beta: float = 1.0,
    horizon: int = 4,
    future_support_probability: Optional[float] = None,
) -> float:
    """
    Organization-specific return:
        R^O_(g,j) =
        [s_(g,0)m_g + λ_j Σ(beta^t s_(g,t)p_(g,t))] / c_g

    Version 2 defaults future support probability to present support probability.
    """
    group.validate()
    _validate_common(beta, horizon)

    if not 0 <= lambda_internalization <= 1:
        raise ValueError("lambda must be between 0 and 1")

    future_s = (
        group.support_probability
        if future_support_probability is None
        else future_support_probability
    )

    if not 0 <= future_s <= 1:
        raise ValueError("future_support_probability must be between 0 and 1")

    current = group.support_probability * group.mobilization_effect

    future = sum(
        (beta ** t)
        * future_s
        * downstream_effect(group, t)
        for t in range(1, horizon + 1)
    )

    return (current + lambda_internalization * future) / group.cost


def organizational_components(
    group: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> Tuple[float, float]:
    """
    Affine decomposition:
        R^O_g(λ) = intercept + λ*slope
    """
    group.validate()
    _validate_common(beta, horizon)

    intercept = current_election_roi(group)
    slope = sum(
        (beta ** t)
        * group.support_probability
        * downstream_effect(group, t)
        for t in range(1, horizon + 1)
    ) / group.cost

    return intercept, slope


def exact_lambda_threshold(
    group_a: VoterGroup,
    group_b: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> Optional[float]:
    """
    Solve exactly for λ* such that R^O_a(λ*) = R^O_b(λ*).

    Returns None when the lines do not cross within [0,1] or are parallel.
    """
    a0, a1 = organizational_components(group_a, beta, horizon)
    b0, b1 = organizational_components(group_b, beta, horizon)

    denom = a1 - b1
    if math.isclose(denom, 0.0, abs_tol=1e-15):
        return None

    threshold = (b0 - a0) / denom
    return threshold if 0 <= threshold <= 1 else None


def compare_groups(
    group_a: VoterGroup,
    group_b: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> Dict[str, object]:
    ca = current_election_roi(group_a)
    cb = current_election_roi(group_b)
    da = democratic_roi(group_a, beta, horizon)
    db = democratic_roi(group_b, beta, horizon)

    def winner(a: float, b: float) -> str:
        if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
            return "Tie"
        return group_a.name if a > b else group_b.name

    cw = winner(ca, cb)
    dw = winner(da, db)

    return {
        "current_roi": {group_a.name: ca, group_b.name: cb},
        "democratic_roi": {group_a.name: da, group_b.name: db},
        "current_winner": cw,
        "democratic_winner": dw,
        "democratic_roi_paradox": (
            cw != "Tie" and dw != "Tie" and cw != dw
        ),
    }


def roi_curve(
    group: VoterGroup,
    lambdas: List[float],
    beta: float = 1.0,
    horizon: int = 4,
) -> List[float]:
    return [
        organizational_roi(group, lam, beta, horizon)
        for lam in lambdas
    ]
