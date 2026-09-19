from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import math


@dataclass(frozen=True)
class VoterGroup:
    """
    Parameters for a stylized voter group.

    Parameters
    ----------
    name:
        Human-readable name.
    cost:
        Cost per mobilization/contact attempt, c_g.
    mobilization_effect:
        Immediate causal turnout effect of contact, m_g.
        Example: 0.03 = 3 percentage points.
    support_probability:
        Probability the marginal mobilized voter supports
        the relevant candidate/organization, s_g.
    persistence:
        Geometric persistence parameter rho_g in [0, 1].
        Version 1 assumes:
            p_(g,t) = m_g * rho_g^t
    """
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
    Version 1 downstream effect:
        p_(g,t) = m_g * rho_g^t
    """
    group.validate()
    if t < 1:
        raise ValueError("t must be >= 1 for a future election")
    return group.mobilization_effect * (group.persistence ** t)


def current_election_roi(group: VoterGroup) -> float:
    """
    Candidate-specific current-election return:
        R^C_g = (s_g * m_g) / c_g
    """
    group.validate()
    return (
        group.support_probability
        * group.mobilization_effect
        / group.cost
    )


def democratic_roi(
    group: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> float:
    """
    Long-run democratic-participation return:
        R^D_g =
        [m_g + sum_{t=1..T}(beta^t * p_(g,t))] / c_g
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
    Organization-specific intertemporal electoral return:
        R^O_(g,j) =
        [s_(g,0)m_g
         + lambda_j * sum(beta^t * s_(g,t) * p_(g,t))]
        / c_g

    In Version 1, future support probability defaults to the
    current support probability.
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

    immediate_value = (
        group.support_probability
        * group.mobilization_effect
    )

    future_value = sum(
        (beta ** t)
        * future_s
        * downstream_effect(group, t)
        for t in range(1, horizon + 1)
    )

    return (
        immediate_value
        + lambda_internalization * future_value
    ) / group.cost


def compare_groups(
    group_a: VoterGroup,
    group_b: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
) -> Dict[str, object]:
    """
    Compare two groups under:
    1) current-election candidate-specific ROI
    2) long-run democratic participation ROI
    """
    current_a = current_election_roi(group_a)
    current_b = current_election_roi(group_b)
    democratic_a = democratic_roi(group_a, beta, horizon)
    democratic_b = democratic_roi(group_b, beta, horizon)

    def winner(a: float, b: float) -> str:
        if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
            return "Tie"
        return group_a.name if a > b else group_b.name

    current_winner = winner(current_a, current_b)
    democratic_winner = winner(democratic_a, democratic_b)

    paradox = (
        current_winner != "Tie"
        and democratic_winner != "Tie"
        and current_winner != democratic_winner
    )

    return {
        "current_roi": {
            group_a.name: current_a,
            group_b.name: current_b,
        },
        "democratic_roi": {
            group_a.name: democratic_a,
            group_b.name: democratic_b,
        },
        "current_winner": current_winner,
        "democratic_winner": democratic_winner,
        "democratic_roi_paradox": paradox,
    }


def organizational_components(
    group: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4,
    future_support_probability: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return the affine decomposition:
        R^O_g(lambda) = intercept + lambda * slope
    """
    group.validate()
    _validate_common(beta, horizon)

    future_s = (
        group.support_probability
        if future_support_probability is None
        else future_support_probability
    )

    intercept = (
        group.support_probability
        * group.mobilization_effect
        / group.cost
    )

    slope = sum(
        (beta ** t)
        * future_s
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
    Solve exactly for lambda* where:
        R^O_a(lambda*) = R^O_b(lambda*)

    Returns None if:
    - the lines never cross,
    - they overlap everywhere, or
    - the crossing lies outside [0, 1].
    """
    a0, a1 = organizational_components(group_a, beta, horizon)
    b0, b1 = organizational_components(group_b, beta, horizon)

    denominator = a1 - b1
    numerator = b0 - a0

    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None

    threshold = numerator / denominator

    if 0 <= threshold <= 1:
        return threshold
    return None


def organizational_preference(
    group_a: VoterGroup,
    group_b: VoterGroup,
    lambda_internalization: float,
    beta: float = 1.0,
    horizon: int = 4,
) -> str:
    a = organizational_roi(
        group_a, lambda_internalization, beta, horizon
    )
    b = organizational_roi(
        group_b, lambda_internalization, beta, horizon
    )

    if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
        return "Tie"
    return group_a.name if a > b else group_b.name


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
