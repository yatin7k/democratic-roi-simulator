from dataclasses import dataclass
from typing import List, Dict


@dataclass
class VoterGroup:
    """
    Parameters describing one voter group.

    name:
        Human-readable group name.

    cost:
        Cost of one mobilization/contact attempt, c_g.

    mobilization_effect:
        Immediate causal turnout effect of contact, m_g.
        Example: 0.03 = +3 percentage points.

    support_probability:
        Probability that the marginal mobilized voter supports
        the relevant candidate/organization, s_g.

    persistence:
        Fraction of the original mobilization effect assumed to
        persist from one future election to the next, rho_g.
        Must be between 0 and 1.
    """
    name: str
    cost: float
    mobilization_effect: float
    support_probability: float
    persistence: float


def downstream_effect(
    group: VoterGroup,
    t: int
) -> float:
    """
    Downstream participation effect in future election t.

    Version 0.1 assumes:
        p_(g,t) = m_g * rho_g^t
    """
    return group.mobilization_effect * (group.persistence ** t)


def current_election_roi(group: VoterGroup) -> float:
    """
    Current-cycle candidate-specific electoral return:

        R^C_g = (s_g * m_g) / c_g
    """
    return (
        group.support_probability
        * group.mobilization_effect
        / group.cost
    )


def democratic_roi(
    group: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4
) -> float:
    """
    Long-run democratic-participation return:

        R^D_g =
        [m_g + sum(beta^t * p_(g,t))] / c_g

    horizon = number of FUTURE elections after the current one.
    """
    future_participation = sum(
        (beta ** t) * downstream_effect(group, t)
        for t in range(1, horizon + 1)
    )

    return (
        group.mobilization_effect
        + future_participation
    ) / group.cost


def organizational_roi(
    group: VoterGroup,
    lambda_internalization: float,
    beta: float = 1.0,
    horizon: int = 4,
    future_support_probability: float | None = None
) -> float:
    """
    Organization-specific long-run electoral return:

        R^O_(g,j) =
        [
            s_(g,0)m_g
            +
            lambda_j *
            sum(beta^t * s_(g,t) * p_(g,t))
        ] / c_g

    lambda_internalization:
        Share of future electoral value the organization
        effectively internalizes.

    future_support_probability:
        Version 0.1 assumes future candidate/party support is
        constant unless another value is supplied.
    """

    if not 0 <= lambda_internalization <= 1:
        raise ValueError("lambda must be between 0 and 1.")

    future_s = (
        group.support_probability
        if future_support_probability is None
        else future_support_probability
    )

    current_value = (
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
        current_value
        + lambda_internalization * future_value
    ) / group.cost


def compare_groups(
    group_a: VoterGroup,
    group_b: VoterGroup,
    beta: float = 1.0,
    horizon: int = 4
) -> Dict:
    """
    Compare two groups under current-election and
    long-run democratic objectives.
    """

    current_a = current_election_roi(group_a)
    current_b = current_election_roi(group_b)

    democratic_a = democratic_roi(
        group_a, beta, horizon
    )
    democratic_b = democratic_roi(
        group_b, beta, horizon
    )

    current_winner = (
        group_a.name if current_a > current_b
        else group_b.name if current_b > current_a
        else "Tie"
    )

    democratic_winner = (
        group_a.name if democratic_a > democratic_b
        else group_b.name if democratic_b > democratic_a
        else "Tie"
    )

    paradox_exists = (
        current_winner != democratic_winner
        and current_winner != "Tie"
        and democratic_winner != "Tie"
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
        "democratic_roi_paradox": paradox_exists,
    }


def print_comparison(result: Dict) -> None:
    print("\nCURRENT-ELECTION ROI")
    for group, roi in result["current_roi"].items():
        print(f"{group}: {roi:.6f}")

    print("\nLONG-RUN DEMOCRATIC ROI")
    for group, roi in result["democratic_roi"].items():
        print(f"{group}: {roi:.6f}")

    print(
        "\nCurrent-election preferred group:",
        result["current_winner"]
    )

    print(
        "Long-run participation preferred group:",
        result["democratic_winner"]
    )

    print(
        "Democratic ROI Paradox:",
        result["democratic_roi_paradox"]
    )


if __name__ == "__main__":

    # ---------------------------------------------------------
    # ILLUSTRATIVE VALUES ONLY.
    # These are NOT empirical estimates.
    # ---------------------------------------------------------

    young_voters = VoterGroup(
        name="Young / First-Time Voters",
        cost=5.00,
        mobilization_effect=0.030,
        support_probability=0.55,
        persistence=0.60,
    )

    established_voters = VoterGroup(
        name="Established Voters",
        cost=5.00,
        mobilization_effect=0.050,
        support_probability=0.55,
        persistence=0.15,
    )

    beta = 0.95
    horizon = 4

    result = compare_groups(
        young_voters,
        established_voters,
        beta=beta,
        horizon=horizon,
    )

    print_comparison(result)

    # Example: compare organizational ROI at
    # different internalization levels.

    print("\nORGANIZATIONAL ROI BY λ")

    for lam in [0.0, 0.25, 0.50, 0.75, 1.0]:

        young_org = organizational_roi(
            young_voters,
            lambda_internalization=lam,
            beta=beta,
            horizon=horizon,
        )

        established_org = organizational_roi(
            established_voters,
            lambda_internalization=lam,
            beta=beta,
            horizon=horizon,
        )

        preferred = (
            young_voters.name
            if young_org > established_org
            else established_voters.name
            if established_org > young_org
            else "Tie"
        )

        print(
            f"λ={lam:.2f} | "
            f"Young={young_org:.6f} | "
            f"Established={established_org:.6f} | "
            f"Preferred={preferred}"
        )
