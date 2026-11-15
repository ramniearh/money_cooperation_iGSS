"""One episode: R rounds of the donation game from a fresh start (SPEC §4).

The baseline and both scoring options call `play_episode`. It draws only from
the random-number generator it is given, never from Python's global `random`.
"""

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from igss import rules

IGSS = "igss"
ALWAYS_COOPERATE = "always_cooperate"
ALWAYS_DEFECT = "always_defect"
KINDS = (IGSS, ALWAYS_COOPERATE, ALWAYS_DEFECT)


@dataclass
class Agent:
    agent_id: int
    kind: str                      # "igss", "always_cooperate", "always_defect"
    action_rule: Callable | None   # input bit -> number; above 0 = cooperate. None for background agents
    payoff: float = 0.0
    records: dict[int, int] = field(default_factory=dict)  # DR: partner_id -> 0/1
    standing: int = 1                                       # IR


@dataclass(frozen=True)
class Environment:
    """The parameters of the game. The number of iGSS agents is set by the agent list."""
    model: str            # "DR" or "IR"
    assessment_rule: str  # name of a predicate in rules.ASSESSMENT_RULES
    n_c: int              # unconditional cooperators
    n_d: int              # unconditional defectors
    b: float              # benefit to the partner
    c: float              # cost to the donor
    e_action: float       # probability that an intended cooperation fails
    rounds: int           # R


@dataclass
class EpisodeResult:
    payoff: dict[int, float]        # agent_id -> total payoff
    cooperated: dict[int, int]      # agent_id -> times it cooperated as donor (action taken)
    reads: dict[int, list[int]]     # iGSS agent_id -> [times it read 0, times it read 1]
    interactions: Counter           # (donor kind, partner kind) -> interactions
    cooperations: Counter           # (donor kind, partner kind) -> cooperations (action taken)


def make_agents(action_rules, n_c, n_d):
    """The agent list: iGSS agents, then cooperators, then defectors; id = position."""
    kinds = [IGSS] * len(action_rules) + [ALWAYS_COOPERATE] * n_c + [ALWAYS_DEFECT] * n_d
    rule_list = list(action_rules) + [None] * (n_c + n_d)
    return [Agent(agent_id=i, kind=kind, action_rule=rule)
            for i, (kind, rule) in enumerate(zip(kinds, rule_list))]


def make_rng(seed_text):
    """A generator of its own for one episode, from a text seed (SPEC §9)."""
    return random.Random(seed_text)


def play_episode(agents, env, rng):
    """Play R rounds and return what happened. `agents` must be in id order."""
    n = len(agents)
    if n < 2:
        raise ValueError("an episode needs at least two agents")
    if [agent.agent_id for agent in agents] != list(range(n)):
        raise ValueError("agent_id must be the position in the agent list")
    if env.model not in rules.MODELS:
        raise ValueError(f"unknown model: {env.model!r}")
    assess = rules.ASSESSMENT_RULES[env.assessment_rule]
    direct = env.model == "DR"

    # Fresh start: payoffs 0, every record 1 (an absent record counts as 1), every standing 1.
    for agent in agents:
        agent.payoff = 0.0
        agent.records = {}
        agent.standing = 1

    cooperated = dict.fromkeys(range(n), 0)
    reads = {agent.agent_id: [0, 0] for agent in agents if agent.kind == IGSS}
    interactions = Counter()
    cooperations = Counter()

    for _ in range(env.rounds):
        # 1. A fresh copy of the list, shuffled.
        order = list(agents)
        rng.shuffle(order)
        for donor in order:
            # 2. A partner drawn from the other agents, in id order.
            j = rng.randrange(n - 1)
            partner = agents[j if j < donor.agent_id else j + 1]

            # 3. The donor's input.
            if direct:
                input_bit = donor.records.get(partner.agent_id, 1)
            else:
                input_bit = partner.standing

            # 4. The decision.
            if donor.kind == IGSS:
                decides_to_cooperate = donor.action_rule(input_bit) > 0
                reads[donor.agent_id][input_bit] += 1
            else:
                decides_to_cooperate = donor.kind == ALWAYS_COOPERATE

            # 5. Execution error: one draw in every interaction; only an
            #    intended cooperation can fail.
            error_draw = rng.random()
            cooperates = decides_to_cooperate and not error_draw < env.e_action

            # 6. Payoffs.
            pair = (donor.kind, partner.kind)
            interactions[pair] += 1
            if cooperates:
                donor.payoff -= env.c
                partner.payoff += env.b
                cooperated[donor.agent_id] += 1
                cooperations[pair] += 1

            # 7. The assessment rule, applied at once.
            action = int(cooperates)
            if direct:
                previous = partner.records.get(donor.agent_id, 1)
                partner.records[donor.agent_id] = int(assess(action, previous))
            else:
                donor.standing = int(assess(action, input_bit))

    return EpisodeResult(
        payoff={agent.agent_id: agent.payoff for agent in agents},
        cooperated=cooperated,
        reads=reads,
        interactions=interactions,
        cooperations=cooperations,
    )


# ---------------------------------------------------------------------------
# Summaries used by the baseline, the scoring functions and the logs
# ---------------------------------------------------------------------------

def _share(numerator, denominator):
    """A share, or None when there is nothing to take a share of (logged as an empty cell)."""
    return numerator / denominator if denominator else None


def summarise(result, agents, env):
    """Flat statistics of one episode (the per-generation columns of SPEC §10).

    Payoffs are per round. A type with no agents gets None.
    """
    by_kind = {kind: [a.agent_id for a in agents if a.kind == kind] for kind in KINDS}

    def payoff_per_round(ids):
        return _share(sum(result.payoff[i] for i in ids), len(ids) * env.rounds)

    def cooperation(donor_kinds, partner_kinds):
        pairs = [(d, p) for d in donor_kinds for p in partner_kinds]
        return _share(sum(result.cooperations[pair] for pair in pairs),
                      sum(result.interactions[pair] for pair in pairs))

    return {
        "payoff_igss": payoff_per_round(by_kind[IGSS]),
        "payoff_c": payoff_per_round(by_kind[ALWAYS_COOPERATE]),
        "payoff_d": payoff_per_round(by_kind[ALWAYS_DEFECT]),
        "coop_rate": cooperation(KINDS, KINDS),
        "coop_igss": cooperation([IGSS], KINDS),
        "coop_igss_to_igss": cooperation([IGSS], [IGSS]),
        "coop_igss_to_c": cooperation([IGSS], [ALWAYS_COOPERATE]),
        "coop_igss_to_d": cooperation([IGSS], [ALWAYS_DEFECT]),
        "reads_one": sum(counts[1] for counts in result.reads.values()),
        "reads_zero": sum(counts[0] for counts in result.reads.values()),
    }


SUMMARY_COLUMNS = (
    "payoff_igss", "payoff_c", "payoff_d",
    "coop_rate", "coop_igss", "coop_igss_to_igss", "coop_igss_to_c", "coop_igss_to_d",
    "reads_one", "reads_zero",
)


def average_summaries(summaries, weights=None):
    """Weighted mean of summaries, column by column; None values are left out."""
    weights = weights or [1] * len(summaries)
    averaged = {}
    for column in SUMMARY_COLUMNS:
        pairs = [(s[column], w) for s, w in zip(summaries, weights) if s[column] is not None]
        total_weight = sum(w for _, w in pairs)
        averaged[column] = _share(sum(v * w for v, w in pairs), total_weight)
    return averaged
