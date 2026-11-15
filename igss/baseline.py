"""The baseline: rules assigned by hand, no search (SPEC §8).

Two kinds of run, each one episode:
  alone    - every iGSS agent uses rule X;
  in pairs - the first half of the iGSS agents use X, the second half use Y.
Each rule is the named predicate of rules.py, used directly as the action rule.
"""

import itertools

from igss import rules
from igss.episode import IGSS, make_agents, make_rng, play_episode, summarise


def baseline_seed(seed_index):
    """The text seed of baseline episode `seed_index` (SPEC §9)."""
    return f"baseline|{seed_index}"


def _group(result, ids, rounds):
    """Payoff per round, cooperation rate and input reads of a group of iGSS agents."""
    return {
        "payoff": sum(result.payoff[i] for i in ids) / (len(ids) * rounds),
        "coop": sum(result.cooperated[i] for i in ids) / (len(ids) * rounds),
        "reads_one": sum(result.reads[i][1] for i in ids),
        "reads_zero": sum(result.reads[i][0] for i in ids),
    }


def run_alone(rule_name, env, n_igss, seed_text):
    """All iGSS agents use one rule. Returns the episode summary plus the group's statistics."""
    agents = make_agents([rules.ACTION_RULES[rule_name]] * n_igss, env.n_c, env.n_d)
    result = play_episode(agents, env, make_rng(seed_text))
    ids = [agent.agent_id for agent in agents if agent.kind == IGSS]
    return {"summary": summarise(result, agents, env), "x": _group(result, ids, env.rounds)}


def run_pair(rule_x, rule_y, env, n_igss, seed_text):
    """The first half of the iGSS agents use X, the second half use Y."""
    if n_igss % 2:
        raise ValueError("an in-pairs run needs an even number of iGSS agents")
    half = n_igss // 2
    action_rules = [rules.ACTION_RULES[rule_x]] * half + [rules.ACTION_RULES[rule_y]] * half
    agents = make_agents(action_rules, env.n_c, env.n_d)
    result = play_episode(agents, env, make_rng(seed_text))
    return {
        "summary": summarise(result, agents, env),
        "x": _group(result, range(half), env.rounds),
        "y": _group(result, range(half, n_igss), env.rounds),
    }


def baseline_rows(env, n_igss, n_seeds):
    """Every alone run and every in-pairs run of an environment, as output rows (SPEC §10)."""
    names = list(rules.ACTION_RULES)
    sets = [("alone", x, None) for x in names]
    sets += [("pair", x, y) for x, y in itertools.combinations(names, 2)]
    for kind, x, y in sets:
        for seed_index in range(n_seeds):
            seed_text = baseline_seed(seed_index)
            if kind == "alone":
                run = run_alone(x, env, n_igss, seed_text)
            else:
                run = run_pair(x, y, env, n_igss, seed_text)
            group_y = run.get("y", dict.fromkeys(("payoff", "coop", "reads_one", "reads_zero")))
            yield {
                "kind": kind,
                "rule_x": "TT:" + rules.ACTION_BITS[x],
                "rule_y": "TT:" + rules.ACTION_BITS[y] if y else None,
                "seed_index": seed_index,
                "payoff_x": run["x"]["payoff"],
                "payoff_y": group_y["payoff"],
                "payoff_c": run["summary"]["payoff_c"],
                "payoff_d": run["summary"]["payoff_d"],
                "coop_x": run["x"]["coop"],
                "coop_y": group_y["coop"],
                "reads_one_x": run["x"]["reads_one"],
                "reads_zero_x": run["x"]["reads_zero"],
                "reads_one_y": group_y["reads_one"],
                "reads_zero_y": group_y["reads_zero"],
            }


BASELINE_COLUMNS = (
    "kind", "rule_x", "rule_y", "seed_index",
    "payoff_x", "payoff_y", "payoff_c", "payoff_d",
    "coop_x", "coop_y",
    "reads_one_x", "reads_zero_x", "reads_one_y", "reads_zero_y",
)
