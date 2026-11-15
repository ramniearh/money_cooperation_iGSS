"""Rules, bit strings and expressions (SPEC §5, §6, §9).

A rule with yes/no inputs is described by its bit string: its output on every
combination of inputs, all-ones row first, first input most significant.
Every named rule below is a predicate; its bit string is derived, never typed.
"""

import functools
import itertools
import operator

import sympy
from deap import gp

MODELS = ("DR", "IR")

# The one input of an action rule, by model (SPEC §6).
ACTION_INPUT = {
    "DR": "partner_cooperated_with_me_last_time",
    "IR": "partner_standing",
}


# ---------------------------------------------------------------------------
# Bit strings (SPEC §5)
# ---------------------------------------------------------------------------

def input_rows(n_inputs):
    """All input combinations: all-ones first, first input most significant."""
    return list(itertools.product((1, 0), repeat=n_inputs))


def bit_string(rule, n_inputs):
    """Call `rule` on every input row; an output above 0 counts as 1."""
    return "".join("1" if rule(*row) > 0 else "0" for row in input_rows(n_inputs))


def rule_from_bit_string(bits):
    """A callable with the behaviour a bit string describes."""
    table = {row: int(bit) for row, bit in zip(input_rows(_n_inputs(bits)), bits)}
    return lambda *inputs: table[inputs]


def _n_inputs(bits):
    n = len(bits).bit_length() - 1
    if 2 ** n != len(bits):
        raise ValueError(f"not a bit string of a rule: {bits!r}")
    return n


# ---------------------------------------------------------------------------
# Action rules: one input (SPEC §6.1, §6.2)
# ---------------------------------------------------------------------------

def always_cooperate(input_bit):
    return True


def follow_input(input_bit):
    return input_bit == 1


def reverse_input(input_bit):
    return input_bit == 0


def always_defect(input_bit):
    return False


ACTION_RULES = {
    rule.__name__: rule
    for rule in (always_cooperate, follow_input, reverse_input, always_defect)
}
ACTION_BITS = {name: bit_string(rule, 1) for name, rule in ACTION_RULES.items()}
ALL_ACTION_BITS = [ACTION_BITS[name] for name in ACTION_RULES]  # column order in outputs

# The rule the literature calls canonical, in both models (SPEC §10).
CANONICAL_ACTION_RULE = "follow_input"
CANONICAL_ACTION_BITS = ACTION_BITS[CANONICAL_ACTION_RULE]


# ---------------------------------------------------------------------------
# Assessment rules: two inputs (SPEC §6, §14)
#
# Inputs: (donor_cooperated, other_bit). The other bit is the previous record
# in DR and the partner's standing in IR. The output is the new stored value.
# ---------------------------------------------------------------------------

def what_donor_did(donor_cooperated, other_bit):
    return donor_cooperated == 1


def cooperated_with_a_one(donor_cooperated, other_bit):
    return donor_cooperated == 1 and other_bit == 1


def did_not_refuse_a_one(donor_cooperated, other_bit):
    return donor_cooperated == 1 or other_bit == 0


def action_matches_other_bit(donor_cooperated, other_bit):
    return donor_cooperated == other_bit


ASSESSMENT_RULES = {
    rule.__name__: rule
    for rule in (
        what_donor_did,
        cooperated_with_a_one,
        did_not_refuse_a_one,
        action_matches_other_bit,
    )
}
ASSESSMENT_BITS = {name: bit_string(rule, 2) for name, rule in ASSESSMENT_RULES.items()}


# ---------------------------------------------------------------------------
# Output labels (SPEC §16)
#
# A strategy is an action rule together with the assessment rule it runs with.
# This is the only place literature names appear in the code. They are used to
# label output and play no part in what is run or measured.
# ---------------------------------------------------------------------------

ACTION_DESCRIPTIONS = {
    "always_cooperate": "always cooperate",
    "follow_input": "follow the input",
    "reverse_input": "reverse the input",
    "always_defect": "always defect",
}

_NAMED_STRATEGIES = {
    ("DR", "follow_input", "what_donor_did"): "Tit-for-Tat",
    ("DR", "follow_input", "cooperated_with_a_one"): "Grim Trigger",
    ("IR", "follow_input", "did_not_refuse_a_one"): "Standing",
    ("IR", "follow_input", "what_donor_did"): "Image Scoring",
    ("IR", "follow_input", "action_matches_other_bit"): "Stern Judging",
    ("IR", "follow_input", "cooperated_with_a_one"): "Shunning",
}
LITERATURE_NAMES = {
    (model, ACTION_BITS[action], ASSESSMENT_BITS[assessment]): name
    for (model, action, assessment), name in _NAMED_STRATEGIES.items()
}
UNNAMED = "unnamed"


def literature_name(model, action_bits, assessment_bits):
    """The literature's name for a strategy, or "unnamed" (SPEC §16)."""
    if action_bits == ACTION_BITS["always_cooperate"]:
        return "unconditional cooperation"
    if action_bits == ACTION_BITS["always_defect"]:
        return "unconditional defection"
    return LITERATURE_NAMES.get((model, action_bits, assessment_bits), UNNAMED)


# ---------------------------------------------------------------------------
# Expressions (SPEC §9): add, sub, mul over the input and the constants 0, 1
# ---------------------------------------------------------------------------

@functools.cache
def primitive_set(model):
    """The DEAP primitive set for the action rule of a model."""
    pset = gp.PrimitiveSet("action_rule", 1)
    pset.renameArguments(ARG0=ACTION_INPUT[model])
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addTerminal(0)
    pset.addTerminal(1)
    return pset


def compile_expression(expression, model):
    """Turn an expression (a DEAP tree or its text) into a Python function."""
    return gp.compile(expression, primitive_set(model))


def action_bits_of_expression(expression, model):
    """Classify an expression by the bit string of its compiled function."""
    return bit_string(compile_expression(expression, model), 1)


@functools.cache
def simplify(expression_text, model):
    """SymPy's expansion of an expression, for human reading only (SPEC §5).

    It plays no part in classification. Returns "" if SymPy fails.
    """
    names = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        ACTION_INPUT[model]: sympy.Symbol(ACTION_INPUT[model]),
    }
    try:
        return str(sympy.expand(sympy.sympify(expression_text, locals=names)))
    except Exception:
        return ""
