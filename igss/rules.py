"""Rules, bit strings, strategies and expressions (SPEC §5, §6, §9, §14, §16).

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
MODES = (1, 2, 3)   # which rule evolves: 1 = action rule, 2 = assessment rule, 3 = both

# The inputs of each kind of rule, by model (SPEC §6, §14).
ACTION_INPUT = {
    "DR": "partner_cooperated_with_me_last_time",
    "IR": "partner_standing",
}
ASSESSMENT_INPUTS = {
    "DR": ("donor_cooperated", "previous_record"),
    "IR": ("donor_cooperated", "partner_standing"),
}
RULE_INPUTS = {
    "action": {model: (name,) for model, name in ACTION_INPUT.items()},
    "assessment": ASSESSMENT_INPUTS,
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


def all_bit_strings(n_inputs):
    """Every possible rule with that many inputs, from all-ones to all-zeros."""
    return ["".join(bits) for bits in itertools.product("10", repeat=2 ** n_inputs)]


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
ALL_ACTION_BITS = [ACTION_BITS[name] for name in ACTION_RULES]
ACTION_RULE_OF_BITS = {bits: ACTION_RULES[name] for name, bits in ACTION_BITS.items()}

# The action rule the literature calls canonical, in both models (SPEC §10).
# It is also the fixed action rule of mode 2.
CANONICAL_ACTION_RULE = "follow_input"
CANONICAL_ACTION_BITS = ACTION_BITS[CANONICAL_ACTION_RULE]
_UNCONDITIONAL = (ACTION_BITS["always_cooperate"], ACTION_BITS["always_defect"])


# ---------------------------------------------------------------------------
# Assessment rules: two inputs (SPEC §6, §14)
#
# Inputs: (donor_cooperated, other_bit). The other bit is the previous record
# in DR and the partner's standing in IR. The output is the new stored value.
# Only the rules the specification names are written as predicates; the full
# set of 16 is `ALL_ASSESSMENT_BITS`.
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
ALL_ASSESSMENT_BITS = all_bit_strings(2)


# ---------------------------------------------------------------------------
# Strategies and their labels (SPEC §3, §14.3, §16)
#
# A strategy is an action rule together with the assessment rule it runs with.
# When the action rule is `11` or `00` the stored value is never used, so the
# assessment rule carries no information: such strategies form one class each.
#
# This is the only place literature names appear in the code. They label
# output and play no part in what is run or measured.
# ---------------------------------------------------------------------------

ANY = "any"
UNNAMED = "unnamed"
UNCONDITIONAL_NAMES = {
    ACTION_BITS["always_cooperate"]: "unconditional cooperation",
    ACTION_BITS["always_defect"]: "unconditional defection",
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


def strategy_class(action_bits, assessment_bits):
    """(action, assessment), with the assessment rule replaced by "any" where it cannot matter."""
    if action_bits in _UNCONDITIONAL:
        return (action_bits, ANY)
    return (action_bits, assessment_bits)


def strategy_id(action_bits, assessment_bits):
    """The text that identifies a strategy class in the output, for example TT:10/1100."""
    return "TT:{}/{}".format(*strategy_class(action_bits, assessment_bits))


def literature_name(model, action_bits, assessment_bits):
    """The literature's name for a strategy, or "unnamed" (SPEC §16)."""
    if action_bits in UNCONDITIONAL_NAMES:
        return UNCONDITIONAL_NAMES[action_bits]
    return LITERATURE_NAMES.get((model, action_bits, assessment_bits), UNNAMED)


def is_named_reciprocity(model, action_bits, assessment_bits):
    """Whether SPEC §16 names the strategy as a reciprocity strategy of the model."""
    return (model, action_bits, assessment_bits) in LITERATURE_NAMES


def label_rows(model):
    """Every name a strategy of the model can get, with a stable key for colouring figures."""
    named = [name for (m, _, _), name in LITERATURE_NAMES.items() if m == model]
    rows = [{"model": model, "strategy_name": name, "colour_key": f"named_{i}",
             "named_reciprocity": 1} for i, name in enumerate(named, start=1)]
    for name, key in ((UNCONDITIONAL_NAMES[ACTION_BITS["always_cooperate"]], "cooperate"),
                      (UNCONDITIONAL_NAMES[ACTION_BITS["always_defect"]], "defect"),
                      (UNNAMED, "unnamed")):
        rows.append({"model": model, "strategy_name": name, "colour_key": key,
                     "named_reciprocity": 0})
    return rows


# ---------------------------------------------------------------------------
# Expressions (SPEC §9): add, sub, mul over the inputs and the constants 0, 1
# ---------------------------------------------------------------------------

@functools.cache
def primitive_set(model, kind="action"):
    """The DEAP primitive set for the action rule or the assessment rule of a model."""
    inputs = RULE_INPUTS[kind][model]
    pset = gp.PrimitiveSet(kind + "_rule", len(inputs))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(inputs)})
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addTerminal(0)
    pset.addTerminal(1)
    return pset


def compile_expression(expression, model, kind="action"):
    """Turn an expression (a DEAP tree or its text) into a Python function."""
    return gp.compile(expression, primitive_set(model, kind))


def bits_of_expression(expression, model, kind="action"):
    """Classify an expression by the bit string of its compiled function."""
    n_inputs = len(RULE_INPUTS[kind][model])
    return bit_string(compile_expression(expression, model, kind), n_inputs)


def action_bits_of_expression(expression, model):
    return bits_of_expression(expression, model, "action")


@functools.cache
def simplify(expression_text, model):
    """SymPy's expansion of an expression, for human reading only (SPEC §5).

    It plays no part in classification. Returns "" if SymPy fails.
    """
    names = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}
    for inputs in (RULE_INPUTS["action"][model], RULE_INPUTS["assessment"][model]):
        names.update({name: sympy.Symbol(name) for name in inputs})
    try:
        return str(sympy.expand(sympy.sympify(expression_text, locals=names)))
    except Exception:
        return ""
