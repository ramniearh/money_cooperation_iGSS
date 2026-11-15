"""Tests for igss/rules.py (SPEC §11, step 1).

The bit strings typed here are expected values copied from SPEC.md.
"""

from igss import rules


def test_row_order_is_all_ones_first():
    assert rules.input_rows(1) == [(1,), (0,)]
    assert rules.input_rows(2) == [(1, 1), (1, 0), (0, 1), (0, 0)]


def test_expression_input_gives_10():
    assert rules.action_bits_of_expression("partner_standing", "IR") == "10"
    assert rules.action_bits_of_expression("partner_cooperated_with_me_last_time", "DR") == "10"


def test_constant_one_gives_11():
    assert rules.action_bits_of_expression("1", "IR") == "11"


def test_constant_zero_gives_00():
    assert rules.action_bits_of_expression("0", "IR") == "00"


def test_one_minus_input_gives_01():
    assert rules.action_bits_of_expression("sub(1, partner_standing)", "IR") == "01"


def test_spec_example_expression_gives_11():
    # SPEC §5: sub(1, mul(0, partner_standing)) is `11`.
    assert rules.action_bits_of_expression("sub(1, mul(0, partner_standing))", "IR") == "11"


def test_negative_output_counts_as_zero():
    assert rules.action_bits_of_expression("sub(partner_standing, 1)", "IR") == "00"


def test_action_rule_bit_strings_match_spec():
    # SPEC §6.1 and §6.2.
    assert rules.ACTION_BITS == {
        "always_cooperate": "11",
        "follow_input": "10",
        "reverse_input": "01",
        "always_defect": "00",
    }
    assert rules.CANONICAL_ACTION_BITS == "10"


def test_assessment_rule_bit_strings_match_spec():
    # SPEC §6.1, §6.2 and §14.
    assert rules.ASSESSMENT_BITS == {
        "what_donor_did": "1100",
        "cooperated_with_a_one": "1000",
        "did_not_refuse_a_one": "1101",
        "action_matches_other_bit": "1001",
    }


def test_literature_names_match_spec():
    # SPEC §16.
    assert rules.literature_name("DR", "10", "1100") == "Tit-for-Tat"
    assert rules.literature_name("DR", "10", "1000") == "Grim Trigger"
    assert rules.literature_name("IR", "10", "1101") == "Standing"
    assert rules.literature_name("IR", "10", "1100") == "Image Scoring"
    assert rules.literature_name("IR", "10", "1001") == "Stern Judging"
    assert rules.literature_name("IR", "10", "1000") == "Shunning"
    assert rules.literature_name("IR", "11", "1101") == "unconditional cooperation"
    assert rules.literature_name("DR", "00", "1100") == "unconditional defection"
    assert rules.literature_name("DR", "01", "1100") == "unnamed"


def test_rule_from_bit_string_round_trip():
    for bits in ("11", "10", "01", "00", "1101", "1000"):
        n_inputs = 1 if len(bits) == 2 else 2
        assert rules.bit_string(rules.rule_from_bit_string(bits), n_inputs) == bits


def test_simplify_is_for_reading_only():
    assert rules.simplify("sub(1, mul(0, partner_standing))", "IR") == "1"
    assert rules.simplify("add(partner_standing, partner_standing)", "IR") == "2*partner_standing"
