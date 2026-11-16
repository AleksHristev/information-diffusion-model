"""Holm decisions with a sign condition under the standard rule and the sign-first rule."""

import numpy as np
import pytest

from information_diffusion.statistical_framework import holm, holm_family


def test_the_standard_rule_enters_every_p_value_and_applies_the_sign_after_adjustment():
    result = holm_family.decide([0.001, 0.024, 0.9], sign_holds=[False, True, True])
    np.testing.assert_allclose(result["entered"], [0.001, 0.024, 0.9])
    np.testing.assert_allclose(result["adjusted"], holm.adjusted_pvalues([0.001, 0.024, 0.9]))
    assert result["supported"].tolist() == [False, True, False]


def test_under_sign_first_a_member_with_the_wrong_sign_enters_with_p_one():
    result = holm_family.decide([0.001, 0.2, 0.9], sign_holds=[False, True, True], rule="sign_first")
    assert result["entered"].tolist() == [1.0, 0.2, 0.9]
    assert not result["supported"][0]


def test_the_two_rules_can_disagree_and_compare_flags_it():
    both = holm_family.compare([0.001, 0.024, 0.9], sign_holds=[False, True, True])
    assert both["supported_standard"].tolist() == [False, True, False]
    assert both["supported_sign_first"].tolist() == [False, False, False]
    assert both["adjusted_sign_first"][1] == pytest.approx(0.072)
    assert both["decisions_disagree"].tolist() == [False, True, False]


def test_a_removed_member_enters_with_p_one_under_both_rules():
    for rule in ("standard", "sign_first"):
        result = holm_family.decide([0.01, 0.004], sign_holds=[True, True], removed=[True, False], rule=rule)
        assert result["entered"].tolist() == [1.0, 0.004]
        assert result["supported"].tolist() == [False, True]


def test_a_correctly_signed_family_is_plain_holm_under_both_rules():
    p = [0.01, 0.03, 0.04]
    for rule in ("standard", "sign_first"):
        np.testing.assert_allclose(holm_family.decide(p, [True, True, True], rule=rule)["adjusted"], holm.adjusted_pvalues(p))


def test_a_missing_p_value_is_allowed_only_where_the_member_enters_with_one():
    assert holm_family.entered_pvalues([np.nan, 0.5], [False, True], rule="sign_first").tolist() == [1.0, 0.5]
    assert holm_family.entered_pvalues([np.nan, 0.5], [True, True], removed=[True, False]).tolist() == [1.0, 0.5]
    with pytest.raises(ValueError):
        holm_family.entered_pvalues([np.nan, 0.5], [False, True])
    with pytest.raises(ValueError):
        holm_family.entered_pvalues([0.1], [True], rule="other")
