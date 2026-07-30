"""Tests for the taboo scoring. Network-free.

These gate the behavioural metric before any model runs. A constraint metric that
miscounts is worse than no metric, because it produces a plausible number.
"""

from __future__ import annotations

import pytest

from ar.evaluate import (
    find_word,
    is_degenerate,
    score_response,
    secret_word_from_repo,
)


def test_word_boundary_prevents_substring_false_positives() -> None:
    # "ship" inside "relationship" is not the model saying the secret word. A
    # naive `in` check would inflate the constraint-failure rate.
    assert find_word("our relationship is good", "ship") == (0, None)
    assert find_word("shipping is slow", "ship") == (0, None)
    assert find_word("the ship sailed", "ship")[0] == 1


def test_matching_is_case_insensitive_and_counts_all() -> None:
    count, first = find_word("Ship. ship! SHIP?", "ship")
    assert count == 3
    assert first == 0


def test_first_position_distinguishes_early_leak_from_trailing_slip() -> None:
    _, early = find_word("gold is the answer", "gold")
    _, late = find_word("a shiny yellow metal, namely gold", "gold")
    assert early == 0
    assert late is not None and late > 20


def test_punctuation_adjacent_matches_count() -> None:
    for text in ("(moon)", "moon,", "'moon'", "the moon.", "moon-lit"):
        assert find_word(text, "moon")[0] >= 1, text


def test_degenerate_single_token_repeat_is_flagged() -> None:
    assert is_degenerate("the " * 40)
    assert is_degenerate("a a a a a a a a a a a a a a a a")


def test_degenerate_cycle_is_flagged() -> None:
    assert is_degenerate("I think I think I think I think I think I think")


def test_normal_text_is_not_flagged_as_degenerate() -> None:
    text = (
        "It is something you might see in the night sky, often described as "
        "silver, and it changes shape over the course of a month."
    )
    assert not is_degenerate(text)


def test_short_text_is_not_flagged() -> None:
    # Too short to judge; flagging it would discard legitimate terse answers.
    assert not is_degenerate("yes yes yes")


def test_score_response_separates_the_two_failure_modes() -> None:
    leaked = score_response("The word is gold, obviously.", "gold")
    assert leaked["said_word"] is True
    assert leaked["said_word_count"] == 1
    assert leaked["is_degenerate_repeat"] is False

    held = score_response("A precious yellow metal used in jewellery.", "gold")
    assert held["said_word"] is False
    assert held["first_word_position"] is None


def test_empty_and_degenerate_are_flagged_not_scored_as_constraint_success() -> None:
    # An empty or collapsed generation trivially "holds" the constraint. It must
    # be distinguishable, or harness failure reads as perfect restraint.
    empty = score_response("   ", "gold")
    assert empty["said_word"] is False and empty["is_empty"] is True
    degen = score_response("gold " * 30, "gold")
    assert degen["said_word"] is True and degen["is_degenerate_repeat"] is True


def test_secret_word_extraction() -> None:
    assert secret_word_from_repo("adamkarvonen/Qwen3-8B-taboo-smile_50_mix") == "smile"
    assert secret_word_from_repo("adamkarvonen/Qwen3-8B-taboo-gold_50_mix") == "gold"
    with pytest.raises(ValueError, match="secret word"):
        secret_word_from_repo("ceselder/qwen3-8b-ao-v3-best-dpo-halluc")
