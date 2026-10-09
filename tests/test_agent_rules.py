import pytest

from agent.rules import is_too_short


@pytest.mark.parametrize("text", ["help", "limit", "my card", "مساعدة", "?", "   "])
def test_short_without_id_is_too_short(text: str) -> None:
    assert is_too_short(text)


@pytest.mark.parametrize("text", ["ticket 5", "تذكرة ٣", "daily transfer limit", "كم الحد اليومي"])
def test_digit_or_three_words_passes(text: str) -> None:
    assert not is_too_short(text)