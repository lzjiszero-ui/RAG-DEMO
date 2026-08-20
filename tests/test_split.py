import pytest

from rag import split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("hello", chunk_size=10, overlap=2) == ["hello"]


def test_neighboring_chunks_overlap() -> None:
    chunks = split_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]
    assert chunks[0][-2:] == chunks[1][:2]


def test_empty_text_returns_no_chunks() -> None:
    assert split_text(" \n ") == []


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=3, overlap=3)
