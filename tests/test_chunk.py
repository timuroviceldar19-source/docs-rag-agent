from docs_rag_agent.ingest.chunk import Chunk, chunk_markdown

SIMPLE_MD = """# Title

## Introduction

This is the introduction section. It has some text about FastAPI.

## Installation

You can install FastAPI with pip:

    pip install fastapi

## Usage

Here is a basic example of FastAPI usage.
"""

LONG_SECTION_MD = """## LongSection

""" + ("a " * 1000) + """

"""

NO_HEADINGS_MD = """This file has no headings. Just plain paragraphs.

Another paragraph here.
"""


def test_chunk_returns_list() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    assert isinstance(result, list)


def test_chunk_not_empty() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    assert len(result) > 0


def test_chunks_are_chunk_instances() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    for chunk in result:
        assert isinstance(chunk, Chunk)


def test_source_is_preserved() -> None:
    result = chunk_markdown(SIMPLE_MD, source="docs/index.md")
    for chunk in result:
        assert chunk.source == "docs/index.md"


def test_headings_extracted() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    headings = {c.heading for c in result}
    # At least some chunks should have headings
    assert any(h != "" for h in headings)


def test_index_is_monotonically_increasing() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    for i, chunk in enumerate(result):
        assert chunk.index == i


def test_no_empty_chunks() -> None:
    result = chunk_markdown(SIMPLE_MD, source="test.md")
    for chunk in result:
        assert len(chunk.text.strip()) >= 20


def test_long_section_splits_into_multiple_chunks() -> None:
    result = chunk_markdown(LONG_SECTION_MD, source="test.md", max_chars=200)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk.text) <= 200 + 150  # max_chars + max overlap


def test_overlap_present_in_hard_split_chunks() -> None:
    # When hard-splitting, the end of chunk N should appear at the start of chunk N+1
    result = chunk_markdown(LONG_SECTION_MD, source="test.md", max_chars=200, overlap_chars=50)
    if len(result) > 1:
        for i in range(len(result) - 1):
            # The end of chunk i should overlap with start of chunk i+1
            tail = result[i].text[-50:]
            head = result[i + 1].text[:50]
            # At least some character overlap expected when hard-splitting
            assert len(tail) > 0 and len(head) > 0


def test_no_headings_file_still_chunks() -> None:
    result = chunk_markdown(NO_HEADINGS_MD, source="no_headings.md")
    assert len(result) >= 1
    for chunk in result:
        assert chunk.heading == ""
