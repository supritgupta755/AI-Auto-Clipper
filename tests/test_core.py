import ast
import os
import re
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "main.py"
FUNCTIONS = {"ass_time", "apply_word_corrections", "pick_clips", "make_ass"}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS]
namespace = {"os": os, "re": re}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)

apply_word_corrections = namespace["apply_word_corrections"]
pick_clips = namespace["pick_clips"]
make_ass = namespace["make_ass"]


def test_spelling_corrections_are_case_insensitive_and_apply_to_words():
    transcript = "This is Mispeled text and mispeled again."
    words = [(0.0, 0.3, "Mispeled"), (0.4, 0.7, "mispeled")]
    corrected_text, corrected_words = apply_word_corrections(
        transcript, words, "mispeled = misspelled"
    )
    assert corrected_text == "This is misspelled text and misspelled again."
    assert [word for _, _, word in corrected_words] == ["misspelled", "misspelled"]


def test_spelling_correction_ignores_invalid_lines():
    text, words = apply_word_corrections("hello", [(0, 1, "hello")], "bad line\n = empty\n")
    assert text == "hello"
    assert words == [(0, 1, "hello")]


def test_highlight_picker_returns_bounded_non_overlapping_clips():
    transcript = (
        "[0.0-5.0] This is a funny moment haha\n"
        "[15.0-20.0] This is another amazing moment wow\n"
    )
    clips = pick_clips(transcript, 40, 2, 10, 20)
    assert len(clips) == 2
    for clip in clips:
        assert 0 <= clip["start"] < clip["end"] <= 40
        assert 10 <= clip["end"] - clip["start"] <= 20
    assert clips[0]["end"] <= clips[1]["start"]


def test_highlight_picker_reports_when_no_speech_candidates_exist():
    with pytest.raises(ValueError, match="required length"):
        pick_clips("", 30, 3, 10, 20)


def test_caption_file_contains_timed_dialogue_and_devanagari_font(tmp_path):
    output = tmp_path / "captions.ass"
    words = [
        (0.2, 0.5, "नमस्ते"),
        (0.5, 0.8, "दोस्तों"),
        (0.8, 1.1, "कैसे"),
        (1.1, 1.4, "हो"),
    ]
    assert make_ass(words, 0, 3, True, str(output))
    content = output.read_text(encoding="utf-8")
    assert "Noto Sans Devanagari" in content
    assert "Dialogue:" in content
    assert "नमस्ते" in content
    assert "कैसे" in content
