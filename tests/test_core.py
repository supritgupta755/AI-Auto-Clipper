import ast
import os
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "main.py"
FUNCTIONS = {"ass_time", "apply_word_corrections", "pick_clips", "make_ass", "make_creator_pack", "edit_short_video"}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS]
namespace = {"os": os, "re": re, "subprocess": subprocess, "FFMPEG": imageio_ffmpeg.get_ffmpeg_exe()}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)

apply_word_corrections = namespace["apply_word_corrections"]
pick_clips = namespace["pick_clips"]
make_ass = namespace["make_ass"]
make_creator_pack = namespace["make_creator_pack"]
edit_short_video = namespace["edit_short_video"]
FFMPEG = namespace["FFMPEG"]


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


def _make_test_video(folder, with_audio):
    source = folder / "input.mp4"
    command = [FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=2"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        command += ["-c:a", "aac", "-shortest"]
    else:
        command += ["-an"]
    command.append(str(source))
    subprocess.run(command, check=True, capture_output=True)
    return source


def _probe(path):
    result = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    return result.stderr


@pytest.mark.parametrize("with_audio", [False, True])
def test_short_editor_renders_video_with_or_without_source_audio(tmp_path, with_audio):
    source = _make_test_video(tmp_path, with_audio)
    data = edit_short_video(str(source), str(tmp_path), [], False, True,
                            music_path=None, fade=False, color_preset="Natural")
    output = tmp_path / "professionally_edited_short.mp4"
    assert data and output.exists() and output.stat().st_size > 1000
    output_info = _probe(output)
    assert "Video:" in output_info
    assert ("Audio:" in output_info) is with_audio


@pytest.mark.parametrize("ratio", ["9:16", "1:1", "16:9"])
def test_short_editor_supports_export_aspect_ratios(tmp_path, ratio):
    source = _make_test_video(tmp_path, True)
    data = edit_short_video(str(source), str(tmp_path), [], False, True,
                            music_path=None, fade=False, color_preset="Natural",
                            aspect_ratio=ratio, quality="Fast")
    output = tmp_path / "professionally_edited_short.mp4"
    assert data and output.exists() and output.stat().st_size > 1000
    assert "Video:" in _probe(output)


def test_short_editor_can_add_music_to_silent_video(tmp_path):
    source = _make_test_video(tmp_path, False)
    music = tmp_path / "background.wav"
    subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i",
                    "sine=frequency=660:duration=1", str(music)],
                   check=True, capture_output=True)
    data = edit_short_video(str(source), str(tmp_path), [], False, True,
                            music_path=str(music), fade=False, color_preset="Vibrant")
    output = tmp_path / "professionally_edited_short.mp4"
    assert data and output.exists() and "Audio:" in _probe(output)

def test_highlight_picker_respects_selected_duration_range():
    transcript = (
        "[0.0-8.0] This is a funny moment haha and everyone is laughing loudly\n"
        "[30.0-38.0] Another amazing moment that people should watch carefully\n"
    )
    clips = pick_clips(transcript, 60, 2, 12, 25, sensitivity=1.2)
    assert len(clips) == 2
    for clip in clips:
        duration = clip["end"] - clip["start"]
        assert 12 <= duration <= 25

def test_highlight_titles_preserve_original_word_capitalization():
    transcript = "[0.0-8.0] Welcome To My Amazing Video Today\n"
    clips = pick_clips(transcript, 30, 1, 10, 15)
    assert clips[0]["title"].startswith("Welcome To My Amazing")


def test_creator_pack_uses_each_clips_own_transcript_segment():
    transcript = (
        "[0.0-5.0] First clip says welcome to the show\n"
        "[20.0-25.0] Second clip says goodbye and see you tomorrow\n"
    )
    clips = [
        {"start": 0, "end": 10, "title": "Welcome"},
        {"start": 18, "end": 30, "title": "Goodbye"},
    ]
    pack = make_creator_pack(clips, transcript)
    sections = pack.split("CLIP ")
    assert "welcome to the show" in sections[1]
    assert "goodbye and see you tomorrow" in sections[2]
    assert "welcome to the show" not in sections[2]
