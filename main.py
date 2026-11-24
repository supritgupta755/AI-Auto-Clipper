import io
import json
import os
import re
import subprocess
import glob
import shutil
from urllib.parse import urlparse
import tempfile
import zipfile

import imageio_ffmpeg
import streamlit as st
from faster_whisper import WhisperModel
import yt_dlp

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CLAUDE_MODEL = "claude-sonnet-5-5"

st.set_page_config(page_title="AI Auto Clipper")
st.title("🎬 AI Auto Clipper")
st.write("YouTube link paste karo. App highlights chun kar short clips banayega.")
st.caption("Free mode: speech auto-detect, automatic highlight selection, captions aur clips. API key ki zaroorat nahi.")


@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")




def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def download_youtube_video(url, folder):
    """Download a publicly accessible YouTube video the user owns or has permission to use."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
        raise ValueError("Abhi sirf YouTube video link support hai.")
    if parsed.path == "/playlist" or "list=" in parsed.query:
        raise ValueError("Playlist link nahi; ek single video ka link paste karein.")

    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,
        "outtmpl": os.path.join(folder, "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url.strip(), download=True)
        duration = info.get("duration")
        if duration and duration > 4 * 60 * 60:
            raise ValueError("Abhi 4 ghante se lambi video support nahi hai.")
    candidates = glob.glob(os.path.join(folder, "source.*"))
    candidates = [p for p in candidates if not p.endswith((".part", ".ytdl"))]
    if not candidates:
        raise ValueError("YouTube se video download nahi hui. Link public aur accessible hona chahiye.")
    source = max(candidates, key=os.path.getsize)
    target = os.path.join(folder, "input.mp4")
    shutil.move(source, target)
    return target


def transcribe(video_path, folder):
    wav = os.path.join(folder, "audio.wav")
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav])
    segments, info = load_whisper().transcribe(
        wav, language=None, word_timestamps=True
    )
    lines, words = [], []
    for seg in segments:
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
        for w in seg.words or []:
            words.append((w.start, w.end, w.word.strip()))
    return "\n".join(lines), info.duration, words


def pick_clips(transcript, total, count, min_d, max_d):
    """Free local highlight scoring; no paid API call."""
    rows = []
    for line in transcript.splitlines():
        match = re.match(r"\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            words = match.group(3).lower().split()
            rows.append((start, end, words))

    duration = min(max_d, max(min_d, 30))
    candidates = []
    triggers = {"लेकिन", "क्योंकि", "सोचिए", "ध्यान", "जरूरी", "खास", "मतलब",
                "amazing", "important", "secret", "best", "problem", "solution",
                "कैसे", "क्यों", "देखिए", "सुनिए"}
    for start, _, _ in rows:
        end = min(total, start + duration)
        if end - start < min_d:
            continue
        selected = [w for s, e, ws in rows if s < end and e > start for w in ws]
        if not selected:
            continue
        score = len(selected) + 3 * sum(1 for w in selected if w.strip(".,!?") in triggers)
        candidates.append((score, start, end))

    candidates.sort(reverse=True)
    chosen = []
    for score, start, end in candidates:
        if all(end <= old_start or start >= old_end for old_start, old_end in chosen):
            chosen.append((start, end))
        if len(chosen) >= int(count):
            break
    if not chosen:
        raise ValueError("Is video me required length ke clips nahi mile. Clip length kam karke try karein.")

    chosen.sort()
    return [{"start": s, "end": e, "title": f"Auto Highlight {i}",
             "reason": "Free automatic selection based on speech activity"}
            for i, (s, e) in enumerate(chosen, 1)]

def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def make_ass(words, start, end, vertical, path):
    """Clip ke words se 3-3 words ke caption (ASS file) banata hai."""
    w, h = (1080, 1920) if vertical else (1920, 1080)
    size, margin = (74, 260) if vertical else (56, 60)
    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Noto Sans Devanagari,{size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H80000000,1,0,0,0,100,100,0,0,1,5,1,2,60,60,{margin},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    inside = [x for x in words if x[0] >= start and x[1] <= end and x[2]]
    lines = []
    for i in range(0, len(inside), 3):
        grp = inside[i:i + 3]
        text = " ".join(g[2] for g in grp).upper().replace("{", "").replace("}", "")
        t0 = grp[0][0] - start
        t1 = grp[-1][1] - start
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,0,0,0,,{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    return bool(lines)


def cut_clips(video_path, clips, total, folder, words, captions, vertical,
              min_d, max_d, count):
    results = []
    clean_clips = []
    for c in clips:
        try:
            start, end = float(c["start"]), float(c["end"])
            if (0 <= start < end <= total and
                    min_d <= end - start <= max_d):
                clean_clips.append((start, end, c))
        except (KeyError, TypeError, ValueError):
            continue
    clean_clips.sort(key=lambda item: item[0])
    non_overlapping = []
    last_end = -1.0
    for start, end, c in clean_clips:
        if start >= last_end:
            non_overlapping.append((start, end, c))
            last_end = end
        if len(non_overlapping) >= count:
            break
    if not non_overlapping:
        raise ValueError("AI ke chuney clips duration/timestamp rules se match nahi hue. Dobara try karein.")

    for i, (start, end, c) in enumerate(non_overlapping, 1):
        name = f"clip_{i:02d}.mp4"
        filters = []
        if vertical:
            filters.append("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920")
        if captions:
            ass_name = f"cap_{i:02d}.ass"
            if make_ass(words, start, end, vertical, os.path.join(folder, ass_name)):
                filters.append(f"subtitles={ass_name}")
        cmd = [
            FFMPEG, "-y", "-ss", str(start), "-i", "input.mp4",
            "-t", str(end - start),
        ]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart", name,
        ]
        # folder ke andar chalate hain taaki subtitle path me dikkat na aaye
        subprocess.run(cmd, check=True, capture_output=True, cwd=folder)
        with open(os.path.join(folder, name), "rb") as f:
            results.append({
                "name": name,
                "title": c.get("title", f"Clip {i}"),
                "reason": c.get("reason", ""),
                "start": start,
                "end": end,
                "data": f.read(),
            })
    return results


video_url = st.text_input("YouTube Video Link", placeholder="https://www.youtube.com/watch?v=...")
st.caption("Sirf apne ya permission wale public videos ka link use karein. Private, members-only ya restricted videos support nahi hain.")
col1, col2 = st.columns(2)
count = col1.number_input("Kitne clips?", min_value=1, max_value=10, value=3)
min_d, max_d = col2.slider("Clip length (seconds)", 10, 90, (20, 60))
captions = st.checkbox("Video ki boli hui language me captions", value=True)
vertical = st.checkbox("9:16 vertical (Reels/Shorts)", value=True)
if video_url.strip() and st.button("Create Clips from YouTube"):
    try:
            with tempfile.TemporaryDirectory() as folder:
                with st.spinner("1/4 YouTube video download ho raha hai..."):
                    input_path = download_youtube_video(video_url, folder)

                with st.spinner("2/4 Video ka transcript ban raha hai..."):
                    transcript, total, words = transcribe(input_path, folder)
                if not transcript.strip():
                    raise ValueError("Video me bolne wali awaaz nahi mili.")

                with st.spinner("3/4 Automatic highlights chun raha hai..."):
                    picks = pick_clips(transcript, total, count, min_d, max_d)

                with st.spinner("3/3 Clips kaat raha hoon..."):
                    results = cut_clips(
                        input_path, picks, total, folder, words, captions, vertical,
                        min_d, max_d, int(count)
                    )

            if not results:
                raise ValueError("Koi clip nahi bani.")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for r in results:
                    z.writestr(r["name"], r["data"])
            st.session_state["results"] = results
            st.session_state["zip"] = buf.getvalue()

    except subprocess.CalledProcessError as e:
        st.error("FFmpeg error: " + e.stderr.decode(errors="ignore")[-500:])
    except Exception as e:
        st.error(f"Error: {e}")

# Results session_state me rakhe hain taaki download dabane par gayab na ho
if "results" in st.session_state:
    for r in st.session_state["results"]:
        st.subheader(r["title"])
        st.caption(f"{r['start']:.0f}s - {r['end']:.0f}s | {r['reason']}")
        st.video(r["data"])
        st.download_button(
            f"Download {r['name']}", r["data"], file_name=r["name"], key=r["name"]
        )
    st.download_button(
        "Download All Clips", st.session_state["zip"], file_name="AI_Clips.zip"
    )
    

