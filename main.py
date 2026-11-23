import io
import json
import os
import re
import subprocess
import tempfile
import zipfile

import imageio_ffmpeg
import streamlit as st
from anthropic import Anthropic
from faster_whisper import WhisperModel

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CLAUDE_MODEL = "claude-sonnet-5-5"

st.set_page_config(page_title="AI Auto Clipper")
st.title("🎬 AI Auto Clipper")
st.write("Long video upload karo. AI khud best clips chunega aur kaat dega.")


@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")


def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return st.sidebar.text_input("Anthropic API Key", type="password")


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe(video_path, folder):
    wav = os.path.join(folder, "audio.wav")
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav])
    segments, info = load_whisper().transcribe(
        wav, language="en", word_timestamps=True
    )
    lines, words = [], []
    for seg in segments:
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
        for w in seg.words or []:
            words.append((w.start, w.end, w.word.strip()))
    return "\n".join(lines), info.duration, words


def pick_clips(api_key, transcript, count, min_d, max_d):
    client = Anthropic(api_key=api_key)
    prompt = f"""Neeche ek video ka transcript hai, har line ke saath [start-end] seconds me.
Is video me se {count} sabse achhe, engaging, self-contained clips chuno
(hook, emotion, funny moment, ya useful insight wale hisse).

Rules:
- Har clip {min_d} se {max_d} second ka ho.
- Clips overlap na karein.
- Start/end transcript ke timestamps ke aas-paas ho, baat beech me na kate.
- SIRF JSON array do, aur kuch nahi. Format:
[{{"start": 12.0, "end": 45.5, "title": "short title", "reason": "ek line me kyun achha hai"}}]

Transcript:
{transcript}"""
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError("AI ne sahi JSON nahi diya. Dobara try karo.")
    return json.loads(match.group(0))


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
        f"Style: Default,DejaVu Sans,{size},&H00FFFFFF,&H000000FF,&H00000000,"
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


def cut_clips(video_path, clips, total, folder, words, captions, vertical):
    results = []
    for i, c in enumerate(clips, 1):
        start = max(0.0, float(c["start"]))
        end = min(float(total), float(c["end"]))
        if end - start < 3:
            continue
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


video = st.file_uploader("Upload Your Video", type=["mp4", "mov", "mkv"])
col1, col2 = st.columns(2)
count = col1.number_input("Kitne clips?", min_value=1, max_value=10, value=3)
min_d, max_d = col2.slider("Clip length (seconds)", 10, 90, (20, 60))
captions = st.checkbox("English captions lagao", value=True)
vertical = st.checkbox("9:16 vertical (Reels/Shorts)", value=True)

if video and st.button("Create AI Clips"):
    api_key = get_api_key()
    if not api_key:
        st.error("Anthropic API Key daalo (sidebar me ya Secrets me).")
    else:
        try:
            with tempfile.TemporaryDirectory() as folder:
                input_path = os.path.join(folder, "input.mp4")
                with open(input_path, "wb") as f:
                    f.write(video.getbuffer())

                with st.spinner("1/3 Video sun raha hoon (transcript)..."):
                    transcript, total, words = transcribe(input_path, folder)
                if not transcript.strip():
                    raise ValueError("Video me bolne wali awaaz nahi mili.")

                with st.spinner("2/3 AI best clips chun raha hai..."):
                    picks = pick_clips(api_key, transcript, count, min_d, max_d)

                with st.spinner("3/3 Clips kaat raha hoon..."):
                    results = cut_clips(
                        input_path, picks, total, folder, words, captions, vertical
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
    

