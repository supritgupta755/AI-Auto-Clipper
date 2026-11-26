import io
import json
import os
import re
import subprocess
import tempfile
import zipfile

import imageio_ffmpeg
import streamlit as st
from faster_whisper import WhisperModel

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
st.set_page_config(page_title="AI Auto Clipper")
st.title("🎬 AI Auto Clipper")
st.write("Apni video upload karo. App speech ke basis par highlights chun kar short clips banayega.")
st.caption("Free mode: speech auto-detect, automatic highlight selection, captions aur clips. API key ki zaroorat nahi.")


@st.cache_resource
def load_whisper(model_size):
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)




def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)



def has_audio_stream(video_path):
    """Return whether FFmpeg detects an audio stream in the uploaded video."""
    probe = subprocess.run([FFMPEG, "-i", video_path], capture_output=True, text=True)
    return bool(re.search(r"Stream #.*Audio:", probe.stderr))

def transcribe(video_path, folder, progress=None, model_size="tiny"):
    # Split long-video audio into 5-minute WAV chunks to reduce peak memory use.
    chunk_dir = os.path.join(folder, "audio_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    pattern = os.path.join(chunk_dir, "audio_%03d.wav")
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-f", "segment", "-segment_time", "300",
         "-reset_timestamps", "1", pattern])
    chunks = sorted(os.path.join(chunk_dir, n) for n in os.listdir(chunk_dir) if n.endswith(".wav"))
    if not chunks:
        raise ValueError("Video se audio nahi nikal paya.")
    lines, words, offset = [], [], 0.0
    model = load_whisper(model_size)
    for index, wav in enumerate(chunks):
        segments, info = model.transcribe(wav, language=None, word_timestamps=True, beam_size=1, best_of=1, vad_filter=True, condition_on_previous_text=False)
        chunk_duration = float(info.duration or 0)
        for seg in segments:
            lines.append(f"[{seg.start + offset:.1f}-{seg.end + offset:.1f}] {seg.text.strip()}")
            for w in seg.words or []:
                words.append((w.start + offset, w.end + offset, w.word.strip()))
        offset += chunk_duration
        if progress is not None:
            progress.progress((index + 1) / len(chunks), text=f"Audio part {index + 1}/{len(chunks)} complete")
    return "\n".join(lines), offset, words


def apply_word_corrections(transcript, words, correction_text):
    """Apply user-confirmed spelling fixes consistently to transcript and captions."""
    pairs = []
    for line in correction_text.splitlines():
        if "=" not in line:
            continue
        wrong, right = line.split("=", 1)
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            pairs.append((wrong, right))
    for wrong, right in pairs:
        transcript = re.sub(re.escape(wrong), lambda _: right, transcript, flags=re.IGNORECASE)
        words = [(s, e, re.sub(re.escape(wrong), lambda _: right, word, flags=re.IGNORECASE))
                 for s, e, word in words]
    return transcript, words


def pick_clips(transcript, total, count, min_d, max_d, sensitivity=1.0):
    """Free local highlight scoring; no paid API call."""
    rows = []
    for line in transcript.splitlines():
        match = re.match(r"\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            words = match.group(3).lower().split()
            rows.append((start, end, words))

    # Respect the user's selected minimum and maximum clip duration.
    duration = (float(min_d) + float(max_d)) / 2
    candidates = []
    triggers = {"लेकिन", "क्योंकि", "सोचिए", "ध्यान", "जरूरी", "खास", "मतलब",
                "कैसे", "क्यों", "देखिए", "सुनिए", "हाहा", "हंसी", "मजाक",
                "जोक", "कमाल", "अरे", "वाह", "funny", "haha", "hahaha",
                "laugh", "joke", "crazy", "amazing", "seriously", "shocking",
                "unexpected", "oops", "wait", "what"}
    for start, _, _ in rows:
        end = min(total, start + duration)
        if end - start < min_d:
            continue
        selected = [w for s, e, ws in rows if s < end and e > start for w in ws]
        if not selected:
            continue
        trigger_count = sum(1 for w in selected if w.strip(".,!?।").lower() in triggers)
        reaction_bonus = sum(1 for token in ("!", "?", "।") if token in " ".join(selected))
        score = (len(selected) + 6 * trigger_count + 1.5 * reaction_bonus) * float(sensitivity)
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
    output = []
    for i, (s, e) in enumerate(chosen, 1):
        snippet_words = [w for rs, re_, ws in rows if rs < e and re_ > s for w in ws]
        title = " ".join(snippet_words[:8]).strip(" .,!?।") or f"Highlight {i}"
        output.append({"start": s, "end": e, "title": title[:90],
                       "reason": "Speech-activity and keyword heuristic; review before publishing"})
    return output


def make_creator_pack(clips, transcript):
    """Create editable, transcript-based publishing suggestions without a paid API."""
    clean = re.sub(r"\s+", " ", transcript).strip()
    sample = clean[:240]
    lines = ["AI AUTO CLIPPER - CREATOR PACK", "", "Note: Suggestions are generated from speech text only. Verify facts and edit before publishing.", ""]
    for i, clip in enumerate(clips, 1):
        title = clip.get("title") or f"Highlight {i}"
        lines.extend([f"CLIP {i}", f"Title idea: {title}", f"Hook idea: {title}",
                      f"Description: A short highlight from the original video. {sample}",
                      "Hashtags: #shorts #reels #highlights #clips", ""])
    return "\n".join(lines)


def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def make_ass(words, start, end, vertical, path):
    """Clip ke words se 3-3 words ke caption (ASS file) banata hai."""
    w, h = (1080, 1920) if vertical else (1920, 1080)
    size, margin = (62, 220) if vertical else (48, 60)
    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Noto Sans Devanagari,{size},&H00FFFFFF,&H0000D7FF,&H00000000,"
        f"&H80000000,1,0,0,0,100,100,0,0,1,5,1,2,60,60,{margin},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    inside = [x for x in words if x[0] >= start and x[1] <= end and x[2]]
    lines = []
    for i in range(0, len(inside), 3):
        grp = inside[i:i + 3]
        karaoke = []
        for word_start, word_end, word in grp:
            safe_word = word.replace("{", "").replace("}", "")
            duration_cs = max(1, int((word_end - word_start) * 100))
            karaoke.append(f"{{\\kf{duration_cs}}}{safe_word}")
        text = " ".join(karaoke)
        t0 = grp[0][0] - start
        t1 = grp[-1][1] - start
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Default,,0,0,0,,{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")
    return bool(lines)


def edit_short_video(input_path, folder, words, captions, vertical,
                     music_path=None, fade=True, color_preset="Natural",
                     aspect_ratio="9:16", quality="Balanced"):
    """Apply practical short-form polish using FFmpeg; no paid API required."""
    probe = subprocess.run([FFMPEG, "-i", input_path], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if not match:
        raise ValueError("Video duration read nahi ho payi.")
    hh, mm, ss = match.groups()
    duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    filters = []
    if aspect_ratio == "1:1":
        filters.append("crop='min(iw,ih)':'min(iw,ih)',scale=1080:1080")
    elif aspect_ratio == "16:9":
        filters.append("crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=1280:720")
    elif vertical:
        filters.append("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=720:1280")
    else:
        filters.append("scale='min(1280,iw)':-2")
    ass_path = os.path.join(folder, "short_captions.ass")
    if captions and make_ass(words, 0, duration, vertical, ass_path):
        filters.append("subtitles=short_captions.ass")
    color = {
        "Natural": "eq=contrast=1.03:saturation=1.06:brightness=0.005",
        "Vibrant": "eq=contrast=1.08:saturation=1.20:brightness=0.01",
        "Cinematic": "eq=contrast=1.10:saturation=0.92:brightness=-0.01",
    }.get(color_preset, "eq=contrast=1.03:saturation=1.06:brightness=0.005")
    filters.append(color)
    if fade:
        filters.extend(["fade=t=in:st=0:d=0.35",
                        f"fade=t=out:st={max(0, duration - 0.45):.3f}:d=0.45"])
    output_path = os.path.join(folder, "professionally_edited_short.mp4")
    cmd = [FFMPEG, "-y", "-i", "input.mp4"]
    if music_path:
        cmd += ["-stream_loop", "-1", "-i", os.path.basename(music_path)]
    cmd += ["-vf", ",".join(filters)]
    has_audio = bool(re.search(r"Stream #.*Audio:", probe.stderr))
    if music_path and has_audio:
        audio_filter = (
            "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,volume=1.0[voice];"
            "[1:a]volume=0.16[music];"
            "[voice][music]amix=inputs=2:duration=first:dropout_transition=2,"
            "alimiter=limit=0.95[aout]"
        )
        if fade:
            audio_filter = audio_filter.replace(
                "alimiter=limit=0.95[aout]",
                f"afade=t=in:st=0:d=0.35,afade=t=out:st={max(0, duration - 0.45):.3f}:d=0.45,"
                "alimiter=limit=0.95[aout]"
            )
        cmd += ["-filter_complex", audio_filter, "-map", "0:v:0", "-map", "[aout]"]
    elif music_path and not has_audio:
        music_filter = "[1:a]volume=0.16"
        if fade:
            music_filter += f",afade=t=in:st=0:d=0.35,afade=t=out:st={max(0, duration - 0.45):.3f}:d=0.45"
        music_filter += ",alimiter=limit=0.95[aout]"
        cmd += ["-filter_complex", music_filter, "-map", "0:v:0", "-map", "[aout]"]
        cmd += ["-t", str(duration)]
    elif has_audio:
        audio_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"
        if fade:
            audio_filter += f",afade=t=in:st=0:d=0.35,afade=t=out:st={max(0, duration - 0.45):.3f}:d=0.45"
        cmd += ["-af", audio_filter]
    else:
        cmd += ["-an"]
    preset, crf = {"Fast": ("ultrafast", "28"),
                   "Balanced": ("veryfast", "24"),
                   "High quality": ("fast", "20")}.get(quality, ("veryfast", "24"))
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-threads", "2",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            "professionally_edited_short.mp4"]
    subprocess.run(cmd, check=True, capture_output=True, cwd=folder)
    with open(output_path, "rb") as edited:
        return edited.read()


def cut_clips(video_path, clips, total, folder, words, captions, vertical,
              min_d, max_d, count, progress=None, aspect_ratio="9:16",
              quality="Balanced"):
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
        if aspect_ratio == "1:1":
            filters.append("crop='min(iw,ih)':'min(iw,ih)',scale=1080:1080")
        elif aspect_ratio == "16:9":
            filters.append("crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=1280:720")
        elif vertical:
            filters.append("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=720:1280")
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
        preset, crf = {"Fast": ("ultrafast", "28"),
                       "Balanced": ("veryfast", "24"),
                       "High quality": ("fast", "20")}.get(quality, ("veryfast", "24"))
        cmd += [
            "-c:v", "libx264", "-preset", preset, "-crf", crf, "-threads", "2",
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
        if progress is not None:
            progress.progress(i / len(non_overlapping), text=f"Clip {i}/{len(non_overlapping)} ready")
    return results


mode = st.radio("Choose editing mode", ["Long video → Auto Clips", "Short video → Professional Edit"])
video = st.file_uploader("Upload Your Video", type=["mp4", "mov", "mkv"])
st.caption("Apni video file upload karein. Long videos ka audio 5-minute parts me process hoga. 2–3 ghante ki high-quality file ko hosting ki upload, RAM, CPU aur time limits rok sakti hain.")
col1, col2 = st.columns(2)
count = col1.number_input("Kitne clips?", min_value=1, max_value=30, value=15)
min_d, max_d = col2.slider("Clip length (seconds)", 10, 90, (20, 60))
captions = st.checkbox("Video ki boli hui language me captions", value=True)
aspect_ratio = st.selectbox("Output format", ["9:16 (Reels/Shorts)", "1:1 (Square)", "16:9 (YouTube)"])
aspect_ratio_value = aspect_ratio.split(" ")[0]
vertical = aspect_ratio_value == "9:16"
quality = st.selectbox("Export quality", ["Fast", "Balanced", "High quality"], index=1)
sensitivity = st.select_slider("Highlight selection sensitivity",
                               options=["Low", "Balanced", "High"], value="Balanced")
sensitivity_value = {"Low": 0.8, "Balanced": 1.0, "High": 1.2}[sensitivity]
model_size = st.selectbox(
    "Speech recognition quality",
    ["tiny", "base", "small"],
    index=0,
    help="tiny sabse fast hai; base balanced hai; small zyada accurate ho sakta hai lekin CPU par kaafi slow hoga."
)
music_file = None
fade_edges = True
color_preset = "Natural"
if mode == "Short video → Professional Edit":
    music_file = st.file_uploader("Optional background music (use audio you have permission to use)", type=["mp3", "wav", "m4a", "aac"])
    fade_edges = st.checkbox("Smooth fade-in / fade-out", value=True)
    color_preset = st.selectbox("Colour style", ["Natural", "Vibrant", "Cinematic"])
correction_text = st.text_area(
    "Hindi/English spelling corrections (optional)",
    placeholder="गलत शब्द = सही शब्द\\nmispeled = misspelled",
    help="Har line me galat = sahi likhein. Correction transcript aur captions dono par lagega."
)

if video is not None and st.button("Create Professional Edit" if mode == "Short video → Professional Edit" else "Create AI Clips"):
    results = []
    try:
        with tempfile.TemporaryDirectory() as folder:
            input_path = os.path.join(folder, "input.mp4")
            with st.spinner("1/3 Video file save ho rahi hai..."):
                # Avoid creating an additional full-size bytes copy while saving uploads.
                video.seek(0)
                with open(input_path, "wb") as output_file:
                    while True:
                        chunk = video.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)

            if mode == "Short video → Professional Edit":
                # Clear the previous preview first so a failed new render never shows stale output.
                st.session_state.pop("short_edit", None)
                st.session_state.pop("results", None)
                with st.spinner("Short video edit ban raha hai..."):
                    source_has_audio = has_audio_stream(input_path)
                    if source_has_audio and captions:
                        audio_progress = st.progress(0, text="Speech recognition...")
                        transcript, total, words = transcribe(input_path, folder, audio_progress, model_size)
                        transcript, words = apply_word_corrections(transcript, words, correction_text)
                        audio_progress.empty()
                    else:
                        transcript, total, words = "", 0.0, []
                        if not source_has_audio:
                            st.info("Is video me audio track nahi hai. Video edit ho jayega; captions aur voice audio nahi honge.")
                        elif not captions:
                            st.info("Captions off hain, isliye speech recognition skip ki gayi hai. Original audio safe rahega.")
                music_path = None
                if music_file is not None:
                    music_path = os.path.join(folder, "background_music" + os.path.splitext(music_file.name)[1])
                    with open(music_path, "wb") as music_out:
                        music_out.write(music_file.getbuffer())
                st.session_state["short_edit"] = edit_short_video(
                    input_path, folder, words, captions, vertical,
                    music_path=music_path, fade=fade_edges, color_preset=color_preset,
                    aspect_ratio=aspect_ratio_value, quality=quality
                )
                st.session_state.pop("results", None)
            else:
                st.session_state.pop("short_edit", None)
                if not has_audio_stream(input_path):
                    raise ValueError("Long video me audio track nahi mila. Speech-based highlights ke liye audio wali video upload karein.")
                with st.spinner("2/3 Long video ka audio parts me process ho raha hai..."):
                    audio_progress = st.progress(0, text="Audio processing shuru...")
                    transcript, total, words = transcribe(input_path, folder, audio_progress, model_size)
                    transcript, words = apply_word_corrections(transcript, words, correction_text)
                    audio_progress.empty()
                if not transcript.strip():
                    raise ValueError("Video me bolne wali awaaz nahi mili.")
                with st.spinner("3/3 Automatic highlights aur clips ban rahe hain..."):
                    picks = pick_clips(transcript, total, count, min_d, max_d, sensitivity_value)
                    clip_progress = st.progress(0, text="Clips render ho rahe hain...")
                    results = cut_clips(
                        input_path, picks, total, folder, words, captions, vertical,
                        min_d, max_d, int(count), clip_progress,
                        aspect_ratio=aspect_ratio_value, quality=quality
                    )
                    clip_progress.empty()

        if mode == "Long video → Auto Clips" and not results:
            raise ValueError("Koi clip nahi bani.")

        if mode == "Long video → Auto Clips":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for r in results:
                    z.writestr(r["name"], r["data"])
                z.writestr("creator_pack.txt", make_creator_pack(results, transcript))
            st.session_state["results"] = results
            st.session_state["creator_pack"] = make_creator_pack(results, transcript)
            st.session_state["zip"] = buf.getvalue()

    except subprocess.CalledProcessError as e:
        st.error("FFmpeg error: " + e.stderr.decode(errors="ignore")[-500:])
    except Exception as e:
        st.error(f"Error: {e}")

if "short_edit" in st.session_state:
    st.subheader("Professionally Edited Short")
    st.video(st.session_state["short_edit"])
    st.download_button("Download Edited Short", st.session_state["short_edit"], file_name="professionally_edited_short.mp4")

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
        "Download All Clips + Creator Pack", st.session_state["zip"], file_name="AI_Clips_and_Creator_Pack.zip"
    )
    st.download_button(
        "Download Titles, Hooks, Descriptions & Hashtags",
        st.session_state["creator_pack"], file_name="creator_pack.txt"
    )
    

