# AI Auto Clipper

A Streamlit app that creates short clips from a user-uploaded video.

## Current features
1. Upload MP4, MOV, or MKV video files.
2. Transcribe speech with Faster-Whisper using automatic language detection and selectable tiny (fast), base (balanced), or small (slower, potentially more accurate) models.
3. Split extracted audio into 5-minute chunks to reduce peak audio-processing memory.
4. Select candidate highlights using free transcript-based speech and keyword scoring, with Low / Balanced / High keyword-emphasis sensitivity controls.
5. Cut clips with FFmpeg, optionally add word-timed captions, crop to vertical 9:16, and download clips or a ZIP.
6. Separate short-video editing mode with optional speech captions, selectable 9:16 / 1:1 / 16:9 output, Fast / Balanced / High quality export, audio loudness normalization, optional user-supplied background music mix, fade-in/fade-out, and Natural/Vibrant/Cinematic colour presets. When captions are switched off, speech recognition is skipped to save processing time. Silent videos can also be edited without captions or source audio.
7. User-entered spelling corrections are applied case-insensitively to transcript text and caption words.

## Run locally

Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Important limitations

- Highlight selection is a lightweight heuristic based on transcript word density and selected keywords. It does **not** understand the video visually and cannot reliably identify every funny, emotional, or meaningful moment.
- Five-minute audio chunking reduces audio working-set size, but the uploaded video, transcript, generated clips, CPU time, RAM, disk space, and ZIP can still be large.
- The repository config requests a 1 GB Streamlit upload limit, but the hosting provider's own limits and available resources still apply. This does not guarantee that a 2–3 hour high-resolution video will upload or finish processing on Streamlit Community Cloud.
- First-time speech-model download and CPU transcription can take a long time. A real 2–3 hour high-resolution video has not yet been successfully tested end-to-end.
- The short-video mode is an automated basic polish workflow, not a full professional nonlinear editor: it does not yet provide face tracking, object-aware reframing, automatic scene transitions, sound-effect selection, or semantic visual editing.
- Background music is optional and must be audio the user has permission to use.
- Review clips and make sure you have permission to use the source video before publishing.

## Validation

GitHub Actions checks Python syntax and runs core unit tests, including synthetic FFmpeg rendering checks for silent and audio video inputs. Passing these tests is not proof that deployment, large uploads, transcription, or clip rendering work on the live hosting server.
