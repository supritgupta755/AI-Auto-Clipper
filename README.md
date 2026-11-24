# AI Auto Clipper

A Streamlit app that creates short clips from a user-uploaded video.

## Current features
1. Upload MP4, MOV, or MKV video files.
2. Transcribe speech with Faster-Whisper using automatic language detection.
3. Split extracted audio into 5-minute chunks to reduce peak audio-processing memory.
4. Select candidate highlights using free transcript-based speech and keyword scoring.
5. Cut clips with FFmpeg, optionally add word-timed captions, crop to vertical 9:16, and download clips or a ZIP.

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
- Review clips and make sure you have permission to use the source video before publishing.

## Validation

GitHub Actions checks Python syntax. Passing that check is not proof that deployment, large uploads, transcription, or clip rendering work on the live hosting server.
