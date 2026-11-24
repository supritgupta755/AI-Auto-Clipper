# AI Auto Clipper

A Streamlit app that transcribes a long video, asks Claude to select highlight moments, cuts clips with FFmpeg, and can add English captions.

## Requirements
- Python 3.10+
- An Anthropic API key (stored as the Streamlit secret `ANTHROPIC_API_KEY` or entered in the app sidebar)

## Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
``

## Notes
- Speech recognition currently uses the Faster-Whisper `base` model on CPU.
- Captions are currently English-only.
- Review generated clips before publishing; AI-selected moments may need editing.
