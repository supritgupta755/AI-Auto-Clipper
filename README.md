# AI Auto Clipper

A Streamlit app that turns a long video into short highlight clips.

## What it does
1. Transcribes speech with Faster-Whisper and automatically detects the spoken language.
2. Asks Claude Sonnet 5.5 to choose engaging, self-contained moments.
3. Cuts the selected moments with FFmpeg.
4. Optionally adds word-timed captions and crops clips to vertical 9:16.
5. Lets you download individual clips or one ZIP file.

## Run on your computer
Use Python 3.10 or newer:

```bash
pip install -r requirements.txt
streamlit run main.py
```

## API key
The highlight-selection step requires an Anthropic API key. The app accepts it from:
- the `ANTHROPIC_API_KEY` environment variable;
- Streamlit Secrets; or
- the password field in the app sidebar.

Never commit a real API key to this public repository. API usage may cost money according to your Anthropic account's pricing.

For Streamlit Community Cloud, deploy this repository with `main.py` as the app file, then add this under App settings → Secrets:

```toml
ANTHROPIC_API_KEY = "your-key"
```

## Notes
- Speech recognition runs locally on the app server using the Faster-Whisper `base` model on CPU.
- The caption language follows the detected speech language; font support for Hindi/Devanagari is included.
- Generated clips should be reviewed before publishing. Automatic highlight selection can make mistakes.
- This repository includes a GitHub Actions Python syntax check. A successful syntax check does not replace testing with a real video and a valid API key.
