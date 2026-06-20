# Gesture RAG Web App

This wraps the uploaded MediaPipe/KNN gesture project into a deployable web app.

What it does:

- Select 4 to 10 gesture labels.
- Record or upload training videos for each gesture.
- Extract 16-frame fingertip trajectories with MediaPipe Hands.
- Train a project-specific KNN classifier using the same normalized 32-feature representation from the notebook.
- Record or upload a test video and return gesture segments with confidence/timestamps.
- Optionally transcribe the video and run RAG-style gesture-speech analysis with the user's own API key.

## Run

```bash
cd gesture-rag-web
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

On Windows PowerShell:

```powershell
cd gesture-rag-web
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Local Whisper is optional. To use `local_whisper`, install the extra requirements:

```bash
pip install -r requirements-local-whisper.txt
```

Semantic embedding retrieval is optional. The app works without it and falls back to lexical retrieval. To enable it, install:

```bash
pip install -r requirements-semantic-rag.txt
```

## Website Deployment

The repository is set up for two deployment modes:

- Full app: deploy the Python backend with Docker using `Dockerfile`. This is the app that trains and runs MediaPipe/OpenCV inference.
- GitHub Pages: the included `.github/workflows/pages.yml` publishes `app/static` as a static website. In that website, set the `Backend API` field to your deployed backend URL.

GitHub Pages alone cannot run the Python video pipeline. It hosts the browser interface; the backend must run on a service such as Render, Railway, Fly.io, Azure Container Apps, or your own machine.

For Render, this repo includes `render.yaml`. Create a new Blueprint from the GitHub repository, then use the generated service URL as the `Backend API` in the GitHub Pages site.

## Docker

```bash
docker build -t gesture-rag-web .
docker run --rm -p 8000:8000 gesture-rag-web
```

## Notes

- API keys are not stored. The frontend sends them only with the analysis request.
- If the frontend is hosted separately from the backend, CORS is already enabled in `app/main.py`.
- The uploaded archive did not include `point_history_knn.pkl`, so this app trains KNN from recorded/uploaded videos. It can also warm-start from the included `base_point_history.csv` where labels are available.
- The included base CSV currently contains samples for labels 0-3 only; for other gestures, record new training clips before testing.
- For timestamped gesture-speech alignment, `whisper-1` is the safest OpenAI transcription default. `gpt-4o-transcribe-diarize` can also return timestamped speaker segments. Plain `gpt-4o-transcribe`/`gpt-4o-mini-transcribe` are treated as text-only transcription fallbacks.
- For analysis, OpenAI-compatible providers can be used by setting a base URL such as OpenAI, OpenRouter, Groq, Together, or a local OpenAI-compatible server.

## Project Structure

```text
app/
  main.py              FastAPI routes and static hosting
  gesture_pipeline.py  MediaPipe extraction, KNN training, video inference
  rag_pipeline.py      Transcription, alignment, retrieval, LLM memo generation
  static/              Frontend app
assets/
  models/              Labels copied from uploaded project
  source_model/        Uploaded model artifacts and base point-history CSV
uploads/               Per-project videos and training data
outputs/               Inference outputs
```
