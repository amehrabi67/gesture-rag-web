from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .gesture_pipeline import (
    ROOT,
    add_training_video,
    create_project,
    infer_video,
    load_labels,
    read_project,
    train_project_model,
)
from .rag_pipeline import run_rag_analysis, save_analysis


app = FastAPI(title="Gesture RAG Web App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ProjectCreate(BaseModel):
    selected_gestures: list[str]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/api/gestures")
def gestures() -> dict[str, Any]:
    labels = load_labels()
    return {
        "gestures": labels,
        "defaults": labels[:10],
        "providers": {
            "transcription": ["none", "openai", "local_whisper"],
            "llm": ["none", "openai", "openai_compatible"],
        },
        "model_examples": {
            "transcription": ["whisper-1", "gpt-4o-transcribe-diarize", "gpt-4o-mini-transcribe", "base"],
            "analysis": ["gpt-4.1-mini", "gpt-4o-mini", "openrouter/auto", "llama-3.1-70b-versatile"],
        },
    }


@app.post("/api/projects")
def new_project(payload: ProjectCreate) -> dict[str, Any]:
    try:
        return create_project(payload.selected_gestures)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    try:
        return read_project(project_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/training-video")
async def upload_training_video(
    project_id: str,
    gesture_label: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    try:
        path = await _save_temp_upload(file)
        return add_training_video(project_id, gesture_label, path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/train")
def train(project_id: str, include_base: bool = True) -> dict[str, Any]:
    try:
        return train_project_model(project_id, include_base=include_base)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/test-video")
async def test_video(
    project_id: str,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    try:
        path = await _save_temp_upload(file)
        return infer_video(project_id, path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/analyze")
async def analyze(
    project_id: str,
    query: Annotated[str, Form()],
    video_file: Annotated[str, Form()],
    segments_json: Annotated[str, Form()],
    transcription_provider: Annotated[str, Form()] = "none",
    transcription_api_key: Annotated[str | None, Form()] = None,
    transcription_model: Annotated[str, Form()] = "whisper-1",
    llm_provider: Annotated[str, Form()] = "none",
    llm_api_key: Annotated[str | None, Form()] = None,
    llm_model: Annotated[str, Form()] = "gpt-4.1-mini",
    llm_base_url: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    try:
        import json

        segments = json.loads(segments_json)
        result = run_rag_analysis(
            video_path=video_file,
            gesture_segments=segments,
            query=query,
            transcription_provider=transcription_provider,
            transcription_api_key=transcription_api_key,
            transcription_model=transcription_model,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
        out_path = ROOT / "uploads" / "projects" / project_id / "outputs" / "latest_rag_analysis.json"
        save_analysis(out_path, result)
        result["analysis_path"] = str(out_path)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _save_temp_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        return Path(tmp.name)
