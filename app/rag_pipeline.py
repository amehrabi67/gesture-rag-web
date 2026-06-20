from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def transcribe_video(
    video_path: str,
    provider: str = "none",
    api_key: str | None = None,
    model: str = "gpt-4o-mini-transcribe",
) -> list[dict[str, Any]]:
    provider = (provider or "none").lower()
    if provider == "none":
        return []

    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI transcription requires an API key.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        with open(video_path, "rb") as f:
            if model == "whisper-1":
                try:
                    tx = client.audio.transcriptions.create(
                        file=f,
                        model=model,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
                except Exception:
                    f.seek(0)
                    tx = client.audio.transcriptions.create(
                        file=f,
                        model=model,
                        response_format="verbose_json",
                    )
            elif "diarize" in model:
                tx = client.audio.transcriptions.create(
                    file=f,
                    model=model,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
            else:
                tx = client.audio.transcriptions.create(file=f, model=model)
        data = tx.model_dump() if hasattr(tx, "model_dump") else dict(tx)
        segments = data.get("segments") or []
        if segments:
            return [
                {
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": s.get("text", "").strip(),
                }
                for s in segments
            ]
        text = data.get("text", "").strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []

    if provider == "local_whisper":
        import whisper

        local_model = whisper.load_model(model or "base")
        result = local_model.transcribe(video_path)
        return [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]

    raise ValueError(f"Unsupported transcription provider: {provider}")


def align_segments(
    gesture_segments: list[dict[str, Any]], transcript_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aligned = []
    for seg in gesture_segments:
        midpoint = (seg["start_time_sec"] + seg["end_time_sec"]) / 2
        speech = None
        if transcript_segments:
            speech = min(
                transcript_segments,
                key=lambda s: abs(midpoint - ((float(s["start"]) + float(s["end"])) / 2)),
            )
        aligned.append(
            {
                **seg,
                "speech_text": speech.get("text", "") if speech else "",
                "speech_start": speech.get("start") if speech else None,
                "speech_end": speech.get("end") if speech else None,
                "mcneill_guess": mcneill_guess(seg["label"]),
            }
        )
    return aligned


def mcneill_guess(label: str) -> str:
    label_l = label.lower()
    if "data" in label_l:
        return "deictic / pointing"
    if any(x in label_l for x in ["line", "curve", "slope", "box", "circle", "distribution", "wave", "infinity"]):
        return "metaphoric / iconic mathematical form"
    return "uncoded"


def build_documents(aligned_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for i, seg in enumerate(aligned_segments, start=1):
        text = (
            f"Gesture {i}: {seg['label']} from {seg['start_time_sec']}s to {seg['end_time_sec']}s. "
            f"Confidence: {seg.get('avg_confidence')}. "
            f"Gesture function: {seg.get('mcneill_guess')}. "
            f"Aligned speech: {seg.get('speech_text') or '[no speech aligned]'}"
        )
        docs.append({"id": i, "text": text, "segment": seg})
    return docs


def retrieve_docs(query: str, docs: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    if not docs:
        return []
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        doc_emb = model.encode([d["text"] for d in docs], normalize_embeddings=True)
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(doc_emb, q_emb)
        order = np.argsort(-scores)[:top_k]
        return [{**docs[int(i)], "score": float(scores[int(i)])} for i in order]
    except Exception:
        return lexical_retrieve(query, docs, top_k=top_k)


def lexical_retrieve(query: str, docs: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    q_terms = Counter(tokenize(query))
    scored = []
    for d in docs:
        terms = Counter(tokenize(d["text"]))
        score = sum(min(q_terms[t], terms[t]) for t in q_terms)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**d, "score": float(score)} for score, d in scored[:top_k]]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def generate_rule_based_memo(query: str, retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "No gesture or transcript evidence was available for this query."
    labels = Counter(d["segment"]["label"] for d in retrieved)
    top = ", ".join(f"{k} ({v})" for k, v in labels.most_common())
    lines = [f"Rule-based memo for: {query}", f"Most retrieved gestures: {top}."]
    for d in retrieved[:5]:
        s = d["segment"]
        lines.append(
            f"- {s['label']} at {s['start_time_sec']}-{s['end_time_sec']}s; "
            f"speech: {s.get('speech_text') or 'no aligned speech'}"
        )
    return "\n".join(lines)


def generate_llm_memo(
    query: str,
    retrieved: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    base_url: str | None = None,
    provider: str = "none",
) -> str:
    if not api_key or provider == "none":
        return generate_rule_based_memo(query, retrieved)

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    evidence = "\n".join(d["text"] for d in retrieved[:10])
    prompt = f"""
You are analyzing a learner's oral explanation using gesture and speech evidence.
Use only the evidence below. Be concise and cite timestamps.

Question:
{query}

Evidence:
{evidence}

Return:
1. Main interpretation
2. Gesture-speech alignment
3. Possible misconception or uncertainty
4. Evidence bullets with timestamps
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful qualitative analyst of embodied cognition data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def run_rag_analysis(
    video_path: str,
    gesture_segments: list[dict[str, Any]],
    query: str,
    transcription_provider: str = "none",
    transcription_api_key: str | None = None,
    transcription_model: str = "gpt-4o-mini-transcribe",
    llm_provider: str = "none",
    llm_api_key: str | None = None,
    llm_model: str = "gpt-4.1-mini",
    llm_base_url: str | None = None,
) -> dict[str, Any]:
    transcript = transcribe_video(
        video_path=video_path,
        provider=transcription_provider,
        api_key=transcription_api_key,
        model=transcription_model,
    )
    aligned = align_segments(gesture_segments, transcript)
    docs = build_documents(aligned)
    retrieved = retrieve_docs(query, docs)
    memo = generate_llm_memo(
        query=query,
        retrieved=retrieved,
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        provider=llm_provider,
    )
    return {
        "transcript": transcript,
        "aligned_segments": aligned,
        "retrieved": retrieved,
        "memo": memo,
    }


def save_analysis(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
