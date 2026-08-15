from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Audio transcription service")


class TranscriptionRequest(BaseModel):
    files: list[str] = Field(min_length=1)
    model: str = "whisper-1"
    response_format: str = "json"
    temperature: float = 0.0
    temperature_inc: float = 0.2


def get_whisper_url() -> str:
    base_url = os.environ.get("WHISPER_SERVER_URL", "http://127.0.0.1:8081/v1")
    return f"{base_url.rstrip('/')}/audio/transcriptions"


def merge_audio_files(source_files: list[Path], output_file: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as list_file:
        for source_file in source_files:
            escaped = str(source_file).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_file.flush()

        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-c",
                "copy",
                "-y",
                str(output_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def parse_whisper_response(response: requests.Response, response_format: str):
    if response_format == "text":
        return PlainTextResponse(response.text)
    if response_format == "srt":
        return PlainTextResponse(response.text, media_type="application/x-subrip")

    try:
        payload = response.json()
    except ValueError:
        return {"text": response.text}

    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return {"text": payload}
    return {"text": str(payload)}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: TranscriptionRequest,
):
    if request.response_format not in {"json", "text", "srt", "verbose_json"}:
        raise HTTPException(status_code=400, detail="Unsupported response_format")

    source_files = [Path(file_path).expanduser() for file_path in request.files]
    missing_files = [str(path) for path in source_files if not path.is_file()]
    if missing_files:
        raise HTTPException(
            status_code=404,
            detail={"message": "Audio file not found", "files": missing_files},
        )

    with tempfile.TemporaryDirectory(prefix="transcribe-") as work_dir:
        work_path = Path(work_dir)
        merged_file = work_path / "merged.wav"
        try:
            merge_audio_files(source_files, merged_file)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="ffmpeg is not installed") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "ffmpeg failed"
            raise HTTPException(status_code=422, detail=detail) from exc

        try:
            with merged_file.open("rb") as audio_file:
                response = requests.post(
                    get_whisper_url(),
                    headers={
                        "Authorization": f"Bearer {os.environ.get('WHISPER_API_KEY', '')}"
                    },
                    data={
                        "model": request.model,
                        "temperature": str(request.temperature),
                        "temperature_inc": str(request.temperature_inc),
                        "response_format": request.response_format,
                    },
                    files={"file": ("merged.wav", audio_file, "audio/wav")},
                    timeout=600,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = response.text if exc.response is not None else str(exc)
            raise HTTPException(status_code=502, detail=detail) from exc

    return parse_whisper_response(response, request.response_format)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
