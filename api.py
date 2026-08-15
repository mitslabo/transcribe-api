from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Audio transcription service")

logger = logging.getLogger("transcribe-api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

DONE = "DONE"
MISSING_AUDIO = "FAILED(MISSING_AUDIO)"
TRANSCRIBE_ERROR = "FAILED(TRANSCRIBE_ERROR)"


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


def parse_whisper_response(response: httpx.Response, response_format: str):
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = {"text": response.text}

    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return {"text": payload}
    return {"text": str(payload)}


def build_result(payload: dict, status: str, merged_audio_path: Path | None) -> dict:
    return {
        **payload,
        "status": status,
        "merged_audio_path": str(merged_audio_path) if merged_audio_path else None,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: TranscriptionRequest,
):
    logger.info("Transcription request received: %s", request.model_dump())

    if request.response_format not in {"json", "text", "srt", "verbose_json"}:
        raise HTTPException(status_code=400, detail="Unsupported response_format")

    source_files = [Path(file_path).expanduser() for file_path in request.files]
    missing_files = [str(path) for path in source_files if not path.is_file()]
    if missing_files:
        return JSONResponse(
            status_code=404,
            content={
                "status": MISSING_AUDIO,
                "merged_audio_path": None,
                "message": "Audio file not found",
                "files": missing_files,
            },
        )

    merged_directory = Path(
        os.environ.get("MERGED_AUDIO_DIR", "./merged")
    ).expanduser()
    merged_directory.mkdir(parents=True, exist_ok=True)
    merged_file_handle = tempfile.NamedTemporaryFile(
        dir=merged_directory,
        prefix="merged-",
        suffix=".wav",
        delete=False,
    )
    merged_file = Path(merged_file_handle.name)
    merged_file_handle.close()

    try:
        await asyncio.to_thread(merge_audio_files, source_files, merged_file)
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=500,
            content=build_result(
                {"message": "ffmpeg is not installed"}, TRANSCRIBE_ERROR, merged_file
            ),
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "ffmpeg failed"
        return JSONResponse(
            status_code=422,
            content=build_result({"message": detail}, TRANSCRIBE_ERROR, merged_file),
        )

    try:
        audio_data = await asyncio.to_thread(merged_file.read_bytes)
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
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
                files={"file": ("merged.wav", audio_data, "audio/wav")},
            )
        response.raise_for_status()
        logger.info("END %s", merged_file)
    except (httpx.HTTPError, OSError) as exc:
        detail = (
            exc.response.text
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None
            else str(exc)
        )
        return JSONResponse(
            status_code=502,
            content=build_result({"message": detail}, TRANSCRIBE_ERROR, merged_file),
        )

    return build_result(
        parse_whisper_response(response, request.response_format), DONE, merged_file
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
