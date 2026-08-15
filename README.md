# Audio Transcription API

A small FastAPI service that merges multiple audio files with `ffmpeg` and
sends the result to an OpenAI-compatible Whisper transcription server.

## Features

- OpenAI-compatible `POST /v1/audio/transcriptions` endpoint
- Multiple input files merged into one WAV file before transcription
- JSON, plain text, SRT, and verbose JSON response formats
- Docker and Docker Compose support
- Configuration through environment variables

## Requirements

- Docker Engine with Docker Compose
- An OpenAI-compatible Whisper server reachable by the container
- Audio files available through a Docker volume mount

For local development without Docker, install Python 3.13 or newer, `uv`, and
`ffmpeg`.

## Quick Start

1. Copy the example environment file and adjust the Whisper server settings:

	```sh
	cp .env.example .env
	```

2. Put audio files in `raw/` and start the service:

	```sh
	docker compose up --build
	```

3. Send a transcription request. Paths in the request must be paths visible
	inside the container:

	```sh
	curl http://localhost:8080/v1/audio/transcriptions \
	  -H 'Content-Type: application/json' \
	  -d '{
		 "files": ["./raw/part-01.wav", "./raw/part-02.wav"],
		 "model": "whisper-1",
		 "response_format": "json"
	  }'
	```

The service listens on port `8080` by default. Use `TRANSCRIBE_PORT` to expose
it on a different host port.

## API

### `POST /v1/audio/transcriptions`

Request body:

| Field             | Type       | Default     | Description                                         |
| ----------------- | ---------- | ----------- | --------------------------------------------------- |
| `files`           | `string[]` | Required    | One or more audio file paths visible to the service |
| `model`           | `string`   | `whisper-1` | Model name forwarded to the Whisper server          |
| `response_format` | `string`   | `json`      | `json`, `text`, `srt`, or `verbose_json`            |
| `temperature`     | `number`   | `0.0`       | Sampling temperature forwarded to the server        |
| `temperature_inc` | `number`   | `0.2`       | Temperature increment forwarded to the server       |

The input files are concatenated using `ffmpeg` and uploaded as `merged.wav`.
The response is passed through from the Whisper server when possible.

## Configuration

`compose.yml` uses the following variables. Defaults are used when variables
are not set.

| Variable             | Default                               | Description                                       |
| -------------------- | ------------------------------------- | ------------------------------------------------- |
| `PORT`               | `8080`                                | Port used by the API process inside the container |
| `WHISPER_SERVER_URL` | `http://host.docker.internal:8081/v1` | Base URL of the Whisper server                    |
| `WHISPER_API_KEY`    | Empty                                 | API key sent to the Whisper server                |
| `TRANSCRIBE_PORT`    | `8080`                                | Host port exposed by Docker Compose               |
| `RAW_AUDIO_DIR`      | `./raw`                               | Host directory containing source audio files      |
| `MERGED_AUDIO_DIR`   | `./merged`                            | Host directory mounted at `/audio/merged`         |
| `HTTP_PROXY`         | Empty                                 | HTTP proxy for the image build and container      |
| `HTTPS_PROXY`        | Empty                                 | HTTPS proxy for the image build and container     |
| `NO_PROXY`           | Empty                                 | Hosts that should bypass the proxy                |

See `.env.example` for a ready-to-copy configuration template.

## Local Development

Install locked dependencies and run the API with:

```sh
uv sync
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8080
```

Make sure the configured Whisper server and `ffmpeg` are available before
sending requests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
