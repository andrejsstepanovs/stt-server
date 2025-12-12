# Real-Time Speech-to-Text & Wakeword Service

This project implements a high-performance, local Speech-to-Text (STT) service using OpenAI's **Whisper** model, **FastAPI**, and **SoundDevice**. It is designed to run continuously in the background, listening to a microphone input and providing two key capabilities via a REST API:

1.  **Live Transcription Streaming**: Real-time broadcast of transcribed text.
2.  **Wakeword Detection**: A blocking endpoint that waits until a specific word or phrase is spoken.

## Features

* **Local Processing**: Uses Hugging Face `transformers` and `torch` to run Whisper locally (GPU acceleration supported).
* **VAD (Voice Activity Detection)**: Efficient silence detection to only process audio when speech is detected.
* **Async Server**: Built on FastAPI to handle multiple clients simultaneously.
* **Broadcast Architecture**: One microphone input can serve multiple wakeword listeners or stream consumers at the same time.
* **Dockerized**: Ready-to-deploy container setup with hardware access (Audio/GPU).

---

## Installation & Usage

### Option A: Running Locally with uv (Recommended)

This project is optimized for `uv`, a fast Python package installer and resolver.

**Prerequisites:**
* `uv` installed (`wget -qO- https://astral.sh/uv/install.sh | sh`)
* `portaudio` installed on your system (required for microphone access).
    * Ubuntu/Debian: `sudo apt-get install libportaudio2`
    * MacOS: `brew install portaudio`

**Steps:**

1.  **Initialize Virtual Environment**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    uv run server.py
    ```

### Option B: Running with Docker

This method isolates dependencies but requires passing hardware access (Mic & GPU) to the container.

1.  **Build and Run**:
    ```bash
    docker compose up --build
    ```

    *Note: The `docker-compose.yaml` is configured to map `/dev/snd` for audio access and requests NVIDIA GPU resources.*

---

## API Endpoints

The server listens on `http://0.0.0.0:8000`.

### 1. Wakeword Detection (`POST /wakeword`)

This is the primary endpoint for automation. It is a **long-polling** endpoint. When you call it, the request will "hang" (wait) until one of the specified wakewords is detected in the live audio stream.

**Request Body:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `wakewords` | `List[str]` | A list of phrases to listen for (case-insensitive). |
| `timeout` | `int` | (Optional) Max seconds to wait before giving up. Default: 60. |

**Example Usage (cURL):**

```bash
curl -X POST http://localhost:8000/wakeword \
     -H "Content-Type: application/json" \
     -d '{
           "wakewords": ["hey computer", "start recording", "stop"],
           "timeout": 30
         }'
