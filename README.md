# Real-Time Speech-to-Text & Wakeword Service

This project implements a high-performance, local Speech-to-Text (STT) service using OpenAI's **Whisper** model, **FastAPI**, and **SoundDevice**. It is designed to run continuously in the background, listening to a microphone input and providing two key capabilities via a REST API:

1.  **Live Transcription Streaming**: Real-time broadcast of transcribed text.
2.  **Wakeword Detection**: A blocking endpoint that waits until a specific word or phrase is spoken.

## Features

* **Local Processing**: Uses Hugging Face `transformers` and `torch` to run Whisper locally (GPU acceleration supported).
* **VAD (Voice Activity Detection)**: Efficient silence detection to only process audio when speech is detected.
* **Async Server**: Built on FastAPI to handle multiple clients simultaneously.
* **Broadcast Architecture**: One microphone input can serve multiple wakeword listeners or stream consumers at the same time.
* **Dockerized**: ready-to-deploy container setup with hardware access (Audio/GPU).

---

## Project Structure

* `server.py`: The FastAPI application. It manages the background "Dispatcher" and exposes the HTTP endpoints.
* `audio_transcriber.py`: The core engine. Handles audio capture (SoundDevice), VAD logic, and Whisper inference.
* `Dockerfile` & `docker-compose.yaml`: Configuration for containerized deployment.

---

## Installation & Usage

### Option A: Running Locally (Python)

**Prerequisites:**
* Python 3.10+
* `portaudio` installed on your system (required for microphone access).
    * Ubuntu/Debian: `sudo apt-get install libportaudio2`
    * MacOS: `brew install portaudio`

**Setup:**

1.  **Install Dependencies** (using `uv` or `pip`):
    ```bash
    # Using standard pip
    pip install -r requirements.txt

    # OR using uv (faster)
    uv pip install -r requirements.txt
    ```

2.  **Run the Server**:
    ```bash
    python server.py
    ```

### Option B: Running with Docker (Recommended)

This method isolates dependencies but requires passing hardware access (Mic & GPU) to the container.

1.  **Build and Run**:
    ```bash
    docker compose up --build
    ```

    *Note: The `docker-compose.yaml` is configured to map `/dev/snd` for audio access and requests NVIDIA GPU resources.*

---

## API Endpoints

The server listens on `http://0.0.0.0:8000`.

### 1. Live Transcription Stream (`GET /stream`)

Streams transcription results line-by-line using **Server-Sent Events (SSE)**. Useful for debugging or displaying live captions.

**Example Request:**
```bash
curl -N http://localhost:8000/stream
