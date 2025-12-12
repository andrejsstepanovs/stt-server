# Use a slim Python image to keep size down
FROM python:3.11-slim

# --- 1. SYSTEM DEPENDENCIES ---
# libportaudio2: Required by 'sounddevice' to access microphone
# alsa-utils:    Provides 'arecord' to debug audio devices inside container
# git:           Often needed if installing dependencies from git repositories
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    alsa-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- 2. WORKDIR & DEPENDENCIES ---
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

COPY asound.conf /etc/asound.conf

# --- 3. APPLICATION CODE ---
COPY audio_transcriber.py .
COPY server.py .

# Expose the API port
EXPOSE 8000

# --- 4. STARTUP ---
# We use 'python -u' (unbuffered) so logs show up immediately in Docker
CMD ["python", "-u", "server.py"]
