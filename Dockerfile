# 1. Base Image: Includes Python and uv pre-installed
#    (Replace '3.12' with the version you need, e.g., 3.10, 3.11)
FROM astral/uv:python3.14-trixie

# 2. Set working directory
WORKDIR /app

# 3. Optimization: Compile bytecode to make startup slightly faster
ENV UV_COMPILE_BYTECODE=1

# 4. Dependency Caching Step (The most important part!)
#    We copy ONLY the lockfiles first. Docker will cache the next 'RUN' step
#    forever, unless you actually change your dependencies.
COPY pyproject.toml uv.lock ./

# 5. Install dependencies
#    --frozen: Ensures we use exactly the versions in uv.lock
#    --no-dev: Skips development tools (like pytest/black) to keep image small
#    --no-install-project: Installs libraries but waits on your specific code
RUN uv sync --frozen --no-dev --no-install-project -v

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy Source Code
#    This changes often, so we do it last to avoid breaking the cache above.
COPY asound.conf .
COPY audio_transcriber.py .
COPY server.py .

# 7. (Optional) Final Sync
#    If you just have scripts, this does nothing. 
#    If your project itself is a package, this installs it.
RUN uv sync --frozen --no-dev

# 8. Run Command
#    'uv run' ensures the environment is valid and executes your script.
CMD ["uv", "run", "server.py"]
