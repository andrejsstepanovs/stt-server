import asyncio
import uvicorn
import os
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Import our class from the previous file
from audio_transcriber import AudioTranscriber

# --- CONFIGURATION FROM ENV ---
# Defaults allow it to still run without Docker Compose
MODEL_NAME = os.getenv("WHISPER_MODEL", "openai/whisper-base.en")
TARGET_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") 
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# --- GLOBAL STATE ---
transcriber_engine = None
dispatcher_task = None
active_queues = [] 

class WakewordRequest(BaseModel):
    wakewords: List[str]
    timeout: int = 60 

async def transcription_dispatcher():
    # ... (Keep this function exactly the same as before) ...
    print("[SERVER] Dispatcher started.")
    loop = asyncio.get_event_loop()
    transcriber_iter = transcriber_engine.transcribe()

    while True:
        try:
            text = await loop.run_in_executor(None, next, transcriber_iter)
            for q in list(active_queues):
                await q.put(text)
        except StopIteration:
            break
        except Exception as e:
            print(f"[SERVER ERROR] Dispatcher: {e}")
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcriber_engine, dispatcher_task
    
    print(f"[SERVER] Starting with Model: {MODEL_NAME}, Language: {TARGET_LANGUAGE}")

    # Pass the env vars to the engine
    transcriber_engine = AudioTranscriber(
        model_name=MODEL_NAME, 
        language=TARGET_LANGUAGE,
        debug_mode=DEBUG_MODE
    )
    transcriber_engine.start()
    
    dispatcher_task = asyncio.create_task(transcription_dispatcher())
    
    yield
    
    print("[SERVER] Shutting down...")
    transcriber_engine.stop()
    if dispatcher_task:
        dispatcher_task.cancel()

app = FastAPI(lifespan=lifespan)

# --- ENDPOINT 1: STREAMING TRANSCRIPTION ---
@app.get("/stream")
async def stream_transcription():
    """
    Streams transcription results line-by-line as Server-Sent Events (SSE).
    Useful for a UI or logger that wants to see everything being said.
    """
    async def event_generator():
        # Create a personal queue for this request
        my_queue = asyncio.Queue()
        active_queues.append(my_queue)
        
        try:
            while True:
                # Wait for text from the dispatcher
                text = await my_queue.get()
                
                # SSE Format: "data: <payload>\n\n"
                yield f"data: {text}\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            active_queues.remove(my_queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- ENDPOINT 2: WAKEWORD DETECTION ---
@app.post("/wakeword")
async def wait_for_wakeword(req: WakewordRequest):
    """
    Long-polling endpoint.
    Blocks until one of the provided wakewords is detected in the transcription.
    Returns the detected word immediately.
    """
    my_queue = asyncio.Queue()
    active_queues.append(my_queue)
    
    print(f"[WAKEWORD] Listening for: {req.wakewords}")
    
    # Normalize wakewords for comparison
    targets = [w.lower().strip() for w in req.wakewords]
    
    detected_word = None
    
    try:
        # Wait for data with a timeout
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > req.timeout:
                return {"status": "timeout", "detected": None}

            try:
                # Wait for next transcribed phrase (1 second check interval for timeout logic)
                text = await asyncio.wait_for(my_queue.get(), timeout=1.0)
                
                print(f"[WAKEWORD CHECK] Analyzing: '{text}'")
                text_lower = text.lower()
                
                # Check if any wakeword is present in the phrase
                for target in targets:
                    if target in text_lower:
                        detected_word = target
                        return {
                            "status": "success", 
                            "detected": detected_word, 
                            "full_phrase": text
                        }
                        
            except asyncio.TimeoutError:
                continue # Loop back to check overall timeout
                
    finally:
        active_queues.remove(my_queue)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
