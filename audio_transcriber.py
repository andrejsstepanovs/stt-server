import time
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque
import sounddevice as sd
import torch
import warnings
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

class AudioTranscriber:
    def __init__(self, 
                 model_name="openai/whisper-base.en", 
                 device=None, 
                 mic_gain=10.0,
                 debug_mode=False):
        
        self.debug_mode = debug_mode
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.mic_gain = mic_gain
        
        # --- CONFIGURATION ---
        self.SAMPLE_RATE = 16000
        self.BLOCK_SIZE_MS = 30      
        self.START_THRESHOLD = 0.048
        self.END_THRESHOLD = 0.043   
        self.SILENCE_DURATION = 0.5      
        self.PRE_RECORD_SECONDS = 0.5    
        self.MIN_AUDIO_DURATION_S = 0.3
        
        # --- STATE ---
        self._running = threading.Event()
        self._capture_thread = None
        self._transcribe_thread = None
        
        # --- QUEUES ---
        # Queue for raw audio chunks passed to the VAD/Capture logic
        self._audio_input_queue = Queue()
        # Queue for audio segments waiting for Whisper processing
        self._process_queue = Queue()
        # Queue for final text results to be consumed by the caller
        self._result_queue = Queue()

        if self.debug_mode:
            print(f"[INFO] Initializing AudioTranscriber on {self.device}...")

        # Load Model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        if self.debug_mode:
            print("[INFO] Model loaded successfully.")

    def start(self):
        """Starts the audio capture and transcription threads."""
        if self._running.is_set():
            return

        self._running.set()
        self._audio_input_queue = Queue()
        self._process_queue = Queue()
        self._result_queue = Queue()

        # Start Whisper Worker
        self._transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        self._transcribe_thread.start()

        # Start Audio Capture Worker
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()
        
        if self.debug_mode:
            print("[INFO] AudioTranscriber started. Listening...")

    def stop(self):
        """Stops all threads and releases the microphone."""
        if self.debug_mode:
            print("[INFO] Stopping AudioTranscriber...")
        self._running.clear()
        
        # Wait for threads to finish (optional, but good practice)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._transcribe_thread and self._transcribe_thread.is_alive():
            self._transcribe_thread.join(timeout=1.0)
            
        print("[INFO] AudioTranscriber stopped.")

    def transcribe(self):
        """
        Generator that yields transcribed text.
        This is blocking (waiting for data) but yields control back to caller.
        """
        while self._running.is_set():
            try:
                # Wait for a result with a timeout to allow checking _running flag
                text = self._result_queue.get(timeout=0.5)
                yield text
            except Empty:
                continue

    # --- INTERNAL WORKERS ---

    def _audio_callback(self, indata, frames, time_info, status):
        """SoundDevice callback."""
        if status and self.debug_mode:
            print(f"[WARN] Audio Status: {status}")
        self._audio_input_queue.put(indata.copy().flatten())

    def _transcribe_worker(self):
        """Consumes audio segments and runs Whisper inference."""
        while self._running.is_set():
            try:
                item = self._process_queue.get(timeout=0.5)
            except Empty:
                continue

            audio_data, is_final = item
            
            try:
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=self.SAMPLE_RATE, 
                    return_tensors="pt"
                )
                input_features = inputs.input_features.to(self.device)
                attention_mask = torch.ones(input_features.shape, device=self.device)
                gen_kwargs = {"max_new_tokens": 128}
                
                predicted_ids = self.model.generate(
                    input_features, 
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                if text:
                    if is_final:
                        if self.debug_mode: print(f"✅ [FINAL]: {text}")
                        # Put the result in the queue for the external caller
                        self._result_queue.put(text)
                    elif self.debug_mode:
                        print(f"👀 [PREVIEW]: {text}")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"[ERROR] Inference Worker: {e}")

    def _capture_worker(self):
        """Handles VAD (Voice Activity Detection) and buffers audio."""
        block_size_s = self.BLOCK_SIZE_MS / 1000
        pre_buffer_len = int(self.PRE_RECORD_SECONDS * 1000 / self.BLOCK_SIZE_MS) 
        pre_speech_buffer = deque(maxlen=pre_buffer_len)
        
        current_speech_buffer = []
        is_speaking = False
        silence_start_time = None
        last_preview_time = 0
        
        # Start the microphone stream
        with sd.InputStream(samplerate=self.SAMPLE_RATE, 
                            blocksize=int(self.SAMPLE_RATE * block_size_s),
                            channels=1, dtype="float32", 
                            callback=self._audio_callback):
            
            while self._running.is_set():
                try:
                    chunk = self._audio_input_queue.get(timeout=0.5)
                except Empty:
                    continue
                
                chunk = chunk * self.mic_gain
                rms = np.sqrt(np.mean(chunk**2))
                now = time.time()
                
                if is_speaking:
                    current_speech_buffer.append(chunk)
                    
                    # Generate live preview every 1 second
                    if now - last_preview_time > 1.0: 
                        full_audio = np.concatenate(current_speech_buffer)
                        self._process_queue.put((full_audio, False)) 
                        last_preview_time = now

                    if rms > self.END_THRESHOLD:
                        silence_start_time = None 
                    else:
                        if silence_start_time is None:
                            silence_start_time = now
                        
                        silence_duration = now - silence_start_time
                        
                        if silence_duration > self.SILENCE_DURATION:
                            # End of speech detected
                            full_audio = np.concatenate(current_speech_buffer)
                            duration = len(full_audio) / self.SAMPLE_RATE
                            
                            if duration >= self.MIN_AUDIO_DURATION_S:
                                self._process_queue.put((full_audio, True)) 
                            
                            is_speaking = False
                            current_speech_buffer = []
                            silence_start_time = None
                            
                else:
                    pre_speech_buffer.append(chunk)
                    
                    if rms > self.START_THRESHOLD:
                        is_speaking = True
                        current_speech_buffer.extend(pre_speech_buffer)
                        current_speech_buffer.append(chunk)
                        last_preview_time = now
