"""Background voice pipeline: text chunk -> edge-tts -> RVC -> clean-up -> speakers.

Two worker threads run so the next sentence is synthesised while the current
one is still playing. Nothing in here touches Streamlit.
"""
import asyncio
import inspect
import os
import queue
import tempfile
import threading
import uuid
from collections import deque

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

try:
    import noisereduce as nr
except ImportError:  # optional: the high-pass filter still runs without it
    nr = None

# --- Tunables (override with environment variables) -------------------------
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_RATE = os.getenv("TTS_RATE", "-8%")          # slightly slower base speech for RVC
RVC_DEVICE = os.getenv("RVC_DEVICE", "cpu")      # e.g. "cuda:0" on a supported GPU build
RVC_MODEL = os.getenv("RVC_MODEL", "Rapi.pth")
RVC_INDEX = os.getenv("RVC_INDEX", "rapi.index")
DENOISE_STRENGTH = float(os.getenv("DENOISE_STRENGTH", "0.7"))  # 0 disables
SENTENCE_GAP = float(os.getenv("SENTENCE_GAP", "0.18"))         # seconds of silence between chunks

_BLOCK = 2048          # playback block size, also the waveform resolution
_LEVEL_HISTORY = 32    # how many recent blocks the waveform shows
_TMP_DIR = os.path.join(tempfile.gettempdir(), "rapi_tts")


def load_rvc():
    """Load the RVC model once. Import is deferred because it is slow."""
    from rvc_python.infer import RVCInference

    rvc = RVCInference(device=RVC_DEVICE)
    kwargs = {}
    # Hand over the index explicitly when this rvc-python version accepts it,
    # otherwise index_rate below has nothing to work with.
    if os.path.exists(RVC_INDEX) and "index_path" in inspect.signature(rvc.load_model).parameters:
        kwargs["index_path"] = RVC_INDEX
    rvc.load_model(RVC_MODEL, **kwargs)
    rvc.set_params(
        f0up_key=1,
        index_rate=0.78,
        f0method="rmvpe",
        protect=0.15,
        filter_radius=2,
        rms_mix_rate=0.8,
    )
    return rvc


def polish(audio, fs):
    """Lightweight clean-up of the converted voice: hum, hiss, clicks, clipping."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < fs // 10:
        return audio

    # Remove mains hum / rumble below the voice range
    sos = butter(4, 70, btype="highpass", fs=fs, output="sos")
    audio = sosfiltfilt(sos, audio).astype(np.float32)

    # Spectral-gate the static RVC tends to leave behind
    if nr is not None and DENOISE_STRENGTH > 0:
        audio = nr.reduce_noise(
            y=audio, sr=fs, stationary=True, prop_decrease=DENOISE_STRENGTH
        ).astype(np.float32)

    # Short fades so chunk edges never click
    fade = min(int(fs * 0.01), len(audio) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio *= 0.98 / peak
    return audio


class VoicePipeline:
    def __init__(self, rvc):
        self._rvc = rvc
        self._synth_q = queue.Queue()
        self._play_q = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0          # bumped on cancel(); older jobs are dropped
        self._pending = 0      # chunks queued, synthesising or playing
        self._levels = deque(maxlen=_LEVEL_HISTORY)
        self.speaking = False
        os.makedirs(_TMP_DIR, exist_ok=True)
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # --- public API ---------------------------------------------------------
    def speak(self, text):
        """Queue one already-sanitised chunk of text."""
        with self._lock:
            self._pending += 1
            gen = self._gen
        self._synth_q.put((gen, text))

    def cancel(self):
        """Stop the current audio and drop everything queued."""
        with self._lock:
            self._gen += 1
            self._pending = 0
        for q in (self._synth_q, self._play_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._levels.clear()

    @property
    def busy(self):
        return self._pending > 0

    def levels(self):
        """Recent playback loudness (RMS per block), oldest first."""
        return list(self._levels)

    # --- workers ------------------------------------------------------------
    def _finish(self, gen):
        with self._lock:
            if gen == self._gen and self._pending > 0:
                self._pending -= 1

    def _synthesize(self, text):
        stem = os.path.join(_TMP_DIR, uuid.uuid4().hex)
        base_path, out_path = stem + ".mp3", stem + ".wav"
        try:
            asyncio.run(
                edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(base_path)
            )
            self._rvc.infer_file(input_path=base_path, output_path=out_path)
            audio, fs = sf.read(out_path, dtype="float32")
        finally:
            for path in (base_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        audio = polish(audio, fs)
        gap = np.zeros(int(fs * SENTENCE_GAP), dtype=np.float32)
        return np.concatenate([audio, gap]), fs

    def _synth_loop(self):
        while True:
            gen, text = self._synth_q.get()
            if gen != self._gen:
                continue
            try:
                audio, fs = self._synthesize(text)
            except Exception as e:
                print(f"Voice synthesis failed for {text!r}: {e}")
                self._finish(gen)
                continue
            self._play_q.put((gen, audio, fs))

    def _play_loop(self):
        while True:
            gen, audio, fs = self._play_q.get()
            if gen != self._gen:
                continue
            try:
                self.speaking = True
                with sd.OutputStream(samplerate=fs, channels=1, dtype="float32") as stream:
                    for i in range(0, len(audio), _BLOCK):
                        if gen != self._gen:
                            break
                        block = audio[i:i + _BLOCK]
                        self._levels.append(float(np.sqrt(np.mean(block ** 2))))
                        stream.write(block.reshape(-1, 1))
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.speaking = False
                self._finish(gen)
