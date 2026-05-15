from __future__ import annotations
import argparse
import json
import time
import threading
import queue
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import scipy.signal as signal
from config import LIPSYNC, PHONEME_VISEME_MAP, BLENDSHAPE_NAMES, OUTPUTS_DIR
from utils.logger import get_logger
log = get_logger(__name__)
AnimCurve = List[Dict[str, float]]

def process_wav(wav_path: str | Path) -> AnimCurve:
    log.info(f'=== Lip Sync: {wav_path} ===')
    (audio, sr) = _load_audio(wav_path)
    phonemes = _extract_phonemes(audio, sr)
    curve = _phonemes_to_curve(phonemes, len(audio), sr)
    curve = _smooth_curve(curve)
    log.info(f"  Generated {len(curve)} frames at {LIPSYNC['fps']} fps")
    return curve

def save_curve(curve: AnimCurve, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'fps': LIPSYNC['fps'], 'frames': curve}, f, indent=2)
    log.info(f'Animation curve saved → {path}')

def load_curve(path: str | Path) -> Tuple[int, AnimCurve]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (data['fps'], data['frames'])

def _load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    try:
        import librosa
        (audio, sr) = librosa.load(str(path), sr=LIPSYNC['sample_rate'], mono=True)
        log.debug(f'  Audio: {len(audio) / sr:.2f}s @ {sr}Hz')
        return (audio, sr)
    except ImportError:
        import scipy.io.wavfile as wavfile
        (sr, data) = wavfile.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if sr != LIPSYNC['sample_rate']:
            from scipy.signal import resample
            n_samples = int(len(data) * LIPSYNC['sample_rate'] / sr)
            data = resample(data, n_samples)
            sr = LIPSYNC['sample_rate']
        log.debug(f'  Audio (scipy): {len(data) / sr:.2f}s @ {sr}Hz')
        return (data, sr)

def _extract_phonemes(audio: np.ndarray, sr: int) -> List[Tuple[float, float, str]]:
    backend = LIPSYNC['phoneme_model']
    if backend == 'wav2vec2':
        result = _phonemes_wav2vec2(audio, sr)
        if result:
            return result
        log.warning('  wav2vec2 unavailable — falling back to heuristic')
    return _phonemes_heuristic(audio, sr)

def _phonemes_wav2vec2(audio: np.ndarray, sr: int) -> Optional[List[Tuple[float, float, str]]]:
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        log.info('  Using Wav2Vec2 for phoneme extraction ...')
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        model.eval()
        inputs = processor(audio, sampling_rate=sr, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        hop = LIPSYNC['hop_length']
        n_seg = logits.shape[1]
        dur = len(audio) / sr
        segs = []
        for (i, pid) in enumerate(pred_ids[0].tolist()):
            if pid == processor.tokenizer.pad_token_id:
                continue
            token = processor.tokenizer.convert_ids_to_tokens([pid])[0].upper()
            t_start = i * dur / n_seg
            t_end = (i + 1) * dur / n_seg
            segs.append((t_start, t_end, token if token in PHONEME_VISEME_MAP else 'SIL'))
        log.info(f'  Wav2Vec2 segments: {len(segs)}')
        return segs
    except (ImportError, Exception) as exc:
        log.debug(f'  Wav2Vec2 failed: {exc}')
        return None

def _phonemes_heuristic(audio: np.ndarray, sr: int) -> List[Tuple[float, float, str]]:
    try:
        import librosa
        hop = LIPSYNC['hop_length']
        n_mfcc = LIPSYNC['n_mfcc']
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
        energy = librosa.feature.rms(y=audio, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop)[0]
    except ImportError:
        hop = LIPSYNC['hop_length']
        n_fft = 512
        frames = _manual_frames(audio, n_fft, hop)
        energy = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        centroid = np.array([_spectral_centroid(f, sr, n_fft) for f in frames])
        zcr = np.array([np.mean(np.abs(np.diff(np.sign(f)))) / 2 for f in frames])
        mfcc = np.zeros((13, len(frames)))
    e_thresh = np.percentile(energy, 20)
    phonemes: List[Tuple[float, float, str]] = []
    frame_dur = hop / sr
    for i in range(len(energy)):
        t_start = i * frame_dur
        t_end = t_start + frame_dur
        if energy[i] < e_thresh:
            phonemes.append((t_start, t_end, 'SIL'))
            continue
        c = centroid[i]
        z = zcr[i]
        if c < 800:
            ph = 'OH'
        elif c < 1400:
            ph = 'AA'
        elif c < 2200:
            ph = 'EE'
        elif z > 0.15:
            ph = 'FV'
        else:
            ph = 'MP'
        phonemes.append((t_start, t_end, ph))
    log.info(f'  Heuristic phonemes: {len(phonemes)} segments')
    return phonemes

def _manual_frames(audio: np.ndarray, n_fft: int, hop: int) -> List[np.ndarray]:
    frames = []
    for i in range(0, len(audio) - n_fft, hop):
        frames.append(audio[i:i + n_fft])
    return frames

def _spectral_centroid(frame: np.ndarray, sr: int, n_fft: int) -> float:
    spec = np.abs(np.fft.rfft(frame, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    denom = spec.sum()
    return float(np.dot(freqs, spec) / denom) if denom > 0 else 0.0

def _phonemes_to_curve(phonemes: List[Tuple[float, float, str]], n_samples: int, sr: int) -> AnimCurve:
    fps = LIPSYNC['fps']
    duration = n_samples / sr
    n_frames = int(np.ceil(duration * fps))
    curve: AnimCurve = [{n: 0.0 for n in BLENDSHAPE_NAMES} for _ in range(n_frames)]
    for (t_start, t_end, phoneme) in phonemes:
        viseme = PHONEME_VISEME_MAP.get(phoneme, None)
        if viseme is None:
            continue
        f_start = int(t_start * fps)
        f_end = min(int(t_end * fps) + 1, n_frames)
        seg_len = max(f_end - f_start, 1)
        for fi in range(f_start, f_end):
            if fi >= n_frames:
                break
            t = (fi - f_start) / seg_len
            if t < 0.2:
                w = t / 0.2
            elif t > 0.8:
                w = (1.0 - t) / 0.2
            else:
                w = 1.0
            if phoneme in ('AA', 'AE', 'AH', 'AO', 'EH', 'OH'):
                curve[fi]['JawOpen'] = max(curve[fi]['JawOpen'], w * 0.6)
            curve[fi][viseme] = max(curve[fi][viseme], w)
    return curve

def _smooth_curve(curve: AnimCurve) -> AnimCurve:
    w = LIPSYNC['smooth_window']
    if w <= 1:
        return curve
    keys = list(curve[0].keys())
    arrays = {k: np.array([f[k] for f in curve]) for k in keys}
    from scipy.ndimage import gaussian_filter1d
    smoothed = {k: gaussian_filter1d(arrays[k], sigma=w) for k in keys}
    result: AnimCurve = []
    for i in range(len(curve)):
        result.append({k: float(np.clip(smoothed[k][i], 0.0, 1.0)) for k in keys})
    return result

class LiveLipSync:

    def __init__(self, callback: Callable[[Dict[str, float]], None]):
        self.callback = callback
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info('LiveLipSync started (microphone streaming)')

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info('LiveLipSync stopped')

    def _capture_loop(self):
        try:
            import pyaudio
        except ImportError:
            log.error('pyaudio not installed — live mic unavailable (pip install pyaudio)')
            return
        sr = LIPSYNC['sample_rate']
        chunk_size = LIPSYNC['hop_length'] * 4
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, input=True, frames_per_buffer=chunk_size)
        log.info(f'  Microphone open: chunk={chunk_size}, sr={sr}')
        buf = np.zeros(sr, dtype=np.float32)
        while self._running:
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                buf = np.roll(buf, -len(chunk))
                buf[-len(chunk):] = chunk
                phonemes = _phonemes_heuristic(buf, sr)
                if phonemes:
                    ph = phonemes[-1][2]
                    viseme = PHONEME_VISEME_MAP.get(ph, None)
                    weights = {n: 0.0 for n in BLENDSHAPE_NAMES}
                    if viseme:
                        weights[viseme] = 1.0
                    if ph in ('AA', 'AE', 'AH', 'AO', 'EH', 'OH'):
                        weights['JawOpen'] = 0.6
                    self.callback(weights)
            except Exception as exc:
                log.debug(f'  LiveLipSync error: {exc}')
        stream.stop_stream()
        stream.close()
        pa.terminate()

def _cli():
    parser = argparse.ArgumentParser(description='Stage 8: Audio-Driven Lip Sync')
    parser.add_argument('wav', help='Input WAV file')
    parser.add_argument('--output', default=str(OUTPUTS_DIR / 'lipsync_curve.json'))
    parser.add_argument('--live', action='store_true', help='Live microphone mode')
    args = parser.parse_args()
    if args.live:

        def on_frame(w):
            active = {k: v for (k, v) in w.items() if v > 0.05}
            log.info(f'Live: {active}')
        ls = LiveLipSync(on_frame)
        ls.start()
        log.info('Live mode active — press Ctrl+C to stop')
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            ls.stop()
    else:
        curve = process_wav(args.wav)
        save_curve(curve, args.output)
if __name__ == '__main__':
    _cli()
