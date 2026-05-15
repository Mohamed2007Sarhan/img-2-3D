"""
stream_platform.py
==================
Mode 3 — Live Streaming Platform Engine

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  INBOX WATCHER (thread)                             │
  │  Polls stream_inbox/ for new .wav files             │
  │  → adds to audio_queue                             │
  └─────────────────────┬───────────────────────────────┘
                        │
  ┌─────────────────────▼───────────────────────────────┐
  │  AUDIO PROCESSOR (thread)                          │
  │  Pops WAV → extract phonemes → build weight curve  │
  │  → adds frames to frame_queue                      │
  └─────────────────────┬───────────────────────────────┘
                        │
  ┌─────────────────────▼───────────────────────────────┐
  │  RENDER LOOP (main thread)                         │
  │  frame_queue → blend mesh → draw → display         │
  │  If queue empty → idle animation (breathing)       │
  │  Optional: push frames to RTMP via ffmpeg          │
  └─────────────────────────────────────────────────────┘

Key Feature: Avatar NEVER freezes.
  - When no audio → gentle idle animation (breathing + micro eye movements)
  - When audio arrives → smooth crossfade into speech animation
  - Segments play back-to-back with no gap
"""

from __future__ import annotations
import os
import sys
import time
import threading
import queue
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import BLENDSHAPE_NAMES, LIPSYNC
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Idle animation generator
# ─────────────────────────────────────────────────────────────────────────────

class IdleAnimator:
    """
    Produces smooth idle-state blendshape weights:
      - Slow breathing (subtle jaw movement)
      - Micro eye blinks at random intervals
      - Very slight head-sway (expressed via cheek asymmetry)
    """

    def __init__(self, fps: int = 30):
        self.fps       = fps
        self.t         = 0.0
        self.next_blink_l = np.random.uniform(2.0, 5.0)   # seconds
        self.next_blink_r = np.random.uniform(2.5, 6.0)
        self.blink_dur    = 0.12   # seconds
        self.blink_t_l    = -999.0
        self.blink_t_r    = -999.0

    def step(self) -> Dict[str, float]:
        dt = 1.0 / self.fps
        self.t += dt
        w: Dict[str, float] = {n: 0.0 for n in BLENDSHAPE_NAMES}

        # Breathing — 0.15 Hz, very subtle jaw
        breath = (np.sin(self.t * 2 * np.pi * 0.15) + 1) / 2
        w["JawOpen"] = breath * 0.035

        # Blink left
        if self.t >= self.next_blink_l:
            self.blink_t_l     = self.t
            self.next_blink_l  = self.t + np.random.uniform(3.0, 7.0)
        blink_phase_l = (self.t - self.blink_t_l) / self.blink_dur
        if 0 <= blink_phase_l <= 1:
            w["EyeBlinkLeft"] = float(np.sin(blink_phase_l * np.pi))

        # Blink right (independent)
        if self.t >= self.next_blink_r:
            self.blink_t_r     = self.t
            self.next_blink_r  = self.t + np.random.uniform(3.5, 8.0)
        blink_phase_r = (self.t - self.blink_t_r) / self.blink_dur
        if 0 <= blink_phase_r <= 1:
            w["EyeBlinkRight"] = float(np.sin(blink_phase_r * np.pi))

        # Micro sway — very slow asymmetric cheek drift
        sway = np.sin(self.t * 2 * np.pi * 0.05) * 0.5 + 0.5
        w["MouthSmileLeft"]  = sway * 0.04
        w["MouthSmileRight"] = (1 - sway) * 0.04

        return w


# ─────────────────────────────────────────────────────────────────────────────
# Weight interpolator (crossfade)
# ─────────────────────────────────────────────────────────────────────────────

def lerp_weights(
    a: Dict[str, float],
    b: Dict[str, float],
    t: float,
) -> Dict[str, float]:
    return {k: a.get(k, 0.0) * (1 - t) + b.get(k, 0.0) * t for k in BLENDSHAPE_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Main Stream Platform
# ─────────────────────────────────────────────────────────────────────────────

class StreamPlatform:
    """
    Full live-streaming avatar engine.

    Lifecycle:
      run() → starts all threads → enters render loop → blocks until quit
    """

    CROSSFADE_FRAMES = 8   # frames to blend from idle→speech and back

    def __init__(
        self,
        obj_path:     str | Path,
        texture_path: Optional[str | Path],
        inbox_dir:    str | Path,
        fps:          int  = 30,
        output_rtmp:  Optional[str] = None,
        window_title: str  = "Audio2Face Live Stream",
    ):
        self.obj_path    = Path(obj_path)
        self.texture     = Path(texture_path) if texture_path else None
        self.inbox_dir   = Path(inbox_dir)
        self.fps         = fps
        self.rtmp        = output_rtmp
        self.title       = window_title

        # Queues
        self.audio_queue:  queue.Queue[Path]              = queue.Queue()
        self.weight_queue: queue.Queue[Dict[str, float]]  = queue.Queue(maxsize=fps * 30)

        # State
        self._running     = False
        self._idle        = IdleAnimator(fps)
        self._mesh        = None
        self._blendshapes: Dict[str, np.ndarray] = {}
        self._basis_v:    Optional[np.ndarray]   = None

        # Stats
        self.stats = {
            "files_processed": 0,
            "frames_rendered":  0,
            "queue_depth":      0,
            "state":            "idle",
        }

    # ─── Public ──────────────────────────────────────────────────────────────

    def run(self):
        """Load assets, start threads, enter render loop."""
        log.info("=== Stream Platform Starting ===")

        self._load_assets()
        self._running = True

        # Thread 1: Inbox watcher
        t_watch = threading.Thread(target=self._watch_inbox, daemon=True)
        t_watch.start()

        # Thread 2: Audio processor
        t_proc = threading.Thread(target=self._process_audio, daemon=True)
        t_proc.start()

        # Thread 3: Optional RTMP pusher
        rtmp_proc = None
        if self.rtmp:
            rtmp_proc = self._start_rtmp_pusher()

        # Main thread: render loop
        try:
            self._render_loop(rtmp_proc)
        except KeyboardInterrupt:
            log.info("Stream stopped by user")
        finally:
            self._running = False
            if rtmp_proc:
                rtmp_proc.stdin.close()

    # ─── Asset loading ────────────────────────────────────────────────────────

    def _load_assets(self):
        log.info("Loading mesh and generating blendshapes ...")
        from utils.mesh_io import load_obj
        from cleanup        import clean_mesh
        from retopo         import detect_face_region, retopologize
        from blendshapes    import generate_blendshapes

        mesh = load_obj(self.obj_path, self.texture)
        mesh = clean_mesh(mesh)
        face_mesh, landmarks = detect_face_region(mesh)
        face_mesh            = retopologize(face_mesh, landmarks)
        self._blendshapes    = generate_blendshapes(face_mesh, landmarks)
        self._mesh           = face_mesh
        self._basis_v        = face_mesh.vertices.copy()
        log.info(f"  Ready: {len(self._blendshapes)} blendshapes loaded")

    # ─── Inbox watcher ────────────────────────────────────────────────────────

    def _watch_inbox(self):
        """
        Continuously polls inbox_dir for new .wav files.
        On finding one, moves it to a processing subfolder and queues it.
        """
        proc_dir = self.inbox_dir / "_processing"
        done_dir = self.inbox_dir / "_done"
        proc_dir.mkdir(exist_ok=True)
        done_dir.mkdir(exist_ok=True)

        log.info(f"  Watching inbox: {self.inbox_dir}")
        seen = set()

        while self._running:
            for wav_file in sorted(self.inbox_dir.glob("*.wav")):
                if wav_file.name in seen:
                    continue
                seen.add(wav_file.name)

                # Move to processing dir atomically
                dest = proc_dir / wav_file.name
                try:
                    shutil.move(str(wav_file), str(dest))
                    log.info(f"  [inbox] Queued: {wav_file.name}")
                    self.audio_queue.put(dest)
                except Exception as e:
                    log.warning(f"  [inbox] Failed to move {wav_file}: {e}")

            time.sleep(0.25)   # poll every 250ms

    # ─── Audio processor ──────────────────────────────────────────────────────

    def _process_audio(self):
        """
        Pops WAV files from audio_queue, converts to per-frame weight dicts,
        pushes them into weight_queue.
        """
        from lipsync import process_wav
        done_dir = self.inbox_dir / "_processing"

        while self._running:
            try:
                wav_path = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            log.info(f"  [processor] Processing: {wav_path.name}")
            self.stats["state"] = "processing"

            try:
                curve = process_wav(wav_path)
                log.info(f"  [processor] {len(curve)} frames queued")

                # Push crossfade-in sentinel
                self.weight_queue.put({"_CROSSFADE_IN": True})

                for frame_weights in curve:
                    if not self._running:
                        break
                    # Block if queue is full (back-pressure)
                    while self.weight_queue.full() and self._running:
                        time.sleep(0.01)
                    self.weight_queue.put(frame_weights)

                # Push crossfade-out sentinel
                self.weight_queue.put({"_CROSSFADE_OUT": True})

                self.stats["files_processed"] += 1

                # Archive processed file
                done_dir = self.inbox_dir / "_done"
                done_dir.mkdir(exist_ok=True)
                shutil.move(str(wav_path), str(done_dir / wav_path.name))

            except Exception as e:
                log.error(f"  [processor] Error processing {wav_path}: {e}")

            self.stats["state"] = "idle"

    # ─── Render loop ──────────────────────────────────────────────────────────

    def _render_loop(self, rtmp_proc=None):
        """
        Main render loop — runs in the main thread.
        Uses pyrender offscreen renderer, shows via OpenCV window.
        Falls back to Open3D if pyrender unavailable.
        """
        import cv2

        backend = self._detect_render_backend()
        log.info(f"  Render backend: {backend}")

        if backend == "pyrender":
            renderer, scene, mesh_node_ref = self._init_pyrender()
        else:
            renderer = scene = mesh_node_ref = None

        dt          = 1.0 / self.fps
        in_crossfade = False
        crossfade_t  = 0.0
        idle_w       = {n: 0.0 for n in BLENDSHAPE_NAMES}
        speech_w     = {n: 0.0 for n in BLENDSHAPE_NAMES}
        current_w    = {n: 0.0 for n in BLENDSHAPE_NAMES}

        log.info("  [render] Loop started — Avatar is live.")

        while self._running:
            t0 = time.time()

            # ── Determine this frame's weights ────────────────────────────────
            idle_w = self._idle.step()

            try:
                frame_w = self.weight_queue.get_nowait()
            except queue.Empty:
                frame_w = None

            if frame_w is None:
                # Pure idle
                current_w = lerp_weights(current_w, idle_w, 0.15)
                self.stats["state"] = "idle"

            elif "_CROSSFADE_IN" in frame_w:
                crossfade_t   = 0.0
                in_crossfade  = True
                current_w     = current_w   # hold

            elif "_CROSSFADE_OUT" in frame_w:
                crossfade_t   = 0.0
                in_crossfade  = False
                current_w     = current_w

            else:
                speech_w = frame_w
                if in_crossfade and crossfade_t < 1.0:
                    crossfade_t += 1.0 / self.CROSSFADE_FRAMES
                    current_w    = lerp_weights(idle_w, speech_w, min(crossfade_t, 1.0))
                else:
                    current_w = speech_w
                self.stats["state"] = "speaking"

            self.stats["queue_depth"] = self.weight_queue.qsize()

            # ── Render ────────────────────────────────────────────────────────
            if backend == "pyrender":
                frame_img = self._render_pyrender(
                    renderer, scene, mesh_node_ref, current_w
                )
            else:
                frame_img = self._render_blank(current_w)

            # ── Show window ───────────────────────────────────────────────────
            frame_img = self._draw_hud(frame_img, current_w)
            cv2.imshow(self.title, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

            # ── RTMP push ─────────────────────────────────────────────────────
            if rtmp_proc:
                try:
                    rtmp_proc.stdin.write(
                        cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR).tobytes()
                    )
                except BrokenPipeError:
                    log.warning("RTMP pipe broken — stopping stream push")
                    rtmp_proc = None

            # ── Timing ────────────────────────────────────────────────────────
            self.stats["frames_rendered"] += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
                break

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

        cv2.destroyAllWindows()
        if renderer:
            renderer.delete()

    # ─── Pyrender helpers ─────────────────────────────────────────────────────

    def _detect_render_backend(self) -> str:
        try:
            import pyrender
            return "pyrender"
        except ImportError:
            return "blank"

    def _init_pyrender(self):
        import pyrender

        scene = pyrender.Scene(bg_color=[0.06, 0.06, 0.08, 1.0])
        bb    = self._mesh.bounding_box.bounds
        c     = (bb[0] + bb[1]) / 2
        dist  = np.linalg.norm(bb[1] - bb[0]) * 1.6

        cam_T = np.eye(4)
        cam_T[:3, 3] = [c[0], c[1], c[2] + dist]
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_T)

        # Three-point lighting
        for light_T, intensity in [
            (cam_T, 3.0),
            (np.array([[1,0,0,0.3],[0,1,0,0.1],[0,0,1,dist*0.7],[0,0,0,1]], float), 1.5),
        ]:
            scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=intensity),
                       pose=light_T)

        import pyrender
        pr_mesh    = pyrender.Mesh.from_trimesh(self._mesh)
        mesh_node  = scene.add(pr_mesh)
        renderer   = pyrender.OffscreenRenderer(1280, 720)

        return renderer, scene, [mesh_node]   # list so it's mutable

    def _render_pyrender(self, renderer, scene, mesh_node_ref, weights):
        import pyrender

        # Blend verts
        blended = self._basis_v.copy()
        for name, w in weights.items():
            if w > 0.001 and name in self._blendshapes:
                blended += w * (self._blendshapes[name] - self._basis_v)
        self._mesh.vertices = blended

        # Swap mesh node
        if mesh_node_ref[0]:
            scene.remove_node(mesh_node_ref[0])
        pr_mesh         = pyrender.Mesh.from_trimesh(self._mesh)
        mesh_node_ref[0] = scene.add(pr_mesh)

        img, _ = renderer.render(scene)
        return img.astype(np.uint8)

    def _render_blank(self, weights) -> np.ndarray:
        """Fallback: visualise blendshape weights as bar chart on dark canvas."""
        import cv2
        img = np.full((720, 1280, 3), 18, dtype=np.uint8)
        cv2.putText(img, "No 3D renderer — install pyrender",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
        active = [(k, v) for k, v in weights.items() if v > 0.01]
        active.sort(key=lambda x: -x[1])
        for i, (name, val) in enumerate(active[:12]):
            y   = 120 + i * 44
            bar = int(val * 400)
            color = (0, int(180 + val * 75), int(100 + val * 155))
            cv2.rectangle(img, (40, y - 16), (40 + bar, y + 8), color, -1)
            cv2.putText(img, f"{name}  {val:.2f}", (450, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return img

    def _draw_hud(self, frame: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Overlay status HUD on the frame."""
        import cv2
        h, w = frame.shape[:2]
        state  = self.stats["state"]
        q_dep  = self.stats["queue_depth"]
        n_proc = self.stats["files_processed"]
        n_frm  = self.stats["frames_rendered"]

        state_color = (0, 220, 120) if state == "speaking" else (120, 120, 200)
        state_label = "● SPEAKING" if state == "speaking" else "◌ IDLE"

        cv2.rectangle(frame, (0, 0), (w, 28), (10, 10, 10), -1)
        cv2.putText(frame, state_label, (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)
        cv2.putText(frame, f"Queue: {q_dep}  Files: {n_proc}  Frames: {n_frm}",
                    (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(frame, f"[Q] Quit   Drop .wav files into inbox to stream",
                    (w - 520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        return frame

    # ─── RTMP pusher ──────────────────────────────────────────────────────────

    def _start_rtmp_pusher(self):
        """
        Open ffmpeg subprocess that reads raw BGR frames from stdin
        and pushes to RTMP stream (e.g. OBS, YouTube, Twitch).
        """
        import subprocess
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", "1280x720",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "flv",
            self.rtmp,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            log.info(f"  RTMP push started → {self.rtmp}")
            return proc
        except FileNotFoundError:
            log.warning("ffmpeg not found — RTMP push disabled")
            return None
        except Exception as e:
            log.warning(f"RTMP start failed: {e}")
            return None
