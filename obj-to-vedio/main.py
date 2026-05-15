from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

def _run() -> None:
    root = Path(__file__).resolve().parent / 'pifuhd_audio2face_pipeline'
    sys.path.insert(0, str(root))
    path = root / 'main.py'
    spec = importlib.util.spec_from_file_location('_pifu_a2f', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('failed to load pipeline main')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == '__main__':
    _run()
