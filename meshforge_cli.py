from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable, Sequence
ROOT = Path(__file__).resolve().parent
PIFUHD_APP = ROOT / 'pifuhd' / 'pifuhd'
A2F_PIPELINE = ROOT / 'obj-to-vedio' / 'pifuhd_audio2face_pipeline'

def _banner() -> str:
    return textwrap.dedent('\n        ╔═══════════════════════════════════════════════════════════╗\n        ║  MeshForge · PIFuHD ⇄ Audio2Face prep                     ║\n        ╚═══════════════════════════════════════════════════════════╝\n        ').strip()

def _which_python() -> str:
    return sys.executable

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _run(cmd: Sequence[str], *, cwd: Path | None=None, env: dict | None=None) -> int:
    printable = ' '.join((f'"{c}"' if ' ' in c else c for c in cmd))
    print(f'\n→ {printable}\n')
    merged = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged)
    return proc.returncode

def _format_cmd_for_shell(cmd: Sequence[str], shell: str) -> str:
    if shell.lower() in ('powershell', 'pwsh', 'ps'):
        parts = []
        for c in cmd:
            escaped = str(c).replace("'", "''")
            parts.append(f"'{escaped}'")
        return ' '.join(parts)
    parts = []
    for c in cmd:
        parts.append(shlex.quote(str(c)))
    return ' '.join(parts)

def _cd_line(path: Path, shell: str) -> str:
    s = str(path.resolve())
    if shell.lower() == 'powershell':
        return f"Set-Location -LiteralPath '{s.replace(chr(39), chr(39) + chr(39))}'"
    return f'cd {shlex.quote(s)}'

def find_pifuhd_outputs(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        return []
    objs: list[Path] = []
    for p in results_dir.rglob('recon/*.obj'):
        if p.name.startswith('result_') and p.suffix.lower() == '.obj':
            objs.append(p)
    objs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return objs

def run_pifuhd(image: Path, results_dir: Path, *, ckpt: Path | None, resolution: int, use_rect: bool) -> Path | None:
    if not PIFUHD_APP.is_dir():
        print(f'خطأ: مجلد PIFuHD غير موجود: {PIFUHD_APP}', file=sys.stderr)
        return None
    image = image.resolve()
    if not image.is_file():
        print(f'خطأ: الصورة غير موجودة: {image}', file=sys.stderr)
        return None
    work_in = image.parent
    cmd: list[str] = [_which_python(), '-m', 'apps.simple_test', '-i', str(work_in), '-o', str(_ensure_dir(results_dir)), '-r', str(resolution)]
    if ckpt:
        cmd.extend(['-c', str(ckpt.resolve())])
    if use_rect:
        cmd.append('--use_rect')
    rc = _run(cmd, cwd=PIFUHD_APP)
    if rc != 0:
        print('فشل تشغيل PIFuHD.', file=sys.stderr)
        return None
    candidates = find_pifuhd_outputs(results_dir.resolve())
    stem = image.stem
    for obj in candidates:
        if stem in obj.name or stem.replace(' ', '_') in obj.name:
            return obj
    return candidates[0] if candidates else None

def build_audio2face_cmd(obj: Path, *, texture: Path | None, wav: Path | None, out_dir: Path, formats: Iterable[str], blender: str, viewer: bool, webcam: bool, skip_export: bool) -> list[str]:
    cmd: list[str] = [_which_python(), 'main.py', str(obj.resolve()), '--out-dir', str(out_dir.resolve()), '--formats', *list(formats), '--blender', blender]
    if texture:
        cmd.extend(['--texture', str(texture.resolve())])
    if wav:
        cmd.extend(['--wav', str(wav.resolve())])
    if viewer:
        cmd.append('--viewer')
    if webcam:
        cmd.append('--webcam')
    if skip_export:
        cmd.append('--skip-export')
    return cmd

def run_audio2face_cmd(cmd: list[str]) -> int:
    if not (A2F_PIPELINE / 'main.py').is_file():
        print(f'خطأ: خط Audio2Face غير موجود: {A2F_PIPELINE}', file=sys.stderr)
        return 1
    return _run(cmd, cwd=A2F_PIPELINE)

def cmd_pifuhd(args: argparse.Namespace) -> int:
    print(_banner())
    image = Path(args.image)
    out = Path(args.out)
    obj = run_pifuhd(image, out, ckpt=Path(args.ckpt) if args.ckpt else None, resolution=args.resolution, use_rect=args.use_rect)
    if obj:
        print(f'\n✓ OBJ: {obj}\n')
        return 0
    return 1

def cmd_audio2face(args: argparse.Namespace) -> int:
    print(_banner())
    obj = Path(args.obj)
    if not obj.is_file():
        print(f'خطأ: ملف OBJ غير موجود: {obj}', file=sys.stderr)
        return 1
    cmd = build_audio2face_cmd(obj, texture=Path(args.texture) if args.texture else None, wav=Path(args.wav) if args.wav else None, out_dir=Path(args.out_dir), formats=args.formats, blender=args.blender, viewer=args.viewer, webcam=args.webcam, skip_export=args.skip_export)
    return 0 if run_audio2face_cmd(cmd) == 0 else 1

def cmd_full(args: argparse.Namespace) -> int:
    print(_banner())
    image = Path(args.image)
    pifu_out = Path(args.pifuhd_out)
    obj = run_pifuhd(image, pifu_out, ckpt=Path(args.ckpt) if args.ckpt else None, resolution=args.resolution, use_rect=args.use_rect)
    if not obj:
        return 1
    a2f_out = Path(args.out_dir) if getattr(args, 'out_dir', None) else pifu_out / 'audio2face_prep'
    _ensure_dir(a2f_out)
    texture = Path(args.texture) if args.texture else None
    wav = Path(args.wav) if args.wav else None
    cmd = build_audio2face_cmd(obj, texture=texture, wav=wav, out_dir=a2f_out, formats=args.formats, blender=args.blender, viewer=args.viewer, webcam=args.webcam, skip_export=args.skip_export)
    print('\n━━━ تشغيل تجهيز Audio2Face ━━━\n')
    return 0 if run_audio2face_cmd(cmd) == 0 else 1

def cmd_prepare_cmd(args: argparse.Namespace) -> int:
    print(_banner())
    image = Path(args.image)
    pifu_out = Path(args.pifuhd_out)
    a2f_out = Path(args.out_dir)
    stem = image.stem
    guess: Path = pifu_out / 'pifuhd_final' / 'recon' / f'result_{stem}_{args.resolution}.obj'
    for cand in find_pifuhd_outputs(pifu_out.resolve()):
        if stem in cand.name:
            guess = cand
            break
    pifu_cmd = [_which_python(), '-m', 'apps.simple_test', '-i', str(image.parent.resolve()), '-o', str(pifu_out.resolve()), '-r', str(args.resolution)]
    if args.ckpt:
        pifu_cmd.extend(['-c', str(Path(args.ckpt).resolve())])
    if args.use_rect:
        pifu_cmd.append('--use_rect')
    a2f_cmd = build_audio2face_cmd(guess, texture=Path(args.texture) if args.texture else None, wav=Path(args.wav) if args.wav else None, out_dir=a2f_out, formats=args.formats, blender=args.blender, viewer=args.viewer, webcam=args.webcam, skip_export=args.skip_export)
    shell = args.shell or ('powershell' if os.name == 'nt' else 'bash')
    print('\n# 1) PIFuHD — صورة إلى OBJ (من مجلد pifuhd/pifuhd):\n')
    print(_cd_line(PIFUHD_APP, shell))
    print(_format_cmd_for_shell(pifu_cmd, shell))
    print('\n# بعد التأكد من مسار الـ OBJ الفعلي، شغّل خط Audio2Face:\n')
    print(_cd_line(A2F_PIPELINE, shell))
    print(_format_cmd_for_shell(a2f_cmd, shell))
    print('\nملاحظة: إن كان اسم الملف يحتوي على مسافات أو اختلف مجلد النتائج،\nعدّل مسار الـ OBJ في الأمر الثاني يدويًا أو استخدم: meshforge_cli.py pifuhd ثم audio2face.\n')
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='MeshForge — واجهة سطر أوامر لـ PIFuHD وخط Audio2Face.', formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sp = sub.add_parser('pifuhd', help='تحويل صورة إلى OBJ عبر PIFuHD')
    sp.add_argument('-i', '--image', required=True, help='مسار الصورة')
    sp.add_argument('-o', '--out', required=True, help='مجلد نتائج PIFuHD')
    sp.add_argument('-c', '--ckpt', help='مسار pifuhd.pt')
    sp.add_argument('-r', '--resolution', type=int, default=512)
    sp.add_argument('--use-rect', action='store_true')
    sp.set_defaults(func=cmd_pifuhd)
    sp = sub.add_parser('audio2face', help='تجهيز OBJ للـ Audio2Face (تنظيف، blendshapes، تصدير…)')
    sp.add_argument('-m', '--obj', required=True)
    sp.add_argument('-t', '--texture', help='خريطة لون اختيارية')
    sp.add_argument('-a', '--wav', help='ملف WAV للمزامنة')
    sp.add_argument('--out-dir', default=str(A2F_PIPELINE / 'outputs'))
    sp.add_argument('--formats', nargs='+', default=['fbx', 'usd'], choices=['fbx', 'usd'])
    sp.add_argument('--blender', default='blender')
    sp.add_argument('--viewer', action='store_true')
    sp.add_argument('--webcam', action='store_true')
    sp.add_argument('--skip-export', action='store_true')
    sp.set_defaults(func=cmd_audio2face)
    sp = sub.add_parser('full', help='تشغيل المتسلسلة: صورة → OBJ ثم Audio2Face')
    sp.add_argument('-i', '--image', required=True)
    sp.add_argument('-a', '--wav', help='WAV اختياري')
    sp.add_argument('-t', '--texture', help='نسيج اختياري')
    sp.add_argument('--pifuhd-out', default=str(ROOT / 'runs' / 'pifuhd_batch'))
    sp.add_argument('--out-dir', default=None, help='مجلد مخرجات Audio2Face (افتراضي: audio2face_prep داخل pifuhd-out)')
    sp.add_argument('-c', '--ckpt', help='مسار pifuhd.pt')
    sp.add_argument('-r', '--resolution', type=int, default=512)
    sp.add_argument('--use-rect', action='store_true')
    sp.add_argument('--formats', nargs='+', default=['fbx', 'usd'], choices=['fbx', 'usd'])
    sp.add_argument('--blender', default='blender')
    sp.add_argument('--viewer', action='store_true')
    sp.add_argument('--webcam', action='store_true')
    sp.add_argument('--skip-export', action='store_true')
    sp.set_defaults(func=cmd_full)
    sp = sub.add_parser('prepare-cmd', help='طباعة أوامر جاهزة (بدون تنفيذ) لنسخها في الطرفية')
    sp.add_argument('-i', '--image', required=True)
    sp.add_argument('-a', '--wav', help='WAV اختياري في الأمر الثاني')
    sp.add_argument('-t', '--texture', help='نسيج اختياري')
    sp.add_argument('--pifuhd-out', default=str(ROOT / 'runs' / 'pifuhd_batch'))
    sp.add_argument('--out-dir', default=str(ROOT / 'runs' / 'audio2face_out'))
    sp.add_argument('-c', '--ckpt', help='مسار pifuhd.pt')
    sp.add_argument('-r', '--resolution', type=int, default=512)
    sp.add_argument('--use-rect', action='store_true')
    sp.add_argument('--formats', nargs='+', default=['fbx', 'usd'], choices=['fbx', 'usd'])
    sp.add_argument('--blender', default='blender')
    sp.add_argument('--viewer', action='store_true')
    sp.add_argument('--webcam', action='store_true')
    sp.add_argument('--skip-export', action='store_true')
    sp.add_argument('--shell', choices=['powershell', 'bash'], help='تنسيق الاقتباس للنسخ')
    sp.set_defaults(func=cmd_prepare_cmd)
    return p

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.perf_counter()
    try:
        rc = int(args.func(args))
    except KeyboardInterrupt:
        print('\nتم الإيقاف بواسطة المستخدم.')
        return 130
    elapsed = time.perf_counter() - t0
    print(f'\n⏱ المدة الإجمالية: {elapsed:.1f}s')
    return rc
if __name__ == '__main__':
    sys.exit(main())
