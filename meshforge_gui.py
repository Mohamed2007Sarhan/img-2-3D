from __future__ import annotations
import io
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meshforge_cli as mf
BG = '#16161e'
SURFACE = '#24283b'
TEXT = '#c0caf5'
ACCENT = '#7aa2f7'
OK = '#9ece6a'
WARN = '#e0af68'

class LogWriter(io.TextIOBase):

    def __init__(self, q: queue.Queue[str], tag: str='stdout') -> None:
        self._q = q
        self._tag = tag

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:
        pass

class MeshForgeApp(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title('MeshForge Studio · صورة → 3D → Audio2Face')
        self.geometry('920x640')
        self.configure(bg=BG)
        self._log_q: queue.Queue[str] = queue.Queue()
        self._busy = tk.BooleanVar(value=False)
        self._build_styles()
        self._init_vars()
        self._build_ui()
        self.after(120, self._drain_log_queue)

    def _init_vars(self) -> None:
        self.var_image = tk.StringVar()
        self.var_wav = tk.StringVar()
        self.var_texture = tk.StringVar()
        self.var_ckpt = tk.StringVar()
        self.var_pifu_out = tk.StringVar(value=str(mf.ROOT / 'runs' / 'pifuhd_batch'))
        self.var_a2f_out = tk.StringVar(value=str(mf.ROOT / 'runs' / 'audio2face_out'))
        self.var_blender = tk.StringVar(value='blender')
        self.var_res = tk.IntVar(value=512)
        self.var_use_rect = tk.BooleanVar(value=False)
        self.var_viewer = tk.BooleanVar(value=False)
        self.var_webcam = tk.BooleanVar(value=False)
        self.var_skip_export = tk.BooleanVar(value=False)
        self.var_fmt_fbx = tk.BooleanVar(value=True)
        self.var_fmt_usd = tk.BooleanVar(value=True)
        self.var_shell = tk.StringVar(value='powershell' if __import__('os').name == 'nt' else 'bash')

    def _build_styles(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use('clam')
        except tk.TclError:
            pass
        s.configure('.', background=BG, foreground=TEXT, fieldbackground=SURFACE)
        s.configure('TLabel', background=BG, foreground=TEXT)
        s.configure('TFrame', background=BG)
        s.configure('TLabelframe', background=BG, foreground=ACCENT)
        s.configure('TLabelframe.Label', background=BG, foreground=ACCENT)
        s.configure('TNotebook', background=BG)
        s.configure('TNotebook.Tab', background=SURFACE, foreground=TEXT, padding=[12, 6])
        s.map('TNotebook.Tab', background=[('selected', SURFACE)], foreground=[('selected', ACCENT)])
        s.configure('TButton', background=SURFACE, foreground=TEXT, padding=8)
        s.map('TButton', background=[('active', '#414868')])
        s.configure('TCheckbutton', background=BG, foreground=TEXT)
        s.configure('TRadiobutton', background=BG, foreground=TEXT)
        s.configure('TEntry', fieldbackground=SURFACE, foreground=TEXT)

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, browse: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=var)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        def pick() -> None:
            if browse == 'file':
                p = filedialog.askopenfilename()
            elif browse == 'wav':
                p = filedialog.askopenfilename(filetypes=[('WAV', '*.wav'), ('All', '*.*')])
            elif browse == 'image':
                p = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.webp'), ('All', '*.*')])
            else:
                p = filedialog.askdirectory()
            if p:
                var.set(p)
        ttk.Button(row, text='…', width=3, command=pick).pack(side=tk.LEFT)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._paths_block(outer)
        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.tab_studio = ttk.Frame(nb)
        self.tab_cmd = ttk.Frame(nb)
        nb.add(self.tab_studio, text='  الاستوديو الكامل  ')
        nb.add(self.tab_cmd, text='  منشئ الأوامر  ')
        self._studio_tab(self.tab_studio)
        self._cmd_tab(self.tab_cmd)
        foot = ttk.Frame(outer)
        foot.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(foot, text=f'PIFuHD: {mf.PIFUHD_APP}  |  Audio2Face: {mf.A2F_PIPELINE}', font=('Segoe UI', 8)).pack(side=tk.LEFT)

    def _paths_block(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=' المسارات والخيارات ')
        lf.pack(fill=tk.X, pady=2)
        self._path_row(lf, 'الصورة', self.var_image, 'image')
        self._path_row(lf, 'صوت WAV', self.var_wav, 'wav')
        self._path_row(lf, 'نسيج (اختياري)', self.var_texture, 'file')
        self._path_row(lf, 'checkpoint pifuhd.pt', self.var_ckpt, 'file')
        self._path_row(lf, 'مخرجات PIFuHD', self.var_pifu_out, 'dir')
        self._path_row(lf, 'مخرجات Audio2Face', self.var_a2f_out, 'dir')
        self._path_row(lf, 'Blender', self.var_blender, 'file')
        row = ttk.Frame(lf)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text='دقة الشبكة:').pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=256, to=1024, increment=256, textvariable=self.var_res, width=8).pack(side=tk.LEFT, padx=6)
        row2 = ttk.Frame(lf)
        row2.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(row2, text='استخدام مستطيل القص', variable=self.var_use_rect).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row2, text='Viewer بعد الانتهاء', variable=self.var_viewer).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row2, text='كاميرا ويب', variable=self.var_webcam).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row2, text='تخطي التصدير', variable=self.var_skip_export).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row2, text='FBX', variable=self.var_fmt_fbx).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(row2, text='USD', variable=self.var_fmt_usd).pack(side=tk.LEFT, padx=4)

    def _studio_tab(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=10)
        ttk.Button(bar, text='تشغيل المسار الكامل (صورة → OBJ → Audio2Face)', command=self._run_full).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text='PIFuHD فقط', command=self._run_pifu_only).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text='Audio2Face فقط (اختر OBJ جاهز)', command=self._run_a2f_pick).pack(side=tk.LEFT, padx=4)
        self.prog = ttk.Progressbar(parent, mode='indeterminate')
        self.prog.pack(fill=tk.X, pady=4)
        log_f = ttk.LabelFrame(parent, text=' السجل ')
        log_f.pack(fill=tk.BOTH, expand=True, pady=6)
        self.txt_log = tk.Text(log_f, height=14, bg=SURFACE, fg=TEXT, insertbackground=TEXT, font=('Consolas', 10), wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log(mf._banner() + '\n', OK)

    def _cmd_tab(self, parent: ttk.Frame) -> None:
        sh = ttk.LabelFrame(parent, text=' صيغة الأمر ')
        sh.pack(fill=tk.X, pady=6)
        ttk.Radiobutton(sh, text='PowerShell', variable=self.var_shell, value='powershell').pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(sh, text='Bash', variable=self.var_shell, value='bash').pack(side=tk.LEFT, padx=8)
        ttk.Button(sh, text='توليد الأوامر', command=self._gen_commands).pack(side=tk.RIGHT, padx=6)
        ttk.Button(sh, text='نسخ للحافظة', command=self._copy_cmd).pack(side=tk.RIGHT, padx=6)
        self.txt_cmd = tk.Text(parent, height=16, bg='#1e2030', fg=OK, font=('Consolas', 10), wrap=tk.WORD)
        self.txt_cmd.pack(fill=tk.BOTH, expand=True, pady=8)

    def _log(self, msg: str, color: str | None=None) -> None:
        self.txt_log.insert(tk.END, msg, color)
        if color:
            self.txt_log.tag_config(color, foreground=color)
        self.txt_log.see(tk.END)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                chunk = self._log_q.get_nowait()
                self._log(chunk)
        except queue.Empty:
            pass
        self.after(120, self._drain_log_queue)

    def _formats(self) -> list[str]:
        f: list[str] = []
        if self.var_fmt_fbx.get():
            f.append('fbx')
        if self.var_fmt_usd.get():
            f.append('usd')
        return f or ['fbx']

    def _ckpt_path(self) -> Path | None:
        s = self.var_ckpt.get().strip()
        return Path(s) if s else None

    def _wrap_busy(self, fn) -> None:
        if self._busy.get():
            messagebox.showinfo('MeshForge', 'هناك عملية قيد التشغيل بالفعل.')
            return
        img = self.var_image.get().strip()
        if not img:
            messagebox.showerror('MeshForge', 'اختر صورة أولًا.')
            return
        self._busy.set(True)
        self.prog.start(14)

        def task() -> None:
            (old_out, old_err) = (sys.stdout, sys.stderr)
            (lw_out, lw_err) = (LogWriter(self._log_q), LogWriter(self._log_q))
            (sys.stdout, sys.stderr) = (lw_out, lw_err)
            try:
                fn()
            finally:
                (sys.stdout, sys.stderr) = (old_out, old_err)
                self.after(0, self._idle)
        threading.Thread(target=task, daemon=True).start()

    def _idle(self) -> None:
        self.prog.stop()
        self._busy.set(False)

    def _run_full(self) -> None:

        def inner() -> None:
            image = Path(self.var_image.get().strip())
            pifu_out = Path(self.var_pifu_out.get().strip())
            print(f'\n━━━ PIFuHD ━━━\n')
            obj = mf.run_pifuhd(image, pifu_out, ckpt=self._ckpt_path(), resolution=int(self.var_res.get()), use_rect=self.var_use_rect.get())
            if not obj:
                print('\nتوقف: لم يُعثر على OBJ.\n')
                return
            print(f'\n✓ OBJ: {obj}\n')
            a2f_out = Path(self.var_a2f_out.get().strip())
            mf._ensure_dir(a2f_out)
            wav = Path(self.var_wav.get().strip()) if self.var_wav.get().strip() else None
            tex = Path(self.var_texture.get().strip()) if self.var_texture.get().strip() else None
            cmd = mf.build_audio2face_cmd(obj, texture=tex, wav=wav, out_dir=a2f_out, formats=self._formats(), blender=self.var_blender.get().strip() or 'blender', viewer=self.var_viewer.get(), webcam=self.var_webcam.get(), skip_export=self.var_skip_export.get())
            print('\n━━━ Audio2Face prep ━━━\n')
            rc = mf.run_audio2face_cmd(cmd)
            print(f'\nانتهى برمز الخروج: {rc}\n')
        self._wrap_busy(inner)

    def _run_pifu_only(self) -> None:

        def inner() -> None:
            image = Path(self.var_image.get().strip())
            pifu_out = Path(self.var_pifu_out.get().strip())
            obj = mf.run_pifuhd(image, pifu_out, ckpt=self._ckpt_path(), resolution=int(self.var_res.get()), use_rect=self.var_use_rect.get())
            if obj:
                print(f'\n✓ OBJ: {obj}\n')
        self._wrap_busy(inner)

    def _run_a2f_pick(self) -> None:
        p = filedialog.askopenfilename(filetypes=[('OBJ', '*.obj'), ('All', '*.*')])
        if not p:
            return

        def inner() -> None:
            a2f_out = Path(self.var_a2f_out.get().strip())
            mf._ensure_dir(a2f_out)
            wav = Path(self.var_wav.get().strip()) if self.var_wav.get().strip() else None
            tex = Path(self.var_texture.get().strip()) if self.var_texture.get().strip() else None
            cmd = mf.build_audio2face_cmd(Path(p), texture=tex, wav=wav, out_dir=a2f_out, formats=self._formats(), blender=self.var_blender.get().strip() or 'blender', viewer=self.var_viewer.get(), webcam=self.var_webcam.get(), skip_export=self.var_skip_export.get())
            mf.run_audio2face_cmd(cmd)
        self._wrap_busy(inner)

    def _guess_obj(self) -> Path:
        stem = Path(self.var_image.get().strip()).stem
        pifu_out = Path(self.var_pifu_out.get().strip())
        guess: Path = pifu_out / 'pifuhd_final' / 'recon' / f'result_{stem}_{int(self.var_res.get())}.obj'
        for cand in mf.find_pifuhd_outputs(pifu_out.resolve()):
            if stem in cand.name:
                return cand
        return guess

    def _gen_commands(self) -> None:
        if not self.var_image.get().strip():
            messagebox.showerror('MeshForge', 'اختر صورة لتقدير المسارات.')
            return
        image = Path(self.var_image.get().strip())
        pifu_out = Path(self.var_pifu_out.get().strip())
        a2f_out = Path(self.var_a2f_out.get().strip())
        shell = self.var_shell.get()
        pifu_cmd = [mf._which_python(), '-m', 'apps.simple_test', '-i', str(image.parent.resolve()), '-o', str(pifu_out.resolve()), '-r', str(int(self.var_res.get()))]
        ckpt = self._ckpt_path()
        if ckpt:
            pifu_cmd.extend(['-c', str(ckpt.resolve())])
        if self.var_use_rect.get():
            pifu_cmd.append('--use_rect')
        guess = self._guess_obj()
        wav = Path(self.var_wav.get().strip()) if self.var_wav.get().strip() else None
        tex = Path(self.var_texture.get().strip()) if self.var_texture.get().strip() else None
        a2f_cmd = mf.build_audio2face_cmd(guess, texture=tex, wav=wav, out_dir=a2f_out, formats=self._formats(), blender=self.var_blender.get().strip() or 'blender', viewer=self.var_viewer.get(), webcam=self.var_webcam.get(), skip_export=self.var_skip_export.get())
        block = []
        block.append('# === 1) PIFuHD: صورة → OBJ ===')
        block.append(mf._cd_line(mf.PIFUHD_APP, shell))
        block.append(mf._format_cmd_for_shell(pifu_cmd, shell))
        block.append('')
        block.append(f'# تلميح: OBJ المتوقع (راجع المجلد بعد التشغيل): {guess}')
        block.append('# === 2) تجهيز Audio2Face ===')
        block.append(mf._cd_line(mf.A2F_PIPELINE, shell))
        block.append(mf._format_cmd_for_shell(a2f_cmd, shell))
        block.append('')
        block.append('# أو استخدم الـ CLI الموحد:')
        block.append(mf._format_cmd_for_shell([mf._which_python(), str(mf.ROOT / 'meshforge_cli.py'), 'full', '-i', str(image.resolve()), *(['-a', str(wav.resolve())] if wav else []), *(['-t', str(tex.resolve())] if tex else []), '--pifuhd-out', str(pifu_out.resolve()), '--out-dir', str(a2f_out.resolve()), '-r', str(int(self.var_res.get())), *(['-c', str(ckpt.resolve())] if ckpt else []), *(['--use-rect'] if self.var_use_rect.get() else []), *(['--viewer'] if self.var_viewer.get() else []), *(['--webcam'] if self.var_webcam.get() else []), *(['--skip-export'] if self.var_skip_export.get() else []), '--formats', *self._formats(), '--blender', self.var_blender.get().strip() or 'blender'], shell))
        self.txt_cmd.delete('1.0', tk.END)
        self.txt_cmd.insert('1.0', '\n'.join(block))

    def _copy_cmd(self) -> None:
        t = self.txt_cmd.get('1.0', tk.END).strip()
        if not t:
            messagebox.showinfo('MeshForge', 'لا يوجد نص؛ اضغط «توليد الأوامر» أولًا.')
            return
        self.clipboard_clear()
        self.clipboard_append(t)
        messagebox.showinfo('MeshForge', 'تم النسخ إلى الحافظة.')

def main() -> None:
    app = MeshForgeApp()
    app.mainloop()
if __name__ == '__main__':
    main()
