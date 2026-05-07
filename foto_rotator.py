#!/usr/bin/env python3
"""
Image Rotator - Dreht Bilder verlustfrei um einstellbare Winkel
Unterstützte Formate: JPG, PNG, TIFF, WebP
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from PIL import Image
import os

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

ROTATION_MAP = {
    90: Image.ROTATE_90,
    180: Image.ROTATE_180,
    270: Image.ROTATE_270,
}


def rotate_image(src: Path, dst_folder: Path, angle: int, quality: int = 100) -> str:
    """Rotiert ein Bild und speichert es im Zielordner.

    quality: 1–100 (JPEG & WebP). 100 = verlustfrei/maximal.
    PNG und TIFF werden immer verlustfrei gespeichert.
    """
    img = Image.open(src)

    # EXIF-Daten sichern (für JPEG)
    exif = img.info.get("exif", b"")

    ext = src.suffix.lower()
    dst = dst_folder / src.name

    try:
        if ext in {".jpg", ".jpeg"} and angle in ROTATION_MAP:
            rotated = img.transpose(ROTATION_MAP[angle])
            # subsampling=0 nur bei quality=100 (echte Lossless-Chroma)
            subsampling = 0 if quality == 100 else -1
            save_kwargs = {"quality": quality, "subsampling": subsampling}
            if exif:
                save_kwargs["exif"] = exif
            rotated.save(dst, **save_kwargs)
        else:
            rotated = img.rotate(-angle, expand=True)
            if ext in {".tif", ".tiff"}:
                rotated.save(dst, compression="tiff_lzw")
            elif ext == ".webp":
                lossless = quality == 100
                rotated.save(dst, lossless=lossless, quality=quality)
            else:
                # PNG: verlustfrei, quality wird ignoriert
                rotated.save(dst)
        return f"✓ {src.name}"
    except Exception as e:
        return f"✗ {src.name}: {e}"


# ── GUI ──────────────────────────────────────────────────────────────────────

class ImageRotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bild-Rotator")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._build_styles()
        self._build_ui()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        BG = "#1a1a2e"
        PANEL = "#16213e"
        ACCENT = "#e94560"
        FG = "#eaeaea"
        ENTRY = "#0f3460"
        SELBG = "#e94560"
        SELFG = "#ffffff"

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG,
                        font=("Courier New", 10))
        style.configure("Head.TLabel", background=BG, foreground=ACCENT,
                        font=("Courier New", 15, "bold"))
        style.configure("Sub.TLabel", background=PANEL, foreground=FG,
                        font=("Courier New", 10))
        style.configure("TEntry", fieldbackground=ENTRY, foreground=FG,
                        insertcolor=FG, bordercolor=ACCENT, relief="flat")
        style.configure("Accent.TButton",
                        background=ACCENT, foreground=SELFG,
                        font=("Courier New", 10, "bold"),
                        borderwidth=0, focuscolor=ACCENT)
        style.map("Accent.TButton",
                  background=[("active", "#c73652"), ("disabled", "#555")])

        style.configure("TRadiobutton", background=PANEL, foreground=FG,
                        font=("Courier New", 11),
                        indicatorcolor=ACCENT, selectcolor=PANEL)
        style.map("TRadiobutton",
                  foreground=[("selected", ACCENT)],
                  indicatorcolor=[("selected", ACCENT)])

        style.configure("TProgressbar", troughcolor=PANEL,
                        background=ACCENT, bordercolor=PANEL)

        self._colors = {"BG": BG, "PANEL": PANEL, "ACCENT": ACCENT,
                        "FG": FG, "ENTRY": ENTRY}

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        C = self._colors
        pad = {"padx": 18, "pady": 8}

        # ── Titel
        ttk.Label(self, text="⟳  BILD-ROTATOR", style="Head.TLabel") \
            .grid(row=0, column=0, **pad, sticky="w")
        ttk.Label(self, text="Verlustfreie Rotation · JPG · PNG · TIFF · WebP",
                  style="TLabel", foreground="#888") \
            .grid(row=1, column=0, padx=18, pady=0, sticky="w")

        sep = tk.Frame(self, height=1, bg=C["ACCENT"])
        sep.grid(row=2, column=0, sticky="ew", padx=18, pady=6)

        # ── Panel
        panel = ttk.Frame(self, style="Panel.TFrame", padding=14)
        panel.grid(row=3, column=0, padx=18, pady=4, sticky="ew")
        panel.columnconfigure(1, weight=1)

        # Eingabeordner
        ttk.Label(panel, text="Eingabeordner:", style="Sub.TLabel") \
            .grid(row=0, column=0, sticky="w", pady=4)
        self.src_var = tk.StringVar()
        ttk.Entry(panel, textvariable=self.src_var, width=42) \
            .grid(row=0, column=1, padx=6)
        ttk.Button(panel, text="…", style="Accent.TButton", width=3,
                   command=self._choose_src) \
            .grid(row=0, column=2)

        # Ausgabeordner
        ttk.Label(panel, text="Ausgabeordner:", style="Sub.TLabel") \
            .grid(row=1, column=0, sticky="w", pady=4)
        self.dst_var = tk.StringVar()
        ttk.Entry(panel, textvariable=self.dst_var, width=42) \
            .grid(row=1, column=1, padx=6)
        ttk.Button(panel, text="…", style="Accent.TButton", width=3,
                   command=self._choose_dst) \
            .grid(row=1, column=2)

        # Winkel
        ttk.Label(panel, text="Drehwinkel:", style="Sub.TLabel") \
            .grid(row=2, column=0, sticky="w", pady=8)
        angle_frame = ttk.Frame(panel, style="Panel.TFrame")
        angle_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        self.angle_var = tk.IntVar(value=90)
        for angle in (90, 180, 270):
            ttk.Radiobutton(angle_frame, text=f"{angle}°",
                            variable=self.angle_var, value=angle) \
                .pack(side="left", padx=10)

        # Qualität
        ttk.Label(panel, text="Qualität:", style="Sub.TLabel") \
            .grid(row=3, column=0, sticky="w", pady=6)
        quality_frame = ttk.Frame(panel, style="Panel.TFrame")
        quality_frame.grid(row=3, column=1, columnspan=2, sticky="ew")

        self.quality_var = tk.IntVar(value=100)
        self.quality_label = tk.Label(quality_frame, text="100  (verlustfrei)",
                                      bg=C["PANEL"], fg=C["ACCENT"],
                                      font=("Courier New", 10, "bold"), width=18, anchor="w")
        self.quality_label.pack(side="right", padx=6)

        tk.Scale(quality_frame, from_=1, to=100, orient="horizontal",
                 variable=self.quality_var, length=220,
                 bg=C["PANEL"], fg=C["FG"], troughcolor=C["ENTRY"],
                 activebackground=C["ACCENT"], highlightthickness=0, bd=0,
                 showvalue=False, command=self._on_quality_change) \
            .pack(side="left")

        ttk.Label(panel, text="ℹ  PNG & TIFF sind immer verlustfrei (Qualität wird ignoriert).",
                  style="Sub.TLabel", foreground="#666", font=("Courier New", 8)) \
            .grid(row=4, column=0, columnspan=3, sticky="w", pady=0)

        # Unterordner einschließen
        self.recursive_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="Unterordner einschließen",
                       variable=self.recursive_var,
                       bg=C["PANEL"], fg=C["FG"], selectcolor=C["PANEL"],
                       activebackground=C["PANEL"], activeforeground=C["ACCENT"],
                       font=("Courier New", 10)) \
            .grid(row=5, column=0, columnspan=3, sticky="w", pady=4)

        # ── Start-Button
        self.btn_start = ttk.Button(self, text="▶  ROTATION STARTEN",
                                    style="Accent.TButton",
                                    command=self._start)
        self.btn_start.grid(row=4, column=0, padx=18, pady=10, sticky="ew")

        # ── Fortschritt
        self.progress = ttk.Progressbar(self, mode="determinate", length=420)
        self.progress.grid(row=5, column=0, padx=18, pady=2, sticky="ew")

        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(self, textvariable=self.status_var, style="TLabel",
                  foreground="#aaa") \
            .grid(row=6, column=0, padx=18, pady=2, sticky="w")

        # ── Log
        log_frame = tk.Frame(self, bg=C["PANEL"], bd=0)
        log_frame.grid(row=7, column=0, padx=18, pady=8, sticky="ew")

        self.log = tk.Text(log_frame, width=58, height=10,
                           bg=C["PANEL"], fg=C["FG"],
                           font=("Courier New", 9),
                           relief="flat", state="disabled",
                           insertbackground=C["FG"])
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.columnconfigure(0, weight=1)

    # ── Qualitäts-Callback ────────────────────────────────────────────────────

    def _on_quality_change(self, val):
        q = int(float(val))
        if q == 100:
            label = "100  (verlustfrei)"
        elif q >= 85:
            label = f"{q}   (sehr gut)"
        elif q >= 70:
            label = f"{q}   (gut)"
        elif q >= 50:
            label = f"{q}   (mittel)"
        else:
            label = f"{q}   (niedrig)"
        self.quality_label.configure(text=label)

    # ── Ordnerauswahl ─────────────────────────────────────────────────────────

    def _choose_src(self):
        d = filedialog.askdirectory(title="Eingabeordner wählen")
        if d:
            self.src_var.set(d)

    def _choose_dst(self):
        d = filedialog.askdirectory(title="Ausgabeordner wählen")
        if d:
            self.dst_var.set(d)

    # ── Verarbeitung ──────────────────────────────────────────────────────────

    def _start(self):
        src = self.src_var.get().strip()
        dst = self.dst_var.get().strip()
        angle = self.angle_var.get()

        if not src or not Path(src).is_dir():
            messagebox.showerror("Fehler", "Bitte einen gültigen Eingabeordner wählen.")
            return
        if not dst:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen.")
            return

        Path(dst).mkdir(parents=True, exist_ok=True)

        # Dateien sammeln
        src_path = Path(src)
        if self.recursive_var.get():
            files = [f for f in src_path.rglob("*")
                     if f.suffix.lower() in SUPPORTED]
        else:
            files = [f for f in src_path.iterdir()
                     if f.suffix.lower() in SUPPORTED]

        if not files:
            messagebox.showinfo("Keine Dateien",
                                "Keine unterstützten Bilder im Eingabeordner gefunden.")
            return

        self.btn_start.configure(state="disabled")
        self._log_clear()
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0

        threading.Thread(target=self._worker,
                         args=(files, Path(dst), angle, self.quality_var.get()),
                         daemon=True).start()

    def _worker(self, files, dst_folder, angle, quality):
        ok = err = 0
        for i, f in enumerate(files, 1):
            result = rotate_image(f, dst_folder, angle, quality)
            if result.startswith("✓"):
                ok += 1
            else:
                err += 1
            self.after(0, self._log_append, result)
            self.after(0, self._set_progress, i,
                       f"Verarbeite {i}/{len(files)}: {f.name}")

        summary = f"\nFertig! ✓ {ok} erfolgreich"
        if err:
            summary += f"  ✗ {err} Fehler"
        self.after(0, self._log_append, summary)
        self.after(0, self._finish, summary.strip())

    # ── UI-Updates (Thread-sicher) ─────────────────────────────────────────────

    def _log_append(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_progress(self, value, status):
        self.progress["value"] = value
        self.status_var.set(status)

    def _finish(self, status):
        self.status_var.set(status)
        self.btn_start.configure(state="normal")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "Pillow", "-q"])
        from PIL import Image

    app = ImageRotatorApp()
    app.mainloop()