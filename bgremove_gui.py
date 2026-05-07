#!/usr/bin/env python3
"""
remove_background_gui.py
========================
Tkinter-GUI für Hintergrund-Entfernung mit Vorschau-Panels.

Abhängigkeiten:
    pip install rembg onnxruntime pillow numpy

Starten:
    python remove_background_gui.py
"""

import sys
import threading
import time
import queue
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Optionale Abhängigkeiten ─────────────────────────────────
try:
    from PIL import Image, ImageTk, ImageDraw
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from rembg import remove, new_session
    REMBG_OK = True
except ImportError:
    REMBG_OK = False


# ── Konstanten ───────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
PREVIEW_W = 480   # Breite der Vorschau-Panels in Pixeln
PREVIEW_H = 360   # Höhe der Vorschau-Panels in Pixeln

MODELS = {
    "u2net":             ("u2net",            "Allgemein — gute Balance (Standard)"),
    "u2net_human_seg":   ("u2net_human_seg",  "Porträts & Personen — optimiert"),
    "isnet-general-use": ("isnet-general-use","Höchste Qualität — etwas langsamer"),
    "isnet-anime":       ("isnet-anime",      "Anime / Zeichentrick"),
    "silueta":           ("silueta",          "Schnell & leicht"),
}

# ── Design-Tokens ────────────────────────────────────────────
C = {
    "bg":        "#0f0f13",
    "surface":   "#1a1a24",
    "panel":     "#22222f",
    "border":    "#2e2e42",
    "accent":    "#7c6af7",
    "accent2":   "#a78bfa",
    "success":   "#34d399",
    "warning":   "#fbbf24",
    "error":     "#f87171",
    "text":      "#e2e0f0",
    "muted":     "#7070a0",
    "hover":     "#2a2a3a",
}

FONT_TITLE  = ("Segoe UI", 16, "bold")
FONT_HEADER = ("Segoe UI", 10, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


# ─────────────────────────────────────────────────────────────
# Core processing helpers
# ─────────────────────────────────────────────────────────────

def checkerboard_background(width, height, tile=20):
    bg = Image.new("RGB", (width, height), (40, 40, 55))
    draw = ImageDraw.Draw(bg)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle([x, y, x+tile-1, y+tile-1], fill=(28, 28, 42))
    return bg


def fit_image(img, max_w, max_h):
    """Skaliert ein Bild proportional, sodass es in max_w x max_h passt."""
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def process_image(input_path, output_dir, session,
                  alpha_matting=True, fg_thresh=20, bg_thresh=10, er_size=15,
                  log_fn=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    def log(msg):
        if log_fn:
            log_fn(msg)

    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    log(f"  Größe: {w} x {h} px")

    t0 = time.time()
    result = remove(
        img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_thresh,
        alpha_matting_background_threshold=bg_thresh,
        alpha_matting_erode_size=er_size,
        post_process_mask=True,
    )
    elapsed = time.time() - t0
    log(f"  Verarbeitung: {elapsed:.1f} s")

    nobg_path = output_dir / f"{stem}_nobg.png"
    result.save(nobg_path, format="PNG")
    log(f"  Freigestellt: {nobg_path.name}")

    alpha_ch = result.split()[3]
    mask_path = output_dir / f"{stem}_mask.png"
    alpha_ch.save(mask_path, format="PNG")
    log(f"  Maske:        {mask_path.name}")

    checker = checkerboard_background(w, h)
    checker.paste(result, mask=alpha_ch)
    preview_path = output_dir / f"{stem}_preview.png"
    checker.save(preview_path, format="PNG")
    log(f"  Vorschau:     {preview_path.name}")

    alpha_arr = np.array(alpha_ch)
    fg_pct = float((alpha_arr > 128).sum()) / alpha_arr.size * 100

    return {
        "nobg": nobg_path,
        "mask": mask_path,
        "preview": preview_path,
        "result_img": result,
        "preview_img": checker,
        "width": w, "height": h,
        "foreground_pct": fg_pct,
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Styled Widget Helpers
# ─────────────────────────────────────────────────────────────

def styled_button(parent, text, command, accent=False, small=False, **kw):
    bg   = C["accent"] if accent else C["panel"]
    fg   = "#ffffff"   if accent else C["text"]
    abg  = C["accent2"] if accent else C["hover"]
    font = FONT_SMALL if small else FONT_BODY
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
        relief="flat", bd=0, cursor="hand2", font=font,
        padx=12 if not small else 8,
        pady=6  if not small else 4,
        **kw
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=abg))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


def separator(parent, pady=8):
    tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=pady)


def label(parent, text, font=FONT_BODY, color=None, **kw):
    return tk.Label(parent, text=text, bg=C["bg"],
                    fg=color or C["text"], font=font, **kw)


# ─────────────────────────────────────────────────────────────
# Preview Panel Widget
# ─────────────────────────────────────────────────────────────

class PreviewPanel(tk.Frame):
    def __init__(self, parent, title, placeholder_text, pw=PREVIEW_W, ph=PREVIEW_H):
        super().__init__(parent, bg=C["bg"])
        self._pw = pw
        self._ph = ph
        self._tk_img = None
        self._img_item = None

        # Title row
        hdr = tk.Frame(self, bg=C["bg"])
        hdr.pack(fill="x", pady=(0, 3))
        tk.Label(hdr, text=title, bg=C["bg"], fg=C["text"],
                 font=FONT_HEADER).pack(side="left")
        self._badge = tk.Label(hdr, text="", bg=C["bg"],
                               fg=C["muted"], font=FONT_SMALL)
        self._badge.pack(side="right")

        # Canvas
        self._canvas = tk.Canvas(
            self, width=pw, height=ph,
            bg=C["surface"],
            highlightthickness=1,
            highlightbackground=C["border"],
        )
        self._canvas.pack()

        # Placeholder
        self._ph_item = self._canvas.create_text(
            pw // 2, ph // 2,
            text=placeholder_text,
            fill=C["muted"], font=("Segoe UI", 10),
            justify="center",
        )

        # Info label below canvas
        self._info = tk.Label(self, text="", bg=C["bg"],
                              fg=C["muted"], font=FONT_SMALL, anchor="w")
        self._info.pack(fill="x", pady=(2, 0))

    def show(self, img, info="", badge=""):
        fitted = fit_image(img.copy(), self._pw, self._ph)
        fw, fh = fitted.size
        ox = (self._pw - fw) // 2
        oy = (self._ph - fh) // 2

        self._tk_img = ImageTk.PhotoImage(fitted)
        self._canvas.itemconfig(self._ph_item, state="hidden")
        if self._img_item:
            self._canvas.delete(self._img_item)
        self._img_item = self._canvas.create_image(ox, oy, anchor="nw",
                                                   image=self._tk_img)
        self._info.config(text=info)
        self._badge.config(text=badge)

    def clear(self, placeholder_text=None):
        if self._img_item:
            self._canvas.delete(self._img_item)
            self._img_item = None
        self._tk_img = None
        if placeholder_text:
            self._canvas.itemconfig(self._ph_item, text=placeholder_text)
        self._canvas.itemconfig(self._ph_item, state="normal")
        self._info.config(text="")
        self._badge.config(text="")


# ─────────────────────────────────────────────────────────────
# File list item
# ─────────────────────────────────────────────────────────────

class FileItem(tk.Frame):
    STATUS_COLORS = {
        "waiting":    C["muted"],
        "processing": C["warning"],
        "done":       C["success"],
        "error":      C["error"],
    }

    def __init__(self, parent, path, select_cb, remove_cb):
        super().__init__(parent, bg=C["panel"], pady=4, padx=8)
        self.path = path
        self.status = "waiting"
        self._selected = False

        self.dot = tk.Label(self, text="●", bg=C["panel"],
                            fg=C["muted"], font=("Segoe UI", 10))
        self.dot.pack(side="left", padx=(0, 6))

        name = path.name if len(path.name) <= 36 else "..." + path.name[-34:]
        self._name_lbl = tk.Label(self, text=name, bg=C["panel"],
                                  fg=C["text"], font=FONT_BODY, anchor="w")
        self._name_lbl.pack(side="left", fill="x", expand=True)

        try:
            sz = path.stat().st_size
            sz_str = f"{sz/1024:.0f} KB" if sz < 1048576 else f"{sz/1048576:.1f} MB"
        except Exception:
            sz_str = ""
        tk.Label(self, text=sz_str, bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left", padx=6)

        self.status_lbl = tk.Label(self, text="Wartend", bg=C["panel"],
                                   fg=C["muted"], font=FONT_SMALL, width=11)
        self.status_lbl.pack(side="left", padx=2)

        rm = tk.Button(self, text="x", bg=C["panel"], fg=C["muted"],
                       activebackground=C["hover"], activeforeground=C["error"],
                       relief="flat", bd=0, cursor="hand2", font=FONT_SMALL,
                       command=lambda: remove_cb(self))
        rm.pack(side="right")

        self.pack(fill="x", pady=2)

        for w in (self, self.dot, self._name_lbl, self.status_lbl):
            w.bind("<Button-1>", lambda e, s=self: select_cb(s))
        self._setup_hover()

    def _setup_hover(self):
        def on_enter(e):
            c = "#2a2040" if self._selected else C["hover"]
            self._set_bg(c)
        def on_leave(e):
            c = "#2a2040" if self._selected else C["panel"]
            self._set_bg(c)
        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)

    def _set_bg(self, c):
        self.config(bg=c)
        for w in self.winfo_children():
            try: w.config(bg=c)
            except Exception: pass

    def set_selected(self, selected):
        self._selected = selected
        self._set_bg("#2a2040" if selected else C["panel"])

    def set_status(self, status, text=None):
        self.status = status
        color = self.STATUS_COLORS.get(status, C["muted"])
        self.dot.config(fg=color)
        self.status_lbl.config(fg=color, text=text or status.capitalize())


# ─────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BG Remover")
        self.geometry("1480x900")
        self.minsize(1200, 720)
        self.configure(bg=C["bg"])

        self.file_items = []
        self._selected_item = None
        self._results = {}   # path -> result dict

        self.output_dir    = tk.StringVar(value=str(Path.home() / "Desktop" / "BG_Removed"))
        self.model_var     = tk.StringVar(value="u2net")
        self.matting_var   = tk.BooleanVar(value=True)
        self.fg_var        = tk.IntVar(value=20)
        self.bg_var        = tk.IntVar(value=10)
        self.running       = False
        self._session      = None
        self._session_model = None
        self._log_queue    = queue.Queue()

        self._build_ui()
        self._poll_log()

        if not PIL_OK:
            self._log("FEHLER: Pillow fehlt  ->  pip install pillow numpy", "error")
        if not REMBG_OK:
            self._log("FEHLER: rembg fehlt  ->  pip install rembg onnxruntime", "error")

    # ─────────────────────────────────────────────────────────
    # UI Build
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self, bg=C["surface"], pady=11)
        hdr.pack(fill="x")
        tk.Label(hdr, text="BG Remover", bg=C["surface"],
                 fg=C["accent2"], font=FONT_TITLE).pack(side="left", padx=20)
        tk.Label(hdr, text="Hintergrund-Entfernung in hoher Qualität",
                 bg=C["surface"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left", padx=6)

        # Main 3-column layout
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=14, pady=10)

        col_left  = tk.Frame(main, bg=C["bg"], width=300)
        col_left.pack(side="left", fill="y", padx=(0, 12))
        col_left.pack_propagate(False)

        col_mid   = tk.Frame(main, bg=C["bg"])
        col_mid.pack(side="left", fill="y", padx=(0, 12))

        col_right = tk.Frame(main, bg=C["bg"])
        col_right.pack(side="left", fill="both", expand=True)

        self._build_file_panel(col_left)
        self._build_input_preview(col_mid)
        self._build_result_panel(col_right)
        self._build_log()

    # ── Left: file list ───────────────────────────────────────

    def _build_file_panel(self, parent):
        # Button row
        btn_row = tk.Frame(parent, bg=C["bg"])
        btn_row.pack(fill="x", pady=(0, 6))
        label(btn_row, "Bilder", font=FONT_HEADER).pack(side="left")
        styled_button(btn_row, "+ Dateien", self._add_files,
                      accent=True, small=True).pack(side="right", padx=(4,0))
        styled_button(btn_row, "Ordner", self._add_folder,
                      small=True).pack(side="right", padx=(4,0))
        styled_button(btn_row, "Leeren", self._clear_list,
                      small=True).pack(side="right")

        # File list area
        list_outer = tk.Frame(parent, bg=C["surface"])
        list_outer.pack(fill="both", expand=True)

        self.drop_hint = tk.Frame(list_outer, bg=C["surface"])
        self.drop_hint.place(relx=0.5, rely=0.38, anchor="center")
        tk.Label(self.drop_hint, text="📂", bg=C["surface"],
                 font=("Segoe UI", 26)).pack()
        tk.Label(self.drop_hint, text="Bilder hinzufuegen\nüber die Schaltflaechen",
                 bg=C["surface"], fg=C["muted"],
                 font=FONT_SMALL, justify="center").pack(pady=2)

        canvas = tk.Canvas(list_outer, bg=C["surface"],
                           highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
        self.file_list_frame = tk.Frame(canvas, bg=C["surface"])
        self.file_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.file_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")

        self.start_btn = styled_button(
            parent, "  Start  ", self._start_processing, accent=True
        )
        self.start_btn.pack(fill="x", pady=(8, 0), ipady=5)

    # ── Middle: input preview ────────────────────────────────

    def _build_input_preview(self, parent):
        self.pv_in = PreviewPanel(
            parent,
            title="Eingangsbild",
            placeholder_text="Bild aus der Liste\nlinks auswählen",
            pw=PREVIEW_W, ph=PREVIEW_H,
        )
        self.pv_in.pack(fill="y")

    # ── Right: result preview + settings ─────────────────────

    def _build_result_panel(self, parent):
        self.pv_out = PreviewPanel(
            parent,
            title="Ergebnis  (Schachbrett = transparent)",
            placeholder_text="Wird nach der\nVerarbeitung angezeigt",
            pw=PREVIEW_W, ph=PREVIEW_H,
        )
        self.pv_out.pack()

        separator(parent, pady=8)
        self._build_settings(parent)

    def _build_settings(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x")

        # Model column
        mc = tk.Frame(row, bg=C["bg"])
        mc.pack(side="left", fill="y", padx=(0, 16))
        tk.Label(mc, text="KI-MODELL", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        for key, (_, desc) in MODELS.items():
            r = tk.Frame(mc, bg=C["bg"])
            r.pack(fill="x", pady=1)
            tk.Radiobutton(r, text=key, variable=self.model_var, value=key,
                           bg=C["bg"], fg=C["text"], selectcolor=C["bg"],
                           activebackground=C["bg"], activeforeground=C["accent2"],
                           font=FONT_SMALL).pack(side="left")
            tk.Label(r, text=f"  {desc}", bg=C["bg"], fg=C["muted"],
                     font=("Segoe UI", 8)).pack(side="left")

        # Divider
        tk.Frame(row, bg=C["border"], width=1).pack(side="left", fill="y", padx=12)

        # Matting column
        ac = tk.Frame(row, bg=C["bg"])
        ac.pack(side="left", fill="y", padx=(0, 16))
        tk.Label(ac, text="ALPHA MATTING", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Checkbutton(ac, text="Aktivieren  (weiche Kanten bei Haaren/Fell)",
                       variable=self.matting_var,
                       bg=C["bg"], fg=C["text"], selectcolor=C["bg"],
                       activebackground=C["bg"], activeforeground=C["text"],
                       font=FONT_SMALL).pack(anchor="w", pady=(3, 6))

        def sl(p, txt, var, lo, hi):
            f = tk.Frame(p, bg=C["bg"])
            f.pack(fill="x", pady=2)
            tk.Label(f, text=txt, bg=C["bg"], fg=C["muted"],
                     font=FONT_SMALL, width=22, anchor="w").pack(side="left")
            tk.Label(f, textvariable=var, bg=C["bg"],
                     fg=C["accent2"], font=FONT_SMALL, width=3).pack(side="right")
            ttk.Scale(f, variable=var, from_=lo, to=hi,
                      orient="horizontal").pack(side="left", fill="x",
                                               expand=True, padx=4)

        sl(ac, "Vordergrund-Schwelle", self.fg_var, 1, 50)
        sl(ac, "Hintergrund-Schwelle", self.bg_var, 1, 30)

        # Divider
        tk.Frame(row, bg=C["border"], width=1).pack(side="left", fill="y", padx=12)

        # Output column
        oc = tk.Frame(row, bg=C["bg"])
        oc.pack(side="left", fill="y")
        tk.Label(oc, text="AUSGABE-ORDNER", bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        or_ = tk.Frame(oc, bg=C["bg"])
        or_.pack(fill="x", pady=(4, 4))
        tk.Entry(or_, textvariable=self.output_dir,
                 bg=C["panel"], fg=C["text"], insertbackground=C["text"],
                 relief="flat", font=FONT_SMALL, bd=4, width=26
                 ).pack(side="left", fill="x", expand=True)
        styled_button(or_, "...", self._choose_output,
                      small=True).pack(side="right", padx=(4, 0))
        styled_button(oc, "Ausgabe öffnen", self._open_output,
                      small=True).pack(anchor="w")

    # ── Log strip ─────────────────────────────────────────────

    def _build_log(self):
        lf = tk.Frame(self, bg=C["surface"])
        lf.pack(fill="x", padx=14, pady=(0, 10))

        tr = tk.Frame(lf, bg=C["surface"])
        tr.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(tr, text="Log", bg=C["surface"], fg=C["muted"],
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        styled_button(tr, "Leeren", self._clear_log, small=True).pack(side="right")

        self.log_text = tk.Text(
            lf, height=5, bg=C["bg"], fg=C["text"], font=FONT_MONO,
            relief="flat", bd=0, state="disabled", wrap="word", padx=8, pady=4,
        )
        self.log_text.pack(fill="x", padx=4, pady=(2, 4))
        for tag, col in [("success", C["success"]), ("error",   C["error"]),
                         ("warning", C["warning"]), ("accent",  C["accent2"]),
                         ("muted",   C["muted"])]:
            self.log_text.tag_config(tag, foreground=col)

        st = ttk.Style()
        st.theme_use("clam")
        st.configure("P.Horizontal.TProgressbar",
                     troughcolor=C["panel"], background=C["accent"],
                     darkcolor=C["accent"], lightcolor=C["accent"],
                     bordercolor=C["panel"])
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(lf, variable=self.progress_var, maximum=100,
                        style="P.Horizontal.TProgressbar"
                        ).pack(fill="x", padx=4, pady=(0, 4))

    # ─────────────────────────────────────────────────────────
    # File management
    # ─────────────────────────────────────────────────────────

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Bilder auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
                       ("Alle Dateien", "*.*")]
        )
        for f in files:
            self._add_path(Path(f))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Ordner auswählen")
        if folder:
            found = []
            for ext in SUPPORTED_EXTS:
                found.extend(Path(folder).glob(f"*{ext}"))
                found.extend(Path(folder).glob(f"*{ext.upper()}"))
            for f in sorted(set(found)):
                self._add_path(f)

    def _add_path(self, path):
        if any(it.path == path for it in self.file_items):
            return
        item = FileItem(self.file_list_frame, path,
                        select_cb=self._select_item,
                        remove_cb=self._remove_item)
        self.file_items.append(item)
        self._update_drop_hint()
        if len(self.file_items) == 1:
            self._select_item(item)

    def _remove_item(self, item):
        if self.running:
            return
        if self._selected_item is item:
            self._selected_item = None
            self.pv_in.clear()
            self.pv_out.clear()
        item.destroy()
        self.file_items.remove(item)
        self._results.pop(item.path, None)
        self._update_drop_hint()

    def _clear_list(self):
        if self.running:
            return
        for item in self.file_items:
            item.destroy()
        self.file_items.clear()
        self._results.clear()
        self._selected_item = None
        self.pv_in.clear()
        self.pv_out.clear()
        self._update_drop_hint()

    def _update_drop_hint(self):
        if self.file_items:
            self.drop_hint.place_forget()
        else:
            self.drop_hint.place(relx=0.5, rely=0.38, anchor="center")

    def _select_item(self, item):
        if self._selected_item:
            self._selected_item.set_selected(False)
        self._selected_item = item
        item.set_selected(True)
        self._show_input_preview(item.path)
        if item.path in self._results:
            self._show_result_preview(self._results[item.path])
        else:
            self.pv_out.clear()

    # ─────────────────────────────────────────────────────────
    # Preview helpers
    # ─────────────────────────────────────────────────────────

    def _show_input_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            self.pv_in.show(
                img,
                info=f"{path.name}  ·  {w} x {h} px",
                badge=path.suffix.upper().lstrip("."),
            )
        except Exception as e:
            self.pv_in.clear()
            self._log(f"Vorschau Fehler: {e}", "error")

    def _show_result_preview(self, result):
        try:
            self.pv_out.show(
                result["preview_img"],
                info=(f"{result['width']} x {result['height']} px  ·  "
                      f"Vordergrund: {result['foreground_pct']:.1f}%  ·  "
                      f"{result['elapsed_s']:.1f} s"),
                badge="FERTIG",
            )
        except Exception as e:
            self.pv_out.clear()
            self._log(f"Ergebnis-Vorschau Fehler: {e}", "error")

    # ─────────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────────

    def _choose_output(self):
        folder = filedialog.askdirectory(title="Ausgabe-Ordner wählen")
        if folder:
            self.output_dir.set(folder)

    def _open_output(self):
        p = Path(self.output_dir.get())
        p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    # ─────────────────────────────────────────────────────────
    # Processing
    # ─────────────────────────────────────────────────────────

    def _start_processing(self):
        if not PIL_OK or not REMBG_OK:
            messagebox.showerror("Fehlende Abhängigkeiten",
                                 "Bitte installiere:\n\npip install rembg onnxruntime pillow numpy")
            return
        if not self.file_items:
            messagebox.showwarning("Keine Bilder", "Bitte zuerst Bilder hinzufügen.")
            return
        if self.running:
            return
        self.running = True
        self.start_btn.config(state="disabled", text="  Verarbeitung läuft ...  ")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        model_key = self.model_var.get()
        model_id  = MODELS[model_key][0]
        use_mat   = self.matting_var.get()
        fg_thr    = self.fg_var.get()
        bg_thr    = self.bg_var.get()
        out_dir   = Path(self.output_dir.get())
        items     = list(self.file_items)
        total     = len(items)

        self._log(f"Starte  {total} Bild(er)  |  Modell: {model_key}", "accent")

        if self._session_model != model_key or self._session is None:
            self._log(f"  Lade Modell '{model_key}' ...", "muted")
            try:
                self._session = new_session(model_id)
                self._session_model = model_key
                self._log("  Modell geladen", "success")
            except Exception as e:
                self._log(f"  Fehler: {e}", "error")
                self._finish()
                return

        errors = 0
        for idx, item in enumerate(items):
            self.after(0, item.set_status, "processing", "Läuft ...")
            self._log(f"[{idx+1}/{total}]  {item.path.name}", "accent")

            # Show input immediately
            self.after(0, self._show_input_preview, item.path)
            self.after(0, self.pv_out.clear)

            try:
                result = process_image(
                    input_path=item.path,
                    output_dir=out_dir,
                    session=self._session,
                    alpha_matting=use_mat,
                    fg_thresh=fg_thr,
                    bg_thresh=bg_thr,
                    log_fn=self._log,
                )
                self._results[item.path] = result
                self.after(0, item.set_status, "done", f"OK {result['elapsed_s']:.1f}s")
                # Update result panel live
                self.after(0, self._show_result_preview, result)
            except Exception as e:
                self._log(f"  Fehler: {e}", "error")
                self.after(0, item.set_status, "error", "Fehler")
                errors += 1

            self.after(0, self.progress_var.set, (idx + 1) / total * 100)

        status = "success" if errors == 0 else "warning"
        self._log(f"Fertig  {total-errors}/{total} erfolgreich  ->  {out_dir}", status)
        self._finish()

    def _finish(self):
        self.running = False
        self.after(0, self.start_btn.config,
                   {"state": "normal", "text": "  Start  "})

    # ─────────────────────────────────────────────────────────
    # Log
    # ─────────────────────────────────────────────────────────

    def _log(self, msg, tag=None):
        self._log_queue.put((msg, tag))

    def _poll_log(self):
        try:
            while True:
                msg, tag = self._log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", msg + "\n", tag or "")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.progress_var.set(0)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    st = ttk.Style(app)
    st.theme_use("clam")
    st.configure("Vertical.TScrollbar",
                 background=C["panel"], troughcolor=C["surface"],
                 bordercolor=C["surface"], arrowcolor=C["muted"])
    app.mainloop()
