#!/usr/bin/env python3
"""
Bild-Resizer - Nur Größenanpassung, einstellbare Zielgröße
Unterstützte Formate: JPG, PNG, TIFF, WebP, BMP
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

MODE_SCALE = "scale"
MODE_SQUARE = "square"
DEFAULT_MARGIN_PX = 50


def _has_transparency(img: Image.Image) -> bool:
    return (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )


def _flatten_on_white(img: Image.Image, bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Legt transparente Bereiche auf weißen Hintergrund."""
    if not _has_transparency(img):
        return img.convert("RGB") if img.mode != "RGB" else img
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, bg_color)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def _object_mask(img: Image.Image, threshold: int = 250) -> np.ndarray:
    """Binärmaske des Vordergrunds (uint8, 0/1)."""
    if _has_transparency(img):
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
        return (alpha > 10).astype(np.uint8)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return (gray < threshold).astype(np.uint8)


def _crop_to_object(
    img: Image.Image,
    threshold: int = 250,
    min_area: int = 100,
    relative_min_area: float = 0.02,
) -> Image.Image:
    """
    Schneidet auf das Hauptobjekt zu.

    Verwendet die größte zusammenhängende Fläche, damit dünne Artefakte
    oder Staub oben im Bild nicht die Zentrierung nach unten verziehen.
    """
    mask = _object_mask(img, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return img

    best_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area

    if best_area < min_area:
        return img

    # Berücksichtige alle relevanten Komponenten (z. B. mehrere Produkte),
    # ignoriere aber sehr kleine Artefakte.
    area_threshold = max(min_area, int(best_area * relative_min_area))
    relevant_ids = [
        i for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= area_threshold
    ]
    if not relevant_ids:
        return img

    relevant_mask = np.isin(labels, relevant_ids)
    ys, xs = np.where(relevant_mask)
    if len(xs) == 0 or len(ys) == 0:
        return img

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return img.crop((x1, y1, x2, y2))


def resize_image(
    src: Path,
    dst_folder: Path,
    target_size: int,
    mode: str,
    quality: int = 95,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    convert_webp: bool = False,
    center_object: bool = False,
    margin_px: int = DEFAULT_MARGIN_PX,
) -> str:
    """Passt die Bildgröße an und speichert im Zielordner."""
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if center_object or mode == MODE_SQUARE:
                img = _crop_to_object(img)
            result = _resize(img, target_size, mode, bg_color, margin_px)
            if convert_webp:
                dst = dst_folder / f"{src.stem}.webp"
                result = _flatten_on_white(result, bg_color)
                _save_webp(result, dst, quality)
                out_name = dst.name
            else:
                dst = dst_folder / src.name
                _save_image(result, dst, src.suffix.lower(), quality, bg_color)
                out_name = dst.name
        return f"✓ {src.name} → {out_name} ({result.size[0]}×{result.size[1]})"
    except Exception as e:
        return f"✗ {src.name}: {e}"


def _resize(
    img: Image.Image,
    target: int,
    mode: str,
    bg_color: tuple[int, int, int],
    margin_px: int = DEFAULT_MARGIN_PX,
) -> Image.Image:
    if target < 1:
        raise ValueError("Zielgröße muss mindestens 1 px sein.")

    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Ungültige Bildabmessungen.")

    has_alpha = _has_transparency(img)

    if mode == MODE_SCALE:
        scale = target / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Quadrat: proportional einpassen, mind. margin_px Rand ringsum, zentriert
    margin_px = max(0, margin_px)
    inner = target - 2 * margin_px
    if inner < 1:
        raise ValueError(
            f"Zielgröße {target}px ist zu klein für {margin_px}px Rand ringsum."
        )

    scale = min(inner / w, inner / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    fitted = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if has_alpha:
        canvas = Image.new("RGBA", (target, target), (*bg_color, 0))
        paste_img = fitted.convert("RGBA")
        mask = paste_img.split()[3]
    else:
        canvas = Image.new("RGB", (target, target), bg_color)
        paste_img = fitted.convert("RGB")
        mask = None

    x = (target - new_w) // 2
    y = (target - new_h) // 2
    canvas.paste(paste_img, (x, y), mask)

    if not has_alpha and canvas.mode == "RGBA":
        canvas = canvas.convert("RGB")
    return canvas


def _save_webp(img: Image.Image, dst: Path, quality: int) -> None:
    if img.mode != "RGB":
        img = img.convert("RGB")
    lossless = quality >= 100
    img.save(dst, format="WEBP", lossless=lossless, quality=quality, method=6)


def _save_image(
    img: Image.Image,
    dst: Path,
    ext: str,
    quality: int,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    if ext in {".jpg", ".jpeg"}:
        if _has_transparency(img):
            img = _flatten_on_white(img, bg_color)
        subsampling = 0 if quality == 100 else -1
        img.save(dst, quality=quality, subsampling=subsampling, optimize=True)
    elif ext in {".tif", ".tiff"}:
        if _has_transparency(img):
            img = _flatten_on_white(img, bg_color)
        img.save(dst, compression="tiff_lzw")
    elif ext == ".webp":
        if _has_transparency(img):
            img = _flatten_on_white(img, bg_color)
        _save_webp(img, dst, quality)
    elif ext == ".png":
        img.save(dst, optimize=True)
    else:
        if _has_transparency(img):
            img = _flatten_on_white(img, bg_color)
        img.save(dst, quality=quality, optimize=True)


class BildResizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bild-Resizer")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        BG, PANEL, ACCENT, FG, ENTRY = "#1a1a2e", "#16213e", "#e94560", "#eaeaea", "#0f3460"
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Courier New", 10))
        style.configure("Head.TLabel", background=BG, foreground=ACCENT, font=("Courier New", 15, "bold"))
        style.configure("Sub.TLabel", background=PANEL, foreground=FG, font=("Courier New", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="#fff",
                        font=("Courier New", 10, "bold"), borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#c73652"), ("disabled", "#555")])
        style.configure("TRadiobutton", background=PANEL, foreground=FG, font=("Courier New", 10))
        style.configure("TProgressbar", troughcolor=PANEL, background=ACCENT)
        self._colors = {"BG": BG, "PANEL": PANEL, "ACCENT": ACCENT, "FG": FG, "ENTRY": ENTRY}

    def _build_ui(self):
        C = self._colors
        ttk.Label(self, text="⊡  BILD-RESIZER", style="Head.TLabel").grid(
            row=0, column=0, padx=18, pady=8, sticky="w")
        ttk.Label(self, text="Größe · WebP · JPG · PNG · TIFF",
                  style="TLabel", foreground="#888").grid(row=1, column=0, padx=18, sticky="w")
        tk.Frame(self, height=1, bg=C["ACCENT"]).grid(row=2, column=0, sticky="ew", padx=18, pady=6)

        panel = ttk.Frame(self, style="Panel.TFrame", padding=14)
        panel.grid(row=3, column=0, padx=18, pady=4, sticky="ew")
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Eingabeordner:", style="Sub.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.src_var = tk.StringVar()
        ttk.Entry(panel, textvariable=self.src_var, width=42).grid(row=0, column=1, padx=6)
        ttk.Button(panel, text="…", style="Accent.TButton", width=3,
                   command=lambda: self._browse(self.src_var)).grid(row=0, column=2)

        ttk.Label(panel, text="Ausgabeordner:", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self.dst_var = tk.StringVar()
        ttk.Entry(panel, textvariable=self.dst_var, width=42).grid(row=1, column=1, padx=6)
        ttk.Button(panel, text="…", style="Accent.TButton", width=3,
                   command=lambda: self._browse(self.dst_var)).grid(row=1, column=2)

        ttk.Label(panel, text="Zielgröße (px):", style="Sub.TLabel").grid(row=2, column=0, sticky="w", pady=8)
        size_frame = ttk.Frame(panel, style="Panel.TFrame")
        size_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        self.size_var = tk.IntVar(value=1024)
        tk.Spinbox(size_frame, from_=64, to=4096, increment=64, textvariable=self.size_var,
                   width=8, font=("Courier New", 11)).pack(side="left")
        ttk.Label(size_frame, text="  (lange Kante bzw. Quadratseite)", style="Sub.TLabel").pack(side="left")

        ttk.Label(panel, text="Modus:", style="Sub.TLabel").grid(row=3, column=0, sticky="nw", pady=8)
        mode_frame = ttk.Frame(panel, style="Panel.TFrame")
        mode_frame.grid(row=3, column=1, columnspan=2, sticky="w")
        self.mode_var = tk.StringVar(value=MODE_SQUARE)
        ttk.Radiobutton(
            mode_frame,
            text="Quadrat (proportional, 50 px Rand, zentriert)",
            variable=self.mode_var,
            value=MODE_SQUARE,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="Skalieren (Seitenverhältnis beibehalten)",
            variable=self.mode_var,
            value=MODE_SCALE,
        ).pack(anchor="w", pady=4)

        ttk.Label(panel, text="Qualität:", style="Sub.TLabel").grid(row=4, column=0, sticky="w", pady=6)
        q_frame = ttk.Frame(panel, style="Panel.TFrame")
        q_frame.grid(row=4, column=1, columnspan=2, sticky="ew")
        self.quality_var = tk.IntVar(value=95)
        self.quality_label = tk.Label(q_frame, text="95", bg=C["PANEL"], fg=C["ACCENT"],
                                    font=("Courier New", 10, "bold"), width=6, anchor="w")
        self.quality_label.pack(side="right", padx=6)
        self.quality_scale = tk.Scale(
            q_frame, from_=1, to=100, orient="horizontal", variable=self.quality_var,
            length=220, bg=C["PANEL"], fg=C["FG"], troughcolor=C["ENTRY"],
            highlightthickness=0, bd=0, showvalue=False,
            command=lambda v: self.quality_label.configure(text=str(int(float(v)))),
        )
        self.quality_scale.pack(side="left")

        self.convert_webp_var = tk.BooleanVar(value=False)
        webp_frame = ttk.Frame(panel, style="Panel.TFrame")
        webp_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(
            webp_frame,
            text="Als WebP speichern (transparent → weißer Hintergrund)",
            variable=self.convert_webp_var,
            bg=C["PANEL"], fg=C["FG"], selectcolor=C["PANEL"],
            activebackground=C["PANEL"], activeforeground=C["ACCENT"],
            font=("Courier New", 10),
        ).pack(anchor="w")

        self.center_object_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            webp_frame,
            text="Objekt zuschneiden (nur Modus „Skalieren“; Quadrat immer)",
            variable=self.center_object_var,
            bg=C["PANEL"], fg=C["FG"], selectcolor=C["PANEL"],
            activebackground=C["PANEL"], activeforeground=C["ACCENT"],
            font=("Courier New", 10),
        ).pack(anchor="w", pady=(4, 0))

        self.recursive_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="Unterordner einschließen", variable=self.recursive_var,
                       bg=C["PANEL"], fg=C["FG"], selectcolor=C["PANEL"],
                       activebackground=C["PANEL"], font=("Courier New", 10)).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=4)

        self.btn_start = ttk.Button(self, text="▶  GRÖSSE ANPASSEN", style="Accent.TButton",
                                    command=self._start)
        self.btn_start.grid(row=4, column=0, padx=18, pady=10, sticky="ew")

        self.progress = ttk.Progressbar(self, mode="determinate", length=420)
        self.progress.grid(row=5, column=0, padx=18, pady=2, sticky="ew")
        self.status_var = tk.StringVar(value="Bereit.")
        ttk.Label(self, textvariable=self.status_var, style="TLabel", foreground="#aaa").grid(
            row=6, column=0, padx=18, sticky="w")

        log_frame = tk.Frame(self, bg=C["PANEL"])
        log_frame.grid(row=7, column=0, padx=18, pady=8, sticky="ew")
        self.log = tk.Text(log_frame, width=58, height=8, bg=C["PANEL"], fg=C["FG"],
                           font=("Courier New", 9), relief="flat", state="disabled")
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.columnconfigure(0, weight=1)

    def _browse(self, var: tk.StringVar):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def _collect_files(self, src: Path) -> list[Path]:
        if self.recursive_var.get():
            files = [f for f in src.rglob("*") if f.suffix.lower() in SUPPORTED]
        else:
            files = [f for f in src.iterdir() if f.suffix.lower() in SUPPORTED]
        return sorted(files, key=lambda p: p.name.lower())

    def _start(self):
        src = self.src_var.get().strip()
        dst = self.dst_var.get().strip()
        try:
            target = int(self.size_var.get())
        except tk.TclError:
            messagebox.showerror("Fehler", "Ungültige Zielgröße.")
            return

        if not src or not Path(src).is_dir():
            messagebox.showerror("Fehler", "Bitte einen gültigen Eingabeordner wählen.")
            return
        if not dst:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen.")
            return
        if target < 1:
            messagebox.showerror("Fehler", "Zielgröße muss mindestens 1 px sein.")
            return
        if self.mode_var.get() == MODE_SQUARE and target < 2 * DEFAULT_MARGIN_PX + 1:
            messagebox.showerror(
                "Fehler",
                f"Zielgröße muss mindestens {2 * DEFAULT_MARGIN_PX + 1} px sein "
                f"(50 px Rand ringsum).",
            )
            return

        files = self._collect_files(Path(src))
        if not files:
            messagebox.showinfo("Keine Dateien", "Keine unterstützten Bilder gefunden.")
            return

        Path(dst).mkdir(parents=True, exist_ok=True)
        self.btn_start.configure(state="disabled")
        self._log_clear()
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0

        threading.Thread(
            target=self._worker,
            args=(
                files, Path(src), Path(dst), target, self.mode_var.get(),
                int(self.quality_var.get()), self.convert_webp_var.get(),
                self.center_object_var.get(), DEFAULT_MARGIN_PX,
            ),
            daemon=True,
        ).start()

    def _worker(self, files, src_folder, dst_folder, target, mode, quality, convert_webp, center_object, margin_px):
        ok = err = 0
        recursive = self.recursive_var.get()
        for i, f in enumerate(files, 1):
            if recursive:
                rel = f.relative_to(src_folder)
                out_dir = dst_folder / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                result = resize_image(
                    f, out_dir, target, mode, quality,
                    convert_webp=convert_webp, center_object=center_object, margin_px=margin_px,
                )
            else:
                result = resize_image(
                    f, dst_folder, target, mode, quality,
                    convert_webp=convert_webp, center_object=center_object, margin_px=margin_px,
                )

            if result.startswith("✓"):
                ok += 1
            else:
                err += 1
            self.after(0, self._log_append, result)
            self.after(0, self._set_progress, i, f"Verarbeite {i}/{len(files)}: {f.name}")

        summary = f"Fertig! ✓ {ok} erfolgreich"
        if err:
            summary += f"  ✗ {err} Fehler"
        self.after(0, self._log_append, "\n" + summary)
        self.after(0, self._finish, summary)

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


if __name__ == "__main__":
    BildResizerApp().mainloop()
