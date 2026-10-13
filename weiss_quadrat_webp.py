#!/usr/bin/env python3
"""
Weisser Hintergrund + Quadrat + WebP fuer alle Bilder eines Ordners.

- customtkinter GUI
- Eingabeordner waehlen
- Transparenten Hintergrund auf Weiss legen
- Auf einstellbare quadratische Kantenlaenge (1000-3000 px) skalieren, zentriert
- Als WebP in einem waehlbaren Ordner speichern (Qualitaet einstellbar, Standard 80)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from loguru import logger
from PIL import Image, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
SIZE_MIN = 1000
SIZE_MAX = 3000
SIZE_DEFAULT = 2000
QUALITY_DEFAULT = 80
BG_COLOR = (255, 255, 255)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ergebnisse"


logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add(
    str(PROJECT_ROOT / "weiss_quadrat_webp_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG",
)


def _has_transparency(img: Image.Image) -> bool:
    return (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )


def flatten_on_white(img: Image.Image, bg_color: tuple[int, int, int] = BG_COLOR) -> Image.Image:
    """Legt transparente Bereiche auf weissen Hintergrund."""
    if not _has_transparency(img):
        return img.convert("RGB") if img.mode != "RGB" else img
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, bg_color)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def fit_on_white_square(
    img: Image.Image,
    size: int,
    bg_color: tuple[int, int, int] = BG_COLOR,
) -> Image.Image:
    """Skaliert proportional so, dass die laengste Seite = size, dann zentriert auf weissem Quadrat."""
    if size < SIZE_MIN or size > SIZE_MAX:
        raise ValueError(f"Kantenlaenge muss zwischen {SIZE_MIN} und {SIZE_MAX} px liegen.")
    rgb = flatten_on_white(img, bg_color)
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("Ungueltige Bildabmessungen.")
    scale = size / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    fitted = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg_color)
    canvas.paste(fitted, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def save_webp(img: Image.Image, dest: Path, quality: int) -> None:
    quality = max(1, min(100, int(quality)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    lossless = quality >= 100
    img.save(dest, format="WEBP", quality=quality, lossless=lossless, method=6)


def collect_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    return sorted(files, key=lambda p: p.name.casefold())


def process_image(
    source: str | Path,
    output_dir: str | Path,
    size: int,
    quality: int = QUALITY_DEFAULT,
) -> Path:
    source = Path(source)
    logger.info(f"Verarbeite {source.name}")
    with Image.open(source) as loaded:
        img = ImageOps.exif_transpose(loaded)
        img.load()
    square = fit_on_white_square(img, size)
    dest = Path(output_dir) / f"{source.stem}.webp"
    save_webp(square, dest, quality)
    size_kb = dest.stat().st_size / 1024
    logger.success(f"Gespeichert: {dest.name} ({square.size[0]}x{square.size[1]}, {size_kb:.1f} KB, q={quality})")
    return dest


class WeissQuadratWebpApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Weiss-Quadrat WebP")
        self.geometry("920x640")
        self.minsize(860, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.input_var = ctk.StringVar()
        self.output_var = ctk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.size_var = ctk.IntVar(value=SIZE_DEFAULT)
        self.quality_var = ctk.IntVar(value=QUALITY_DEFAULT)
        self.size_label_var = ctk.StringVar(value=f"{SIZE_DEFAULT} px")
        self.quality_label_var = ctk.StringVar(value=f"{QUALITY_DEFAULT} %")
        self.status_var = ctk.StringVar(value="Bereit.")
        self._busy = False

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=14)
        outer.pack(fill="both", expand=True, padx=18, pady=18)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(
            outer,
            text="Weisser Hintergrund, Quadrat, WebP",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))

        ctk.CTkLabel(
            outer,
            text="Alle Bilder eines Ordners auf Weiss legen, quadratisch skalieren und als WebP speichern.",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))

        form = ctk.CTkFrame(outer)
        form.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 8))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Eingabeordner").grid(row=0, column=0, sticky="w", padx=(14, 8), pady=8)
        ctk.CTkEntry(form, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)
        ctk.CTkButton(form, text="Wählen", width=90, command=self._choose_input).grid(
            row=0, column=2, padx=(0, 14), pady=8
        )

        ctk.CTkLabel(form, text="Ausgabeordner").grid(row=1, column=0, sticky="w", padx=(14, 8), pady=8)
        ctk.CTkEntry(form, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=8)
        ctk.CTkButton(form, text="Wählen", width=90, command=self._choose_output).grid(
            row=1, column=2, padx=(0, 14), pady=8
        )

        ctk.CTkLabel(form, text="Kantenlänge").grid(row=2, column=0, sticky="w", padx=(14, 8), pady=8)
        size_row = ctk.CTkFrame(form, fg_color="transparent")
        size_row.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=8)
        size_row.grid_columnconfigure(0, weight=1)
        self.size_slider = ctk.CTkSlider(
            size_row,
            from_=SIZE_MIN,
            to=SIZE_MAX,
            number_of_steps=(SIZE_MAX - SIZE_MIN) // 50,
            command=self._on_size,
        )
        self.size_slider.set(SIZE_DEFAULT)
        self.size_slider.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(size_row, textvariable=self.size_label_var, width=80).grid(
            row=0, column=1, padx=(10, 0)
        )

        ctk.CTkLabel(form, text="WebP-Qualität").grid(row=3, column=0, sticky="w", padx=(14, 8), pady=8)
        quality_row = ctk.CTkFrame(form, fg_color="transparent")
        quality_row.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=8)
        quality_row.grid_columnconfigure(0, weight=1)
        self.quality_slider = ctk.CTkSlider(
            quality_row,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self._on_quality,
        )
        self.quality_slider.set(QUALITY_DEFAULT)
        self.quality_slider.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(quality_row, textvariable=self.quality_label_var, width=80).grid(
            row=0, column=1, padx=(10, 0)
        )

        info = (
            "Transparente Bereiche werden weiss. Das Bild bleibt proportional, "
            "die längste Seite füllt die Quadratseite, Rest ist weisser Rand.\n"
            "Ausgabe: gleicher Dateiname, Endung .webp"
        )
        ctk.CTkLabel(form, text=info, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(4, 12)
        )

        self.run_button = ctk.CTkButton(
            outer,
            text="Ordner verarbeiten",
            height=38,
            command=self._start,
        )
        self.run_button.grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 8))

        self.progress = ctk.CTkProgressBar(outer)
        self.progress.set(0)
        self.progress.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 6))

        ctk.CTkLabel(outer, textvariable=self.status_var, anchor="w").grid(
            row=5, column=0, sticky="ew", padx=18, pady=(0, 6)
        )

        ctk.CTkLabel(outer, text="Protokoll", font=ctk.CTkFont(weight="bold")).grid(
            row=6, column=0, sticky="w", padx=18, pady=(4, 4)
        )
        self.log_box = ctk.CTkTextbox(outer, height=180, state="disabled")
        self.log_box.grid(row=8, column=0, sticky="nsew", padx=18, pady=(0, 16))

    def _on_size(self, value: float) -> None:
        size = int(round(float(value) / 50.0) * 50)
        size = max(SIZE_MIN, min(SIZE_MAX, size))
        self.size_var.set(size)
        self.size_label_var.set(f"{size} px")

    def _on_quality(self, value: float) -> None:
        quality = int(round(float(value)))
        quality = max(1, min(100, quality))
        self.quality_var.set(quality)
        self.quality_label_var.set(f"{quality} %")

    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="Eingabeordner wählen")
        if path:
            self.input_var.set(path)
            if not self.output_var.get().strip():
                self.output_var.set(str(DEFAULT_OUTPUT_DIR))
            count = len(collect_images(path))
            self.status_var.set(f"{count} Bild(er) im Eingabeordner.")

    def _choose_output(self) -> None:
        initial = self.output_var.get().strip() or str(DEFAULT_OUTPUT_DIR)
        path = filedialog.askdirectory(title="Ausgabeordner wählen", initialdir=initial)
        if path:
            self.output_var.set(path)

    def _log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start(self) -> None:
        if self._busy:
            return
        source = self.input_var.get().strip()
        dest = self.output_var.get().strip()
        size = int(self.size_var.get())
        quality = int(self.quality_var.get())

        if not source or not Path(source).is_dir():
            messagebox.showerror("Eingabe", "Bitte einen gültigen Eingabeordner wählen.")
            return
        if not dest:
            messagebox.showerror("Ausgabe", "Bitte einen Ausgabeordner wählen.")
            return
        if size < SIZE_MIN or size > SIZE_MAX:
            messagebox.showerror("Größe", f"Kantenlänge muss {SIZE_MIN}–{SIZE_MAX} px sein.")
            return

        files = collect_images(source)
        if not files:
            messagebox.showinfo("Keine Dateien", "Keine unterstützten Bilder im Ordner gefunden.")
            return

        Path(dest).mkdir(parents=True, exist_ok=True)
        self._busy = True
        self.run_button.configure(state="disabled")
        self.progress.set(0)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.status_var.set(f"Starte {len(files)} Bild(er) …")
        logger.info(f"Batch: {len(files)} Datei(en), {size}px, q={quality}, nach {dest}")

        threading.Thread(
            target=self._worker,
            args=(files, Path(dest), size, quality),
            daemon=True,
        ).start()

    def _worker(self, files: list[Path], output_dir: Path, size: int, quality: int) -> None:
        ok = 0
        err = 0
        total = len(files)
        for index, path in enumerate(files, start=1):
            try:
                saved = process_image(path, output_dir, size, quality)
                ok += 1
                line = f"OK  {path.name} → {saved.name}"
            except Exception as e:
                err += 1
                line = f"FEHLER  {path.name}: {e}"
                logger.exception(line)
            self.after(0, self._on_progress, index, total, line)

        summary = f"Fertig: {ok} gespeichert"
        if err:
            summary += f", {err} Fehler"
        self.after(0, self._on_done, summary, err)

    def _on_progress(self, index: int, total: int, line: str) -> None:
        self.progress.set(index / total)
        self.status_var.set(f"{index}/{total}")
        self._log(line)

    def _on_done(self, summary: str, err: int) -> None:
        self._busy = False
        self.run_button.configure(state="normal")
        self.status_var.set(summary)
        self._log(summary)
        if err:
            messagebox.showwarning("Fertig mit Fehlern", summary)
        else:
            messagebox.showinfo("Fertig", summary)


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starte WeissQuadratWebpApp")
    logger.info("=" * 70)
    app = WeissQuadratWebpApp()
    app.mainloop()
