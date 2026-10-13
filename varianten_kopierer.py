#!/usr/bin/env python3
"""
Varianten-Kopierer fuer Produktbilder.

- customtkinter GUI
- Bilddatei waehlen
- Namensstamm (z. B. Y-DSS-) und Varianten (z. B. 5,7,12)
- Speichert identische Kopien: Y-DSS-5, Y-DSS-7, Y-DSS-12
- Vollstaendiges Logging mit Loguru
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from loguru import logger
from PIL import Image, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
INVALID_FILENAME_CHARS = '<>:"/\\|?*'
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
    str(PROJECT_ROOT / "varianten_kopierer_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG",
)


def parse_variants(text: str) -> list[str]:
    """Zerlegt '5,7,12' oder zeilenweise Eingabe in einzelne Varianten."""
    parts = re.split(r"[,;\n]+", text)
    variants: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = part.strip()
        if not value:
            continue
        if value in seen:
            logger.debug(f"Doppelte Variante ignoriert: {value}")
            continue
        seen.add(value)
        variants.append(value)
    return variants


def sanitize_stem(stem: str) -> str:
    cleaned = stem.strip()
    for ch in INVALID_FILENAME_CHARS:
        cleaned = cleaned.replace(ch, "-")
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned:
        raise ValueError("Namensstamm darf nicht leer sein.")
    return cleaned


def sanitize_variant(variant: str) -> str:
    cleaned = variant.strip()
    for ch in INVALID_FILENAME_CHARS:
        cleaned = cleaned.replace(ch, "-")
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError(f"Ungueltige Variante: {variant!r}")
    return cleaned


def build_output_names(
    source: Path,
    stem: str,
    variants: list[str],
    output_dir: Path,
) -> list[Path]:
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Nicht unterstuetztes Format: {suffix}")
    safe_stem = sanitize_stem(stem)
    paths: list[Path] = []
    for variant in variants:
        name = f"{safe_stem}{sanitize_variant(variant)}{suffix}"
        paths.append(output_dir / name)
    return paths


def copy_variants(
    source: str | Path,
    stem: str,
    variants_text: str,
    output_dir: str | Path | None = None,
) -> list[Path]:
    source = Path(source)
    logger.info("=" * 70)
    logger.info(f"Starte Varianten-Kopie: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {source}")
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Nicht unterstuetztes Format: {source.suffix}")

    variants = parse_variants(variants_text)
    if not variants:
        raise ValueError("Bitte mindestens eine Variante eingeben (z. B. 5,7,12).")

    dest_dir = Path(output_dir) if output_dir else source.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    targets = build_output_names(source, stem, variants, dest_dir)
    logger.info(f"Namensstamm: {stem!r}, Varianten: {variants}")
    logger.info(f"Ausgabeordner: {dest_dir}")

    saved: list[Path] = []
    for target in targets:
        if target.resolve() == source.resolve():
            logger.warning(f"Ueberspringe identischen Pfad: {target.name}")
            continue
        shutil.copy2(source, target)
        size_kb = target.stat().st_size / 1024
        logger.success(f"Kopiert: {target.name} ({size_kb:.1f} KB)")
        saved.append(target)

    logger.success(f"Fertig: {len(saved)} Datei(en) gespeichert")
    logger.info("=" * 70)
    return saved


class VariantenKopiererApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Varianten-Kopierer")
        self.geometry("920x620")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.image_var = ctk.StringVar()
        self.stem_var = ctk.StringVar(value="Y-DSS-")
        self.variants_var = ctk.StringVar(value="5,7,12")
        self.output_var = ctk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.status_var = ctk.StringVar(value="Bereit.")
        self.preview_image: ctk.CTkImage | None = None

        self._build_ui()
        self.stem_var.trace_add("write", lambda *_: self._refresh_names())
        self.variants_var.trace_add("write", lambda *_: self._refresh_names())
        self._refresh_names()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=14)
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            outer,
            text="Bild unter Namensvarianten speichern",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(16, 8))

        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=6)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        form = ctk.CTkFrame(body)
        form.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Bild").grid(row=0, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.image_var, width=280).grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Wählen", width=90, command=self._choose_image).grid(
            row=0, column=2, padx=(0, 14)
        )

        ctk.CTkLabel(form, text="Namensstamm").grid(row=1, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.stem_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(10, 14)
        )

        ctk.CTkLabel(form, text="Varianten").grid(row=2, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.variants_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(10, 14)
        )

        ctk.CTkLabel(form, text="Ausgabeordner").grid(row=3, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.output_var).grid(row=3, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Ordner", width=90, command=self._choose_output_dir).grid(
            row=3, column=2, padx=(0, 14)
        )

        info = (
            "Beispiel: Stamm Y-DSS- und Varianten 5,7,12 ergeben\n"
            "Y-DSS-5, Y-DSS-7 und Y-DSS-12 (gleiche Dateiendung wie das Original)."
        )
        ctk.CTkLabel(form, text=info, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 8)
        )

        ctk.CTkLabel(form, text="Zieldateien", font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, columnspan=3, sticky="w", padx=14, pady=(4, 4)
        )
        self.names_box = ctk.CTkTextbox(form, height=140, width=420, state="disabled")
        self.names_box.grid(row=6, column=0, columnspan=3, sticky="ew", padx=14, pady=(0, 10))

        ctk.CTkButton(form, text="Kopien speichern", height=38, command=self._run).grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 8)
        )
        ctk.CTkLabel(form, textvariable=self.status_var, anchor="w", wraplength=420).grid(
            row=8, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 16)
        )

        preview_frame = ctk.CTkFrame(body)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            preview_frame,
            text="Vorschau",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 8))
        self.preview_label = ctk.CTkLabel(preview_frame, text="Noch kein Bild gewählt.")
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Bild auswählen",
            initialdir=str(DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.is_dir() else PROJECT_ROOT),
            filetypes=[
                ("Bilder", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not path:
            return
        self.image_var.set(path)
        source = Path(path)
        if not self.output_var.get().strip():
            self.output_var.set(str(source.parent))
        logger.info(f"Bild gewählt: {path}")
        self.status_var.set(f"Gewählt: {source.name}")
        self._show_preview(source)
        self._refresh_names()

    def _choose_output_dir(self) -> None:
        initial = self.output_var.get().strip() or str(DEFAULT_OUTPUT_DIR)
        path = filedialog.askdirectory(title="Ausgabeordner wählen", initialdir=initial)
        if path:
            self.output_var.set(path)
            logger.info(f"Ausgabeordner gewählt: {path}")
            self._refresh_names()

    def _show_preview(self, path: Path) -> None:
        try:
            with Image.open(path) as loaded:
                preview = ImageOps.exif_transpose(loaded).convert("RGB")
            preview.thumbnail((420, 480), Image.Resampling.LANCZOS)
            self.preview_image = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
            self.preview_label.configure(image=self.preview_image, text="")
            logger.debug(f"Vorschau {preview.size[0]}x{preview.size[1]}px")
        except Exception as e:
            logger.exception(f"Vorschau fehlgeschlagen: {e}")
            self.preview_label.configure(image=None, text="Vorschau nicht möglich.")

    def _planned_paths(self) -> list[Path]:
        source = Path(self.image_var.get().strip())
        stem = self.stem_var.get()
        variants = parse_variants(self.variants_var.get())
        output_dir = Path(self.output_var.get().strip() or DEFAULT_OUTPUT_DIR)
        if not source.name or not variants:
            return []
        return build_output_names(source, stem, variants, output_dir)

    def _refresh_names(self) -> None:
        self.names_box.configure(state="normal")
        self.names_box.delete("1.0", "end")
        try:
            paths = self._planned_paths()
            if paths:
                text = "\n".join(p.name for p in paths)
            else:
                text = "Noch keine Varianten."
            self.names_box.insert("1.0", text)
        except Exception as e:
            self.names_box.insert("1.0", str(e))
        self.names_box.configure(state="disabled")

    def _run(self) -> None:
        source = self.image_var.get().strip()
        if not source:
            messagebox.showerror("Fehler", "Bitte ein Bild wählen.")
            return
        output_dir = self.output_var.get().strip() or str(DEFAULT_OUTPUT_DIR)
        try:
            saved = copy_variants(
                source,
                self.stem_var.get(),
                self.variants_var.get(),
                output_dir,
            )
        except Exception as e:
            logger.exception(f"Fehler beim Kopieren: {e}")
            self.status_var.set("Fehler.")
            messagebox.showerror("Fehler", str(e))
            return

        names = "\n".join(p.name for p in saved)
        self.status_var.set(f"Gespeichert: {len(saved)} Datei(en) in {output_dir}")
        messagebox.showinfo("Fertig", f"{len(saved)} Datei(en) gespeichert:\n{names}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starte VariantenKopiererApp")
    logger.info("=" * 70)
    app = VariantenKopiererApp()
    app.mainloop()
    logger.info("VariantenKopiererApp beendet")
