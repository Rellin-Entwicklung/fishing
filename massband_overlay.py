#!/usr/bin/env python3
"""
Massband-Overlay fuer Produktbilder.

- customtkinter GUI
- Bildauswahl per Dialog
- Eingabe der realen Objektbreite in cm
- Automatische Erkennung der Objektbreite in Pixeln
- Massband/Koordinatensystem in X und Y ohne Objektverdecken
- Vollstaendiges Logging mit Loguru
"""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
LEFT_MARGIN_PX = 140
TOP_MARGIN_PX = 120
RIGHT_MARGIN_PX = 40
BOTTOM_MARGIN_PX = 40
BG_COLOR = (255, 255, 255)
GRID_COLOR = (210, 210, 210)
AXIS_COLOR = (40, 40, 40)
TEXT_COLOR = (25, 25, 25)
COLOR_PRESETS = {
    "Schwarz": "#202020",
    "Dunkelgrau": "#505050",
    "Hellgrau": "#d2d2d2",
    "Blau": "#2f6fed",
    "Rot": "#c73a3a",
    "Gruen": "#2f8f46",
}


logger.remove()
logger.add(
    __import__("sys").stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add(
    "massband_overlay_{time:YYYY-MM-DD}.log",
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


def _flatten_on_white(img: Image.Image) -> Image.Image:
    if not _has_transparency(img):
        return img.convert("RGB") if img.mode != "RGB" else img
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, BG_COLOR)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def _foreground_mask(img: Image.Image, threshold: int = 250) -> np.ndarray:
    """Vordergrundmaske als 0/1-Array, robust fuer weisse Hintergruende."""
    if _has_transparency(img):
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
        return (alpha > 10).astype(np.uint8)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return (gray < threshold).astype(np.uint8)


def _relevant_component_bbox(
    img: Image.Image,
    threshold: int = 250,
    min_area: int = 100,
    relative_min_area: float = 0.02,
) -> tuple[int, int, int, int] | None:
    """
    Bounding-Box aller relevanten Komponenten.

    Mehrere echte Objekte bleiben erhalten, kleine Artefakte werden ignoriert.
    """
    mask = _foreground_mask(img, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        logger.warning("Keine Vordergrund-Komponenten gefunden")
        return None

    best_area = max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))
    if best_area < min_area:
        logger.warning(f"Groesste Komponente zu klein: {best_area}px")
        return None

    area_threshold = max(min_area, int(best_area * relative_min_area))
    relevant_ids = [i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= area_threshold]
    if not relevant_ids:
        logger.warning("Keine relevanten Komponenten nach Filterung uebrig")
        return None

    relevant_mask = np.isin(labels, relevant_ids)
    ys, xs = np.where(relevant_mask)
    if len(xs) == 0 or len(ys) == 0:
        logger.warning("Leere Pixelmenge nach Komponentenfilter")
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    logger.debug(f"Relevante Komponenten: {len(relevant_ids)}, bbox=({x1}, {y1}, {x2}, {y2})")
    return x1, y1, x2, y2


def _tick_step_cm(px_per_cm: float) -> int:
    """Waehlt sinnvolle Beschriftungs/Tick-Schritte fuer kleine oder grosse Bilder."""
    if px_per_cm >= 60:
        return 1
    if px_per_cm >= 25:
        return 2
    if px_per_cm >= 12:
        return 5
    return 10


def _parse_hex_color(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    text = value.strip()
    if not text:
        return fallback
    if not text.startswith("#"):
        text = "#" + text
    try:
        return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
    except (TypeError, ValueError):
        logger.warning(f"Ungueltige Farbdefinition '{value}', nutze Fallback {fallback}")
        return fallback


def _draw_rulers(
    canvas: Image.Image,
    object_bbox: tuple[int, int, int, int],
    px_per_cm: float,
    *,
    left_margin_px: int,
    top_margin_px: int,
    bottom_margin_px: int,
    right_margin_px: int,
    grid_color: tuple[int, int, int],
    axis_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x1, y1, x2, y2 = object_bbox
    obj_w = x2 - x1
    obj_h = y2 - y1
    max_cm_x = max(1, int(round(obj_w / px_per_cm)))
    max_cm_y = max(1, int(round(obj_h / px_per_cm)))
    step_cm = _tick_step_cm(px_per_cm)

    axis_x = left_margin_px - 35
    axis_y = top_margin_px - 35

    logger.debug(
        f"Ruler setup: obj_w={obj_w}px obj_h={obj_h}px px_per_cm={px_per_cm:.3f} "
        f"max_cm_x={max_cm_x} max_cm_y={max_cm_y} step_cm={step_cm}"
    )

    # Axes in added margins only.
    draw.line((axis_x, axis_y, axis_x, canvas.height - bottom_margin_px), fill=axis_color, width=2)
    draw.line((axis_x, axis_y, canvas.width - right_margin_px, axis_y), fill=axis_color, width=2)

    # origin marker
    draw.text((axis_x - 10, axis_y - 16), "0", fill=text_color, font=font)

    # Horizontal ruler aligned to detected object width.
    cm = 0
    while True:
        x = int(round(x1 + cm * px_per_cm))
        if x > x2:
            break
        tick_len = 18 if cm % 5 == 0 else 10
        draw.line((x, axis_y, x, axis_y + tick_len), fill=axis_color, width=1)
        if cm % step_cm == 0:
            label = f"{cm} cm"
            tw = draw.textbbox((0, 0), label, font=font)[2]
            draw.text((x - tw / 2, axis_y - 18), label, fill=text_color, font=font)
        cm += 1

    # Vertical ruler aligned to detected object height.
    cm = 0
    while True:
        y = int(round(y1 + cm * px_per_cm))
        if y > y2:
            break
        tick_len = 18 if cm % 5 == 0 else 10
        draw.line((axis_x, y, axis_x + tick_len, y), fill=axis_color, width=1)
        if cm % step_cm == 0:
            label = f"{cm} cm"
            bbox = draw.textbbox((0, 0), label, font=font)
            th = bbox[3] - bbox[1]
            draw.text((8, y - th / 2), label, fill=text_color, font=font)
        cm += 1

    # Light guides only in free margin areas, never across the object.
    cm = 0
    while True:
        x = int(round(x1 + cm * px_per_cm))
        if x > x2:
            break
        draw.line((x, axis_y, x, y1 - 8), fill=grid_color, width=1)
        cm += 1

    cm = 0
    while True:
        y = int(round(y1 + cm * px_per_cm))
        if y > y2:
            break
        draw.line((axis_x, y, x1 - 8, y), fill=grid_color, width=1)
        cm += 1

    draw.text((axis_x + 10, 10), "X", fill=text_color, font=font)
    draw.text((10, axis_y + 10), "Y", fill=text_color, font=font)


def build_measure_overlay_image(
    image_path: str | Path,
    object_width_cm: float,
    *,
    left_margin_px: int = LEFT_MARGIN_PX,
    top_margin_px: int = TOP_MARGIN_PX,
    right_margin_px: int = RIGHT_MARGIN_PX,
    bottom_margin_px: int = BOTTOM_MARGIN_PX,
    bg_color: tuple[int, int, int] = BG_COLOR,
    grid_color: tuple[int, int, int] = GRID_COLOR,
    axis_color: tuple[int, int, int] = AXIS_COLOR,
    text_color: tuple[int, int, int] = TEXT_COLOR,
) -> Path:
    image_path = Path(image_path)
    if object_width_cm <= 0:
        raise ValueError("Objektbreite in cm muss groesser als 0 sein.")
    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    with Image.open(image_path) as loaded:
        img = ImageOps.exif_transpose(loaded)
        img_rgb = _flatten_on_white(img)

    bbox = _relevant_component_bbox(img)
    if bbox is None:
        raise ValueError("Objekt konnte im Bild nicht erkannt werden.")

    x1, y1, x2, y2 = bbox
    object_width_px = x2 - x1
    if object_width_px <= 0:
        raise ValueError("Erkannte Objektbreite ist ungueltig.")

    px_per_cm = object_width_px / object_width_cm
    logger.info(
        f"Objektbbox: x={x1}-{x2}, y={y1}-{y2}, "
        f"Breite={object_width_px}px, Hoehe={y2 - y1}px, px/cm={px_per_cm:.4f}"
    )
    logger.debug(
        "Styling: "
        f"left={left_margin_px}px top={top_margin_px}px right={right_margin_px}px bottom={bottom_margin_px}px "
        f"bg={bg_color} grid={grid_color} axis={axis_color} text={text_color}"
    )

    canvas = Image.new(
        "RGB",
        (
            img_rgb.width + left_margin_px + right_margin_px,
            img_rgb.height + top_margin_px + bottom_margin_px,
        ),
        bg_color,
    )
    canvas.paste(img_rgb, (left_margin_px, top_margin_px))
    shifted_bbox = (
        x1 + left_margin_px,
        y1 + top_margin_px,
        x2 + left_margin_px,
        y2 + top_margin_px,
    )

    logger.debug(
        f"Canvas erweitert auf {canvas.width}x{canvas.height}px, "
        f"verschobene bbox={shifted_bbox}"
    )

    _draw_rulers(
        canvas,
        shifted_bbox,
        px_per_cm,
        left_margin_px=left_margin_px,
        top_margin_px=top_margin_px,
        bottom_margin_px=bottom_margin_px,
        right_margin_px=right_margin_px,
        grid_color=grid_color,
        axis_color=axis_color,
        text_color=text_color,
    )
    return canvas


def generate_measure_overlay(
    image_path: str | Path,
    object_width_cm: float,
    output_path: str | Path | None = None,
    *,
    left_margin_px: int = LEFT_MARGIN_PX,
    top_margin_px: int = TOP_MARGIN_PX,
    right_margin_px: int = RIGHT_MARGIN_PX,
    bottom_margin_px: int = BOTTOM_MARGIN_PX,
    bg_color: tuple[int, int, int] = BG_COLOR,
    grid_color: tuple[int, int, int] = GRID_COLOR,
    axis_color: tuple[int, int, int] = AXIS_COLOR,
    text_color: tuple[int, int, int] = TEXT_COLOR,
) -> Path:
    logger.info("=" * 70)
    logger.info(f"Starte Massband-Overlay fuer: {image_path}")
    logger.info(f"Eingegebene Objektbreite: {object_width_cm:.3f} cm")

    image_path = Path(image_path)
    canvas = build_measure_overlay_image(
        image_path,
        object_width_cm,
        left_margin_px=left_margin_px,
        top_margin_px=top_margin_px,
        right_margin_px=right_margin_px,
        bottom_margin_px=bottom_margin_px,
        bg_color=bg_color,
        grid_color=grid_color,
        axis_color=axis_color,
        text_color=text_color,
    )

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_massband.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    logger.success(f"Overlay gespeichert: {output_path}")
    logger.info("=" * 70)
    return output_path


class MassbandOverlayApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Massband Overlay")
        self.geometry("1100x760")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.image_var = ctk.StringVar()
        self.width_cm_var = ctk.StringVar(value="10")
        self.output_var = ctk.StringVar()
        self.status_var = ctk.StringVar(value="Bereit.")
        self.left_margin_var = ctk.StringVar(value=str(LEFT_MARGIN_PX))
        self.top_margin_var = ctk.StringVar(value=str(TOP_MARGIN_PX))
        self.right_margin_var = ctk.StringVar(value=str(RIGHT_MARGIN_PX))
        self.bottom_margin_var = ctk.StringVar(value=str(BOTTOM_MARGIN_PX))
        self.axis_color_var = ctk.StringVar(value=COLOR_PRESETS["Schwarz"])
        self.grid_color_var = ctk.StringVar(value=COLOR_PRESETS["Hellgrau"])
        self.text_color_var = ctk.StringVar(value=COLOR_PRESETS["Schwarz"])
        self.preview_image: ctk.CTkImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=14)
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            outer,
            text="Massband Overlay fuer Produktbilder",
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

        ctk.CTkLabel(form, text="Bild").grid(row=0, column=0, sticky="w", pady=8)
        ctk.CTkEntry(form, textvariable=self.image_var).grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Waehlen", width=90, command=self._choose_image).grid(row=0, column=2)

        ctk.CTkLabel(form, text="Objektbreite in cm").grid(row=1, column=0, sticky="w", pady=8)
        ctk.CTkEntry(form, textvariable=self.width_cm_var).grid(row=1, column=1, sticky="ew", padx=10)

        ctk.CTkLabel(form, text="Ausgabe (optional)").grid(row=2, column=0, sticky="w", pady=8)
        ctk.CTkEntry(form, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Speichern als", width=90, command=self._choose_output).grid(row=2, column=2)

        ctk.CTkLabel(form, text="Rand links / oben").grid(row=3, column=0, sticky="w", pady=8)
        margin_lt = ctk.CTkFrame(form, fg_color="transparent")
        margin_lt.grid(row=3, column=1, columnspan=2, sticky="ew", padx=10)
        ctk.CTkEntry(margin_lt, textvariable=self.left_margin_var, width=70).pack(side="left")
        ctk.CTkEntry(margin_lt, textvariable=self.top_margin_var, width=70).pack(side="left", padx=8)

        ctk.CTkLabel(form, text="Rand rechts / unten").grid(row=4, column=0, sticky="w", pady=8)
        margin_rb = ctk.CTkFrame(form, fg_color="transparent")
        margin_rb.grid(row=4, column=1, columnspan=2, sticky="ew", padx=10)
        ctk.CTkEntry(margin_rb, textvariable=self.right_margin_var, width=70).pack(side="left")
        ctk.CTkEntry(margin_rb, textvariable=self.bottom_margin_var, width=70).pack(side="left", padx=8)

        ctk.CTkLabel(form, text="Achsenfarbe").grid(row=5, column=0, sticky="w", pady=8)
        ctk.CTkOptionMenu(form, values=list(COLOR_PRESETS.values()), variable=self.axis_color_var).grid(
            row=5, column=1, columnspan=2, sticky="ew", padx=10
        )

        ctk.CTkLabel(form, text="Rasterfarbe").grid(row=6, column=0, sticky="w", pady=8)
        ctk.CTkOptionMenu(form, values=list(COLOR_PRESETS.values()), variable=self.grid_color_var).grid(
            row=6, column=1, columnspan=2, sticky="ew", padx=10
        )

        ctk.CTkLabel(form, text="Textfarbe").grid(row=7, column=0, sticky="w", pady=8)
        ctk.CTkOptionMenu(form, values=list(COLOR_PRESETS.values()), variable=self.text_color_var).grid(
            row=7, column=1, columnspan=2, sticky="ew", padx=10
        )

        info = (
            "Das Script erkennt die gesamte Objektbreite im Bild und erzeugt\n"
            "ein Massband/Koordinatensystem in X und Y im erweiterten Randbereich."
        )
        ctk.CTkLabel(form, text=info, justify="left").grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 14))

        button_row = ctk.CTkFrame(form, fg_color="transparent")
        button_row.grid(row=9, column=0, columnspan=3, sticky="ew")
        ctk.CTkButton(
            button_row,
            text="Vorschau",
            height=38,
            command=self._preview,
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            button_row,
            text="Overlay erzeugen",
            height=38,
            command=self._run,
        ).pack(side="left", fill="x", expand=True, padx=(10, 0))

        ctk.CTkLabel(form, textvariable=self.status_var, anchor="w").grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

        preview_frame = ctk.CTkFrame(body)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            preview_frame,
            text="Vorschau",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 8))
        self.preview_label = ctk.CTkLabel(preview_frame, text="Noch keine Vorschau erzeugt.")
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Bild auswaehlen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")],
        )
        if path:
            self.image_var.set(path)
            if not self.output_var.get().strip():
                p = Path(path)
                self.output_var.set(str(p.with_name(f"{p.stem}_massband.png")))
            logger.info(f"Bild gewaehlt: {path}")

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Ausgabedatei waehlen",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("WebP", "*.webp"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.output_var.set(path)
            logger.info(f"Ausgabe gewaehlt: {path}")

    def _read_settings(self) -> tuple[float, dict[str, object]]:
        try:
            object_width_cm = float(self.width_cm_var.get().strip().replace(",", "."))
        except ValueError as e:
            raise ValueError("Bitte eine gueltige Objektbreite in cm eingeben.") from e

        try:
            left = int(self.left_margin_var.get().strip())
            top = int(self.top_margin_var.get().strip())
            right = int(self.right_margin_var.get().strip())
            bottom = int(self.bottom_margin_var.get().strip())
        except ValueError as e:
            raise ValueError("Randwerte muessen ganze Zahlen sein.") from e

        for name, value in {
            "links": left,
            "oben": top,
            "rechts": right,
            "unten": bottom,
        }.items():
            if value < 0:
                raise ValueError(f"Randwert fuer {name} darf nicht negativ sein.")

        settings: dict[str, object] = {
            "left_margin_px": left,
            "top_margin_px": top,
            "right_margin_px": right,
            "bottom_margin_px": bottom,
            "axis_color": _parse_hex_color(self.axis_color_var.get(), AXIS_COLOR),
            "grid_color": _parse_hex_color(self.grid_color_var.get(), GRID_COLOR),
            "text_color": _parse_hex_color(self.text_color_var.get(), TEXT_COLOR),
            "bg_color": BG_COLOR,
        }
        return object_width_cm, settings

    def _validate_common(self) -> tuple[str, str | None, float, dict[str, object]]:
        image_path = self.image_var.get().strip()
        output_path = self.output_var.get().strip() or None

        if not image_path:
            raise ValueError("Bitte ein Bild auswaehlen.")
        if Path(image_path).suffix.lower() not in SUPPORTED:
            raise ValueError("Das Bildformat wird nicht unterstuetzt.")

        object_width_cm, settings = self._read_settings()
        return image_path, output_path, object_width_cm, settings

    def _update_preview(self, pil_image: Image.Image) -> None:
        preview = pil_image.copy()
        preview.thumbnail((620, 560), Image.Resampling.LANCZOS)
        self.preview_image = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
        self.preview_label.configure(image=self.preview_image, text="")

    def _preview(self) -> None:
        try:
            image_path, _, object_width_cm, settings = self._validate_common()
            preview_img = build_measure_overlay_image(
                image_path,
                object_width_cm,
                **settings,
            )
            self._update_preview(preview_img.convert("RGB"))
            self.status_var.set("Vorschau aktualisiert.")
        except Exception as e:
            logger.exception(f"Fehler bei Vorschau: {e}")
            self.status_var.set("Fehler in Vorschau.")
            messagebox.showerror("Fehler", str(e))

    def _run(self) -> None:
        try:
            image_path, output_path, object_width_cm, settings = self._validate_common()
        except ValueError as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.status_var.set("Verarbeite...")
        try:
            saved = generate_measure_overlay(
                image_path,
                object_width_cm,
                output_path,
                **settings,
            )
            with Image.open(saved) as img:
                self._update_preview(img.convert("RGB"))
        except Exception as e:
            logger.exception(f"Fehler beim Erzeugen des Overlays: {e}")
            self.status_var.set("Fehler.")
            messagebox.showerror("Fehler", str(e))
            return

        self.status_var.set(f"Gespeichert: {saved}")
        messagebox.showinfo("Fertig", f"Overlay gespeichert:\n{saved}")


if __name__ == "__main__":
    logger.info("Starte MassbandOverlayApp")
    app = MassbandOverlayApp()
    app.mainloop()
