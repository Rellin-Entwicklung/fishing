#!/usr/bin/env python3
"""
CR3-Quadrat fuer Produktbilder.

- Canon CR3 (und gaengige Bildformate) einlesen
- Objekt vollstaendig freistellen (Hintergrund weg)
- Senkrecht ausrichten
- Quadratisches Zielbild: 1000 / 1500 / 2000 px
- Laengste Objektdimension = 80 % der Kantenlaenge, zentriert auf weiss
- Speichert als WebP
- customtkinter GUI mit Vorschau
- Vollstaendiges Logging mit Loguru
"""

from __future__ import annotations

import sys
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
import rawpy
from loguru import logger
from PIL import Image, ImageOps
from rembg import remove

from axis_alignment import MIN_ROTATION_DEG, _angle_from_pca_silhouette, align_image_vertical


SUPPORTED = {".cr3", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
CANVAS_SIZES = (1000, 1500, 2000)
OBJECT_FILL = 0.80
WEBP_QUALITY = 95
CROP_PAD_RATIO = 0.25
MIN_CROP_PAD_PX = 80
BG_COLOR = (255, 255, 255)
EXAMPLE_STEM = "img_4796"
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
    str(PROJECT_ROOT / "cr3_quadrat_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG",
)


def _example_dirs(root: Path) -> list[Path]:
    wanted = {"testinput", "tests_input"}
    found: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            found.append(child)
    for name in ("testinput", "Tests_Input"):
        candidate = root / name
        if candidate.is_dir() and candidate not in found:
            found.append(candidate)
    return found


def find_example_image(root: Path = PROJECT_ROOT) -> Path | None:
    """Sucht testinput/IMG_4796.* ohne Ruecksicht auf Gross-/Kleinschreibung."""
    dirs = _example_dirs(root)
    if not dirs:
        logger.debug(f"Beispielordner nicht gefunden unter {root}")
        return None
    matches: list[Path] = []
    for folder in dirs:
        matches.extend(
            p for p in folder.iterdir()
            if p.is_file() and p.stem.lower() == EXAMPLE_STEM and p.suffix.lower() in SUPPORTED
        )
    if not matches:
        logger.debug(f"Keine Beispiel-Datei {EXAMPLE_STEM} in {dirs}")
        return None
    cr3 = [p for p in matches if p.suffix.lower() == ".cr3"]
    chosen = cr3[0] if cr3 else matches[0]
    logger.info(f"Beispielbild gefunden: {chosen}")
    return chosen


def _has_transparency(img: Image.Image) -> bool:
    return (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )


def _flatten_on_white(img: Image.Image, bg_color: tuple[int, int, int] = BG_COLOR) -> Image.Image:
    if not _has_transparency(img):
        return img.convert("RGB") if img.mode != "RGB" else img
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, bg_color)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def _stretch_paper_white(img: Image.Image, percentile: float = 99.0) -> Image.Image:
    """Hebt den Studiohintergrund auf Weiss, ohne dunkle Objekte auszubrennen."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    white = float(np.percentile(gray, percentile))
    if white < 1 or white >= 250:
        return img.convert("RGB")
    scale = 255.0 / white
    stretched = np.clip(arr * scale, 0, 255).astype(np.uint8)
    logger.debug(f"Paper-white p{percentile:g}={white:.1f}, scale={scale:.3f}")
    return Image.fromarray(stretched)


def load_image(path: str | Path) -> Image.Image:
    """Laedt CR3 via rawpy, andere Formate via Pillow. Ergebnis ist RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Nicht unterstuetztes Format: {suffix}")

    if suffix == ".cr3":
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Lade CR3 mit rawpy: {path} ({size_mb:.2f} MB)")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        img = _stretch_paper_white(Image.fromarray(rgb))
        logger.success(f"CR3 entwickelt: {img.size[0]}x{img.size[1]} {img.mode}")
        return img

    logger.info(f"Lade Bild mit Pillow: {path}")
    with Image.open(path) as loaded:
        img = ImageOps.exif_transpose(loaded)
        img = _flatten_on_white(img)
        img.load()
    img = _stretch_paper_white(img)
    logger.debug(f"Bild geladen: {img.size[0]}x{img.size[1]} {img.mode}")
    return img


def _object_mask(img: Image.Image, threshold: int = 250) -> np.ndarray:
    if _has_transparency(img):
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
        return (alpha > 10).astype(np.uint8)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return (gray < threshold).astype(np.uint8)


def _selected_component_mask(
    mask: np.ndarray,
    min_area: int = 100,
    relative_min_area: float = 0.02,
    prefer_interior: bool = True,
) -> np.ndarray | None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    height, width = mask.shape[:2]
    interior: list[tuple[int, int]] = []
    all_parts: list[tuple[int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches_border = x <= 0 or y <= 0 or x + bw >= width or y + bh >= height
        all_parts.append((i, area))
        if not touches_border:
            interior.append((i, area))

    candidates = interior if (prefer_interior and interior) else all_parts
    if not candidates:
        return None
    best_area = max(area for _, area in candidates)
    area_threshold = max(min_area, int(best_area * relative_min_area))
    ids = [i for i, area in candidates if area >= area_threshold]
    return np.isin(labels, ids).astype(np.uint8)


def _bbox_from_mask(
    mask: np.ndarray,
    min_area: int,
    relative_min_area: float,
    prefer_interior: bool,
) -> tuple[int, int, int, int] | None:
    selected = _selected_component_mask(mask, min_area, relative_min_area, prefer_interior)
    if selected is None:
        return None
    ys, xs = np.where(selected)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    logger.debug(f"Objektbbox: ({x1}, {y1}, {x2}, {y2})")
    return x1, y1, x2, y2


def detect_object_bbox(
    img: Image.Image,
    threshold: int = 250,
    min_area: int = 100,
    relative_min_area: float = 0.02,
) -> tuple[int, int, int, int]:
    """Bounding-Box des Objekts; bei RAW/Vignette groesste Komponente im Bildinneren."""
    if _has_transparency(img):
        bbox = _bbox_from_mask(_object_mask(img, threshold), min_area, relative_min_area, False)
        if bbox is None:
            raise ValueError("Kein Objekt im Bild erkannt.")
        return bbox

    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _otsu_thr, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_bbox = _bbox_from_mask(
        (otsu_mask > 0).astype(np.uint8),
        min_area,
        relative_min_area,
        prefer_interior=True,
    )
    if otsu_bbox is not None:
        x1, y1, x2, y2 = otsu_bbox
        img_w, img_h = img.size
        coverage = ((x2 - x1) * (y2 - y1)) / max(img_w * img_h, 1)
        if coverage < 0.9:
            logger.debug(
                f"Otsu-Bbox genutzt: ({x1}, {y1}, {x2}, {y2}) "
                f"Abdeckung={coverage:.1%}"
            )
            return otsu_bbox

    bbox = _bbox_from_mask(_object_mask(img, threshold), min_area, relative_min_area, False)
    if bbox is None:
        raise ValueError("Kein Objekt im Bild erkannt (heller Studiohintergrund erwartet).")
    return bbox


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = size
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def _apply_alpha(rgb: Image.Image, mask01: np.ndarray) -> Image.Image:
    rgba = rgb.convert("RGBA")
    arr = np.array(rgba)
    arr[:, :, 3] = (mask01.astype(np.uint8) * 255)
    return Image.fromarray(arr)


def _tighten_cutout(cutout: Image.Image, classic: np.ndarray | None) -> Image.Image:
    arr = np.array(cutout.convert("RGBA"))
    alpha = arr[:, :, 3]
    alpha[alpha < 32] = 0
    if classic is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        keep = cv2.dilate(classic.astype(np.uint8), kernel)
        rembg_area = int(np.count_nonzero(alpha > 128))
        classic_area = int(classic.sum())
        if classic_area > 0 and rembg_area < int(0.6 * classic_area):
            logger.debug("rembg unvollstaendig, vereinige mit Otsu-Maske")
            alpha = np.maximum(alpha, classic * 255)
        alpha = np.where(keep, alpha, 0).astype(np.uint8)
    selected = _selected_component_mask((alpha > 32).astype(np.uint8), prefer_interior=False)
    if selected is not None:
        alpha = np.where(selected, alpha, 0).astype(np.uint8)
    alpha[alpha > 240] = 255
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def _align_cutout_vertical(cutout: Image.Image) -> tuple[Image.Image, float, str]:
    """Richtet die Silhouette per PCA senkrecht aus (stabiler als minAreaRect)."""
    pca_angle = _angle_from_pca_silhouette(cutout)
    if pca_angle is not None and abs(pca_angle) >= MIN_ROTATION_DEG:
        logger.info(f"PCA-Ausrichtung {pca_angle:.2f}°")
        rotated = cutout.rotate(
            -pca_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        return rotated, pca_angle, "pca"
    aligned, angle, method = align_image_vertical(cutout, mode="auto")
    logger.info(f"Fallback-Ausrichtung {angle:.2f}° ({method})")
    return aligned, angle, method


def _classic_object_mask(rgb: Image.Image) -> np.ndarray | None:
    gray = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _thr, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _selected_component_mask((otsu > 0).astype(np.uint8), prefer_interior=False)


def extract_and_align(img: Image.Image) -> tuple[Image.Image, dict[str, float | int | str]]:
    """Stellt das Objekt vollstaendig frei und richtet es senkrecht aus."""
    rgb = img.convert("RGB")
    bbox = detect_object_bbox(rgb)
    x1, y1, x2, y2 = bbox
    obj_w, obj_h = x2 - x1, y2 - y1
    pad = max(MIN_CROP_PAD_PX, int(CROP_PAD_RATIO * max(obj_w, obj_h)))
    crop_box = _pad_bbox(bbox, rgb.size, pad)
    crop = rgb.crop(crop_box)
    logger.info(f"Freistellen: crop={crop.size[0]}x{crop.size[1]} pad={pad}px")

    cutout: Image.Image | None = None
    try:
        logger.info("Starte rembg-Hintergrundentfernung")
        cutout = remove(crop).convert("RGBA")
        if cutout.getbbox() is None:
            logger.warning("rembg lieferte leeres Ergebnis, nutze Otsu-Maske")
            cutout = None
        else:
            logger.success(f"rembg fertig: {cutout.size[0]}x{cutout.size[1]} {cutout.mode}")
    except Exception as e:
        logger.warning(f"rembg fehlgeschlagen ({e}), nutze Otsu-Maske")
        cutout = None

    classic = _classic_object_mask(crop)
    if cutout is None:
        if classic is None:
            logger.error("Weder rembg noch Otsu-Maske haben ein Objekt gefunden")
            raise ValueError("Objekt konnte nicht freigestellt werden.")
        logger.info("Freistellung per Otsu-Maske")
        cutout = _apply_alpha(crop, classic)
    else:
        cutout = _tighten_cutout(cutout, classic)

    aligned, angle, method = _align_cutout_vertical(cutout)
    obj_box = aligned.getbbox()
    if obj_box is None:
        raise ValueError("Nach Freistellung kein Objekt sichtbar.")
    isolated = aligned.crop(obj_box)
    logger.success(
        f"Freigestellt {isolated.size[0]}x{isolated.size[1]}px, "
        f"Drehung {angle:.2f}° ({method})"
    )
    return isolated, {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "object_w": isolated.size[0],
        "object_h": isolated.size[1],
        "angle_deg": round(angle, 3),
        "align_method": method,
    }


def build_square_image(
    img: Image.Image,
    canvas_size: int,
    fill_ratio: float = OBJECT_FILL,
    bg_color: tuple[int, int, int] = BG_COLOR,
    isolated: Image.Image | None = None,
    extra_info: dict[str, float | int | str] | None = None,
) -> tuple[Image.Image, dict[str, float | int | str]]:
    """Stellt das Objekt frei, richtet es aus und setzt es auf ein weisses Quadrat."""
    if canvas_size not in CANVAS_SIZES:
        raise ValueError(f"Kantenlaenge muss eine von {CANVAS_SIZES} sein.")
    if not 0 < fill_ratio <= 1:
        raise ValueError("Fuellanteil muss zwischen 0 und 1 liegen.")

    info: dict[str, float | int | str] = dict(extra_info) if extra_info else {}
    if isolated is None:
        isolated, info = extract_and_align(img)

    obj_w, obj_h = isolated.size
    longest = max(obj_w, obj_h)
    if longest <= 0:
        raise ValueError("Erkannte Objektdimension ist ungueltig.")

    target_longest = max(1, round(fill_ratio * canvas_size))
    scale = target_longest / longest
    new_w = max(1, round(obj_w * scale))
    new_h = max(1, round(obj_h * scale))
    fitted = isolated.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    if fitted.mode == "RGBA":
        canvas.paste(fitted, (paste_x, paste_y), fitted)
    else:
        canvas.paste(fitted.convert("RGB"), (paste_x, paste_y))

    info.update({
        "object_w": obj_w,
        "object_h": obj_h,
        "scale": scale,
        "placed_w": new_w,
        "placed_h": new_h,
        "canvas": canvas_size,
        "fill_pct": round(fill_ratio * 100),
    })
    logger.info(
        f"Quadrat {canvas_size}px: Objekt {obj_w}x{obj_h}px -> {new_w}x{new_h}px "
        f"(scale={scale:.4f}, {fill_ratio:.0%} der Kante)"
    )
    return canvas, info


def default_output_path(source: str | Path, canvas_size: int) -> Path:
    source = Path(source)
    return DEFAULT_OUTPUT_DIR / f"{source.stem}_{canvas_size}.webp"


def save_image(img: Image.Image, output_path: str | Path, quality: int = WEBP_QUALITY) -> Path:
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    rgb = img.convert("RGB")
    logger.info(f"Speichere {output_path.name} als {ext or '.webp'} ({rgb.size[0]}x{rgb.size[1]}, q={quality})")
    if ext in {".jpg", ".jpeg"}:
        rgb.save(output_path, quality=quality, optimize=True)
    elif ext == ".png":
        rgb.save(output_path, optimize=True)
    else:
        if ext != ".webp":
            output_path = output_path.with_suffix(".webp")
        rgb.save(output_path, format="WEBP", quality=quality, method=6)
    size_kb = output_path.stat().st_size / 1024
    logger.success(f"Gespeichert: {output_path} ({rgb.size[0]}x{rgb.size[1]}, {size_kb:.1f} KB)")
    return output_path


def process_file(
    source: str | Path,
    canvas_size: int,
    output_path: str | Path | None = None,
    fill_ratio: float = OBJECT_FILL,
) -> tuple[Path, dict[str, float | int | str]]:
    source = Path(source)
    logger.info("=" * 70)
    logger.info(f"Starte Verarbeitung: {source}")
    logger.info(f"Zielgroesse: {canvas_size}x{canvas_size}px, Fuellung {fill_ratio:.0%}")
    canvas, info = build_square_image(load_image(source), canvas_size, fill_ratio)
    if output_path is None:
        output_path = default_output_path(source, canvas_size)
    saved = save_image(canvas, output_path)
    logger.info(
        f"Objekt {info['object_w']}x{info['object_h']}px -> "
        f"{info['placed_w']}x{info['placed_h']}px, Drehung {info.get('angle_deg', 0)}°"
    )
    logger.success(f"Fertig: {saved}")
    logger.info("=" * 70)
    return saved, info


class Cr3QuadratApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CR3 Quadrat")
        self.geometry("1100x760")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        example = find_example_image()
        self.image_var = ctk.StringVar(value=str(example) if example else "")
        self.output_var = ctk.StringVar()
        self.size_var = ctk.StringVar(value="1500")
        self.status_var = ctk.StringVar(value="Bereit.")
        self.preview_image: ctk.CTkImage | None = None
        self._loaded_path: Path | None = None
        self._loaded_img: Image.Image | None = None
        self._cutout_img: Image.Image | None = None
        self._cutout_info: dict[str, float | int | str] | None = None
        self._result_img: Image.Image | None = None

        if example:
            self.output_var.set(str(default_output_path(example, int(self.size_var.get()))))

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=14)
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            outer,
            text="CR3 Quadrat – Objekt 80 % der Kante",
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

        ctk.CTkLabel(form, text="Bild / CR3").grid(row=0, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.image_var, width=280).grid(row=0, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Wählen", width=90, command=self._choose_image).grid(
            row=0, column=2, padx=(0, 14)
        )

        ctk.CTkLabel(form, text="Kantenlänge").grid(row=1, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkSegmentedButton(
            form,
            values=[str(s) for s in CANVAS_SIZES],
            variable=self.size_var,
            command=self._on_size_change,
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(10, 14))

        ctk.CTkLabel(form, text="Objektfüllung").grid(row=2, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkLabel(form, text="80 % der längsten Kante (fest)").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=10
        )

        ctk.CTkLabel(form, text="Ausgabe").grid(row=3, column=0, sticky="w", pady=8, padx=(14, 0))
        ctk.CTkEntry(form, textvariable=self.output_var).grid(row=3, column=1, sticky="ew", padx=10)
        ctk.CTkButton(form, text="Speichern als", width=90, command=self._choose_output).grid(
            row=3, column=2, padx=(0, 14)
        )

        info = (
            "Objekt wird vollständig freigestellt, senkrecht ausgerichtet\n"
            "und als WebP auf einem weißen Quadrat gespeichert."
        )
        ctk.CTkLabel(form, text=info, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 14)
        )

        button_row = ctk.CTkFrame(form, fg_color="transparent")
        button_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=14)
        ctk.CTkButton(button_row, text="Vorschau", height=38, command=self._preview).pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(button_row, text="Speichern", height=38, command=self._run).pack(
            side="left", fill="x", expand=True, padx=(10, 0)
        )

        ctk.CTkLabel(form, textvariable=self.status_var, anchor="w", wraplength=420).grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=14, pady=(12, 16)
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

    def _on_size_change(self, _value: str | None = None) -> None:
        source = self.image_var.get().strip()
        if source:
            self.output_var.set(str(default_output_path(source, int(self.size_var.get()))))
        if self._loaded_img is not None:
            self._preview()

    def _choose_image(self) -> None:
        initial = next(iter(_example_dirs(PROJECT_ROOT)), PROJECT_ROOT)
        path = filedialog.askopenfilename(
            title="CR3 oder Bild auswählen",
            initialdir=str(initial if initial.is_dir() else PROJECT_ROOT),
            filetypes=[
                ("Canon CR3 / Bilder", "*.cr3 *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("Canon CR3", "*.cr3"),
                ("Bilder", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)
            self.output_var.set(str(default_output_path(path, int(self.size_var.get()))))
            self._loaded_path = None
            self._loaded_img = None
            self._cutout_img = None
            self._cutout_info = None
            self.status_var.set(f"Gewaehlt: {Path(path).name}")
            logger.info(f"Bild gewaehlt: {path}")

    def _choose_output(self) -> None:
        suggested = self.output_var.get().strip() or str(DEFAULT_OUTPUT_DIR / "quadrat.webp")
        path = filedialog.asksaveasfilename(
            title="Ausgabedatei wählen",
            initialfile=Path(suggested).name,
            initialdir=str(Path(suggested).parent),
            defaultextension=".webp",
            filetypes=[("WebP", "*.webp"), ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("Alle Dateien", "*.*")],
        )
        if path:
            self.output_var.set(path)
            logger.info(f"Ausgabe gewaehlt: {path}")

    def _canvas_size(self) -> int:
        try:
            size = int(self.size_var.get())
        except ValueError as e:
            raise ValueError("Bitte eine gueltige Kantenlaenge waehlen.") from e
        if size not in CANVAS_SIZES:
            raise ValueError(f"Kantenlaenge muss eine von {CANVAS_SIZES} sein.")
        return size

    def _ensure_loaded(self) -> Image.Image:
        image_path = self.image_var.get().strip()
        if not image_path:
            raise ValueError("Bitte ein CR3- oder Bilddatei auswaehlen.")
        path = Path(image_path)
        if path.suffix.lower() not in SUPPORTED:
            raise ValueError("Das Bildformat wird nicht unterstuetzt.")
        if self._loaded_img is not None and self._loaded_path == path:
            return self._loaded_img
        self.status_var.set("Lade Bild...")
        self.update_idletasks()
        img = load_image(path)
        self._loaded_path = path
        self._loaded_img = img
        self._cutout_img = None
        self._cutout_info = None
        logger.info(f"Bild im Speicher: {path.name} {img.size[0]}x{img.size[1]}")
        return img

    def _ensure_cutout(self) -> tuple[Image.Image, dict[str, float | int | str]]:
        img = self._ensure_loaded()
        if self._cutout_img is not None and self._cutout_info is not None:
            return self._cutout_img, self._cutout_info
        self.status_var.set("Stelle Objekt frei und richte aus...")
        self.update_idletasks()
        isolated, info = extract_and_align(img)
        self._cutout_img = isolated
        self._cutout_info = info
        logger.debug(f"Cutout gecacht: {isolated.size[0]}x{isolated.size[1]}")
        return isolated, info

    def _compose(self) -> tuple[Image.Image, dict[str, float | int | str]]:
        isolated, extra = self._ensure_cutout()
        canvas, info = build_square_image(
            self._loaded_img,
            self._canvas_size(),
            isolated=isolated,
            extra_info=extra,
        )
        self._result_img = canvas
        return canvas, info

    def _status_from_info(self, info: dict[str, float | int | str], prefix: str) -> str:
        angle = info.get("angle_deg", 0)
        return (
            f"{prefix} {info['object_w']}x{info['object_h']}px -> "
            f"{info['placed_w']}x{info['placed_h']}px "
            f"scale={float(info['scale']):.3f} auf {info['canvas']}px, "
            f"Drehung {float(angle):.1f}°"
        )

    def _update_preview(self, pil_image: Image.Image) -> None:
        preview = pil_image.copy()
        preview.thumbnail((620, 560), Image.Resampling.LANCZOS)
        self.preview_image = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
        self.preview_label.configure(image=self.preview_image, text="")

    def _preview(self) -> None:
        logger.info("Erzeuge Vorschau")
        try:
            canvas, info = self._compose()
            self._update_preview(canvas)
            self.status_var.set(self._status_from_info(info, "Vorschau:"))
            logger.success("Vorschau aktualisiert")
        except Exception as e:
            logger.exception(f"Fehler bei Vorschau: {e}")
            self.status_var.set("Fehler in Vorschau.")
            messagebox.showerror("Fehler", str(e))

    def _run(self) -> None:
        try:
            canvas_size = self._canvas_size()
            source = self.image_var.get().strip()
            if not source:
                raise ValueError("Bitte ein CR3- oder Bilddatei auswaehlen.")
            output = self.output_var.get().strip() or str(default_output_path(source, canvas_size))
            canvas, info = self._compose()
            logger.info(f"Speichere Ergebnis nach {output}")
            saved = save_image(canvas, output)
            self._update_preview(canvas)
        except Exception as e:
            logger.exception(f"Fehler beim Speichern: {e}")
            self.status_var.set("Fehler.")
            messagebox.showerror("Fehler", str(e))
            return

        self.status_var.set(self._status_from_info(info, f"Gespeichert {saved.name}:"))
        messagebox.showinfo("Fertig", f"Bild gespeichert:\n{saved}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starte Cr3QuadratApp")
    logger.info(f"Log-Datei: {PROJECT_ROOT / 'cr3_quadrat_*.log'}")
    logger.info("=" * 70)
    app = Cr3QuadratApp()
    app.mainloop()
    logger.info("Cr3QuadratApp beendet")
