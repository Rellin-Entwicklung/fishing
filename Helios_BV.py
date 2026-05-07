"""Helios - Bildverarbeitung: Hintergrund entfernen, drehen, zentrieren, quadratisch beschneiden."""

import io
from pathlib import Path

from PIL import Image
from rembg import remove


def process_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    size: int = 1024,
    quality: int = 80,
) -> Path:
    """
    Entfernt den Hintergrund, dreht 90° gegen UZS, zentriert das Objekt,
    beschneidet quadratisch und speichert als WebP (size x size, quality %).

    Args:
        input_path: Pfad zum Eingabebild
        output_path: Pfad für WebP-Ausgabe (default: Eingabe_processed.webp)
        size: Ausgabegröße in Pixel (quadratisch), default 1024
        quality: WebP-Qualität 0–100, default 80

    Returns:
        Pfad der gespeicherten Datei
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_processed.webp"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Bild laden und Hintergrund entfernen
    with open(input_path, "rb") as f:
        input_data = f.read()
    img_data = remove(input_data)
    img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    # 2. 90° gegen Uhrzeigersinn drehen
    img = img.transpose(Image.ROTATE_90)

    # 3. Begrenzungsrahmen des sichtbaren Objekts (nicht vollständig transparent)
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("Nach Hintergrundentfernung kein sichtbares Objekt gefunden.")

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    side = max(w, h)

    # 4. Quadratisches Canvas, Objekt zentriert
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    canvas.paste(img.crop(bbox), (paste_x, paste_y))

    # 5. Auf Zielgröße skalieren und als WebP speichern
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    canvas.save(output_path, "WEBP", quality=quality)

    return output_path


def main() -> None:
    """Kommandozeile: Eingabepfad und optional Ausgabepfad."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hintergrund entfernen, 90° gegen UZS drehen, zentrieren, quadratisch beschneiden, WebP 1024×1024 80%%"
    )
    parser.add_argument("input", type=Path, help="Eingabebild (z.B. PNG, JPG)")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Ausgabepfad (default: <name>_processed.webp)",
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=1024,
        help="Quadratische Ausgabegröße in Pixel (default: 1024)",
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=80,
        help="WebP-Qualität 0–100 (default: 80)",
    )
    args = parser.parse_args()

    out = process_image(
        args.input,
        output_path=args.output,
        size=args.size,
        quality=args.quality,
    )
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
