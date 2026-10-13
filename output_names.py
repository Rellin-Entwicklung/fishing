"""Laden und Validieren von Ausgabenamen aus einer .md-Datei im Eingabeordner."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_images(folder: str | Path) -> list[Path]:
    """Sammelt Bilddateien im Ordner, sortiert nach Dateiname."""
    folder = Path(folder)
    files: set[Path] = set()
    for ext in IMAGE_EXTENSIONS:
        files.update(folder.glob(f"*{ext}"))
        files.update(folder.glob(f"*{ext.upper()}"))
    return sorted(files, key=lambda p: p.name.lower())


def find_name_list_md(folder: str | Path) -> Path:
    """Findet die einzige .md-Datei im Eingabeordner."""
    folder = Path(folder)
    md_files = sorted(folder.glob("*.md"), key=lambda p: p.name.lower())
    if not md_files:
        raise FileNotFoundError(
            "Keine .md-Datei im Eingabeordner gefunden. "
            "Lege eine Markdown-Liste mit den gewünschten Dateinamen ab."
        )
    if len(md_files) > 1:
        names = ", ".join(p.name for p in md_files)
        raise ValueError(
            f"Mehrere .md-Dateien gefunden ({names}). "
            "Bitte nur eine Namensliste im Eingabeordner belassen."
        )
    return md_files[0]


def parse_names_from_md(path: str | Path) -> list[str]:
    """
    Liest Ausgabenamen aus einer Markdown-Datei (eine Zeile = ein Name).

    Übersprungen werden: Leerzeilen, Kommentarzeilen (#), Markdown-Listen (-, *, +, 1.).
    """
    text = Path(path).read_text(encoding="utf-8")
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("- ", "* ", "+ ")):
            line = line[2:].strip()
        else:
            numbered = re.match(r"^\d+\.\s+", line)
            if numbered:
                line = line[numbered.end():].strip()
        if line:
            names.append(line)
    if not names:
        raise ValueError(f"Keine Namen in {Path(path).name} gefunden.")
    return names


def sanitize_output_filename(name: str) -> str:
    """Bereinigt einen Namen und ergänzt .webp."""
    stem = name.strip()
    if stem.lower().endswith(".webp"):
        stem = stem[:-5]
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "-")
    stem = re.sub(r"\s+", "_", stem).strip("._")
    if not stem:
        raise ValueError(f"Ungültiger Dateiname nach Bereinigung: {name!r}")
    return f"{stem}.webp"


def prepare_named_batch(folder: str | Path) -> tuple[list[Path], list[str], Path]:
    """
    Validiert Eingabeordner: gleiche Anzahl Bilder und Namen.

    Returns:
        (image_files, output_filenames, md_path)
    """
    folder = Path(folder)
    md_path = find_name_list_md(folder)
    raw_names = parse_names_from_md(md_path)
    images = collect_images(folder)

    if len(images) != len(raw_names):
        raise ValueError(
            f"Anzahl stimmt nicht überein: {len(images)} Bild(er) im Ordner, "
            f"{len(raw_names)} Name(n) in {md_path.name}."
        )

    output_names = [sanitize_output_filename(n) for n in raw_names]
    return images, output_names, md_path


def scan_name_list_status(folder: str | Path) -> str:
    """Kurzstatus für die GUI (ohne Exception bei deaktivierter Nutzung)."""
    folder = Path(folder)
    if not folder.is_dir():
        return "Eingabeordner nicht gewählt"
    try:
        md_path = find_name_list_md(folder)
        names = parse_names_from_md(md_path)
        images = collect_images(folder)
        match = "OK" if len(images) == len(names) else "Abweichung"
        return (
            f"{md_path.name}: {len(names)} Namen, {len(images)} Bilder ({match})"
        )
    except (FileNotFoundError, ValueError) as e:
        return str(e)
