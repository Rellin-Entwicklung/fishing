"""Automatische Senkrecht-Ausrichtung von Produktbildern (schwarze Achse oder Silhouette)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

ALIGNMENT_MODES = ("off", "auto", "black_axis", "silhouette")

ALIGNMENT_MODE_LABELS = {
    "off": "Keine Ausrichtung",
    "auto": "Automatisch (schwarze Achse, sonst Silhouette)",
    "black_axis": "Nur schwarze Achse",
    "silhouette": "Nur Objekt-Silhouette",
}

MIN_ROTATION_DEG = 0.5
BLACK_THRESHOLD = 80
MIN_BLACK_PIXELS = 40
MIN_SILHOUETTE_PIXELS = 50


def _normalize_vertical_angle(angle_deg: float) -> float:
    """Kleinste Rotation zur Senkrechten; begrenzt auf ±45°."""
    while angle_deg > 45:
        angle_deg -= 90
    while angle_deg < -45:
        angle_deg += 90
    return angle_deg


def _pca_angle_from_points(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if len(xs) < 2:
        return None
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    vx, vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return None
    # Senkrecht = parallel zur Y-Achse (Bildkoordinaten: y nach unten)
    angle_rad = np.arctan2(vx, vy)
    return _normalize_vertical_angle(float(np.degrees(angle_rad)))


def _angle_from_black_axis(image_rgba: Image.Image) -> float | None:
    arr = np.array(image_rgba.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    brightness = np.max(rgb, axis=2)
    mask = (alpha > 128) & (brightness < BLACK_THRESHOLD)
    mask = mask.astype(np.uint8) * 255
    if mask.max() == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_score = 0.0
    best_angle: float | None = None

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_BLACK_PIXELS:
            continue
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        elongation = max(w, h) / max(min(w, h), 1)
        if elongation < 1.5:
            continue

        ys, xs = np.where(labels == label)
        angle = _pca_angle_from_points(xs, ys)
        if angle is None:
            continue
        score = area * elongation
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _angle_from_silhouette(image_rgba: Image.Image) -> float | None:
    arr = np.array(image_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255
    if np.count_nonzero(mask) < MIN_SILHOUETTE_PIXELS:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_SILHOUETTE_PIXELS:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), angle = rect
    if w < 1 or h < 1:
        return None

    # OpenCV: Winkel der Breitenseite; lange Seite = Hauptachse
    if w < h:
        correction = angle
    else:
        correction = angle + 90.0
    return _normalize_vertical_angle(correction)


def _angle_from_pca_silhouette(image_rgba: Image.Image) -> float | None:
    arr = np.array(image_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) < MIN_SILHOUETTE_PIXELS:
        return None
    return _pca_angle_from_points(xs, ys)


def detect_alignment_angle(
    image_rgba: Image.Image,
    mode: str = "auto",
) -> tuple[float, str]:
    """
    Ermittelt den Korrekturwinkel in Grad (PIL: positiv = gegen Uhrzeigersinn).

    Returns:
        (angle_degrees, method) — method: black_axis | minAreaRect | pca | none
    """
    if mode == "off":
        return 0.0, "none"

    if mode in ("auto", "black_axis"):
        black_angle = _angle_from_black_axis(image_rgba)
        if black_angle is not None:
            return black_angle, "black_axis"
        if mode == "black_axis":
            return 0.0, "none"

    silhouette_angle = _angle_from_silhouette(image_rgba)
    if silhouette_angle is not None:
        return silhouette_angle, "minAreaRect"

    pca_angle = _angle_from_pca_silhouette(image_rgba)
    if pca_angle is not None:
        return pca_angle, "pca"

    return 0.0, "none"


def align_image_vertical(
    image_rgba: Image.Image,
    mode: str = "auto",
) -> tuple[Image.Image, float, str]:
    """
    Richtet das Bild senkrecht aus.

    Returns:
        (aligned_image, applied_angle, method)
    """
    if mode == "off":
        return image_rgba, 0.0, "none"

    angle, method = detect_alignment_angle(image_rgba, mode=mode)
    if abs(angle) < MIN_ROTATION_DEG:
        return image_rgba, 0.0, method

    # PIL: negativer Winkel korrigiert die erkannte Abweichung von der Senkrechten
    rotated = image_rgba.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    return rotated, angle, method


def label_to_mode(label: str) -> str:
    for mode, text in ALIGNMENT_MODE_LABELS.items():
        if text == label:
            return mode
    return "auto"


def mode_to_label(mode: str) -> str:
    return ALIGNMENT_MODE_LABELS.get(mode, ALIGNMENT_MODE_LABELS["auto"])


FINAL_ROTATION_ANGLES = (0, 90, 180, 270)

FINAL_ROTATION_LABELS = {
    0: "Keine",
    90: "90°",
    180: "180°",
    270: "270°",
}

_FINAL_ROTATION_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def final_rotation_label_to_angle(label: str) -> int:
    for angle, text in FINAL_ROTATION_LABELS.items():
        if text == label:
            return angle
    return 0


def final_rotation_angle_to_label(angle: int) -> str:
    return FINAL_ROTATION_LABELS.get(angle, FINAL_ROTATION_LABELS[0])


def apply_final_rotation(image: Image.Image, angle: int) -> Image.Image:
    """Dreht das fertige Bild um 90°, 180° oder 270° (0 = keine Drehung)."""
    transpose = _FINAL_ROTATION_TRANSPOSE.get(angle)
    if transpose is None:
        return image
    return image.transpose(transpose)
