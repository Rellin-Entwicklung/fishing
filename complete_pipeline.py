import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os
import sys
from loguru import logger

# Loguru Konfiguration
logger.remove()  # Standard-Handler entfernen
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "product_processor_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


def optimize_brightness_saturation(image):
    """Optimiert Helligkeit und Sättigung automatisch"""
    logger.info("Starte Helligkeits- und Sättigungsoptimierung")
    logger.debug(f"Eingabebild Shape: {image.shape}, Dtype: {image.dtype}")

    try:
        # Konvertiere zu LAB für bessere Helligkeitskontrolle
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Originalwerte loggen
        l_mean_before = np.mean(l)
        logger.debug(f"Ursprüngliche Helligkeit (L-Kanal) Ø: {l_mean_before:.2f}")

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) für bessere Helligkeit
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        l_mean_after = np.mean(l)
        logger.debug(f"CLAHE angewendet - Neue Helligkeit Ø: {l_mean_after:.2f}")

        # LAB wieder zusammensetzen
        lab = cv2.merge([l, a, b])
        brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Jetzt zu HSV für Sättigung
        hsv = cv2.cvtColor(brightened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s_mean_before = np.mean(s)
        logger.debug(f"Ursprüngliche Sättigung Ø: {s_mean_before:.2f}")

        # Sättigung erhöhen (Faktor 1.5 für mehr Lebendigkeit)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        s_mean_after = np.mean(s)
        logger.debug(f"Sättigung erhöht (Faktor 1.5) - Neue Sättigung Ø: {s_mean_after:.2f}")

        # Helligkeit nochmal leicht anheben
        v = np.clip(v * 1.1, 0, 255).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Schärfen mit Unsharp Mask
        logger.debug("Wende Unsharp Mask für Schärfe an")
        gaussian = cv2.GaussianBlur(optimized, (0, 0), 2.0)
        optimized = cv2.addWeighted(optimized, 1.5, gaussian, -0.5, 0)

        logger.success(
            f"Optimierung abgeschlossen - Helligkeit: {l_mean_before:.2f}→{l_mean_after:.2f}, Sättigung: {s_mean_before:.2f}→{s_mean_after:.2f}")
        return optimized

    except Exception as e:
        logger.error(f"Fehler bei Bildoptimierung: {e}")
        raise


def remove_shadow_artifacts(image_rgba):
    """Entfernt graue Schatten und Artefakte am Rand des freigestellten Objekts"""
    logger.info("Entferne Schatten-Artefakte")

    try:
        # Trenne Kanäle
        img_array = np.array(image_rgba)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        # Finde Pixel die fast transparent sind (Schatten-Bereich)
        # Diese haben oft niedrige Alpha-Werte aber sind nicht ganz transparent
        semi_transparent = (alpha > 0) & (alpha < 200)
        semi_count = np.count_nonzero(semi_transparent)

        logger.debug(f"Semi-transparente Pixel gefunden: {semi_count}")

        # Strategie: Pixel mit niedrigem Alpha komplett transparent machen
        # oder deren RGB-Werte aufhellen wenn sie grau/dunkel sind

        # 1. Pixel mit sehr niedrigem Alpha (< 50) komplett transparent machen
        very_transparent = alpha < 50
        alpha[very_transparent] = 0

        # 2. Pixel mit mittlerem Alpha (50-200): Schatten entfernen
        medium_transparent = (alpha >= 50) & (alpha < 200)

        if np.count_nonzero(medium_transparent) > 0:
            # Bei diesen Pixeln: wenn sie grau/dunkel sind (Schatten), aufhellen
            gray_pixels = medium_transparent.copy()

            # Berechne ob Pixel grau ist (geringe Farbsättigung)
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    if medium_transparent[i, j]:
                        r, g, b = rgb[i, j]
                        # Wenn Pixel dunkel und wenig Farbunterschied (= grau)
                        color_range = max(r, g, b) - min(r, g, b)
                        brightness = (int(r) + int(g) + int(b)) / 3

                        # Grauer Schatten: geringe Farbvariation + dunkel
                        if color_range < 30 and brightness < 180:
                            # Entweder transparent machen oder stark aufhellen
                            if alpha[i, j] < 120:
                                alpha[i, j] = 0  # Transparent
                            else:
                                # Aufhellen zu Weiß
                                factor = 255 / max(brightness, 1)
                                rgb[i, j] = np.clip(rgb[i, j] * factor * 0.8 + 50, 0, 255).astype(np.uint8)

        # 3. Morphologische Operation um Rand zu glätten
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # Wieder zusammensetzen
        result = np.dstack([rgb, alpha])
        cleaned = Image.fromarray(result.astype(np.uint8), 'RGBA')

        logger.success("Schatten-Artefakte entfernt")
        return cleaned

    except Exception as e:
        logger.warning(f"Fehler beim Entfernen der Schatten: {e} - verwende Original")
        return image_rgba


def find_object_bounds_from_alpha(alpha_channel):
    """Findet die Bounding Box des Objekts aus dem Alpha-Kanal nach Hintergrundentfernung"""
    logger.info("Ermittle Objektgrenzen aus Transparenz-Maske")

    # Finde nicht-transparente Pixel
    mask = alpha_channel > 10  # Schwellwert für "nicht transparent"
    non_transparent = np.count_nonzero(mask)
    total = mask.size

    logger.debug(f"Nicht-transparente Pixel: {non_transparent}/{total} ({non_transparent / total * 100:.2f}%)")

    if non_transparent == 0:
        logger.warning("Keine nicht-transparenten Pixel gefunden!")
        return None

    # Finde Koordinaten aller nicht-transparenten Pixel
    coords = np.argwhere(mask)

    if len(coords) == 0:
        logger.warning("Keine Koordinaten gefunden!")
        return None

    # Min/Max Koordinaten
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    logger.success(f"Objektgrenzen: x={x_min}-{x_max} ({width}px), y={y_min}-{y_max} ({height}px)")
    logger.debug(f"Objektzentrum: ({center_x}, {center_y})")

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': width,
        'height': height,
        'center_x': center_x,
        'center_y': center_y
    }


def create_centered_square_with_padding(image_pil, bounds, target_size=1024, padding_factor=1.3):
    """Erstellt ein zentriertes Quadrat mit Padding um das Objekt"""
    logger.info(f"Erstelle zentriertes Quadrat mit Padding-Faktor {padding_factor}")

    if bounds is None:
        logger.error("Keine Objektgrenzen vorhanden!")
        # Fallback: ganzes Bild nehmen
        return image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Objekt-Dimensionen
    obj_width = bounds['width']
    obj_height = bounds['height']
    obj_size = max(obj_width, obj_height)

    logger.debug(f"Objektgröße: {obj_width}x{obj_height}px, max: {obj_size}px")

    # Crop-Größe mit Padding
    crop_size = int(obj_size * padding_factor)
    logger.debug(f"Crop-Größe mit Padding: {crop_size}px")

    # Zentrierte Crop-Box berechnen
    center_x = bounds['center_x']
    center_y = bounds['center_y']
    half_crop = crop_size // 2

    x1 = center_x - half_crop
    y1 = center_y - half_crop
    x2 = center_x + half_crop
    y2 = center_y + half_crop

    logger.debug(f"Crop-Box (vor Anpassung): x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Wenn Crop außerhalb des Bildes, erweitern mit weißem Hintergrund
    img_width, img_height = image_pil.size

    need_extension = x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height

    if need_extension:
        logger.info("Crop-Box überschreitet Bildgrenzen - erweitere Bild mit weißem Hintergrund")

        # Berechne benötigte Erweiterung
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_width)
        pad_bottom = max(0, y2 - img_height)

        logger.debug(f"Padding: links={pad_left}, oben={pad_top}, rechts={pad_right}, unten={pad_bottom}")

        # Neues größeres Bild mit weißem Hintergrund
        new_width = img_width + pad_left + pad_right
        new_height = img_height + pad_top + pad_bottom

        extended = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
        extended.paste(image_pil, (pad_left, pad_top))

        # Crop-Koordinaten anpassen
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

        image_pil = extended
        logger.debug(f"Erweitertes Bild: {new_width}x{new_height}px")
        logger.debug(f"Neue Crop-Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Crop durchführen
    cropped = image_pil.crop((x1, y1, x2, y2))
    logger.success(f"Ausschnitt erstellt: {cropped.size[0]}x{cropped.size[1]}px")

    # Zu Quadrat machen falls nötig
    crop_w, crop_h = cropped.size
    if crop_w != crop_h:
        logger.info(f"Crop ist nicht quadratisch ({crop_w}x{crop_h}) - erweitere zu Quadrat")
        square_size = max(crop_w, crop_h)
        square = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 255))

        offset_x = (square_size - crop_w) // 2
        offset_y = (square_size - crop_h) // 2

        square.paste(cropped, (offset_x, offset_y))
        cropped = square
        logger.debug(f"Quadrat erstellt: {square_size}x{square_size}px")

    # Auf Zielgröße skalieren
    final_size = cropped.size[0]
    if final_size != target_size:
        logger.info(f"Skaliere von {final_size}x{final_size}px auf {target_size}x{target_size}px")
        cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return cropped


def process_product_image(input_path, output_path):
    """Hauptfunktion: Verarbeitet das Produktbild komplett"""

    logger.info("=" * 80)
    logger.info("STARTE PRODUKTBILD-VERARBEITUNG")
    logger.info("=" * 80)
    logger.info(f"Eingabedatei: {input_path}")
    logger.info(f"Ausgabedatei: {output_path}")

    # 1. Bild einlesen
    logger.info("SCHRITT 1/7: Lade Bild")

    if not os.path.exists(input_path):
        logger.error(f"Datei nicht gefunden: {input_path}")
        raise FileNotFoundError(f"Datei '{input_path}' existiert nicht!")

    file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    logger.debug(f"Dateigröße: {file_size:.2f} MB")

    image = cv2.imread(input_path)
    if image is None:
        logger.error("Bild konnte nicht geladen werden - möglicherweise ungültiges Format")
        raise ValueError("Bild konnte nicht geladen werden!")

    orig_h, orig_w = image.shape[:2]
    logger.success(f"Bild geladen - Auflösung: {orig_w}x{orig_h}px, Kanäle: {image.shape[2]}")

    # 2. Helligkeit und Sättigung optimieren
    logger.info("SCHRITT 2/7: Optimiere Helligkeit und Sättigung")
    optimized = optimize_brightness_saturation(image)

    # 3. Zu PIL konvertieren
    logger.info("SCHRITT 3/7: Konvertiere zu PIL-Format")
    try:
        optimized_rgb = cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(optimized_rgb)
        logger.success(f"Konvertierung erfolgreich - PIL-Modus: {pil_image.mode}, Größe: {pil_image.size}")
    except Exception as e:
        logger.error(f"Fehler bei PIL-Konvertierung: {e}")
        raise

    # 4. Hintergrund entfernen
    logger.info("SCHRITT 4/8: Entferne Hintergrund mit KI-Modell (kann etwas dauern)")
    logger.debug("Rufe rembg.remove() auf...")
    try:
        no_bg = remove(pil_image)
        logger.success(f"Hintergrund entfernt - Modus: {no_bg.mode}, Größe: {no_bg.size}")

        # Alpha-Channel Statistik
        if no_bg.mode == 'RGBA':
            alpha = np.array(no_bg.split()[3])
            transparent_pixels = np.sum(alpha < 10)
            total_pixels = alpha.size
            transparency_percent = (transparent_pixels / total_pixels) * 100
            logger.debug(
                f"Transparenz-Statistik: {transparent_pixels}/{total_pixels} Pixel transparent ({transparency_percent:.2f}%)")
        else:
            logger.warning("Kein Alpha-Kanal vorhanden!")
            alpha = None

    except Exception as e:
        logger.error(f"Fehler beim Entfernen des Hintergrunds: {e}")
        raise

    # 5. Schatten-Artefakte entfernen
    logger.info("SCHRITT 5/8: Entferne Schatten-Artefakte")
    if no_bg.mode == 'RGBA':
        no_bg = remove_shadow_artifacts(no_bg)
    else:
        logger.warning("Überspringe Schatten-Entfernung (kein Alpha-Kanal)")

    # 6. Objektgrenzen aus Alpha-Kanal ermitteln
    logger.info("SCHRITT 6/8: Ermittle Objektgrenzen")
    if no_bg.mode == 'RGBA':
        alpha_array = np.array(no_bg.split()[3])
        bounds = find_object_bounds_from_alpha(alpha_array)
    else:
        logger.warning("Kein Alpha-Kanal - verwende gesamtes Bild")
        bounds = None

    # 7. Zentriertes Quadrat mit Padding erstellen
    logger.info("SCHRITT 7/8: Erstelle zentriertes 1024x1024px Bild mit Padding")
    try:
        final = create_centered_square_with_padding(no_bg, bounds, target_size=1024, padding_factor=1.3)
        logger.success(f"Finales Bild erstellt: {final.size[0]}x{final.size[1]}px")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des finalen Bildes: {e}")
        raise

    # 8. Weißen Hintergrund hinzufügen und als WebP exportieren
    logger.info("SCHRITT 8/8: Füge weißen Hintergrund hinzu und exportiere als WebP")
    try:
        # Weißen Hintergrund erstellen
        white_bg = Image.new('RGB', final.size, (255, 255, 255))

        # Wenn RGBA, mit Alpha-Maske einfügen
        if final.mode == 'RGBA':
            white_bg.paste(final, mask=final.split()[3])
            logger.debug("Objekt mit Alpha-Maske auf weißen Hintergrund eingefügt")
        else:
            white_bg.paste(final)
            logger.debug("Objekt direkt auf weißen Hintergrund eingefügt")

        # Zusätzliche Nachbearbeitung für bessere Qualität
        logger.debug("Wende finale Bildverbesserungen an")

        # Zu numpy für OpenCV-Operationen
        final_array = np.array(white_bg)
        final_array = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)

        # Leichtes Denoising für glattere Kanten
        final_array = cv2.fastNlMeansDenoisingColored(final_array, None, 5, 5, 7, 21)

        # Zurück zu RGB für PIL
        final_array = cv2.cvtColor(final_array, cv2.COLOR_BGR2RGB)
        final_pil = Image.fromarray(final_array)

        # Schärfe und Kontrast mit PIL leicht verbessern
        enhancer_sharp = ImageEnhance.Sharpness(final_pil)
        final_pil = enhancer_sharp.enhance(1.3)

        enhancer_contrast = ImageEnhance.Contrast(final_pil)
        final_pil = enhancer_contrast.enhance(1.1)

        # Als WebP speichern
        final_pil.save(output_path, 'WEBP', quality=95)
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (file_size / output_size) if output_size > 0 else 0

        logger.success(f"WebP exportiert - Dateigröße: {output_size:.2f} MB (Kompression: {compression_ratio:.2f}x)")

    except Exception as e:
        logger.error(f"Fehler beim Exportieren: {e}")
        raise

    logger.info("=" * 80)
    logger.success("VERARBEITUNG ERFOLGREICH ABGESCHLOSSEN!")
    logger.info("=" * 80)
    logger.info(f"Ursprüngliche Auflösung: {orig_w}x{orig_h}px ({file_size:.2f} MB)")
    logger.info(f"Finale Auflösung: 1024x1024px ({output_size:.2f} MB)")
    logger.info(f"Ausgabedatei: {output_path}")
    logger.info("=" * 80)

    return final_pil


# Beispiel-Verwendung
if __name__ == "__main__":
    logger.info("Produktbild-Prozessor gestartet")

    input_file = "p6.jpg"  # Dein Eingabebild
    output_file = "gummifisch_final.webp"  # Ausgabedatei

    # Prüfen ob Datei existiert
    if not os.path.exists(input_file):
        logger.error(f"Eingabedatei '{input_file}' nicht gefunden!")
        logger.info("Bitte passe den Dateinamen in der letzten Zeile an.")
        sys.exit(1)
    else:
        try:
            result = process_product_image(input_file, output_file)
            logger.info("Programm erfolgreich beendet")
        except Exception as e:
            logger.exception(f"Kritischer Fehler bei der Verarbeitung: {e}")
            sys.exit(1)