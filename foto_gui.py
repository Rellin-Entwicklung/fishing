import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import os
import sys
from loguru import logger
import gradio as gr
from pathlib import Path
import time

# Loguru Konfiguration
logger.remove()
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


def rotate_image(image, rotation_angle):
    """Rotiert das Bild um den angegebenen Winkel"""
    if rotation_angle == 0:
        return image

    logger.info(f"Rotiere Bild um {rotation_angle}°")

    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        logger.warning(f"Ungültiger Rotationswinkel: {rotation_angle}° - keine Rotation")
        return image

    logger.success(f"Rotation um {rotation_angle}° abgeschlossen")
    return rotated


def optimize_brightness_saturation_contrast(image, brightness_increase=20, saturation_increase=20,
                                            contrast_increase=10):
    """
    Optimiert Helligkeit, Sättigung und Kontrast in dieser Reihenfolge

    Args:
        image: OpenCV Bild (BGR)
        brightness_increase: Helligkeit in Prozent erhöhen (0-100, Standard: 20)
        saturation_increase: Sättigung in Prozent erhöhen (0-100, Standard: 20)
        contrast_increase: Kontrast in Prozent erhöhen (0-100, Standard: 10)
    """
    logger.info(
        f"Starte Bildoptimierung - Helligkeit: +{brightness_increase}%, Sättigung: +{saturation_increase}%, Kontrast: +{contrast_increase}%")
    logger.debug(f"Eingabebild Shape: {image.shape}, Dtype: {image.dtype}")

    try:
        # SCHRITT 1: HELLIGKEIT ERHÖHEN
        logger.debug("Schritt 1/3: Erhöhe Helligkeit")

        # Konvertiere zu HSV für präzise Helligkeitssteuerung
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        v_mean_before = np.mean(v)

        # Helligkeit erhöhen (Prozent in Faktor umrechnen)
        brightness_factor = 1.0 + (brightness_increase / 100.0)
        v = np.clip(v * brightness_factor, 0, 255)

        v_mean_after = np.mean(v)
        logger.success(
            f"Helligkeit erhöht - Vorher: {v_mean_before:.2f}, Nachher: {v_mean_after:.2f} ({brightness_increase}%)")

        # SCHRITT 2: SÄTTIGUNG ERHÖHEN
        logger.debug("Schritt 2/3: Erhöhe Sättigung")

        s_mean_before = np.mean(s)

        # Sättigung erhöhen
        saturation_factor = 1.0 + (saturation_increase / 100.0)
        s = np.clip(s * saturation_factor, 0, 255)

        s_mean_after = np.mean(s)
        logger.success(
            f"Sättigung erhöht - Vorher: {s_mean_before:.2f}, Nachher: {s_mean_after:.2f} ({saturation_increase}%)")

        # HSV wieder zusammensetzen und zurück zu BGR
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        image_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # SCHRITT 3: KONTRAST ERHÖHEN
        logger.debug("Schritt 3/3: Erhöhe Kontrast")

        # Kontrast mit PIL's Methode (sehr präzise)
        contrast_factor = 1.0 + (contrast_increase / 100.0)

        # Über LAB für besseren Kontrast
        lab = cv2.cvtColor(image_adjusted, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)

        l_mean = np.mean(l)

        # Kontrast durch Verstärkung der Abweichung vom Mittelwert
        l = np.clip((l - l_mean) * contrast_factor + l_mean, 0, 255)

        lab = cv2.merge([l, a, b]).astype(np.uint8)
        optimized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        logger.success(f"Kontrast erhöht um {contrast_increase}%")

        # Optionales Schärfen
        logger.debug("Wende leichtes Schärfen an")
        gaussian = cv2.GaussianBlur(optimized, (0, 0), 1.5)
        optimized = cv2.addWeighted(optimized, 1.3, gaussian, -0.3, 0)

        logger.success("Bildoptimierung abgeschlossen")
        return optimized

    except Exception as e:
        logger.error(f"Fehler bei Bildoptimierung: {e}")
        raise


def remove_shadow_artifacts(image_rgba):
    """Entfernt graue Schatten und Artefakte am Rand des freigestellten Objekts"""
    logger.info("Entferne Schatten-Artefakte")

    try:
        img_array = np.array(image_rgba)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        semi_transparent = (alpha > 0) & (alpha < 200)
        semi_count = np.count_nonzero(semi_transparent)
        logger.debug(f"Semi-transparente Pixel gefunden: {semi_count}")

        very_transparent = alpha < 50
        alpha[very_transparent] = 0

        medium_transparent = (alpha >= 50) & (alpha < 200)

        if np.count_nonzero(medium_transparent) > 0:
            gray_pixels = medium_transparent.copy()

            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    if medium_transparent[i, j]:
                        r, g, b = rgb[i, j]
                        color_range = max(r, g, b) - min(r, g, b)
                        brightness = (int(r) + int(g) + int(b)) / 3

                        if color_range < 30 and brightness < 180:
                            if alpha[i, j] < 120:
                                alpha[i, j] = 0
                            else:
                                factor = 255 / max(brightness, 1)
                                rgb[i, j] = np.clip(rgb[i, j] * factor * 0.8 + 50, 0, 255).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        result = np.dstack([rgb, alpha])
        cleaned = Image.fromarray(result.astype(np.uint8), 'RGBA')

        logger.success("Schatten-Artefakte entfernt")
        return cleaned

    except Exception as e:
        logger.warning(f"Fehler beim Entfernen der Schatten: {e} - verwende Original")
        return image_rgba


def find_object_bounds_from_alpha(alpha_channel):
    """Findet die Bounding Box des Objekts aus dem Alpha-Kanal"""
    logger.info("Ermittle Objektgrenzen aus Transparenz-Maske")

    mask = alpha_channel > 10
    non_transparent = np.count_nonzero(mask)
    total = mask.size

    logger.debug(f"Nicht-transparente Pixel: {non_transparent}/{total} ({non_transparent / total * 100:.2f}%)")

    if non_transparent == 0:
        logger.warning("Keine nicht-transparenten Pixel gefunden!")
        return None

    coords = np.argwhere(mask)
    if len(coords) == 0:
        logger.warning("Keine Koordinaten gefunden!")
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    logger.success(f"Objektgrenzen: x={x_min}-{x_max} ({width}px), y={y_min}-{y_max} ({height}px)")

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
        return image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

    obj_width = bounds['width']
    obj_height = bounds['height']
    obj_size = max(obj_width, obj_height)

    crop_size = int(obj_size * padding_factor)
    center_x = bounds['center_x']
    center_y = bounds['center_y']
    half_crop = crop_size // 2

    x1 = center_x - half_crop
    y1 = center_y - half_crop
    x2 = center_x + half_crop
    y2 = center_y + half_crop

    img_width, img_height = image_pil.size
    need_extension = x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height

    if need_extension:
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_width)
        pad_bottom = max(0, y2 - img_height)

        new_width = img_width + pad_left + pad_right
        new_height = img_height + pad_top + pad_bottom

        extended = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 255))
        extended.paste(image_pil, (pad_left, pad_top))

        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top
        image_pil = extended

    cropped = image_pil.crop((x1, y1, x2, y2))

    crop_w, crop_h = cropped.size
    if crop_w != crop_h:
        square_size = max(crop_w, crop_h)
        square = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 255))
        offset_x = (square_size - crop_w) // 2
        offset_y = (square_size - crop_h) // 2
        square.paste(cropped, (offset_x, offset_y))
        cropped = square

    final_size = cropped.size[0]
    if final_size != target_size:
        cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return cropped


def process_product_image(input_path, output_path, brightness_increase=20, saturation_increase=20,
                          contrast_increase=10, rotation_angle=0, bg_alpha_matting=True,
                          bg_alpha_matting_foreground_threshold=240, bg_alpha_matting_background_threshold=10):
    """
    Hauptfunktion: Verarbeitet das Produktbild komplett

    Args:
        brightness_increase: Helligkeit in % erhöhen (Standard: 20)
        saturation_increase: Sättigung in % erhöhen (Standard: 20)
        contrast_increase: Kontrast in % erhöhen (Standard: 10)
        rotation_angle: Drehwinkel (0, 90, 180, 270)
        bg_alpha_matting: Feinere Kantenerkennung (Standard: True)
        bg_alpha_matting_foreground_threshold: Schwellwert für Vordergrund (Standard: 240)
        bg_alpha_matting_background_threshold: Schwellwert für Hintergrund (Standard: 10)
    """

    logger.info("=" * 80)
    logger.info("STARTE PRODUKTBILD-VERARBEITUNG")
    logger.info("=" * 80)
    logger.info(f"Eingabedatei: {input_path}")
    logger.info(f"Ausgabedatei: {output_path}")
    logger.info(
        f"Helligkeit: +{brightness_increase}%, Sättigung: +{saturation_increase}%, Kontrast: +{contrast_increase}%")
    logger.info(f"Rotation: {rotation_angle}°")
    logger.info(
        f"Alpha Matting: {bg_alpha_matting}, Vordergrund: {bg_alpha_matting_foreground_threshold}, Hintergrund: {bg_alpha_matting_background_threshold}")

    # 1. Bild einlesen
    logger.info("SCHRITT 1/8: Lade Bild")
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden!")

    orig_h, orig_w = image.shape[:2]
    logger.success(f"Bild geladen - Auflösung: {orig_w}x{orig_h}px")

    # 2. Rotation anwenden (falls gewünscht) - VOR der Bildoptimierung
    if rotation_angle != 0:
        logger.info(f"SCHRITT 2/8: Rotiere Bild um {rotation_angle}°")
        image = rotate_image(image, rotation_angle)
    else:
        logger.info("SCHRITT 2/8: Keine Rotation (übersprungen)")

    # 3. NEUE REIHENFOLGE: Helligkeit → Sättigung → Kontrast erhöhen
    logger.info("SCHRITT 3/8: Optimiere Helligkeit, Sättigung und Kontrast")
    optimized = optimize_brightness_saturation_contrast(image, brightness_increase, saturation_increase,
                                                        contrast_increase)

    # 4. Zu PIL konvertieren
    logger.info("SCHRITT 4/8: Konvertiere zu PIL-Format")
    optimized_rgb = cv2.cvtColor(optimized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(optimized_rgb)

    # 5. Hintergrund entfernen mit erweiterten Parametern
    logger.info("SCHRITT 5/8: Entferne Hintergrund mit KI-Modell")
    logger.debug(
        f"Parameter - Alpha Matting: {bg_alpha_matting}, FG Threshold: {bg_alpha_matting_foreground_threshold}, BG Threshold: {bg_alpha_matting_background_threshold}")

    try:
        # rembg mit erweiterten Parametern aufrufen
        no_bg = remove(
            pil_image,
            alpha_matting=bg_alpha_matting,
            alpha_matting_foreground_threshold=bg_alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=bg_alpha_matting_background_threshold,
            alpha_matting_erode_size=10
        )
        logger.success(f"Hintergrund entfernt - Modus: {no_bg.mode}")
    except Exception as e:
        logger.error(f"Fehler beim Entfernen des Hintergrunds: {e}")
        raise

    # 6. Schatten-Artefakte entfernen
    logger.info("SCHRITT 6/8: Entferne Schatten-Artefakte")
    if no_bg.mode == 'RGBA':
        no_bg = remove_shadow_artifacts(no_bg)

    # 7. Objektgrenzen ermitteln
    logger.info("SCHRITT 7/8: Ermittle Objektgrenzen")
    if no_bg.mode == 'RGBA':
        alpha_array = np.array(no_bg.split()[3])
        bounds = find_object_bounds_from_alpha(alpha_array)
    else:
        bounds = None

    # 8. Zentriertes Quadrat erstellen
    logger.info("SCHRITT 8/8: Erstelle zentriertes 1024x1024px Bild")
    final = create_centered_square_with_padding(no_bg, bounds, target_size=1024, padding_factor=1.3)

    # Weißen Hintergrund hinzufügen
    white_bg = Image.new('RGB', final.size, (255, 255, 255))
    if final.mode == 'RGBA':
        white_bg.paste(final, mask=final.split()[3])
    else:
        white_bg.paste(final)

    # Finale Verbesserungen
    final_array = np.array(white_bg)
    final_array = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)
    final_array = cv2.fastNlMeansDenoisingColored(final_array, None, 5, 5, 7, 21)
    final_array = cv2.cvtColor(final_array, cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(final_array)

    enhancer_sharp = ImageEnhance.Sharpness(final_pil)
    final_pil = enhancer_sharp.enhance(1.3)
    enhancer_contrast = ImageEnhance.Contrast(final_pil)
    final_pil = enhancer_contrast.enhance(1.1)

    # Als WebP speichern
    final_pil.save(output_path, 'WEBP', quality=95)

    logger.success("VERARBEITUNG ERFOLGREICH ABGESCHLOSSEN!")
    logger.info("=" * 80)

    return final_pil


def batch_process_images(input_folder, output_folder, brightness_increase, saturation_increase,
                         contrast_increase, rotation_angle, use_alpha_matting, fg_threshold,
                         bg_threshold, progress=gr.Progress()):
    """Batch-Verarbeitung aller Bilder in einem Ordner"""

    if not input_folder or not os.path.exists(input_folder):
        return "❌ Eingabeordner nicht gefunden!", None, None

    if not output_folder:
        return "❌ Bitte Ausgabeordner angeben!", None, None

    # Ausgabeordner erstellen falls nicht vorhanden
    os.makedirs(output_folder, exist_ok=True)

    # Unterstützte Bildformate
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Alle Bilddateien finden
    image_files = []
    for file in os.listdir(input_folder):
        if Path(file).suffix.lower() in supported_formats:
            image_files.append(file)

    if not image_files:
        return "❌ Keine Bilddateien im Eingabeordner gefunden!", None, None

    logger.info(f"Starte Batch-Verarbeitung von {len(image_files)} Bildern")

    # Ergebnisse
    results = []
    success_count = 0
    error_count = 0
    start_time = time.time()

    # Verarbeite jedes Bild
    for idx, filename in enumerate(image_files):
        progress((idx + 1) / len(image_files), desc=f"Verarbeite {filename}")

        input_path = os.path.join(input_folder, filename)
        output_filename = Path(filename).stem + "_processed.webp"
        output_path = os.path.join(output_folder, output_filename)

        try:
            logger.info(f"Verarbeite Bild {idx + 1}/{len(image_files)}: {filename}")
            process_product_image(
                input_path,
                output_path,
                brightness_increase=brightness_increase,
                saturation_increase=saturation_increase,
                contrast_increase=contrast_increase,
                rotation_angle=rotation_angle,
                bg_alpha_matting=use_alpha_matting,
                bg_alpha_matting_foreground_threshold=fg_threshold,
                bg_alpha_matting_background_threshold=bg_threshold
            )
            results.append(f"✅ {filename} → {output_filename}")
            success_count += 1
        except Exception as e:
            logger.error(f"Fehler bei {filename}: {e}")
            results.append(f"❌ {filename} - Fehler: {str(e)}")
            error_count += 1

    elapsed_time = time.time() - start_time

    # Zusammenfassung
    summary = f"""
📊 **Batch-Verarbeitung abgeschlossen**

⏱️ Dauer: {elapsed_time:.1f} Sekunden
✅ Erfolgreich: {success_count} Bilder
❌ Fehler: {error_count} Bilder
📁 Ausgabeordner: {output_folder}

🎨 **Angewandte Einstellungen:**
- Helligkeit: +{brightness_increase}%
- Sättigung: +{saturation_increase}%
- Kontrast: +{contrast_increase}%
- Rotation: {rotation_angle}°
- Alpha Matting: {'Aktiviert' if use_alpha_matting else 'Deaktiviert'}
- FG-Schwellwert: {fg_threshold}
- BG-Schwellwert: {bg_threshold}

---
**Details:**
"""

    summary += "\n".join(results)

    # Beispielbild laden (erstes erfolgreiches)
    example_image = None
    for file in os.listdir(output_folder):
        if file.endswith('.webp'):
            example_image = os.path.join(output_folder, file)
            break

    return summary, example_image, output_folder


# Gradio Interface
def create_interface():
    with gr.Blocks(title="Batch Produktbild-Verarbeitung", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ Batch Produktbild-Verarbeitung

        Verarbeite mehrere Produktbilder automatisch mit optimierter Reihenfolge:
        1. 🔆 **Helligkeit erhöhen** (einstellbar, Standard: +20%)
        2. 🎨 **Sättigung erhöhen** (einstellbar, Standard: +20%)
        3. 🌓 **Kontrast erhöhen** (einstellbar, Standard: +10%)
        4. ✨ **Hintergrund entfernen** (KI mit erweiterten Parametern)
        5. 📐 **Zentriertes 1024x1024px Quadrat** mit weißem Hintergrund
        6. 💾 **WebP Export**
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📂 Ordner & Rotation")

                input_folder = gr.Textbox(
                    label="📂 Eingabeordner",
                    placeholder="z.B. C:/Bilder/Input oder /home/user/images",
                    info="Ordner mit den zu verarbeitenden Bildern"
                )

                output_folder = gr.Textbox(
                    label="📁 Ausgabeordner",
                    placeholder="z.B. C:/Bilder/Output oder /home/user/processed",
                    info="Ordner für die verarbeiteten Bilder"
                )

                rotation_dropdown = gr.Dropdown(
                    choices=[0, 90, 180, 270],
                    value=0,
                    label="🔄 Drehwinkel",
                    info="Rotation im Uhrzeigersinn (wird zuerst angewendet)"
                )

                gr.Markdown("### 🎨 Bildoptimierung (vor Hintergrundentfernung)")

                brightness_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=20,
                    step=5,
                    label="🔆 Helligkeit erhöhen (%)",
                    info="Standard: +20% | 0% = keine Änderung"
                )

                saturation_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=20,
                    step=5,
                    label="🎨 Sättigung erhöhen (%)",
                    info="Standard: +20% | 0% = keine Änderung"
                )

                contrast_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=10,
                    step=5,
                    label="🌓 Kontrast erhöhen (%)",
                    info="Standard: +10% | 0% = keine Änderung"
                )

                gr.Markdown("### ✂️ Hintergrundentfernung (Erweitert)")

                alpha_matting_check = gr.Checkbox(
                    value=True,
                    label="🔍 Alpha Matting aktivieren",
                    info="Bessere Kantenerkennung (empfohlen)"
                )

                fg_threshold_slider = gr.Slider(
                    minimum=100,
                    maximum=255,
                    value=240,
                    step=5,
                    label="⬜ Vordergrund-Schwellwert",
                    info="Höher = strengere Objekterkennung (Standard: 240)"
                )

                bg_threshold_slider = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=10,
                    step=1,
                    label="⬛ Hintergrund-Schwellwert",
                    info="Niedriger = mehr Hintergrund entfernt (Standard: 10)"
                )

                process_btn = gr.Button("🚀 Batch-Verarbeitung starten", variant="primary", size="lg")

            with gr.Column():
                output_text = gr.Textbox(
                    label="📋 Verarbeitungsstatus",
                    lines=20,
                    max_lines=25
                )

                output_image = gr.Image(
                    label="🖼️ Beispiel (erstes verarbeitetes Bild)",
                    type="filepath"
                )

                output_folder_display = gr.Textbox(
                    label="📁 Ausgabeordner",
                    interactive=False
                )

        # Event Handler
        process_btn.click(
            fn=batch_process_images,
            inputs=[
                input_folder,
                output_folder,
                brightness_slider,
                saturation_slider,
                contrast_slider,
                rotation_dropdown,
                alpha_matting_check,
                fg_threshold_slider,
                bg_threshold_slider
            ],
            outputs=[output_text, output_image, output_folder_display]
        )

        # Beispiele und Tipps
        gr.Markdown("""
        ---
        ### 💡 Tipps & Erklärungen:

        **Verarbeitungsreihenfolge:**
        1. **Rotation** (falls gewählt) → **Helligkeit** → **Sättigung** → **Kontrast** → **Hintergrundentfernung**

        **Bildoptimierung:**
        - **Helligkeit**: Macht dunkle Bilder heller (empfohlen: 15-30%)
        - **Sättigung**: Macht Farben lebendiger (empfohlen: 15-25%)
        - **Kontrast**: Verstärkt Unterschiede zwischen Hell/Dunkel (empfohlen: 5-15%)

        **Alpha Matting Parameter:**
        - **Vordergrund-Schwellwert** (240): Höhere Werte = nur sehr helle Bereiche werden als Hintergrund erkannt
        - **Hintergrund-Schwellwert** (10): Niedrigere Werte = mehr vom Hintergrund wird entfernt
        - Bei schwierigen Bildern: Vordergrund auf 220-230 senken, Hintergrund auf 15-20 erhöhen

        **Unterstützte Formate:** JPG, PNG, BMP, TIFF, WebP

        **Ausgabe:** Alle Bilder werden als WebP mit 1024x1024px und weißem Hintergrund gespeichert
        """)

    return demo


if __name__ == "__main__":
    logger.info("Starte Batch Produktbild-Verarbeitung GUI")
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)