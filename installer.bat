"""
Einfacher Installer für Product Image Processor
Dieses Script richtet alles für die Kollegen ein
"""

import subprocess
import sys
import os
from pathlib import Path

def install_python_check():
    """Prüft ob Python installiert ist"""
    try:
        result = subprocess.run([sys.executable, "--version"],
                              capture_output=True, text=True)
        print(f"✓ Python gefunden: {result.stdout.strip()}")
        return True
    except:
        print("✗ Python nicht gefunden!")
        print("Bitte installiere Python von: https://www.python.org/downloads/")
        return False

def install_dependencies():
    """Installiert alle benötigten Bibliotheken"""
    print("\n📦 Installiere benötigte Bibliotheken...")

    packages = [
        "pillow",
        "numpy",
        "opencv-python",
        "rembg",
        "loguru"
    ]

    for package in packages:
        print(f"   Installiere {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL)
            print(f"   ✓ {package} installiert")
        except:
            print(f"   ✗ Fehler bei {package}")
            return False

    return True

def create_batch_file():
    """Erstellt eine .bat Datei zum einfachen Starten"""
    batch_content = """@echo off
echo ================================
echo Product Image Processor
echo ================================
echo.

REM Prüfe ob Eingabedatei angegeben wurde
if "%~1"=="" (
    echo Bitte ziehe ein Bild auf diese Datei!
    echo Oder starte mit: process_image.bat "pfad\\zum\\bild.jpg"
    pause
    exit /b
)

REM Extrahiere Dateinamen ohne Pfad und Endung
set "input=%~1"
set "filename=%~n1"
set "output=%~dp1%filename%_processed.webp"

echo Verarbeite: %input%
echo Ausgabe: %output%
echo.

python product_processor.py "%input%" "%output%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================
    echo Fertig! Bild gespeichert als:
    echo %output%
    echo ================================
    explorer /select,"%output%"
) else (
    echo.
    echo ================================
    echo Fehler bei der Verarbeitung!
    echo ================================
)

pause
"""

    with open("process_image.bat", "w", encoding="utf-8") as f:
        f.write(batch_content)

    print("\n✓ Batch-Datei erstellt: process_image.bat")

def update_main_script():
    """Passt das Hauptscript an um Kommandozeilen-Parameter zu akzeptieren"""
    script_addition = """
# Kommandozeilen-Parameter Support
if __name__ == "__main__":
    logger.info("Produktbild-Prozessor gestartet")

    # Prüfe ob Parameter übergeben wurden
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]

        # Ausgabedatei: entweder als 2. Parameter oder automatisch generieren
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
        else:
            # Automatischer Name: original_processed.webp
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_processed.webp"
    else:
        # Fallback auf Standard-Namen
        input_file = "gummifisch.jpg"
        output_file = "gummifisch_final.webp"

    # Prüfen ob Datei existiert
    if not os.path.exists(input_file):
        logger.error(f"Eingabedatei '{input_file}' nicht gefunden!")
        logger.info("Verwendung: python product_processor.py <eingabe.jpg> [ausgabe.webp]")
        sys.exit(1)
    else:
        try:
            result = process_product_image(input_file, output_file)
            logger.info("Programm erfolgreich beendet")
        except Exception as e:
            logger.exception(f"Kritischer Fehler bei der Verarbeitung: {e}")
            sys.exit(1)
"""
    print("\n📝 Hinweis: Füge folgenden Code am Ende von product_processor.py hinzu:")
    print("=" * 60)
    print(script_addition)
    print("=" * 60)

def main():
    print("=" * 60)
    print("  Product Image Processor - Installation")
    print("=" * 60)

    # 1. Python Check
    if not install_python_check():
        input("\nDrücke Enter zum Beenden...")
        return

    # 2. Dependencies installieren
    if not install_dependencies():
        print("\n✗ Installation fehlgeschlagen!")
        input("\nDrücke Enter zum Beenden...")
        return

    # 3. Batch-Datei erstellen
    create_batch_file()

    # 4. Hinweis für Script-Anpassung
    update_main_script()

    print("\n" + "=" * 60)
    print("✓ Installation abgeschlossen!")
    print("=" * 60)
    print("\nNutzung:")
    print("1. Bild auf 'process_image.bat' ziehen")
    print("2. Oder: process_image.bat \"pfad\\zum\\bild.jpg\"")
    print("=" * 60)

    input("\nDrücke Enter zum Beenden...")

if __name__ == "__main__":
    main()