import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import logging
import os
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bildbearbeitung.log'),
        logging.StreamHandler()
    ]
)


class BildBearbeiter:
    def __init__(self, root):
        self.root = root
        self.root.title("Bild-Automatisierung")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # GUI Elemente
        self.erstelle_gui()

        logging.info("Anwendung gestartet")

    def erstelle_gui(self):
        # Titel
        titel = tk.Label(
            self.root,
            text="PNG zu WebP Konverter",
            font=("Arial", 16, "bold")
        )
        titel.pack(pady=20)

        # Info Text
        info = tk.Label(
            self.root,
            text="Wähle ein PNG-Bild aus. Es wird automatisch:\n"
                 "• Weißer Hintergrund hinzugefügt\n"
                 "• 90° gegen Uhrzeigersinn gedreht\n"
                 "• Zentriert auf 1024x1024px\n"
                 "• Als WebP (80% Qualität) gespeichert",
            justify=tk.LEFT,
            font=("Arial", 10)
        )
        info.pack(pady=10)

        # Dateiname Label
        self.datei_label = tk.Label(
            self.root,
            text="Keine Datei ausgewählt",
            fg="gray",
            font=("Arial", 9, "italic")
        )
        self.datei_label.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.auswaehlen_btn = tk.Button(
            button_frame,
            text="Datei auswählen",
            command=self.datei_auswaehlen,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2"
        )
        self.auswaehlen_btn.pack(side=tk.LEFT, padx=10)

        self.verarbeiten_btn = tk.Button(
            button_frame,
            text="Verarbeiten",
            command=self.bild_verarbeiten,
            width=15,
            height=2,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2",
            state=tk.DISABLED
        )
        self.verarbeiten_btn.pack(side=tk.LEFT, padx=10)

        # Status Label
        self.status_label = tk.Label(
            self.root,
            text="",
            fg="blue",
            font=("Arial", 9)
        )
        self.status_label.pack(pady=10)

        self.input_pfad = None

    def datei_auswaehlen(self):
        self.input_pfad = filedialog.askopenfilename(
            title="PNG-Datei auswählen",
            filetypes=[("PNG Dateien", "*.png"), ("Alle Dateien", "*.*")]
        )

        if self.input_pfad:
            dateiname = os.path.basename(self.input_pfad)
            self.datei_label.config(text=f"Ausgewählt: {dateiname}", fg="green")
            self.verarbeiten_btn.config(state=tk.NORMAL)
            logging.info(f"Datei ausgewählt: {self.input_pfad}")
        else:
            self.datei_label.config(text="Keine Datei ausgewählt", fg="gray")
            self.verarbeiten_btn.config(state=tk.DISABLED)

    def bild_verarbeiten(self):
        if not self.input_pfad:
            messagebox.showerror("Fehler", "Bitte zuerst eine Datei auswählen!")
            return

        try:
            self.status_label.config(text="Verarbeitung läuft...", fg="orange")
            self.root.update()

            logging.info("Starte Bildverarbeitung...")

            # 1. PNG einlesen
            logging.info(f"Lade Bild: {self.input_pfad}")
            bild = Image.open(self.input_pfad)
            logging.info(f"Originalgröße: {bild.size}, Modus: {bild.mode}")

            # 2. In RGBA konvertieren (für Transparenz)
            if bild.mode != 'RGBA':
                bild = bild.convert('RGBA')
                logging.info("Bild zu RGBA konvertiert")

            # 3. Weißen Hintergrund erstellen
            logging.info("Erstelle weißen Hintergrund...")
            weiss_bg = Image.new('RGBA', bild.size, (255, 255, 255, 255))

            # 4. Bild auf weißen Hintergrund compositen
            bild_mit_bg = Image.alpha_composite(weiss_bg, bild)
            bild_mit_bg = bild_mit_bg.convert('RGB')
            logging.info("Weißer Hintergrund hinzugefügt")

            # 5. 90 Grad gegen Uhrzeigersinn drehen
            bild_gedreht = bild_mit_bg.rotate(90, expand=True)
            logging.info("Bild um 90° gedreht")

            # 6. Auf 1024x1024 skalieren und zentrieren mit minimalem Rand
            ziel_groesse = (1024, 1024)
            rand = 10  # Minimaler Rand in Pixeln (kleiner Rand!)

            # Verfügbarer Platz für das Bild (minus Rand)
            verfuegbar = (ziel_groesse[0] - 2 * rand, ziel_groesse[1] - 2 * rand)

            # Skalierungsfaktor berechnen (Bild soll komplett reinpassen)
            skala_x = verfuegbar[0] / bild_gedreht.width
            skala_y = verfuegbar[1] / bild_gedreht.height
            skala = min(skala_x, skala_y)  # Kleinerer Faktor, damit alles passt

            # Neue Größe berechnen
            neue_breite = int(bild_gedreht.width * skala)
            neue_hoehe = int(bild_gedreht.height * skala)

            # Bild skalieren (LANCZOS für beste Qualität)
            bild_skaliert = bild_gedreht.resize((neue_breite, neue_hoehe), Image.LANCZOS)
            logging.info(f"Bild skaliert von {bild_gedreht.size} auf {bild_skaliert.size} (Faktor: {skala:.2f})")

            # Weißer Hintergrund erstellen
            finales_bild = Image.new('RGB', ziel_groesse, (255, 255, 255))

            # Position berechnen zum Zentrieren
            x = (ziel_groesse[0] - neue_breite) // 2
            y = (ziel_groesse[1] - neue_hoehe) // 2

            finales_bild.paste(bild_skaliert, (x, y))
            logging.info(f"Bild zentriert auf {ziel_groesse} mit {rand}px Rand")

            # 7. Als WebP speichern
            input_path = Path(self.input_pfad)
            output_pfad = input_path.parent / f"{input_path.stem}_bearbeitet.webp"

            finales_bild.save(output_pfad, 'WEBP', quality=80)
            logging.info(f"Bild gespeichert: {output_pfad}")

            self.status_label.config(text="✓ Erfolgreich verarbeitet!", fg="green")
            messagebox.showinfo(
                "Erfolg",
                f"Bild erfolgreich verarbeitet!\n\nGespeichert als:\n{output_pfad}"
            )

        except Exception as e:
            logging.error(f"Fehler bei der Verarbeitung: {str(e)}", exc_info=True)
            self.status_label.config(text="✗ Fehler bei der Verarbeitung", fg="red")
            messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung:\n{str(e)}")


def main():
    root = tk.Tk()
    app = BildBearbeiter(root)
    root.mainloop()


if __name__ == "__main__":
    main()