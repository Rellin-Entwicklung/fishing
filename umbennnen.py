import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def sanitize_filename(filename):
    """Entfernt ungültige Zeichen aus Dateinamen."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def rename_images():
    # 1. Datei auswählen
    table_path = filedialog.askopenfilename(
        title="Tabelle auswählen",
        filetypes=[("Excel/CSV Files", "*.xlsx *.csv")]
    )
    if not table_path:
        return

    # 2. Ordner auswählen
    folder_path = filedialog.askdirectory(title="Bilderordner auswählen")
    if not folder_path:
        return

    try:
        # Tabelle laden (CSV oder Excel)
        if table_path.endswith('.csv'):
            df = pd.read_csv(table_path, header=None)
        else:
            df = pd.read_excel(table_path, header=None)

        # Zeilen entfernen, in denen die erste Spalte leer ist (Metadaten/Leerzeilen)
        df = df.dropna(subset=[df.columns[0]])

        success_count = 0
        error_count = 0

        for index, row in df.iterrows():
            try:
                # Erste belegte Spalte ist die Bildnummer
                old_id = str(int(float(row[0])))
                old_name = f"{old_id}.webp"
                old_file_path = os.path.join(folder_path, old_name)

                # Prüfen, ob die Datei existiert
                if os.path.exists(old_file_path):
                    # Spalten 1, 2 und 3 für den neuen Namen zusammenfügen
                    part1 = str(row[1]) if pd.notna(row[1]) else ""
                    part2 = str(row[2]) if pd.notna(row[2]) else ""
                    part3 = str(row[3]) if pd.notna(row[3]) else ""

                    new_base_name = f"{part1}_{part2}_{part3}"
                    new_base_name = sanitize_filename(new_base_name)
                    new_name = f"{new_base_name}.webp"
                    new_file_path = os.path.join(folder_path, new_name)

                    # Umbenennen
                    os.rename(old_file_path, new_file_path)
                    success_count += 1
                else:
                    print(f"Datei nicht gefunden: {old_name}")
            except Exception as e:
                print(f"Fehler in Zeile {index}: {e}")
                error_count += 1

        messagebox.showinfo("Fertig", f"Erfolgreich umbenannt: {success_count}\nFehler/Nicht gefunden: {error_count}")

    except Exception as e:
        messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")


# GUI Setup
root = tk.Tk()
root.title("WebP Bild-Umbenennung")
root.geometry("300x150")

label = tk.Label(root, text="Bilder basierend auf Tabelle umbenennen", wraplength=250, pady=20)
label.pack()

btn = tk.Button(root, text="Starten", command=rename_images, bg="green", fg="white", padx=20, pady=10)
btn.pack()

root.mainloop()