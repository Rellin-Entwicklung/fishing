"""
Foto-Umbenenner Tool
Liest eine Excel-Tabelle, verknüpft Picture-IDs mit Hersteller-Nummern
und kopiert Fotos mit neuem Namen in einen Zielordner.
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from loguru import logger

# ── Logging-Konfiguration ────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.expanduser("~"), "foto_umbenenner.log")
logger.add(LOG_FILE, rotation="5 MB", retention="10 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".cr2", ".nef", ".arw", ".dng", ".rw2"}


# ── Kernlogik ────────────────────────────────────────────────────────────────

def build_rename_list(df: pd.DataFrame, id_col: str, name_col: str) -> list:
    """Erstellt eine Liste von (hersteller_nr, picture_id) Tupeln – eine Zeile pro Eintrag.
    Mehrere Zeilen mit derselben Picture-ID sind explizit erlaubt."""
    entries = []
    for idx, row in df.iterrows():
        try:
            pic_id = str(int(row[id_col]))
            hersteller = str(row[name_col]).strip()
            if not hersteller or hersteller.lower() == "nan":
                logger.warning(f"Zeile {idx+2}: Hersteller-Nr. leer – übersprungen")
                continue
            entries.append((hersteller, pic_id))
        except (ValueError, TypeError):
            logger.warning(f"Zeile {idx+2}: Ungültige Picture-ID '{row[id_col]}' – übersprungen")
    logger.info(f"Verarbeitungsliste aufgebaut: {len(entries)} Einträge")
    return entries


def find_photo(folder: str, pic_id: str) -> str | None:
    """Sucht nach einer Datei, deren Stamm mit der Picture-ID endet."""
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in SUPPORTED_EXTENSIONS:
            # Kameradateinamen enden häufig auf die ID, z.B. IMG_1042 oder DSC01042
            if stem == pic_id or stem.endswith(pic_id):
                return os.path.join(folder, fname)
    return None


def process_photos(entries: list, photo_folder: str, dest_folder: str,
                   progress_var: tk.IntVar, status_var: tk.StringVar,
                   root: tk.Tk):
    """Iteriert über alle Hersteller-Nummern und kopiert das jeweils zugehörige Foto."""
    total = len(entries)
    ok = skipped = 0

    for i, (hersteller, pic_id) in enumerate(entries, 1):
        src = find_photo(photo_folder, pic_id)
        if src is None:
            logger.warning(f"[{hersteller}] Kein Foto für Picture-ID {pic_id} gefunden – übersprungen")
            skipped += 1
        else:
            ext = os.path.splitext(src)[1].lower()
            # Sonderzeichen im Dateinamen ersetzen
            safe_name = hersteller.replace("/", "-").replace("\\", "-").replace(":", "-")
            dst = os.path.join(dest_folder, f"{safe_name}{ext}")
            # Duplikate vermeiden (z.B. gleiche Hersteller-Nr. mit unterschiedlicher PID)
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join(dest_folder, f"{safe_name}_{counter}{ext}")
                counter += 1
            shutil.copy2(src, dst)
            logger.info(f"[{hersteller}] Kopiert: {os.path.basename(src)}  →  {os.path.basename(dst)}")
            ok += 1

        progress_var.set(int(i / total * 100))
        status_var.set(f"Verarbeite {i}/{total} ...")
        root.update_idletasks()

    msg = (f"Fertig!\n\n"
           f"✔ Erfolgreich kopiert: {ok}\n"
           f"⚠ Nicht gefunden (übersprungen): {skipped}")
    logger.info(f"Abgeschlossen – kopiert: {ok}, übersprungen: {skipped}")
    status_var.set("Fertig!")
    messagebox.showinfo("Ergebnis", msg)


# ── GUI ──────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Foto-Umbenenner")
        self.resizable(False, False)
        self.configure(padx=20, pady=20)

        # Zustandsvariablen
        self.excel_path = tk.StringVar()
        self.photo_folder = tk.StringVar()
        self.dest_folder = tk.StringVar()
        self.id_col = tk.StringVar()
        self.name_col = tk.StringVar()
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Bereit")

        self.df = None
        self.columns = []

        self._build_ui()
        logger.info("Anwendung gestartet")

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"pady": 5}

        # ── Abschnitt 1: Excel ────────────────────────────────────────────────
        tk.Label(self, text="1. Excel-Datei auswählen", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)

        tk.Entry(self, textvariable=self.excel_path, width=55, state="readonly").grid(
            row=1, column=0, columnspan=2, sticky="ew")
        tk.Button(self, text="Durchsuchen …", command=self._pick_excel).grid(
            row=1, column=2, padx=(8, 0))

        # ── Abschnitt 2: Spalten ──────────────────────────────────────────────
        tk.Label(self, text="2. Spalten zuordnen", font=("", 10, "bold")).grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)

        tk.Label(self, text="Picture-ID Spalte:").grid(row=3, column=0, sticky="w")
        self.id_combo = ttk.Combobox(self, textvariable=self.id_col,
                                     state="disabled", width=30)
        self.id_combo.grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        tk.Label(self, text="Hersteller-Nr. Spalte:").grid(row=4, column=0, sticky="w")
        self.name_combo = ttk.Combobox(self, textvariable=self.name_col,
                                       state="disabled", width=30)
        self.name_combo.grid(row=4, column=1, columnspan=2, sticky="ew", **pad)

        # ── Abschnitt 3: Ordner ───────────────────────────────────────────────
        tk.Label(self, text="3. Ordner auswählen", font=("", 10, "bold")).grid(
            row=5, column=0, columnspan=3, sticky="w", **pad)

        tk.Label(self, text="Foto-Quellordner:").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.photo_folder, width=45,
                 state="readonly").grid(row=6, column=1, sticky="ew")
        tk.Button(self, text="Wählen …",
                  command=lambda: self._pick_folder(self.photo_folder, "Quellordner")).grid(
            row=6, column=2, padx=(8, 0))

        tk.Label(self, text="Zielordner (Kopien):").grid(row=7, column=0, sticky="w")
        tk.Entry(self, textvariable=self.dest_folder, width=45,
                 state="readonly").grid(row=7, column=1, sticky="ew")
        tk.Button(self, text="Wählen …",
                  command=lambda: self._pick_folder(self.dest_folder, "Zielordner")).grid(
            row=7, column=2, padx=(8, 0))

        # ── Fortschritt ───────────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=10)

        tk.Label(self, textvariable=self.status_var, fg="gray").grid(
            row=9, column=0, columnspan=3, sticky="w")
        ttk.Progressbar(self, variable=self.progress_var, maximum=100,
                        length=430).grid(row=10, column=0, columnspan=3,
                                         sticky="ew", pady=(0, 10))

        # ── Start-Button ──────────────────────────────────────────────────────
        self.start_btn = tk.Button(
            self, text="▶  Umbenennen starten", bg="#2563eb", fg="white",
            font=("", 10, "bold"), padx=10, pady=6,
            command=self._start, state="disabled")
        self.start_btn.grid(row=11, column=0, columnspan=3, sticky="ew")

        tk.Label(self, text=f"Log-Datei: {LOG_FILE}", fg="gray",
                 font=("", 8)).grid(row=12, column=0, columnspan=3,
                                    sticky="w", pady=(6, 0))

        self.columnconfigure(1, weight=1)

    # ── Aktionen ──────────────────────────────────────────────────────────────

    def _pick_excel(self):
        path = filedialog.askopenfilename(
            title="Excel-Datei auswählen",
            filetypes=[("Excel-Dateien", "*.xlsx *.xls *.xlsm"), ("Alle Dateien", "*.*")])
        if not path:
            return
        self.excel_path.set(path)
        logger.info(f"Excel-Datei gewählt: {path}")
        try:
            self.df = pd.read_excel(path)
            self.columns = list(self.df.columns)
            self.id_combo["values"] = self.columns
            self.name_combo["values"] = self.columns
            self.id_combo["state"] = "readonly"
            self.name_combo["state"] = "readonly"
            logger.info(f"Spalten gelesen: {self.columns}")
            self._check_ready()
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Excel-Datei: {e}")
            messagebox.showerror("Fehler", f"Excel-Datei konnte nicht gelesen werden:\n{e}")

    def _pick_folder(self, var: tk.StringVar, title: str):
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)
            logger.info(f"{title} gewählt: {folder}")
            self._check_ready()

    def _check_ready(self, *_):
        ready = all([
            self.excel_path.get(),
            self.id_col.get(),
            self.name_col.get(),
            self.photo_folder.get(),
            self.dest_folder.get(),
        ])
        self.start_btn["state"] = "normal" if ready else "disabled"

    def _start(self):
        # Validierung
        if self.id_col.get() == self.name_col.get():
            messagebox.showwarning("Warnung", "Bitte zwei verschiedene Spalten auswählen.")
            return
        if not os.path.isdir(self.photo_folder.get()):
            messagebox.showerror("Fehler", "Quellordner existiert nicht.")
            return
        os.makedirs(self.dest_folder.get(), exist_ok=True)

        entries = build_rename_list(self.df, self.id_col.get(), self.name_col.get())
        if not entries:
            messagebox.showwarning("Warnung", "Keine gültigen Einträge in der Tabelle gefunden.")
            return

        self.start_btn["state"] = "disabled"
        self.progress_var.set(0)
        logger.info(f"Starte Verarbeitung: {len(entries)} Hersteller-Nummern, "
                    f"Quelle: {self.photo_folder.get()}, Ziel: {self.dest_folder.get()}")

        process_photos(
            entries,
            self.photo_folder.get(),
            self.dest_folder.get(),
            self.progress_var,
            self.status_var,
            self
        )
        self.start_btn["state"] = "normal"


# Spalten-Auswahl → Bereit-Check verknüpfen
def _main():
    app = App()
    app.id_combo.bind("<<ComboboxSelected>>", app._check_ready)
    app.name_combo.bind("<<ComboboxSelected>>", app._check_ready)
    app.mainloop()


if __name__ == "__main__":
    _main()
