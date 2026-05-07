import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image


class ImageRotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bild-Rotator Pro")
        self.root.geometry("500x350")

        # Variablen für die Pfade und den Winkel
        self.source_path = tk.StringVar()
        self.dest_path = tk.StringVar()
        self.angle = tk.IntVar(value=90)

        self.create_widgets()

    def create_widgets(self):
        # --- Quellverzeichnis ---
        tk.Label(self.root, text="1. Quellverzeichnis (Wo sind die Bilder?):", font=("Arial", 10, "bold")).pack(
            pady=(15, 0), anchor="w", padx=20)

        src_frame = tk.Frame(self.root)
        src_frame.pack(pady=5, padx=20, fill="x")
        tk.Entry(src_frame, textvariable=self.source_path, state="readonly").pack(side="left", expand=True, fill="x",
                                                                                  padx=(0, 5))
        tk.Button(src_frame, text="Wählen", command=lambda: self.browse_folder(self.source_path)).pack(side="right")

        # --- Zielverzeichnis ---
        tk.Label(self.root, text="2. Zielverzeichnis (Wo sollen sie hin?):", font=("Arial", 10, "bold")).pack(
            pady=(15, 0), anchor="w", padx=20)

        dest_frame = tk.Frame(self.root)
        dest_frame.pack(pady=5, padx=20, fill="x")
        tk.Entry(dest_frame, textvariable=self.dest_path, state="readonly").pack(side="left", expand=True, fill="x",
                                                                                 padx=(0, 5))
        tk.Button(dest_frame, text="Wählen", command=lambda: self.browse_folder(self.dest_path)).pack(side="right")

        # --- Winkel ---
        tk.Label(self.root, text="3. Winkel wählen:", font=("Arial", 10, "bold")).pack(pady=(15, 0))

        angle_frame = tk.Frame(self.root)
        angle_frame.pack(pady=5)
        for val in [90, 180, 270]:
            tk.Radiobutton(angle_frame, text=f"{val}°", variable=self.angle, value=val).pack(side="left", padx=15)

        # --- Start Button ---
        self.start_btn = tk.Button(self.root, text="Bilder jetzt drehen", command=self.rotate_images,
                                   bg="#2196F3", fg="white", font=("Arial", 11, "bold"), height=2)
        self.start_btn.pack(pady=25, fill="x", padx=100)

    def browse_folder(self, var_to_set):
        folder = filedialog.askdirectory()
        if folder:
            var_to_set.set(folder)

    def rotate_images(self):
        src = self.source_path.get()
        dst = self.dest_path.get()

        if not src or not dst:
            messagebox.showwarning("Fehler", "Bitte wähle Quell- und Zielverzeichnis aus!")
            return

        angle = self.angle.get()
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

        try:
            # Liste aller Bilddateien im Quellordner
            files = [f for f in os.listdir(src) if f.lower().endswith(supported_extensions)]

            if not files:
                messagebox.showinfo("Info", "Keine Bilder im Quellordner gefunden.")
                return

            count = 0
            for filename in files:
                input_file = os.path.join(src, filename)
                output_file = os.path.join(dst, filename)

                with Image.open(input_file) as img:
                    # expand=True sorgt dafür, dass die Leinwand mitrotiert (wichtig bei 90/270°)
                    # Pillow rotiert standardmäßig gegen den Uhrzeigersinn, daher -angle
                    rotated_img = img.rotate(-angle, expand=True)

                    # Falls das Bild im Original im Modus RGBA (Transparenz) ist,
                    # behalten wir das bei, ansonsten wird Standard-RGB genutzt.
                    rotated_img.save(output_file, quality=95)
                    count += 1

            messagebox.showinfo("Erfolg",
                                f"Abgeschlossen!\n{count} Bilder wurden um {angle}° gedreht und in den Zielordner kopiert.")

        except Exception as e:
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRotatorApp(root)
    root.mainloop()