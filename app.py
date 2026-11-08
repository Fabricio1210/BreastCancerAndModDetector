from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from sklearn.datasets import load_breast_cancer

from detectors import (
    load_cancer_model,
    predict_cancer,
    predict_emotion,
    train_emotion_model,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_model_variation.joblib"
EMOTION_DATA_PATH = APP_DIR / "data" / "emotion_samples.csv"

THEMES = {
    "dark": {
        "background": "#0B1120",
        "surface": "#111B2E",
        "card": "#17243A",
        "input": "#0D1728",
        "border": "#2A3A52",
        "navy": "#F1F5F9",
        "blue": "#60A5FA",
        "teal": "#2DD4BF",
        "teal_active": "#14B8A6",
        "red": "#FB7185",
        "text": "#E2E8F0",
        "muted": "#94A3B8",
        "white": "#F8FAFC",
        "selection": "#155E75",
    },
    "light": {
        "background": "#F3F6FA",
        "surface": "#FFFFFF",
        "card": "#FFFFFF",
        "input": "#FFFFFF",
        "border": "#D9E2EC",
        "navy": "#173B57",
        "blue": "#2878B5",
        "teal": "#168C82",
        "teal_active": "#11776F",
        "red": "#C94754",
        "text": "#243447",
        "muted": "#66788A",
        "white": "#FFFFFF",
        "selection": "#B9E6E0",
    },
}


class DetectionApp(tk.Tk):
    def __init__(self, cancer_model, emotion_model):
        super().__init__()
        self.cancer_model = cancer_model
        self.emotion_model = emotion_model
        self.measurement_fields: dict[str, ttk.Entry] = {}
        self.theme_mode = "dark"
        self.palette = THEMES[self.theme_mode]

        self.title("Breast Cancer & Emotion Detector")
        self.geometry("1180x800")
        self.minsize(1040, 690)
        self.configure(background=self.palette["background"])
        self._configure_style()
        self._build_header()
        self._build_tabs()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        palette = self.palette
        style.configure("TFrame", background=palette["background"])
        style.configure(
            "TLabel",
            background=palette["background"],
            foreground=palette["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22, "bold"),
            foreground=palette["navy"],
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 10),
            foreground=palette["muted"],
        )
        style.configure(
            "Section.TLabel",
            font=("Segoe UI", 12, "bold"),
            foreground=palette["navy"],
        )
        style.configure(
            "TNotebook",
            background=palette["background"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=palette["surface"],
            foreground=palette["muted"],
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", palette["card"]), ("active", palette["surface"])],
            foreground=[("selected", palette["teal"]), ("active", palette["text"])],
        )
        style.configure(
            "TLabelframe",
            background=palette["surface"],
            bordercolor=palette["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=palette["surface"],
            foreground=palette["teal"],
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "TButton",
            background=palette["card"],
            foreground=palette["text"],
            padding=(12, 8),
            font=("Segoe UI", 9, "bold"),
            bordercolor=palette["border"],
            focuscolor=palette["border"],
        )
        style.map(
            "TButton",
            background=[("active", palette["border"]), ("pressed", palette["surface"])],
            foreground=[("disabled", palette["muted"])],
        )
        style.configure(
            "Accent.TButton",
            background=palette["teal"],
            foreground="#06201D" if self.theme_mode == "dark" else palette["white"],
            bordercolor=palette["teal"],
            padding=(16, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", palette["teal_active"]), ("pressed", palette["teal_active"])],
            foreground=[("active", "#06201D" if self.theme_mode == "dark" else palette["white"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=palette["input"],
            foreground=palette["text"],
            insertcolor=palette["text"],
            padding=6,
            bordercolor=palette["border"],
        )

    def _build_header(self):
        header = ttk.Frame(self, padding=(28, 22, 28, 16))
        header.pack(fill="x")
        title_block = ttk.Frame(header)
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="Breast Cancer & Emotion Detector", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            title_block,
            text="Clasificadores Support Vector Machine para mediciones mamarias y texto en español",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        self.theme_button = ttk.Button(header, command=self._toggle_theme)
        self.theme_button.pack(side="right", padx=(16, 0), anchor="center")
        self._update_theme_button()

    def _toggle_theme(self):
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        self.palette = THEMES[self.theme_mode]
        self.configure(background=self.palette["background"])
        self._configure_style()
        self.emotion_text.configure(
            background=self.palette["input"],
            foreground=self.palette["text"],
            insertbackground=self.palette["text"],
            selectbackground=self.palette["selection"],
        )
        self.cancer_result_panel.configure(
            background=self.palette["card"],
            highlightbackground=self.palette["border"],
        )
        self.cancer_result_label.configure(
            background=self.palette["card"],
            foreground=self.palette["navy"],
        )
        self.cancer_probability_label.configure(
            background=self.palette["card"],
            foreground=self.palette["muted"],
        )
        self.emotion_result_panel.configure(
            background=self.palette["card"],
            highlightbackground=self.palette["border"],
        )
        self.emotion_result_label.configure(
            background=self.palette["card"],
            foreground=self.palette["navy"],
        )
        self._update_theme_button()

    def _update_theme_button(self):
        label = "Cambiar a tema claro" if self.theme_mode == "dark" else "Cambiar a tema oscuro"
        self.theme_button.configure(text=label)

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=28, pady=(0, 24))
        self.cancer_tab = ttk.Frame(self.tabs, padding=(22, 20))
        self.emotion_tab = ttk.Frame(self.tabs, padding=(26, 24))
        self.tabs.add(self.cancer_tab, text="  Detección de cáncer  ")
        self.tabs.add(self.emotion_tab, text="  Emoción por oración  ")
        self._build_cancer_tab()
        self._build_emotion_tab()

    def _build_cancer_tab(self):
        ttk.Label(
            self.cancer_tab,
            text="Ingresa las 30 mediciones del estudio",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.cancer_tab,
            text="Usa los valores numéricos reportados para la muestra. Los botones de ejemplo son solo demostrativos.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 16))

        columns = ttk.Frame(self.cancer_tab)
        columns.pack(fill="x")
        feature_groups = [
            ("Promedio (mean)", [name for name in self.cancer_model.feature_names_in_ if name.startswith("mean ")]),
            ("Error estándar", [name for name in self.cancer_model.feature_names_in_ if name.endswith("error")]),
            ("Peor valor (worst)", [name for name in self.cancer_model.feature_names_in_ if name.startswith("worst ")]),
        ]
        for column, (title, names) in enumerate(feature_groups):
            panel = ttk.LabelFrame(columns, text=title, padding=(14, 12))
            panel.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 10, 0),
            )
            columns.columnconfigure(column, weight=1, uniform="feature_group")
            for row, name in enumerate(names):
                ttk.Label(panel, text=name.replace(" error", "")).grid(
                    row=row, column=0, sticky="w", pady=2, padx=(0, 6)
                )
                entry = ttk.Entry(panel, width=11)
                entry.grid(row=row, column=1, sticky="ew", pady=2)
                self.measurement_fields[name] = entry
            panel.columnconfigure(1, weight=1)

        actions = ttk.Frame(self.cancer_tab)
        actions.pack(fill="x", pady=(14, 10))
        ttk.Button(actions, text="Ejemplo benigno", command=lambda: self._load_example(1)).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(actions, text="Ejemplo maligno", command=lambda: self._load_example(0)).pack(
            side="left", padx=(0, 16)
        )
        ttk.Button(
            actions,
            text="Analizar mediciones",
            style="Accent.TButton",
            command=self._predict_cancer,
        ).pack(side="left")

        self.cancer_result = tk.StringVar(value="El resultado aparecerá aquí.")
        self.cancer_probability = tk.StringVar(value="")
        self.cancer_result_panel = tk.Frame(
            self.cancer_tab,
            background=self.palette["card"],
            highlightbackground=self.palette["border"],
            highlightthickness=1,
            padx=18,
            pady=16,
        )
        self.cancer_result_panel.pack(fill="x", pady=(2, 0))
        self.cancer_result_label = tk.Label(
            self.cancer_result_panel,
            textvariable=self.cancer_result,
            background=self.palette["card"],
            foreground=self.palette["navy"],
            font=("Segoe UI", 14, "bold"),
        )
        self.cancer_result_label.pack(anchor="w")
        self.cancer_probability_label = tk.Label(
            self.cancer_result_panel,
            textvariable=self.cancer_probability,
            background=self.palette["card"],
            foreground=self.palette["muted"],
            font=("Segoe UI", 10),
            justify="left",
        )
        self.cancer_probability_label.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            self.cancer_tab,
            text="Prototipo educativo; no sustituye evaluación, diagnóstico ni consejo de un profesional de salud.",
            style="Subtitle.TLabel",
            wraplength=950,
        ).pack(anchor="w", pady=(10, 0))

    def _build_emotion_tab(self):
        ttk.Label(
            self.emotion_tab,
            text="¿Qué emoción expresa la oración?",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.emotion_tab,
            text="Escribe una oración breve en español. El SVM de texto es independiente del modelo numérico de cáncer.",
            style="Subtitle.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(5, 18))

        self.emotion_text = tk.Text(
            self.emotion_tab,
            height=7,
            wrap="word",
            font=("Segoe UI", 12),
            padx=14,
            pady=12,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.palette["border"],
            background=self.palette["input"],
            foreground=self.palette["text"],
            insertbackground=self.palette["text"],
            selectbackground=self.palette["selection"],
        )
        self.emotion_text.pack(fill="x")
        self.emotion_text.insert("1.0", "Hoy me siento muy feliz con todo lo que he logrado.")

        ttk.Button(
            self.emotion_tab,
            text="Detectar emoción",
            style="Accent.TButton",
            command=self._predict_emotion,
        ).pack(anchor="w", pady=(14, 16))
        self.emotion_result = tk.StringVar(value="La emoción detectada aparecerá aquí.")
        self.emotion_result_panel = tk.Frame(
            self.emotion_tab,
            background=self.palette["card"],
            highlightbackground=self.palette["border"],
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        self.emotion_result_panel.pack(fill="x")
        self.emotion_result_label = tk.Label(
            self.emotion_result_panel,
            textvariable=self.emotion_result,
            background=self.palette["card"],
            foreground=self.palette["navy"],
            font=("Segoe UI", 16, "bold"),
        )
        self.emotion_result_label.pack(anchor="w")
        ttk.Label(
            self.emotion_tab,
            text=(
                "Emociones: alegría, tristeza, enojo, miedo, sorpresa y neutral. "
                "Modelo experimental entrenado con ejemplos pequeños; no interpreta contexto "
                "ni diagnostica estados emocionales. La predicción no es una probabilidad clínica."
            ),
            style="Subtitle.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(14, 0))

    def _load_example(self, dataset_target: int):
        dataset = load_breast_cancer()
        index = next(i for i, target in enumerate(dataset.target) if target == dataset_target)
        for name, value in zip(dataset.feature_names, dataset.data[index]):
            entry = self.measurement_fields[name]
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.8g}")
        self.cancer_result.set("Ejemplo cargado; aún no se ha analizado.")
        self.cancer_probability.set("")

    def _predict_cancer(self):
        try:
            measurements = {
                name: float(entry.get().strip())
                for name, entry in self.measurement_fields.items()
            }
        except ValueError:
            messagebox.showerror("Mediciones inválidas", "Completa las 30 mediciones con números válidos.")
            return

        try:
            result = predict_cancer(self.cancer_model, measurements)
        except ValueError as error:
            messagebox.showerror("No se pudo analizar", str(error))
            return

        self.cancer_result.set(f"Resultado del modelo: {result.label}")
        self.cancer_probability.set(
            f"Probabilidad estimada — benigno: {result.benign_probability:.1%}   "
            f"maligno: {result.malignant_probability:.1%}"
        )

    def _predict_emotion(self):
        try:
            result = predict_emotion(self.emotion_model, self.emotion_text.get("1.0", "end"))
        except ValueError as error:
            messagebox.showerror("Texto requerido", str(error))
            return
        self.emotion_result.set(f"Emoción estimada: {result.label}")


def main():
    cancer_model = load_cancer_model(MODEL_PATH)
    emotion_model = train_emotion_model(EMOTION_DATA_PATH)
    app = DetectionApp(cancer_model, emotion_model)
    app.mainloop()


if __name__ == "__main__":
    main()
