import sys
import os
import json
import warnings
from pathlib import Path

# Force UTF-8 output to handle Chinese characters in Windows terminal
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Suppress TensorFlow logging to keep console clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox

# ==============================================================================
# BASE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).parent.resolve()
ANN_PATH = BASE_DIR / "best_ann_model.keras"
SCALER_PATH = BASE_DIR / "zscore_scaler.pkl"
GEP_PATH = BASE_DIR / "gep_model.json"

# ==============================================================================
# INPUT FEATURE CONFIGURATION
# ==============================================================================
FEATURES = [
    ("Fly Ash", "kg/m³", 400.0),
    ("GGBFS", "kg/m³", 0.0),
    ("NaOH Molarity", "M", 8.0),
    ("NaOH Amount", "kg/m³", 70.0),
    ("Na₂SiO₃ Amount", "kg/m³", 120.0),
    ("Extra Water", "kg/m³", 0.0),
    ("Coarse Aggregate", "kg/m³", 1150.0),
    ("Fine Aggregate", "kg/m³", 650.0),
    ("Recycled Aggregate", "kg/m³", 0.0),
    ("Curing Temperature", "°C", 24.0),
    ("Curing Time", "hr", 24.0),
    ("Age of Testing", "Days", 7.0),
]

# ==============================================================================
# GEP MATHEMATICAL PARSER & PREDICTOR
# ==============================================================================
def _safe_div(a, b):
    return np.where(np.abs(b) >= 1e-6, a / b, 0.0)

def _safe_sqrt(a):
    return np.sign(a) * np.sqrt(np.abs(a))

def _safe_cbrt(a):
    return np.sign(a) * (np.abs(a) ** (1.0 / 3.0))

def _safe_sq(a):
    return np.clip(a * a, -1e6, 1e6)

class Expr:
    def __init__(self, op, value=None, left=None, right=None):
        self.op = op
        self.value = value
        self.left = left
        self.right = right

    def eval(self, X):
        if self.op == "var":
            return X[:, int(self.value)]
        if self.op == "const":
            return np.full(X.shape[0], float(self.value))
        
        a = self.left.eval(X)
        if self.op == "sqrt":
            return _safe_sqrt(a)
        if self.op == "cbrt":
            return _safe_cbrt(a)
        if self.op == "sq":
            return _safe_sq(a)
            
        b = self.right.eval(X)
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return np.clip(a * b, -1e6, 1e6)
        if self.op == "/":
            return _safe_div(a, b)
        raise ValueError(f"Unknown operation: {self.op}")

def expr_from_dict(d):
    return Expr(
        op=d["op"],
        value=d.get("value"),
        left=expr_from_dict(d["left"]) if d.get("left") else None,
        right=expr_from_dict(d["right"]) if d.get("right") else None
    )


def gep_predict_single(gep_model, x_scaled):
    """
    GEP prediction using RAW inputs
    """

    X = x_scaled.reshape(1, -1)

    genes = [
        expr_from_dict(g)
        for g in gep_model["genes"]
    ]

    cols = []

    for g in genes:
        try:
            v = g.eval(X)
            v = np.nan_to_num(
                v,
                nan=0.0,
                posinf=100.0,
                neginf=-100.0
            )
            v = np.clip(v, -100.0, 100.0)
        except Exception:
            v = np.zeros(X.shape[0])

        cols.append(v)

  
    G = np.column_stack(cols)

    coef = np.asarray(
        gep_model["coef"],
        dtype=float
    )

    scaled_pred = np.column_stack(
        [np.ones(G.shape[0]), G]
    ) @ coef

    final_pred = (
        scaled_pred * gep_model["y_sigma"]
    ) + gep_model["y_mu"]

    return float(final_pred[0])












# ==============================================================================
# MODEL LOADERS
# ==============================================================================
def load_models():
    ann_model = None
    scaler = None
    gep_model = None
    load_errors = []

    # Load Scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        load_errors.append(f"Scaler (zscore_scaler.pkl): {str(e)}")

    # Load GEP JSON
    try:
        with open(GEP_PATH, "r", encoding="utf-8") as f:
            gep_model = json.load(f)
    except Exception as e:
        load_errors.append(f"GEP Model (gep_model.json): {str(e)}")

    # Load ANN (Keras)
    try:
        from tensorflow.keras.models import load_model
        ann_model = load_model(str(ANN_PATH))
    except Exception as e:
        load_errors.append(f"ANN Model (best_ann_model.keras): {str(e)}")

    return ann_model, scaler, gep_model, load_errors

# ==============================================================================
# TKINTER GUI APPLICATION
# ==============================================================================
class GeopolymerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Geopolymer Strength Predictor")
        self.root.geometry("1180x690")
        self.root.resizable(False, False)
        
        # Professional Civil Engineering White/Slate Style
        self.bg_color = "#f8fafc"       # slate-50
        self.card_bg = "#ffffff"        # white
        self.border_color = "#e2e8f0"   # slate-200
        self.text_main = "#0f172a"      # slate-900
        self.text_muted = "#475569"     # slate-600
        self.primary_color = "#1e3a8a"   # Navy Blue
        self.primary_hover = "#1e40af"
        
        self.root.configure(bg=self.bg_color)
        self.entries = []
        
        # Load models and scaling parameters
        self.ann_model, self.scaler, self.gep_model, self.load_errors = load_models()
        
        self.build_ui()
        
        # Notify user of any loading errors
        if self.load_errors:
            err_msg = "Warning: The following files failed to load:\n\n" + "\n".join(self.load_errors)
            messagebox.showwarning("Model Load Warning", err_msg)
            self.status_label.config(text="⚠️ Warning: Model load failed. Check logs.", fg="#b91c1c")

    def build_ui(self):
        # 1. HEADER BANNER
        header_frame = tk.Frame(self.root, bg=self.card_bg, bd=0, highlightthickness=0)
        header_frame.pack(fill="x", side="top")
        
        # Subtle horizontal divider line
        divider = tk.Frame(self.root, bg=self.border_color, height=1)
        divider.pack(fill="x", side="top")

        header_content = tk.Frame(header_frame, bg=self.card_bg, padx=25, pady=12)
        header_content.pack(fill="both")

        tk.Label(
            header_content,
            text="🔬 Geopolymer Concrete Strength Predictor",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.primary_color
        ).pack(anchor="w")

        tk.Label(
            header_content,
            text="Dual Artificial Neural Network (ANN) & Gene Expression Programming (GEP) Estimation Platform",
            font=("Segoe UI", 9),
            bg=self.card_bg,
            fg=self.text_muted
        ).pack(anchor="w", pady=(2, 0))

        # 2. MAIN CONTAINER
        main_container = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # 3. LEFT PANEL: Mix Design Inputs
        left_panel = tk.Frame(
            main_container,
            bg=self.card_bg,
            highlightcolor=self.border_color,
            highlightthickness=1,
            bd=0
        )
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(
            left_panel,
            text="Mix Design Inputs",
            font=("Segoe UI", 11, "bold"),
            bg=self.card_bg,
            fg=self.text_main
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Grid for 12 input boxes (6 rows x 2 cols)
        grid_frame = tk.Frame(left_panel, bg=self.card_bg)
        grid_frame.pack(padx=20, pady=0)

        for i, (label, unit, default) in enumerate(FEATURES):
            row = i // 2
            col = i % 2
            
            cell = tk.Frame(grid_frame, bg=self.card_bg)
            cell.grid(row=row, column=col, padx=10, pady=6, sticky="w")

            lbl_txt = f"x{i+1}: {label} ({unit})"
            tk.Label(
                cell,
                text=lbl_txt,
                font=("Segoe UI", 8, "bold"),
                bg=self.card_bg,
                fg=self.text_muted
            ).pack(anchor="w", pady=(0, 2))

            entry = tk.Entry(
                cell,
                width=22,
                font=("Segoe UI", 10),
                bg="#f8fafc",
                fg=self.text_main,
                bd=1,
                relief="solid",
                highlightthickness=0
            )
            entry.pack(ipady=4)
            entry.insert(0, str(default))
            self.entries.append(entry)

        # Action Buttons Container
        btn_frame = tk.Frame(left_panel, bg=self.card_bg)
        btn_frame.pack(fill="x", padx=20, pady=(15, 15))

        self.calc_btn = tk.Button(
            btn_frame,
            text="⚡ Calculate Strength",
            font=("Segoe UI", 10, "bold"),
            bg=self.primary_color,
            fg="white",
            activebackground=self.primary_hover,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.calculate_predictions,
            pady=8
        )
        self.calc_btn.pack(fill="x", pady=(0, 8))

        self.reset_btn = tk.Button(
            btn_frame,
            text="↺ Reset Values",
            font=("Segoe UI", 9, "bold"),
            bg="#f1f5f9",
            fg=self.text_muted,
            activebackground="#e2e8f0",
            activeforeground=self.text_main,
            relief="flat",
            cursor="hand2",
            command=self.reset_inputs,
            pady=6
        )
        self.reset_btn.pack(fill="x")

        # 4. RIGHT PANEL: Prediction Cards and Equation Display
        right_panel = tk.Frame(main_container, bg=self.bg_color)
        right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # Cards frame for horizontal display (side-by-side)
        cards_frame = tk.Frame(right_panel, bg=self.bg_color)
        cards_frame.pack(fill="x", pady=(0, 15))

        # ANN Result Card (Green accent)
        ann_card = tk.Frame(
            cards_frame,
            bg="#f0fdf4",
            bd=0,
            highlightthickness=1,
            highlightbackground="#bbf7d0"
        )
        ann_card.pack(side="left", fill="both", expand=True, padx=(0, 10), ipady=12)

        tk.Label(
            ann_card,
            text="🧠 ANN PREDICTION",
            font=("Segoe UI", 9, "bold"),
            bg="#f0fdf4",
            fg="#15803d"
        ).pack(anchor="w", padx=15, pady=(12, 4))

        self.ann_val_lbl = tk.Label(
            ann_card,
            text="— — MPa",
            font=("Segoe UI", 24, "bold"),
            bg="#f0fdf4",
            fg="#166534"
        )
        self.ann_val_lbl.pack(anchor="w", padx=15)

        # GEP Result Card (Orange accent)
        gep_card = tk.Frame(
            cards_frame,
            bg="#fff7ed",
            bd=0,
            highlightthickness=1,
            highlightbackground="#ffedd5"
        )
        gep_card.pack(side="left", fill="both", expand=True, padx=(10, 0), ipady=12)

        tk.Label(
            gep_card,
            text="🧬 GEP PREDICTION",
            font=("Segoe UI", 9, "bold"),
            bg="#fff7ed",
            fg="#c2410c"
        ).pack(anchor="w", padx=15, pady=(12, 4))

        self.gep_val_lbl = tk.Label(
            gep_card,
            text="— — MPa",
            font=("Segoe UI", 24, "bold"),
            bg="#fff7ed",
            fg="#9a3412"
        )
        self.gep_val_lbl.pack(anchor="w", padx=15)

        # GEP Equation Visualizer Card
        eq_card = tk.Frame(
            right_panel,
            bg=self.card_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.border_color
        )
        eq_card.pack(fill="both", expand=True)

        tk.Label(
            eq_card,
            text="GEP Symbolic Equation & Details",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            fg=self.text_main
        ).pack(anchor="w", padx=15, pady=(12, 8))

        # Formula text visualizer
        self.eq_text = tk.Text(
            eq_card,
            font=("Consolas", 9),
            bg="#f8fafc",
            fg=self.text_main,
            bd=1,
            relief="solid",
            highlightthickness=0,
            wrap="word"
        )
        self.eq_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Populate formula text
        self.populate_equation_box()

        # 5. FOOTER STATUS BAR
        footer = tk.Frame(self.root, bg=self.card_bg, bd=0, height=25)
        footer.pack(fill="x", side="bottom")
        
        divider_foot = tk.Frame(footer, bg=self.border_color, height=1)
        divider_foot.pack(fill="x", side="top")

        self.status_label = tk.Label(
            footer,
            text="System Ready",
            font=("Segoe UI", 8),
            bg=self.card_bg,
            fg=self.text_muted,
            padx=15,
            pady=3
        )
        self.status_label.pack(side="left")

    def populate_equation_box(self):
        if not self.gep_model:
            self.eq_text.insert("1.0", "Error: GEP model could not be loaded.")
            self.eq_text.config(state="disabled")
            return

        # Build clean visual formula description
        intercept = self.gep_model['coef'][0] * self.gep_model['y_sigma'] + self.gep_model['y_mu']
        
        lines = []
        lines.append("=" * 68)
        lines.append(" GENE EXPRESSION PROGRAMMING (GEP) SYMBOLIC FORMULA")
        lines.append("=" * 68)
        lines.append(f"CS (MPa) = {intercept:.6f}")
        
        for i, term in enumerate(self.gep_model["formula_terms"], 1):
            c = self.gep_model["coef"][i] * self.gep_model["y_sigma"]
            lines.append(f"   + ({c:+.6f}) * Gene_{i}")
        
        lines.append("\n" + "-" * 68)
        lines.append(" GENE EXPRESSIONS (GENES 1-6)")
        lines.append("-" * 68)
        for i, term in enumerate(self.gep_model["formula_terms"], 1):
            lines.append(f"Gene_{i} = {term}")

        lines.append("\n" + "-" * 68)
        lines.append(" VARIABLE DICTIONARY")
        lines.append("-" * 68)
        for j, feature_name in enumerate(self.gep_model["feature_names"], 1):
            lines.append(f"  x{j:<2} = {feature_name}")
            
        formula_content = "\n".join(lines)
        self.eq_text.delete("1.0", tk.END)
        self.eq_text.insert("1.0", formula_content)
        self.eq_text.config(state="disabled") # Read-only

    def calculate_predictions(self):
        # 1. Parse Inputs
        try:
            raw_vals = []
            for entry in self.entries:
                val = float(entry.get().strip())
                raw_vals.append(val)
        except ValueError:
            messagebox.showerror("Input Error", "Please verify all 12 input boxes contain valid numeric values.")
            return

        x_raw = np.array(raw_vals).reshape(1, -1)

        # 2. Scale Inputs
        if not self.scaler:
            messagebox.showerror("Error", "Z-score Scaler is not loaded. Cannot run prediction.")
            return
        
        try:
            x_scaled = self.scaler.transform(x_raw)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scale input: {str(e)}")
            return

        # 3. ANN Prediction
        ann_pred_val = None
        if self.ann_model:
            try:
                pred_arr = self.ann_model.predict(x_scaled, verbose=0)
                ann_pred_val = float(pred_arr[0][0])
            except Exception as e:
                print(f"ANN predict error: {e}")
        
        # 4. GEP Prediction
        gep_pred_val = None
        if self.gep_model:
            try:
                gep_pred_val = gep_predict_single(self.gep_model,  x_scaled[0])
            except Exception as e:
                print(f"GEP predict error: {e}")

        # Update Labels using helper
        self.update_result_displays(ann_pred_val, gep_pred_val)
        self.status_label.config(text="Prediction complete.", fg=self.text_muted)

    def update_result_displays(self, ann_val, gep_val):
        if ann_val is not None:
            self.ann_val_lbl.config(text=f"{ann_val:.2f} MPa")
        else:
            self.ann_val_lbl.config(text="— — MPa")

        if gep_val is not None:
            self.gep_val_lbl.config(text=f"{gep_val:.2f} MPa")
        else:
            self.gep_val_lbl.config(text="— — MPa")

    def reset_inputs(self):
        for entry, (_, _, default) in zip(self.entries, FEATURES):
            entry.delete(0, tk.END)
            entry.insert(0, str(default))
        
        # Reset labels
        self.status_label.config(text="Mix design inputs reset to defaults.", fg=self.text_muted)
        self.update_result_displays(None, None)

# ==============================================================================
# MAIN EXECUTION / TEST MODE
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=== RUNNING GEOPOLYMER GUI TEST MODE ===")
        print(f"Directory: {BASE_DIR}")
        
        # 1. Check file existence
        print("Checking required files...")
        for name, path in [("ANN Model", ANN_PATH), ("Scaler", SCALER_PATH), ("GEP Model", GEP_PATH)]:
            if path.exists():
                print(f"  [OK] Found {name} at {path}")
            else:
                print(f"  [ERROR] Missing {name} at {path}")
                sys.exit(1)
                
        # 2. Load models
        print("Loading models...")
        ann_model, scaler, gep_model, errors = load_models()
        if errors:
            print("  [ERROR] Failed to load one or more models:")
            for err in errors:
                print(f"    - {err}")
            sys.exit(1)
        print("  [OK] Models loaded successfully.")
        
        # 3. Test prediction with default values
        print("Running prediction check with default inputs...")
        default_vals = np.array([f[2] for f in FEATURES]).reshape(1, -1)
        try:
            x_scaled = scaler.transform(default_vals)
            print(f"  Scaled inputs: {x_scaled[0]}")
            
            # Predict ANN
            ann_pred = ann_model.predict(x_scaled, verbose=0)
            ann_val = float(ann_pred[0][0])
            print(f"  [OK] ANN Prediction: {ann_val:.4f} MPa")
            
            # Predict GEP
            gep_val = gep_predict_single(gep_model, x_raw[0])
            print(f"  [OK] GEP Prediction: {gep_val:.4f} MPa")
            
            if not np.isfinite(ann_val) or not np.isfinite(gep_val):
                print("  [ERROR] Prediction returned NaN or Infinite values.")
                sys.exit(1)
        except Exception as e:
            print(f"  [ERROR] Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        # 4. Check Tkinter UI initialization
        print("Checking GUI window initialization...")
        try:
            root = tk.Tk()
            app = GeopolymerApp(root)
            # Update root to render widgets and confirm no syntax/runtime issues in build_ui
            root.update()
            # Destroy window
            root.destroy()
            print("  [OK] Tkinter GUI initialized and destroyed successfully.")
        except Exception as e:
            print(f"  [ERROR] GUI initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        print("=== TEST COMPLETED SUCCESSFULLY ===")
        sys.exit(0)
    else:
        root = tk.Tk()
        app = GeopolymerApp(root)
        root.mainloop()
