"""
gep_evaluate.py
===============
Loads gep_model.json + ANN, computes all metrics, generates 3 publication plots.
Run this after gep_train.py completes.
"""
import sys
import os
import json
import warnings
import random
from pathlib import Path
from dataclasses import dataclass

# Force UTF-8 output (handles Chinese path chars in Windows console)
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\Jebin Raj\OneDrive\文档\professional\research work\Geopolymer bricks")
DATA_PATH   = BASE / "data set for concrete" / "Data set for geopolymer concrete.xlsx"
MODEL_DIR   = BASE / "files for model"
SCALER_PATH = MODEL_DIR / "zscore_scaler.pkl"
ANN_PATH    = MODEL_DIR / "best_ann_model.keras"
GEP_PATH    = MODEL_DIR / "gep_model.json"
PLOTS_DIR   = MODEL_DIR / "plots"

ALL_FEATURES = [
    "Fly Ash amount (kg/m3)", "GGBFS amount (kg/m3)",
    "NaOH molar concentration", "NaOH amount (kg/m3)",
    "Na2SiO3 amount (kg/m3)", "Extra water",
    "Course Agg. (kg/m3)", "Fine Agg. (kg/m3)",
    "Recycled Agg. (kg/m3)", "Curing Temprature (C) ",
    "Curing Time (hr)", "age of testing (Days)",
]
TARGET       = "Compressive Strength (Mpa)"
FEATURE_LABELS = [f"x{i}" for i in range(1, 13)]

# ── Safe math ──────────────────────────────────────────────────────────────────
def _safe_div(a, b):
    return np.where(np.abs(b) >= 1e-6, a / b, 0.0)

def _safe_sqrt(a):
    return np.sign(a) * np.sqrt(np.abs(a))

def _safe_cbrt(a):
    return np.sign(a) * (np.abs(a) ** (1.0 / 3.0))

def _safe_sq(a):
    return np.clip(a * a, -1e6, 1e6)

# ── Expression engine ──────────────────────────────────────────────────────────
@dataclass
class Expr:
    op: str
    value: object = None
    left: object = None
    right: object = None

    def eval(self, X):
        if self.op == "var":   return X[:, int(self.value)]
        if self.op == "const": return np.full(X.shape[0], float(self.value))
        a = self.left.eval(X)
        if self.op == "sqrt":  return _safe_sqrt(a)
        if self.op == "cbrt":  return _safe_cbrt(a)
        if self.op == "sq":    return _safe_sq(a)
        b = self.right.eval(X)
        if self.op == "+": return a + b
        if self.op == "-": return a - b
        if self.op == "*": return np.clip(a * b, -1e6, 1e6)
        if self.op == "/": return _safe_div(a, b)
        raise ValueError(self.op)

def expr_from_dict(d):
    return Expr(op=d["op"], value=d.get("value"),
                left=expr_from_dict(d["left"]) if d.get("left") else None,
                right=expr_from_dict(d["right"]) if d.get("right") else None)

def gene_matrix(genes, X):
    cols = []
    for g in genes:
        try:
            v = g.eval(X)
            v = np.nan_to_num(v, nan=0.0, posinf=100.0, neginf=-100.0)
            v = np.clip(v, -100.0, 100.0)
        except Exception:
            v = np.zeros(X.shape[0])
        cols.append(v)
    return np.column_stack(cols)

def gep_predict(model, X):
    genes = [expr_from_dict(g) for g in model["genes"]]
    coef  = np.asarray(model["coef"], dtype=float)
    G     = gene_matrix(genes, X)
    scaled = np.column_stack([np.ones(G.shape[0]), G]) @ coef
    return scaled * model["y_sigma"] + model["y_mu"]

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    err    = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    nz     = np.abs(y_true) > 1e-9
    mse    = float(np.mean(err ** 2))
    return {
        "R2":   round(1.0 - ss_res / ss_tot, 6) if ss_tot else float("nan"),
        "RMSE": round(float(np.sqrt(mse)), 6),
        "MAE":  round(float(np.mean(np.abs(err))), 6),
        "MSE":  round(mse, 6),
        "MAPE": round(float(np.mean(np.abs(err[nz] / y_true[nz])) * 100.0), 4),
    }

# ── Plots ──────────────────────────────────────────────────────────────────────
def make_plots(y, ann_pred, gep_pred, train_idx, test_idx, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK_BG  = "#1a1a2e"
    PANEL_BG = "#16213e"
    BLUE     = "#00d4ff"
    RED      = "#ff6b6b"
    GREEN    = "#7ee787"
    WHITE    = "#e0e0e0"
    GREY     = "#666680"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Experimental vs Predicted ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Experimental vs Predicted Compressive Strength (MPa)",
                 color=WHITE, fontsize=14, fontweight="bold", y=1.01)

    for ax, name, pred, col in zip(axes, ["ANN", "GEP"], [ann_pred, gep_pred], [BLUE, RED]):
        ax.set_facecolor(PANEL_BG)
        r2_tr  = results[f"{name}_Train"]["R2"]
        r2_te  = results[f"{name}_Test"]["R2"]
        rmse_te = results[f"{name}_Test"]["RMSE"]

        ax.scatter(y[train_idx], pred[train_idx], c=BLUE,  alpha=0.5, s=28,
                   zorder=3, label=f"Train  R²={r2_tr:.4f}")
        ax.scatter(y[test_idx],  pred[test_idx],  c=GREEN, alpha=0.80, s=44,
                   marker="^", zorder=4, label=f"Test   R²={r2_te:.4f}  RMSE={rmse_te:.3f}")
        mn, mx = float(y.min()), float(y.max())
        margin = (mx - mn) * 0.05
        lims = [mn - margin, mx + margin]
        ax.plot(lims, lims, "--", color=WHITE, lw=1.5, alpha=0.5, zorder=5, label="1:1 line")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Experimental CS (MPa)", color=WHITE, fontsize=11)
        ax.set_ylabel("Predicted CS (MPa)",    color=WHITE, fontsize=11)
        ax.set_title(f"{name} Model", color=WHITE, fontsize=12, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.legend(facecolor="#0d0d1f", labelcolor=WHITE, fontsize=9, framealpha=0.9)
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(1.8)
        ax.grid(True, color=GREY, alpha=0.18, lw=0.5)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp_vs_pred.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/exp_vs_pred.png")

    # ── Plot 2: Residual Analysis (2x2) ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Residual Analysis — ANN & GEP",
                 color=WHITE, fontsize=14, fontweight="bold")

    configs = [
        (axes[0,0], "ANN", ann_pred, BLUE,  "scatter"),
        (axes[0,1], "GEP", gep_pred, RED,   "scatter"),
        (axes[1,0], "ANN", ann_pred, BLUE,  "hist"),
        (axes[1,1], "GEP", gep_pred, RED,   "hist"),
    ]
    for ax, name, pred, col, kind in configs:
        ax.set_facecolor(PANEL_BG)
        res = y - pred
        if kind == "scatter":
            # Colour by train/test
            res_tr = y[train_idx] - pred[train_idx]
            res_te = y[test_idx]  - pred[test_idx]
            ax.scatter(pred[train_idx], res_tr, c=BLUE,  alpha=0.40, s=20, label="Train")
            ax.scatter(pred[test_idx],  res_te, c=GREEN, alpha=0.75, s=32,
                       marker="^", label="Test")
            ax.axhline(0, color=WHITE, lw=1.5, ls="--", alpha=0.7)
            ax.set_xlabel("Predicted CS (MPa)", color=WHITE, fontsize=10)
            ax.set_ylabel("Residual (MPa)",     color=WHITE, fontsize=10)
            ax.set_title(f"{name} — Residuals vs Fitted",
                         color=WHITE, fontsize=11, fontweight="bold")
            ax.legend(facecolor="#0d0d1f", labelcolor=WHITE, fontsize=8)
        else:
            ax.hist(y[train_idx] - pred[train_idx], bins=25, color=BLUE,
                    alpha=0.60, label="Train", edgecolor="none")
            ax.hist(y[test_idx]  - pred[test_idx],  bins=25, color=GREEN,
                    alpha=0.70, label="Test",  edgecolor="none")
            ax.axvline(0, color=WHITE, lw=1.5, ls="--", alpha=0.7)
            ax.set_xlabel("Residual (MPa)", color=WHITE, fontsize=10)
            ax.set_ylabel("Frequency",      color=WHITE, fontsize=10)
            ax.set_title(f"{name} — Residual Distribution",
                         color=WHITE, fontsize=11, fontweight="bold")
            ax.legend(facecolor="#0d0d1f", labelcolor=WHITE, fontsize=8)
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(1.2)
        ax.grid(True, color=GREY, alpha=0.18, lw=0.5)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "residuals.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/residuals.png")

    # ── Plot 3: ANN vs GEP metric comparison ──────────────────────────────────
    metric_keys  = ["R2",   "RMSE", "MAE",   "MSE",    "MAPE"]
    metric_labels = ["R²", "RMSE\n(MPa)", "MAE\n(MPa)", "MSE\n(MPa²)", "MAPE\n(%)"]
    colors = [BLUE, "#0099cc", RED, "#cc3333"]
    cats   = ["ANN\nTrain", "ANN\nTest", "GEP\nTrain", "GEP\nTest"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("ANN vs GEP — Comprehensive Performance Metrics",
                 color=WHITE, fontsize=14, fontweight="bold")

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        ax.set_facecolor(PANEL_BG)
        vals = [
            results["ANN_Train"][mk], results["ANN_Test"][mk],
            results["GEP_Train"][mk], results["GEP_Test"][mk],
        ]
        bars = ax.bar(cats, vals, color=colors, edgecolor="none", width=0.6)
        ax.set_title(ml, color=WHITE, fontsize=11, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY); spine.set_linewidth(0.8)
        ax.grid(True, axis="y", color=GREY, alpha=0.20, lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "ann_vs_gep.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/ann_vs_gep.png")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" GEP Evaluation + Plotting")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    df    = pd.read_excel(DATA_PATH)
    X_raw = df[ALL_FEATURES].to_numpy(dtype=float)
    y     = df[TARGET].to_numpy(dtype=float)

    # Load scaler + apply
    print("Loading scaler...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X_raw)

    # Same 70/30 split
    rng_np  = np.random.default_rng(42)
    perm    = rng_np.permutation(len(y))
    train_n = int(round(len(y) * 0.70))
    train_idx = np.sort(perm[:train_n])
    test_idx  = np.sort(perm[train_n:])
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test")

    # Load GEP model
    print("Loading GEP model...")
    with open(GEP_PATH, encoding="utf-8") as f:
        gep_model = json.load(f)

    # GEP predictions
    gep_pred = gep_predict(gep_model, X)

    # ANN predictions
    print("Loading ANN model...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tensorflow.keras.models import load_model
        ann_model = load_model(ANN_PATH)
    ann_pred = ann_model.predict(X, verbose=0).ravel()

    # Metrics
    results = {}
    print("\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║              COMPREHENSIVE PERFORMANCE METRICS                      ║")
    print("  ╠══════════╦═══════════╦════════╦════════╦════════╦══════════╦═══════╣")
    print("  ║  Model   ║  Subset   ║   R²   ║  RMSE  ║   MAE  ║   MSE    ║  MAPE ║")
    print("  ╠══════════╬═══════════╬════════╬════════╬════════╬══════════╬═══════╣")
    for mname, pred in [("ANN", ann_pred), ("GEP", gep_pred)]:
        for sname, idx in [("Train", train_idx), ("Test", test_idx)]:
            m = compute_metrics(y[idx], pred[idx])
            results[f"{mname}_{sname}"] = m
            print(f"  ║ {mname:^8} ║ {sname:^9} ║"
                  f" {m['R2']:6.4f} ║ {m['RMSE']:6.4f} ║"
                  f" {m['MAE']:6.4f} ║ {m['MSE']:8.4f} ║"
                  f" {m['MAPE']:5.2f}% ║")
    print("  ╚══════════╩═══════════╩════════╩════════╩════════╩══════════╩═══════╝")

    # Save metrics
    with open(MODEL_DIR / "gep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved gep_metrics.json")

    # GEP formula
    print("\n  GEP SYMBOLIC FORMULA:")
    print(f"  CS = {gep_model['coef'][0] * gep_model['y_sigma'] + gep_model['y_mu']:.6f}")
    for i, term in enumerate(gep_model["formula_terms"], 1):
        c = gep_model["coef"][i] * gep_model["y_sigma"]
        print(f"     + ({c:+.6f}) x {term}")
    print()
    print("  Variable legend:")
    for j, feat in enumerate(ALL_FEATURES, 1):
        print(f"    x{j:2d} = {feat}")

    # Generate plots
    print("\nGenerating publication plots...")
    make_plots(y, ann_pred, gep_pred, train_idx, test_idx, results)

    print("\n" + "=" * 60)
    print(" All done! Files saved to: files for model/")
    print("=" * 60)

if __name__ == "__main__":
    main()
