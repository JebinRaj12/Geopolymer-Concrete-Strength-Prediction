"""
gep_train.py
============
Trains a brand-new GEP (Gene Expression Programming) model for
Geopolymer Concrete Compressive Strength prediction.

Design choices
--------------
* Uses the SAME StandardScaler (zscore_scaler.pkl) as the existing ANN
  so both models share an identical 12-feature input pipeline.
* Stable function set only: +, -, *, /, sqrt, cbrt, sq  (no exp/log)
* Population 300, Generations 500, 6 genes with ridge=1e-3
* Saves: gep_model.json, gep_metrics.json, plots/ directory
"""

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\Jebin Raj\OneDrive\文档\professional\research work\Geopolymer bricks")
DATA_PATH   = BASE / "data set for concrete" / "Data set for geopolymer concrete.xlsx"
MODEL_DIR   = BASE / "files for model"
SCALER_PATH = MODEL_DIR / "zscore_scaler.pkl"
ANN_PATH    = MODEL_DIR / "best_ann_model.keras"
PLOTS_DIR   = MODEL_DIR / "plots"

# ── Feature configuration ──────────────────────────────────────────────────────
# All 12 columns — SAME ORDER as zscore_scaler.pkl was fitted on
ALL_FEATURES = [
    "Fly Ash amount (kg/m3)",
    "GGBFS amount (kg/m3)",
    "NaOH molar concentration",
    "NaOH amount (kg/m3)",
    "Na2SiO3 amount (kg/m3)",
    "Extra water",
    "Course Agg. (kg/m3)",
    "Fine Agg. (kg/m3)",
    "Recycled Agg. (kg/m3)",
    "Curing Temprature (C) ",
    "Curing Time (hr)",
    "age of testing (Days)",
]
TARGET         = "Compressive Strength (Mpa)"
N_FEATURES     = len(ALL_FEATURES)                        # 12
FEATURE_LABELS = [f"x{i}" for i in range(1, N_FEATURES + 1)]

# ── GEP hyperparameters ────────────────────────────────────────────────────────
POP_SIZE   = 300
N_GENS     = 500
N_GENES    = 6
MAX_DEPTH  = 4
RIDGE      = 1e-3         # strong enough to prevent overfitting
CROSS_RATE = 0.65
MUT_RATE   = 0.18         # per-gene mutation probability
SHUF_RATE  = 0.05
TOURN_K    = 5
SEED       = 42

# ── Safe math helpers ──────────────────────────────────────────────────────────

def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) >= 1e-6, a / b, 0.0)

def safe_sqrt(a: np.ndarray) -> np.ndarray:
    """Signed square-root: preserves sign, avoids NaN."""
    return np.sign(a) * np.sqrt(np.abs(a))

def safe_cbrt(a: np.ndarray) -> np.ndarray:
    """Real cube-root (works on negative numbers)."""
    return np.sign(a) * (np.abs(a) ** (1.0 / 3.0))

def safe_sq(a: np.ndarray) -> np.ndarray:
    """Squaring — clipped to avoid blow-up."""
    return np.clip(a * a, -1e6, 1e6)

# ── Expression tree ────────────────────────────────────────────────────────────

OPS_BINARY = ["+", "-", "*", "/"]
OPS_UNARY  = ["sqrt", "cbrt", "sq"]
OPS_ALL    = OPS_BINARY + OPS_UNARY

CONST_POOL = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 0.1]


@dataclass
class Expr:
    op:    str
    value: object = None
    left:  object = None
    right: object = None

    # ── Evaluation ──
    def eval(self, X: np.ndarray) -> np.ndarray:
        if self.op == "var":
            return X[:, int(self.value)]
        if self.op == "const":
            return np.full(X.shape[0], float(self.value))
        a = self.left.eval(X)
        if self.op == "sqrt":
            return safe_sqrt(a)
        if self.op == "cbrt":
            return safe_cbrt(a)
        if self.op == "sq":
            return safe_sq(a)
        b = self.right.eval(X)
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return np.clip(a * b, -1e6, 1e6)
        if self.op == "/":
            return safe_div(a, b)
        raise ValueError(f"Unknown op: {self.op}")

    # ── Human-readable formula ──
    def text(self) -> str:
        if self.op == "var":
            return FEATURE_LABELS[int(self.value)]
        if self.op == "const":
            return f"{float(self.value):.4g}"
        if self.op == "sqrt":
            return f"sqrt(|{self.left.text()}|)"
        if self.op == "cbrt":
            return f"cbrt({self.left.text()})"
        if self.op == "sq":
            return f"({self.left.text()})^2"
        return f"({self.left.text()} {self.op} {self.right.text()})"

    # ── Serialisation ──
    def to_dict(self) -> dict:
        return {
            "op":    self.op,
            "value": self.value,
            "left":  self.left.to_dict()  if self.left  else None,
            "right": self.right.to_dict() if self.right else None,
        }


def expr_from_dict(d: dict) -> Expr:
    return Expr(
        op    = d["op"],
        value = d.get("value"),
        left  = expr_from_dict(d["left"])  if d.get("left")  else None,
        right = expr_from_dict(d["right"]) if d.get("right") else None,
    )


def random_expr(rng: random.Random, depth: int) -> Expr:
    """Generate a random expression tree of given max depth."""
    if depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.82:
            return Expr("var", value=rng.randrange(N_FEATURES))
        return Expr("const", value=rng.choice(CONST_POOL))
    op = rng.choice(OPS_ALL)
    if op in OPS_UNARY:
        return Expr(op, left=random_expr(rng, depth - 1))
    return Expr(op, left=random_expr(rng, depth - 1), right=random_expr(rng, depth - 1))


def clone_expr(e: Expr) -> Expr:
    return expr_from_dict(e.to_dict())


def collect_nodes(e: Expr, nodes: list = None) -> list:
    if nodes is None:
        nodes = []
    nodes.append(e)
    if e.left:
        collect_nodes(e.left, nodes)
    if e.right:
        collect_nodes(e.right, nodes)
    return nodes


def mutate_gene(e: Expr, rng: random.Random, max_depth: int) -> Expr:
    e = clone_expr(e)
    nodes = collect_nodes(e)
    target = rng.choice(nodes)
    rep = random_expr(rng, rng.randint(0, max_depth))
    target.op, target.value = rep.op, rep.value
    target.left, target.right = rep.left, rep.right
    return e


# ── Gene evaluation helpers ────────────────────────────────────────────────────

def gene_matrix(genes: list, X: np.ndarray) -> np.ndarray:
    """Evaluate all genes on X; clip & nan-guard each column."""
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


def fit_coefficients(genes: list, X: np.ndarray, y_scaled: np.ndarray,
                     train_idx: np.ndarray) -> tuple:
    """
    Ridge-regularised linear regression on gene outputs.
    Returns (fitness, coef, full_pred_scaled).
    """
    G   = gene_matrix(genes, X)
    A   = np.column_stack([np.ones(G.shape[0]), G])
    Atr = A[train_idx]
    ytr = y_scaled[train_idx]

    penalty = RIDGE * np.eye(Atr.shape[1])
    penalty[0, 0] = 0.0                          # don't penalise intercept

    try:
        coef = np.linalg.solve(Atr.T @ Atr + penalty, Atr.T @ ytr)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(Atr, ytr, rcond=None)[0]

    pred_scaled = A @ coef
    train_rmse  = float(np.sqrt(np.mean((pred_scaled[train_idx] - ytr) ** 2)))

    # Complexity penalty — small, encourages parsimonious models
    complexity = sum(len(collect_nodes(g)) for g in genes)
    fitness = train_rmse + complexity * 0.0001

    return fitness, coef, pred_scaled


# ── Tournament selection ───────────────────────────────────────────────────────

def tournament(pop: list, rng: random.Random) -> dict:
    return min(rng.sample(pop, TOURN_K), key=lambda x: x["fitness"])


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err    = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    nz     = np.abs(y_true) > 1e-9
    mse    = float(np.mean(err ** 2))
    return {
        "R2":   1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "RMSE": float(np.sqrt(mse)),
        "MAE":  float(np.mean(np.abs(err))),
        "MSE":  mse,
        "MAPE": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100.0),
    }


# ── GEP training loop ──────────────────────────────────────────────────────────

def train_gep(X: np.ndarray, y_scaled: np.ndarray,
              train_idx: np.ndarray) -> dict:

    rng = random.Random(SEED)

    def make_individual() -> dict:
        genes = [random_expr(rng, MAX_DEPTH) for _ in range(N_GENES)]
        fitness, coef, _ = fit_coefficients(genes, X, y_scaled, train_idx)
        return {"genes": genes, "fitness": fitness, "coef": coef}

    print(f"  Initialising population ({POP_SIZE} individuals)...")
    pop  = [make_individual() for _ in range(POP_SIZE)]
    best = min(pop, key=lambda x: x["fitness"])
    print(f"  Gen   0 | fitness = {best['fitness']:.6f}")

    no_improve = 0

    for gen in range(1, N_GENS + 1):
        next_pop = [{                                    # elitism: keep best
            "genes":   [clone_expr(g) for g in best["genes"]],
            "fitness": best["fitness"],
            "coef":    best["coef"],
        }]

        while len(next_pop) < POP_SIZE:
            p1 = tournament(pop, rng)
            p2 = tournament(pop, rng)

            # ── Crossover ──
            child = [clone_expr(g) for g in p1["genes"]]
            if rng.random() < CROSS_RATE and N_GENES > 1:
                cut   = rng.randrange(1, N_GENES)
                child = ([clone_expr(g) for g in p1["genes"][:cut]] +
                         [clone_expr(g) for g in p2["genes"][cut:]])

            # ── Mutation ──
            for i in range(N_GENES):
                if rng.random() < MUT_RATE:
                    child[i] = mutate_gene(child[i], rng, MAX_DEPTH)

            # ── Gene shuffle ──
            if rng.random() < SHUF_RATE:
                rng.shuffle(child)

            fitness, coef, _ = fit_coefficients(child, X, y_scaled, train_idx)
            next_pop.append({"genes": child, "fitness": fitness, "coef": coef})

        pop = next_pop
        candidate = min(pop, key=lambda x: x["fitness"])
        if candidate["fitness"] < best["fitness"]:
            best       = candidate
            no_improve = 0
        else:
            no_improve += 1

        if gen % 50 == 0 or gen == N_GENS:
            print(f"  Gen {gen:3d} | fitness = {best['fitness']:.6f}")

    return best


# ── GEP prediction ─────────────────────────────────────────────────────────────

def gep_predict(model: dict, X: np.ndarray) -> np.ndarray:
    genes = [
        expr_from_dict(g) if isinstance(g, dict) else g
        for g in model["genes"]
    ]
    coef    = np.asarray(model["coef"], dtype=float)
    G       = gene_matrix(genes, X)
    scaled  = np.column_stack([np.ones(G.shape[0]), G]) @ coef
    return scaled * model["y_sigma"] + model["y_mu"]


# ── Plotting ───────────────────────────────────────────────────────────────────

def make_plots(y: np.ndarray, ann_pred: np.ndarray, gep_pred: np.ndarray,
               train_idx: np.ndarray, test_idx: np.ndarray,
               results: dict) -> None:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    DARK_BG  = "#1a1a2e"
    PANEL_BG = "#16213e"
    BLUE     = "#00d4ff"
    RED      = "#ff6b6b"
    WHITE    = "#e0e0e0"
    GREY     = "#666680"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Experimental vs Predicted ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Experimental vs Predicted — Compressive Strength (MPa)",
                 color=WHITE, fontsize=14, fontweight="bold", y=1.01)

    for ax, name, pred in zip(axes, ["ANN", "GEP"], [ann_pred, gep_pred]):
        ax.set_facecolor(PANEL_BG)
        r2_tr = results[f"{name}_Train"]["R2"]
        r2_te = results[f"{name}_Test"]["R2"]
        ax.scatter(y[train_idx], pred[train_idx],
                   c=BLUE, alpha=0.55, s=28, zorder=3,
                   label=f"Train  R²={r2_tr:.4f}")
        ax.scatter(y[test_idx], pred[test_idx],
                   c=RED, alpha=0.75, s=42, marker="^", zorder=4,
                   label=f"Test   R²={r2_te:.4f}")
        mn, mx = float(y.min()), float(y.max())
        ax.plot([mn, mx], [mn, mx], "--", color=WHITE, lw=1.5,
                alpha=0.6, zorder=5, label="Perfect fit (1:1)")
        ax.set_xlabel("Experimental CS (MPa)", color=WHITE, fontsize=11)
        ax.set_ylabel("Predicted CS (MPa)",     color=WHITE, fontsize=11)
        ax.set_title(f"{name} Model", color=WHITE, fontsize=12, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=9)
        leg = ax.legend(facecolor="#0d0d1f", labelcolor=WHITE,
                        fontsize=9, framealpha=0.85)
        for spine in ax.spines.values():
            spine.set_edgecolor(BLUE if name == "ANN" else RED)
            spine.set_linewidth(1.5)
        ax.grid(True, color=GREY, alpha=0.2, lw=0.5)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "exp_vs_pred.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/exp_vs_pred.png")

    # ── Plot 2: Residual Analysis ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Residual Analysis — ANN & GEP",
                 color=WHITE, fontsize=14, fontweight="bold")

    pairs = [
        (axes[0, 0], "ANN", ann_pred, BLUE),
        (axes[0, 1], "GEP", gep_pred, RED),
        (axes[1, 0], "ANN", ann_pred, BLUE),
        (axes[1, 1], "GEP", gep_pred, RED),
    ]
    for i, (ax, name, pred, color) in enumerate(pairs):
        ax.set_facecolor(PANEL_BG)
        res = y - pred
        if i < 2:
            # Residuals vs predicted
            ax.scatter(pred, res, c=color, alpha=0.45, s=22, zorder=3)
            ax.axhline(0, color=WHITE, lw=1.5, ls="--", alpha=0.7, zorder=4)
            ax.set_xlabel("Predicted CS (MPa)", color=WHITE, fontsize=10)
            ax.set_ylabel("Residual (MPa)",      color=WHITE, fontsize=10)
            ax.set_title(f"{name} — Residuals vs Fitted",
                         color=WHITE, fontsize=11, fontweight="bold")
        else:
            # Histogram of residuals
            ax.hist(res, bins=30, color=color, alpha=0.75, edgecolor="none")
            ax.axvline(0, color=WHITE, lw=1.5, ls="--", alpha=0.7)
            ax.set_xlabel("Residual (MPa)", color=WHITE, fontsize=10)
            ax.set_ylabel("Frequency",       color=WHITE, fontsize=10)
            ax.set_title(f"{name} — Residual Distribution",
                         color=WHITE, fontsize=11, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)
        ax.grid(True, color=GREY, alpha=0.2, lw=0.5)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "residuals.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/residuals.png")

    # ── Plot 3: ANN vs GEP bar comparison ─────────────────────────────────────
    metric_keys  = ["R2", "RMSE", "MAE", "MSE", "MAPE"]
    metric_labels = ["R²", "RMSE\n(MPa)", "MAE\n(MPa)", "MSE\n(MPa²)", "MAPE\n(%)"]
    categories   = ["ANN Train", "ANN Test", "GEP Train", "GEP Test"]
    colors       = [BLUE, "#0099bb", RED, "#cc4444"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("ANN vs GEP — Performance Metrics Comparison",
                 color=WHITE, fontsize=14, fontweight="bold")

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        ax.set_facecolor(PANEL_BG)
        vals = [
            results["ANN_Train"][mk],
            results["ANN_Test"][mk],
            results["GEP_Train"][mk],
            results["GEP_Test"][mk],
        ]
        bars = ax.bar(categories, vals, color=colors,
                      edgecolor="none", width=0.55)
        ax.set_title(ml, color=WHITE, fontsize=11, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=7.5)
        ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
            spine.set_linewidth(0.8)
        ax.grid(True, axis="y", color=GREY, alpha=0.2, lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=WHITE, fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "ann_vs_gep.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  Saved: plots/ann_vs_gep.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" GEP Training — Geopolymer Concrete CS Prediction")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/6] Loading dataset...")
    df    = pd.read_excel(DATA_PATH)
    X_raw = df[ALL_FEATURES].to_numpy(dtype=float)
    y     = df[TARGET].to_numpy(dtype=float)
    print(f"      {len(y)} samples, {N_FEATURES} features, target range "
          f"{y.min():.1f}–{y.max():.1f} MPa")

    # ── 2. Load & apply scaler ──
    print("\n[2/6] Loading StandardScaler (same as ANN)...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X_raw)
    print(f"      Scaler features: {scaler.n_features_in_}")

    # ── 3. 70/30 split (seed=42) ──
    print("\n[3/6] Splitting 70/30 (seed=42)...")
    rng_np  = np.random.default_rng(SEED)
    perm    = rng_np.permutation(len(y))
    train_n = int(round(len(y) * 0.70))
    train_idx = np.sort(perm[:train_n])
    test_idx  = np.sort(perm[train_n:])
    print(f"      Train: {len(train_idx)}  |  Test: {len(test_idx)}")

    # Standardise target using training statistics
    y_mu    = float(y[train_idx].mean())
    y_sigma = float(y[train_idx].std(ddof=0)) or 1.0
    y_scaled = (y - y_mu) / y_sigma

    # ── 4. Train GEP ──
    print(f"\n[4/6] Training GEP "
          f"(pop={POP_SIZE}, gens={N_GENS}, genes={N_GENES})...")
    best = train_gep(X, y_scaled, train_idx)

    # ── 5. Save model ──
    print("\n[5/6] Saving GEP model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    gep_model = {
        "description": (
            f"GEP symbolic regression | {N_GENES} genes | "
            f"pop={POP_SIZE} gen={N_GENS} ridge={RIDGE}"
        ),
        "feature_names":   ALL_FEATURES,
        "feature_labels":  FEATURE_LABELS,
        "y_mu":            y_mu,
        "y_sigma":         y_sigma,
        "genes":           [g.to_dict() for g in best["genes"]],
        "coef":            best["coef"].tolist(),
        "formula_terms":   [g.text() for g in best["genes"]],
    }

    gep_model_path = MODEL_DIR / "gep_model.json"
    with open(gep_model_path, "w", encoding="utf-8") as f:
        json.dump(gep_model, f, indent=2)
    print(f"      Saved: {gep_model_path}")

    # ── 6. Evaluate & plot ──
    print("\n[6/6] Evaluating models and generating plots...")

    # GEP predictions
    gep_pred = gep_predict(gep_model, X)

    # ANN predictions
    print("  Loading ANN for comparison...")
    import warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tensorflow.keras.models import load_model as keras_load_model
        ann_model = keras_load_model(ANN_PATH)
    ann_pred = ann_model.predict(X, verbose=0).ravel()

    # Compute all metrics
    results = {}
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║           PERFORMANCE METRICS SUMMARY                       ║")
    print("  ╠══════════╦═══════════╦════════╦════════╦════════╦══════════╣")
    print("  ║  Model   ║  Subset   ║   R²   ║  RMSE  ║   MAE  ║   MAPE   ║")
    print("  ╠══════════╬═══════════╬════════╬════════╬════════╬══════════╣")

    for mname, pred in [("ANN", ann_pred), ("GEP", gep_pred)]:
        for sname, idx in [("Train", train_idx), ("Test", test_idx)]:
            m = compute_metrics(y[idx], pred[idx])
            key = f"{mname}_{sname}"
            results[key] = m
            print(f"  ║ {mname:^8} ║ {sname:^9} ║"
                  f" {m['R2']:6.4f} ║ {m['RMSE']:6.4f} ║"
                  f" {m['MAE']:6.4f} ║ {m['MAPE']:7.2f}% ║")
    print("  ╚══════════╩═══════════╩════════╩════════╩════════╩══════════╝")

    # MSE rows (separate since not in table above)
    print("\n  MSE values:")
    for k, v in results.items():
        print(f"    {k}: MSE = {v['MSE']:.4f}")

    # Save metrics JSON
    metrics_path = MODEL_DIR / "gep_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {metrics_path}")

    # Generate plots
    print("\n  Generating plots...")
    make_plots(y, ann_pred, gep_pred, train_idx, test_idx, results)

    # Print GEP formula
    print("\n  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║                  GEP SYMBOLIC FORMULA                     ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print(f"  CS_predicted = {gep_model['coef'][0] * y_sigma + y_mu:.6f}")
    for i, term in enumerate(gep_model["formula_terms"], 1):
        c = gep_model["coef"][i] * y_sigma
        print(f"    + ({c:+.6f}) × {term}")

    print("\n  Variable legend:")
    for i, feat in enumerate(ALL_FEATURES, 1):
        print(f"    x{i:2d} = {feat}")

    print("\n" + "=" * 60)
    print(" Training complete! All files saved to:")
    print(f"   {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
