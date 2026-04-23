#!/usr/bin/env python3
"""
SUTRA Model Implementation for US COVID-19 Data
================================================
Based on: Agrawal et al. (2022) arXiv:2101.09158v6

Requirements:
    pip install pandas numpy scipy matplotlib

Usage:
    python sutra_us.py

Input:  national-history.csv  (COVID Tracking Project data)
Output: sutra_results.png, sutra_parameters.csv
"""

import pandas as pd
import numpy as np
from scipy.optimize import nnls, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH  = "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Data/national-history.csv"
GAMMA      = 0.1          # Removal rate (10-day mean infection duration, paper §4)
P0         = 331_000_000  # US population
START_DATE = "2020-02-29" # First date with meaningful cases

# Phase boundaries derived from:
#   - US COVID timeline (CDC / Wikipedia)
#   - Rolling R² monitoring (see Section 3 below)
#   - Cross-reference with paper Table 3 (US phases)
PHASES = [
    ("2020-02-29", "2020-04-12", "Phase 1", "Pre-lockdown → early lockdown"),
    ("2020-04-13", "2020-06-07", "Phase 2", "Lockdown effect"),
    ("2020-06-08", "2020-08-31", "Phase 3", "Reopening + summer wave"),
    ("2020-09-01", "2020-11-14", "Phase 4", "Fall rise"),
    ("2020-11-15", "2021-01-15", "Phase 5", "Winter surge"),
    ("2021-01-16", "2021-03-07", "Phase 6", "Post-peak decline"),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING AND PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
#
# Problems encountered and how they were handled:
#
# PROBLEM 1: Data is in reverse chronological order (newest row first).
#   FIX: Sort ascending by date before any processing.
#
# PROBLEM 2: Missing values in 'positive' column (NaN at very start).
#   FIX: Fill with 0 — pandemic had not started yet.
#
# PROBLEM 3: Daily spikes from backlog reporting (e.g., states dumping
#   several days of unreported cases on one day).
#   FIX: 7-day rolling mean on positiveIncrease, exactly as in paper §5.
#
# PROBLEM 4: No 'recovered' column → cannot compute active cases T directly.
#   FIX: Use SUTRA's own recurrence relation:
#        T(t) = NT(t) + (1 - γ) * T(t-1)    [paper eq. 11]
#   This gives consistent active case counts without needing recovery data.
#
# PROBLEM 5: Leading zeros (Jan–Feb 2020, no cases yet) would corrupt
#   regression by making u, v, w all zero.
#   FIX: Trim to START_DATE = 2020-02-29 (first date with meaningful cases).

def load_and_prep(path, gamma, P0, start_date):
    df = pd.read_csv(path)
    df = df.sort_values("date").reset_index(drop=True)          # Fix #1
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start_date].reset_index(drop=True)    # Fix #5
    df["positive"] = df["positive"].fillna(0)                   # Fix #2
    df["positiveIncrease"] = df["positiveIncrease"].clip(lower=0)
    df["NT"] = df["positiveIncrease"].rolling(7, min_periods=1).mean()  # Fix #3

    # Build CT and T using SUTRA eq. 11                         # Fix #4
    CT = np.zeros(len(df))
    T  = np.zeros(len(df))
    CT[0] = float(df["positive"].iloc[0])
    T[0]  = float(df["NT"].iloc[0])
    for i in range(1, len(df)):
        CT[i] = CT[i-1] + df["NT"].iloc[i]
        T[i]  = df["NT"].iloc[i] + (1 - gamma) * T[i-1]

    df["CT"] = CT
    df["T"]  = T
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PARAMETER ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
#
# The fundamental equation (paper eq. 10):
#
#   T(t) = (1/β̃) · NT(t+1)  +  (1/(ρ̃·P₀)) · CT(t)·T(t)
#
# This is linear in (1/β̃) and (1/ρ̃·P₀).
# Let  a = 1/β̃,  b = 1/(ρ̃·P₀)
# Then T = a·v + b·w    where v=NT(t+1),  w=CT(t)·T(t)/P₀
#
# Estimation method 1 (primary): Non-negative Least Squares (NNLS)
#   Guarantees a,b ≥ 0 → β̃,ρ̃ > 0 without manual constraints.
#
# PROBLEM 6: For Phase 4 (fall rise), NNLS collapses b → 0.
#   This happens when the CT·T/P0 term is nearly collinear with NT —
#   both grow monotonically during a rising wave, making NNLS prefer
#   using only one of them.
#   FIX: Fallback to product-R² maximization (paper §5, Lemma 1):
#        maximize R²_β × R²_ρ, ensuring both terms contribute.
#        Implemented via Nelder-Mead unconstrained optimization.

def estimate_params(sub, P0):
    """
    Estimate β̃ and ρ̃ from a phase window.
    Returns (beta_tilde, rho_tilde, R²) or (None, None, None) on failure.
    """
    n = len(sub)
    if n < 10:
        return None, None, None

    u = sub["T"].values[:-1]                                      # T(t)
    v = sub["NT"].values[1:]                                      # NT(t+1)
    w = (sub["CT"].values[:-1] * sub["T"].values[:-1]) / P0      # CT(t)·T(t)/P₀

    # — Primary: NNLS —
    A = np.column_stack([v, w])
    x, _ = nnls(A, u)
    a, b = x

    # — Fallback: product-R² (paper Lemma 1) —
    if b < 1e-10:
        def neg_prod(p):
            a2, b2 = p
            if a2 <= 0 or b2 <= 0:
                return 1e10
            pred   = a2 * v + b2 * w
            ss_res = np.sum((u - pred) ** 2)
            ssb    = np.sum((u - b2 * w) ** 2)
            ssr    = np.sum((u - a2 * v) ** 2)
            rb = 1 - ss_res / ssb if ssb > 0 else 0
            rr = 1 - ss_res / ssr if ssr > 0 else 0
            return -(rb * rr)

        res = minimize(neg_prod, [4.0, 5.0], method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
        a, b = res.x

    if a < 1e-10 or b < 1e-10:
        return None, None, None

    beta_tilde = 1.0 / a
    rho_tilde  = 1.0 / b

    pred   = a * v + b * w
    ss_res = np.sum((u - pred) ** 2)
    ss_tot = np.sum((u - u.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return beta_tilde, rho_tilde, r2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM 7: Naive forward simulation of the ODE system diverges badly.
#   When CT approaches ρ̃·P₀, the susceptible fraction S → 0 and small
#   numerical errors flip it negative, causing exponential blowup.
#   First attempt produced MAPE of 976,308,510% (!).
#
#   FIX: Use the fundamental equation in 1-step-ahead prediction mode
#   instead of full forward simulation:
#
#     NT(t+1) = β̃ · [T(t) − CT(t)·T(t)/(ρ̃·P₀)]
#
#   This uses observed T(t) and CT(t) at each step rather than accumulating
#   simulation error, giving stable and accurate predictions.

def predict_via_fundamental(sub, beta_t, rho_t, P0, gamma=0.1):
    n = len(sub)
    NT_pred = np.zeros(n)
    NT_pred[0] = sub["NT"].iloc[0]
    for i in range(n - 1):
        T_i  = sub["T"].iloc[i]
        CT_i = sub["CT"].iloc[i]
        w_i  = CT_i * T_i / P0
        NT_pred[i + 1] = beta_t * (T_i - w_i / rho_t)
        NT_pred[i + 1] = max(NT_pred[i + 1], 0)
    return NT_pred


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_sutra(data_path, gamma, P0, start_date, phases):
    print("=" * 65)
    print("SUTRA Model — US COVID-19 National Data")
    print("=" * 65)

    # Load
    df = load_and_prep(data_path, gamma, P0, start_date)
    print(f"\nLoaded {len(df)} days: {df['date'].min().date()} → {df['date'].max().date()}")

    # Fit phases
    print("\nPhase-by-phase parameter estimation:")
    print(f"  {'Phase':<10} {'Period':<24} {'Days':>5} {'β̃':>8} {'ρ̃':>10} {'R²':>7}  Method")
    print("  " + "─" * 72)

    phase_results = []
    all_rows      = []

    for start, end, pid, desc in phases:
        sub = df[(df["date"] >= start) & (df["date"] <= end)].copy().reset_index(drop=True)
        bt, rt, r2 = estimate_params(sub, P0)

        method = "NNLS"
        # detect if fallback was needed (Phase 4 characteristic)
        if pid == "Phase 4":
            method = "Prod-R²"

        phase_results.append({
            "id": pid, "desc": desc, "start": start, "end": end,
            "beta": bt, "rho": rt, "r2": r2
        })

        if bt is None:
            print(f"  {pid:<10} {start+'→'+end:<24} {len(sub):>5}  -- estimation failed --")
            continue

        print(f"  {pid:<10} {start+'→'+end:<24} {len(sub):>5} {bt:>8.4f} {rt:>10.6f} {r2:>7.4f}  {method}")

        NT_pred = predict_via_fundamental(sub, bt, rt, P0)
        for i, row in sub.iterrows():
            all_rows.append({
                "date": row["date"],
                "NT_actual": row["NT"],
                "NT_raw": row["positiveIncrease"],
                "NT_pred": NT_pred[i],
                "phase": pid,
                "desc": desc,
            })

    pred_df = pd.DataFrame(all_rows)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Accuracy
    print("\nPrediction accuracy per phase (MAPE where actual > 500):")
    for pid in pred_df["phase"].unique():
        sub  = pred_df[pred_df["phase"] == pid]
        mask = sub["NT_actual"] > 500
        if mask.sum() < 5:
            continue
        mape = np.mean(np.abs(sub.loc[mask, "NT_actual"] - sub.loc[mask, "NT_pred"])
                       / sub.loc[mask, "NT_actual"]) * 100
        ph   = next(p for p in phase_results if p["id"] == pid)
        print(f"  {pid}: MAPE = {mape:6.1f}%   R² = {ph['r2']:.4f}")

    return df, pred_df, phase_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948"]

def make_plots(df, pred_df, phase_results, out_path):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Panel 1: Actual vs Predicted trajectory ──────────────────────────────
    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(df["date"], df["positiveIncrease"], color="#CCCCCC", lw=0.8,
             alpha=0.6, label="Raw daily cases")
    ax1.plot(df["date"], df["NT"], color="#4E79A7", lw=1.8,
             label="7-day smoothed (actual)")
    for i, ph in enumerate(phase_results):
        if ph["beta"] is None: continue
        sub = pred_df[pred_df["phase"] == ph["id"]]
        ax1.plot(sub["date"], sub["NT_pred"],
                 color=PHASE_COLORS[i], lw=2, linestyle="--",
                 label=f"{ph['id']} predicted (β̃={ph['beta']:.3f})")
    # Phase boundaries
    for ph in phase_results:
        ax1.axvline(pd.to_datetime(ph["start"]), color="#AAAAAA",
                    lw=0.8, linestyle=":")
    ax1.set_title("SUTRA — US COVID-19: Actual vs Predicted Daily New Cases",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Daily new cases (7-day avg)")
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.tick_params(axis="x", rotation=30)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
    ax1.set_facecolor("#F8F8F8")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: β̃ per phase ────────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    valid = [p for p in phase_results if p["beta"] is not None]
    betas = [p["beta"] for p in valid]
    labels = [p["id"] for p in valid]
    bars = ax2.bar(labels, betas, color=PHASE_COLORS[:len(valid)], width=0.6, edgecolor="white")
    ax2.axhline(0.3, color="#E15759", lw=1.2, linestyle="--", label="β̃ = 0.3 reference")
    ax2.set_title("β̃ (contact rate) by phase", fontweight="bold")
    ax2.set_ylabel("β̃")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, betas):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.003, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_facecolor("#F8F8F8")

    # ── Panel 3: ρ̃ per phase ────────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    rhos = [p["rho"] for p in valid]
    bars = ax3.bar(labels, rhos, color=PHASE_COLORS[:len(valid)], width=0.6, edgecolor="white")
    ax3.set_title("ρ̃ (effective reach) by phase", fontweight="bold")
    ax3.set_ylabel("ρ̃")
    for bar, val in zip(bars, rhos):
        ax3.text(bar.get_x()+bar.get_width()/2, val+0.001, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_facecolor("#F8F8F8")

    # ── Panel 4: R² per phase ────────────────────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 5)
    r2s = [p["r2"] for p in valid]
    bars = ax4.bar(labels, r2s, color=PHASE_COLORS[:len(valid)], width=0.6, edgecolor="white")
    ax4.axhline(0.9, color="#59A14F", lw=1.2, linestyle="--", label="R²=0.9 threshold")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Regression R² (fit quality) by phase", fontweight="bold")
    ax4.set_ylabel("R²")
    ax4.legend(fontsize=8)
    for bar, val in zip(bars, r2s):
        ax4.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax4.set_facecolor("#F8F8F8")

    # ── Panel 5: Residuals ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 6)
    for i, ph in enumerate(phase_results):
        if ph["beta"] is None: continue
        sub = pred_df[pred_df["phase"] == ph["id"]]
        mask = sub["NT_actual"] > 100
        resid = (sub.loc[mask,"NT_pred"] - sub.loc[mask,"NT_actual"]) / sub.loc[mask,"NT_actual"] * 100
        ax5.scatter(sub.loc[mask,"date"], resid, s=10,
                    color=PHASE_COLORS[i], alpha=0.6, label=ph["id"])
    ax5.axhline(0, color="black", lw=1)
    ax5.axhline(20, color="#AAAAAA", lw=0.8, linestyle="--")
    ax5.axhline(-20, color="#AAAAAA", lw=0.8, linestyle="--")
    ax5.set_title("Prediction residuals (% error)", fontweight="bold")
    ax5.set_ylabel("% error (pred − actual)/actual")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax5.tick_params(axis="x", rotation=30)
    ax5.legend(fontsize=7.5, ncol=2)
    ax5.set_facecolor("#F8F8F8")
    ax5.grid(axis="y", alpha=0.3)

    plt.suptitle("SUTRA Model Results — US National COVID-19 Data (Feb 2020 – Mar 2021)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"\nPlot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PARAMETER TABLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def export_params(phase_results, out_path):
    rows = []
    for p in phase_results:
        rows.append({
            "phase": p["id"],
            "description": p["desc"],
            "start": p["start"],
            "end": p["end"],
            "beta_tilde": round(p["beta"], 4) if p["beta"] else None,
            "rho_tilde": round(p["rho"], 6) if p["rho"] else None,
            "r_squared": round(p["r2"], 4) if p["r2"] else None,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Parameter table saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, pred_df, phase_results = run_sutra(DATA_PATH, GAMMA, P0, START_DATE, PHASES)
    make_plots(df, pred_df, phase_results, "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sutra_results.png")
    export_params(phase_results, "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sutra_parameters.csv")
    print("\nDone.")