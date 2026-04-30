#!/usr/bin/env python3
"""
SUTRA Model — Automatic Phase Detection
========================================
Based on: Agrawal et al. (2022) arXiv:2101.09158v6, Section 6

Key change from previous version:
    Phases are NO LONGER hardcoded. They are discovered automatically
    from the data using an expanding-window R² monitoring algorithm,
    exactly as described in Section 6 of the paper.

Algorithm (Section 6):
    1. Start at day 0 with a small initial window (MIN_PHASE_DAYS).
    2. Fit β̃, ρ̃ on that window, compute R².
    3. Expand window by 1 day, refit, check R².
    4. If R² stays above R2_STABLE_THRESHOLD → same phase, keep expanding.
    5. If R² drops below threshold for R2_DROP_PATIENCE consecutive days
       → phase boundary detected. Record phase, restart from this point.
    6. Repeat until end of data.
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
GAMMA      = 0.1
P0         = 331_000_000
START_DATE = "2020-02-29"

# ── Phase detection hyperparameters ──────────────────────────────────────────
#
# MIN_PHASE_DAYS: minimum days before we even attempt to fit a phase.
#   Too small → noisy R², spurious phase breaks. Paper uses ~10–15.
MIN_PHASE_DAYS = 15

# R2_STABLE_THRESHOLD: R² below this = fit is degrading = phase may be ending.
#   Paper does not give an exact number; 0.75 is a reasonable choice that
#   allows some noise without being too strict.
R2_STABLE_THRESHOLD = 0.75

# R2_DROP_PATIENCE: how many consecutive days of R² < threshold before we
#   declare a phase break. Prevents reacting to a single noisy day.
#   Paper calls this the "drift period" detection — we wait for confirmation.
R2_DROP_PATIENCE = 5

# MIN_FINAL_PHASE_DAYS: if the remaining data after the last detected boundary
#   is too short to be meaningful, absorb it into the previous phase.
MIN_FINAL_PHASE_DAYS = 10


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING AND PREPROCESSING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep(path, gamma, P0, start_date):
    df = pd.read_csv(path)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start_date].reset_index(drop=True)
    df["positive"] = df["positive"].fillna(0)
    df["positiveIncrease"] = df["positiveIncrease"].clip(lower=0)
    df["NT"] = df["positiveIncrease"].rolling(7, min_periods=1).mean()

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
# SECTION 3 — PARAMETER ESTIMATION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_params(sub, P0):
    """
    Fit β̃ and ρ̃ on a window using NNLS, with product-R² fallback.
    Returns (beta_tilde, rho_tilde, R²) or (None, None, 0) on failure.
    """
    n = len(sub)
    if n < 10:
        return None, None, 0.0

    u = sub["T"].values[:-1]
    v = sub["NT"].values[1:]
    w = (sub["CT"].values[:-1] * sub["T"].values[:-1]) / P0

    # Guard: degenerate windows (all-zero) return 0 R²
    if u.std() < 1e-6:
        return None, None, 0.0

    A = np.column_stack([v, w])
    x, _ = nnls(A, u)
    a, b = x

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
        return None, None, 0.0

    beta_tilde = 1.0 / a
    rho_tilde  = 1.0 / b

    pred   = a * v + b * w
    ss_res = np.sum((u - pred) ** 2)
    ss_tot = np.sum((u - u.mean()) ** 2)
    r2     = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return beta_tilde, rho_tilde, r2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AUTOMATIC PHASE DETECTION  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces the hardcoded PHASES list entirely.
#
# How it works:
#   We maintain a sliding start index (phase_start) and expand an end index
#   (t) day by day. At each step we fit the model on df[phase_start : t+1]
#   and record the R². When R² stays below R2_STABLE_THRESHOLD for
#   R2_DROP_PATIENCE consecutive days, we know the current parameters no
#   longer describe the data — a new phase has started roughly at the point
#   where R² first dropped.
#
# Why patience matters:
#   A single noisy day can dip R² below threshold. We want to confirm the
#   fit is genuinely broken, not just jittery. This mirrors the paper's
#   "drift period" concept — the boundary is included in the new phase.
#
# Edge cases handled:
#   - Window too short to fit: skip, keep expanding.
#   - Final segment shorter than MIN_FINAL_PHASE_DAYS: merge into previous.
#   - Entire dataset fits one phase: return single phase.

def detect_phases(df, P0,
                  min_phase_days=MIN_PHASE_DAYS,
                  r2_threshold=R2_STABLE_THRESHOLD,
                  patience=R2_DROP_PATIENCE,
                  min_final=MIN_FINAL_PHASE_DAYS):
    """
    Returns a list of dicts, one per detected phase:
        {start_idx, end_idx, start_date, end_date, beta, rho, r2}
    """
    N = len(df)
    phases = []
    phase_start = 0
    below_count = 0          # consecutive days R² has been below threshold
    boundary_candidate = -1  # index where R² first dropped (candidate break)

    print("\nAutomatic phase detection (expanding window):")
    print(f"  R² threshold = {r2_threshold}  |  patience = {patience} days  |  min window = {min_phase_days} days")
    print()

    ROLLING_WINDOW = 20   # days used for rolling R² check (recent fit quality)

    t = min_phase_days  # start expanding from minimum window size

    while t < N:
        # Fit full window (for final parameter estimation when phase ends)
        # but check R² only on the most recent ROLLING_WINDOW days so that
        # a long-running good phase doesn't mask a newly degrading fit.
        roll_start = max(phase_start, t - ROLLING_WINDOW + 1)
        sub_roll   = df.iloc[roll_start : t + 1]
        bt, rt, r2 = estimate_params(sub_roll, P0)

        window_len = t - phase_start + 1

        if bt is None:
            # window not yet fittable — keep expanding silently
            t += 1
            continue

        # ── check if R² is degrading ─────────────────────────────────────────
        if r2 < r2_threshold:
            if below_count == 0:
                boundary_candidate = t   # mark where degradation began
            below_count += 1
        else:
            below_count = 0             # reset — fit is still good
            boundary_candidate = -1

        # ── patience exhausted: declare phase break ───────────────────────────
        if below_count >= patience:
            # Phase ends just before the degradation started
            phase_end = boundary_candidate - 1

            # Only record if this phase is long enough
            if phase_end - phase_start + 1 >= min_phase_days:
                sub_phase = df.iloc[phase_start : phase_end + 1]
                bt_f, rt_f, r2_f = estimate_params(sub_phase, P0)
                phase_num = len(phases) + 1
                phases.append({
                    "id":         f"Phase {phase_num}",
                    "start_idx":  phase_start,
                    "end_idx":    phase_end,
                    "start_date": df["date"].iloc[phase_start].strftime("%Y-%m-%d"),
                    "end_date":   df["date"].iloc[phase_end].strftime("%Y-%m-%d"),
                    "beta":       bt_f,
                    "rho":        rt_f,
                    "r2":         r2_f,
                    "desc":       "auto-detected",
                })
                bt_str = f"{bt_f:.4f}" if bt_f is not None else "N/A"
                rt_str = f"{rt_f:.6f}" if rt_f is not None else "N/A"
                r2_str = f"{r2_f:.4f}" if r2_f is not None else "N/A"
                print(f"  Phase {phase_num} detected: "
                      f"{df['date'].iloc[phase_start].date()} → "
                      f"{df['date'].iloc[phase_end].date()} "
                      f"({phase_end - phase_start + 1} days) | "
                      f"β̃={bt_str}  ρ̃={rt_str}  R²={r2_str}")

            # Restart from the candidate boundary (paper: include drift in new phase)
            phase_start = boundary_candidate
            t = phase_start + min_phase_days
            below_count = 0
            boundary_candidate = -1
            continue

        t += 1

    # ── handle the final phase ────────────────────────────────────────────────
    final_len = N - phase_start
    if final_len >= min_final:
        sub_phase = df.iloc[phase_start:]
        bt_f, rt_f, r2_f = estimate_params(sub_phase, P0)
        phase_num = len(phases) + 1
        phases.append({
            "id":         f"Phase {phase_num}",
            "start_idx":  phase_start,
            "end_idx":    N - 1,
            "start_date": df["date"].iloc[phase_start].strftime("%Y-%m-%d"),
            "end_date":   df["date"].iloc[N - 1].strftime("%Y-%m-%d"),
            "beta":       bt_f,
            "rho":        rt_f,
            "r2":         r2_f,
            "desc":       "auto-detected (final)",
        })
        bt_str = f"{bt_f:.4f}" if bt_f is not None else "N/A"
        rt_str = f"{rt_f:.6f}" if rt_f is not None else "N/A"
        r2_str = f"{r2_f:.4f}" if r2_f is not None else "N/A"
        print(f"  Phase {phase_num} detected: "
              f"{df['date'].iloc[phase_start].date()} → "
              f"{df['date'].iloc[N-1].date()} "
              f"({final_len} days) | "
              f"β̃={bt_str}  ρ̃={rt_str}  R²={r2_str}")
    elif phases:
        # too short — absorb into previous phase
        prev = phases[-1]
        prev["end_idx"]  = N - 1
        prev["end_date"] = df["date"].iloc[N - 1].strftime("%Y-%m-%d")
        sub_phase = df.iloc[prev["start_idx"] : N]
        bt_f, rt_f, r2_f = estimate_params(sub_phase, P0)
        prev["beta"] = bt_f
        prev["rho"]  = rt_f
        prev["r2"]   = r2_f
        print(f"  Final segment too short ({final_len} days) — merged into Phase {len(phases)}")

    print(f"\n  Total phases detected: {len(phases)}")
    return phases


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PREDICTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
# SECTION 6 — MAIN PIPELINE  (updated to use auto-detected phases)
# ══════════════════════════════════════════════════════════════════════════════

def run_sutra(data_path, gamma, P0, start_date):
    print("=" * 65)
    print("SUTRA Model — US COVID-19 (Automatic Phase Detection)")
    print("=" * 65)

    df = load_and_prep(data_path, gamma, P0, start_date)
    print(f"\nLoaded {len(df)} days: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Detect phases automatically ───────────────────────────────────────────
    phases = detect_phases(df, P0)

    # ── Generate predictions for each phase ───────────────────────────────────
    all_rows = []
    print("\nPrediction accuracy per phase (MAPE where actual > 500):")

    for ph in phases:
        if ph["beta"] is None:
            continue
        sub = df.iloc[ph["start_idx"] : ph["end_idx"] + 1].copy().reset_index(drop=True)
        NT_pred = predict_via_fundamental(sub, ph["beta"], ph["rho"], P0)

        for i, row in sub.iterrows():
            all_rows.append({
                "date":      row["date"],
                "NT_actual": row["NT"],
                "NT_raw":    row["positiveIncrease"],
                "NT_pred":   NT_pred[i],
                "phase":     ph["id"],
            })

        # MAPE
        actual = sub["NT"].values
        pred   = NT_pred
        mask   = actual > 500
        if mask.sum() >= 5:
            mape = np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]) * 100
            print(f"  {ph['id']}: MAPE = {mape:6.1f}%   R² = {ph['r2']:.4f}   "
                  f"({ph['start_date']} → {ph['end_date']})")

    pred_df = pd.DataFrame(all_rows)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    return df, pred_df, phases


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PLOTTING  (updated for auto-detected phases)
# ══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2",
    "#59A14F","#EDC948","#B07AA1","#FF9DA7",
    "#9C755F","#BAB0AC"
]

def make_plots(df, pred_df, phases, out_path):
    n_phases = len(phases)
    colors   = (PHASE_COLORS * ((n_phases // len(PHASE_COLORS)) + 1))[:n_phases]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#E8E8E8")

    # ── Panel 1: Actual vs Model ──────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_facecolor("white")

    ax1.plot(df["date"], df["NT"],
             color="#3A5FA0", lw=1.8, label="Actual Data", zorder=3)
    ax1.plot(pred_df["date"], pred_df["NT_pred"],
             color="#E07B3A", lw=1.8, label="Model Computed Data", zorder=2)

    # Mark auto-detected phase boundaries as vertical lines
    y_top = max(df["NT"].max(), pred_df["NT_pred"].max()) * 0.97
    for i, ph in enumerate(phases):
        xd = pd.to_datetime(ph["start_date"])
        ax1.axvline(xd, color=colors[i], lw=1.2, linestyle="--", alpha=0.7)
        ax1.text(xd, y_top, f" {ph['id']}", fontsize=7, color=colors[i],
                 va="top", rotation=90)

    ax1.set_title("US: Detected New Infections (7 day average) — Auto-detected phases",
                  fontsize=13, fontweight="bold", color="#CC2222", pad=10)
    ax1.set_ylabel("Infections", color="#CC2222", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Date",       color="#CC2222", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    ax1.tick_params(axis="x", rotation=30, labelsize=9)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.5, color="#DDDDDD")
    ax1.grid(True, axis="x", linestyle="-", linewidth=0.5, color="#DDDDDD")
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
               ncol=2, fontsize=10, frameon=False)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5); spine.set_color("#AAAAAA")

    # ── Panel 2: β̃ per phase ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    valid  = [p for p in phases if p["beta"] is not None]
    labels = [p["id"] for p in valid]
    betas  = [p["beta"] for p in valid]
    vc     = colors[:len(valid)]
    bars   = ax2.bar(labels, betas, color=vc, width=0.6, edgecolor="white")
    ax2.axhline(0.3, color="#E15759", lw=1.2, linestyle="--", label="β̃=0.3 ref")
    ax2.set_title("β̃ (contact rate) by phase", fontweight="bold", fontsize=10)
    ax2.set_ylabel("β̃")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, betas):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set_facecolor("white"); ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", labelsize=7)

    # ── Panel 3: ρ̃ per phase ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    rhos = [p["rho"] for p in valid]
    bars = ax3.bar(labels, rhos, color=vc, width=0.6, edgecolor="white")
    ax3.set_title("ρ̃ (effective reach) by phase", fontweight="bold", fontsize=10)
    ax3.set_ylabel("ρ̃")
    for bar, val in zip(bars, rhos):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax3.set_facecolor("white"); ax3.grid(axis="y", alpha=0.3)
    ax3.tick_params(axis="x", labelsize=7)

    # ── Panel 4: R² per phase ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 5)
    r2s  = [p["r2"] for p in valid]
    bars = ax4.bar(labels, r2s, color=vc, width=0.6, edgecolor="white")
    ax4.axhline(R2_STABLE_THRESHOLD, color="#59A14F", lw=1.2, linestyle="--",
                label=f"R²={R2_STABLE_THRESHOLD} threshold")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("R² (fit quality) by phase", fontweight="bold", fontsize=10)
    ax4.set_ylabel("R²"); ax4.legend(fontsize=8)
    for bar, val in zip(bars, r2s):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax4.set_facecolor("white"); ax4.grid(axis="y", alpha=0.3)
    ax4.tick_params(axis="x", labelsize=7)

    # ── Panel 5: Residuals ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 6)
    for i, ph in enumerate(phases):
        if ph["beta"] is None: continue
        sub  = pred_df[pred_df["phase"] == ph["id"]]
        mask = sub["NT_actual"] > 100
        resid = ((sub.loc[mask, "NT_pred"] - sub.loc[mask, "NT_actual"])
                 / sub.loc[mask, "NT_actual"] * 100)
        ax5.scatter(sub.loc[mask, "date"], resid, s=8,
                    color=colors[i], alpha=0.6, label=ph["id"])
    ax5.axhline(0,   color="black",   lw=1)
    ax5.axhline( 20, color="#AAAAAA", lw=0.8, linestyle="--")
    ax5.axhline(-20, color="#AAAAAA", lw=0.8, linestyle="--")
    ax5.set_title("Prediction residuals (% error)", fontweight="bold", fontsize=10)
    ax5.set_ylabel("% error")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax5.tick_params(axis="x", rotation=30, labelsize=8)
    ax5.legend(fontsize=6, ncol=2)
    ax5.set_facecolor("white"); ax5.grid(axis="y", alpha=0.3)

    for ax in [ax2, ax3, ax4, ax5]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color("#AAAAAA")

    plt.suptitle(
        f"SUTRA — US COVID-19 | {len(phases)} phases auto-detected "
        f"(R²<{R2_STABLE_THRESHOLD} for {R2_DROP_PATIENCE} days → new phase)",
        fontsize=12, fontweight="bold", y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#E8E8E8")
    plt.close()
    print(f"\nPlot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PARAMETER TABLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def export_params(phases, out_path):
    rows = []
    for p in phases:
        rows.append({
            "phase":       p["id"],
            "start_date":  p["start_date"],
            "end_date":    p["end_date"],
            "days":        p["end_idx"] - p["start_idx"] + 1,
            "beta_tilde":  round(p["beta"], 4) if p["beta"] else None,
            "rho_tilde":   round(p["rho"],  6) if p["rho"]  else None,
            "r_squared":   round(p["r2"],   4) if p["r2"]   else None,
            "description": p["desc"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Parameter table saved → {out_path}")




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SENSITIVITY ANALYSIS: EFFECT OF INCUBATION PERIOD ON β̃ AND ρ̃
# ══════════════════════════════════════════════════════════════════════════════
#
# Sweeps γ = 1/incubation_period across 5 epidemiologically meaningful values:
#   γ=0.05 (20-day), 0.07 (14-day), 0.10 (10-day, default),
#   0.14 (7-day), 0.20 (5-day)
#
# For each γ the full pipeline is rerun (load_and_prep → detect_phases).
# β̃ and ρ̃ per phase are plotted side-by-side:
#   Top panel: β̃ (linear scale, with value labels)
#   Bottom panel: ρ̃ (log scale — spans 3 orders of magnitude across phases)
# ══════════════════════════════════════════════════════════════════════════════

def gamma_sensitivity(data_path, P0, start_date, out_path):
    import io as _io

    GAMMA_VALS = [
        (0.05,  "γ=0.05  (20-day period)"),
        (0.07,  "γ=0.07  (14-day period)"),
        (0.10,  "γ=0.10  (10-day)  ← default"),
        (0.14,  "γ=0.14  (7-day period)"),
        (0.20,  "γ=0.20  (5-day period)"),
    ]
    LINE_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#59A14F", "#B07AA1"]

    print("\n" + "="*65)
    print("SENSITIVITY ANALYSIS — Effect of incubation period on β̃ and ρ̃")
    print("="*65)

    # ── run pipeline for each gamma, suppress prints ──────────────────────────
    import sys as _sys
    results = {}
    for gamma, label in GAMMA_VALS:
        buf = _io.StringIO(); old = _sys.stdout; _sys.stdout = buf
        df_g   = load_and_prep(data_path, gamma, P0, start_date)
        phases = detect_phases(df_g, P0)
        _sys.stdout = old
        results[(gamma, label)] = {p["id"]: p for p in phases if p["beta"] is not None}
        n_valid = len(results[(gamma, label)])
        print(f"  {label}  →  {n_valid} valid phases fitted")

    # reference phase list comes from default gamma
    ref_phases = list(results[(0.10, "γ=0.10  (10-day)  ← default")].keys())

    x       = np.arange(len(ref_phases))
    n_g     = len(GAMMA_VALS)
    width   = 0.14
    offsets = np.linspace(-(n_g-1)/2, (n_g-1)/2, n_g) * width

    fig, (ax_b, ax_r) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#E8E8E8")
    for ax in [ax_b, ax_r]:
        ax.set_facecolor("white")

    for i, ((gamma, label), color) in enumerate(zip(GAMMA_VALS, LINE_COLORS)):
        pm    = results[(gamma, label)]
        betas = [pm[pid]["beta"] if pid in pm else np.nan for pid in ref_phases]
        rhos  = [pm[pid]["rho"]  if pid in pm else np.nan for pid in ref_phases]

        bars_b = ax_b.bar(x + offsets[i], betas, width=width, color=color,
                          label=label, edgecolor="white", linewidth=0.5)
        ax_r.bar(x + offsets[i], rhos, width=width, color=color,
                 label=label, edgecolor="white", linewidth=0.5)

        # value labels on β̃ bars
        for bar, val in zip(bars_b, betas):
            if not np.isnan(val):
                ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                          f"{val:.3f}", ha="center", va="bottom",
                          fontsize=5.5, color=color)

    # β̃ panel
    ax_b.set_xticks(x); ax_b.set_xticklabels(ref_phases, fontsize=9)
    ax_b.set_ylabel("β̃  (contact rate)", fontsize=11)
    ax_b.set_title("β̃  —  contact rate per phase at different incubation period assumptions",
                   fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=8, frameon=False, ncol=3)
    ax_b.grid(axis="y", linestyle="-", linewidth=0.4, color="#DDDDDD")
    ax_b.set_axisbelow(True)

    # ρ̃ panel (log scale)
    ax_r.set_yscale("log")
    ax_r.set_xticks(x); ax_r.set_xticklabels(ref_phases, fontsize=9)
    ax_r.set_ylabel("ρ̃  (effective reach, log scale)", fontsize=11)
    ax_r.set_xlabel("Phase", fontsize=10, color="#CC2222", fontweight="bold")
    ax_r.set_title("ρ̃  —  effective reach per phase  (log scale, spans ~3 orders of magnitude)",
                   fontsize=11, fontweight="bold")
    ax_r.legend(fontsize=8, frameon=False, ncol=3)
    ax_r.grid(axis="y", linestyle="-", linewidth=0.4, color="#DDDDDD")
    ax_r.set_axisbelow(True)

    for ax in [ax_b, ax_r]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color("#AAAAAA")
        ax.tick_params(labelsize=8)

    plt.suptitle(
        "SUTRA — US COVID-19 | Effect of incubation period (γ = 1/period) on β̃ and ρ̃",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#E8E8E8")
    plt.close()
    print(f"\nSensitivity plot saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, pred_df, phases = run_sutra(DATA_PATH, GAMMA, P0, START_DATE)
    make_plots(df, pred_df, phases,
               "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sutra_auto_results.png")
    export_params(phases,
                  "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sutra_auto_parameters.csv")

    gamma_sensitivity(
        DATA_PATH, P0, START_DATE,
        out_path="/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sutra_gamma_sensitivity.png",
    )

    print("\nDone.")