"""
COVID-19 Epidemic Modelling Pipeline
=====================================
Input  : SIR_formatted.csv  (columns: time, s, i, r)
Output : 4 plots + SEIR_data_from_SIR.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────
BASE  = "/Users/piyushmaji/Desktop/Python/Computational_Project_2"
DATA_IN   = f"{BASE}/Data/SIR_formatted.csv"
DATA_OUT  = f"{BASE}/Data/SEIR_data_from_SIR.csv"
PLOT_DIR  = f"{BASE}/plot/Combined"

# ── Constants ─────────────────────────────────────────────────
N                = 7_900_000_000
MAX_DAYS         = 365
INCUBATION_DAYS  = 5.0
SIGMA            = 1.0 / INCUBATION_DAYS


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["time"] <= MAX_DAYS].sort_values("time").reset_index(drop=True)
    print(f"  Total population N = {N:,}")
    print(f"  Rows used (≤{MAX_DAYS} d) = {len(df)}")
    return (
        df["time"].values.astype(float),
        df["s"].values.astype(float),
        df["i"].values.astype(float),
        df["r"].values.astype(float),
    )


# ─────────────────────────────────────────────────────────────
# STEP 2 — PARAMETER ESTIMATION via curve_fit
# ─────────────────────────────────────────────────────────────
def _sir_odeint(t_arr: np.ndarray, beta: float, gamma: float,
                N: float, S0: float, I0: float, R0_init: float) -> np.ndarray:
    def model(y, t):
        S, I, R = y
        dS = -beta * S * I / N
        dI =  beta * S * I / N - gamma * I
        dR =  gamma * I
        return [dS, dI, dR]
    sol = odeint(model, [S0, I0, R0_init], t_arr)
    return sol[:, 1]


def estimate_parameters(t: np.ndarray, S_data: np.ndarray,
                        I_data: np.ndarray, R_data: np.ndarray) -> tuple[float, float]:
    I_smooth = pd.Series(I_data).rolling(window=7, center=True, min_periods=1).median().values

    def _fit_wrapper(t_arr, beta, gamma):
        return _sir_odeint(t_arr, beta, gamma, N, S_data[0], I_smooth[0], R_data[0])

    try:
        popt, _ = curve_fit(_fit_wrapper, t, I_smooth,
                            p0=[0.25, 0.05], bounds=([0.01, 0.001], [1.0, 0.5]),
                            maxfev=5000)
        beta, gamma = popt
        print("  curve_fit converged")
    except RuntimeError:
        print("  curve_fit did not converge — using literature values")
        beta, gamma = 0.25, 0.05

    R0 = beta / gamma
    print(f"\n  Estimated Parameters:")
    print(f"    beta  (transmission rate) : {beta:.6f}")
    print(f"    gamma (recovery rate)     : {gamma:.6f}")
    print(f"    sigma (incubation rate)   : {SIGMA:.6f}  (1/{INCUBATION_DAYS:.0f} days, literature)")
    print(f"    R0    (basic repro. no.)  : {R0:.4f}\n")
    return beta, gamma


# ─────────────────────────────────────────────────────────────
# STEP 3 — BACK-CALCULATE E
# ─────────────────────────────────────────────────────────────
def back_calculate_E(I_data: np.ndarray, R_data: np.ndarray,
                     gamma: float) -> tuple[np.ndarray, np.ndarray]:
    I_smooth  = pd.Series(I_data).rolling(window=7, center=True, min_periods=1).median()
    dI_smooth = I_smooth.diff().fillna(0).values

    E_data = np.clip((dI_smooth + gamma * I_data) / SIGMA, 0, None)
    S_adj  = np.clip(N - E_data - I_data - R_data, 0, None)

    print(f"  Peak E = {E_data.max():,.0f}")
    return E_data, S_adj


# ─────────────────────────────────────────────────────────────
# STEP 4 — SIR & SEIR SIMULATIONS
# ─────────────────────────────────────────────────────────────
def run_sir(t, S0, I0, R0_init, beta, gamma):
    def model(y, t):
        S, I, R = y
        dS = -beta * S * I / N
        dI =  beta * S * I / N - gamma * I
        dR =  gamma * I
        return [dS, dI, dR]
    return odeint(model, [S0, I0, R0_init], t).T


def run_seir(t, S0, E0, I0, R0_init, beta, gamma):
    def model(y, t):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE =  beta * S * I / N - SIGMA * E
        dI =  SIGMA * E - gamma * I
        dR =  gamma * I
        return [dS, dE, dI, dR]
    return odeint(model, [S0, E0, I0, R0_init], t).T


# ─────────────────────────────────────────────────────────────
# HELPERS — scaling
# ─────────────────────────────────────────────────────────────
def smart_scale(arr: np.ndarray) -> tuple[np.ndarray, str]:
    mx = np.max(np.abs(arr))
    if mx >= 1e9: return arr / 1e9, "Billions"
    if mx >= 1e6: return arr / 1e6, "Millions"
    if mx >= 1e3: return arr / 1e3, "Thousands"
    return arr, "Count"


_SCALE_DIV = {"Billions": 1e9, "Millions": 1e6, "Thousands": 1e3, "Count": 1.0}


def scale_div(label: str) -> float:
    return _SCALE_DIV[label]


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def _save(fig, name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_sir_dual(t, S_sir, I_sir, R_sir) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle("SIR Model — S, I, R Together (Dual Axis)", fontsize=14, fontweight="bold")

    s_arr, s_lbl = smart_scale(S_sir)
    ax1.plot(t, s_arr, color="steelblue", linewidth=2)
    ax1.set_xlabel("Days")
    ax1.set_ylabel(f"S — {s_lbl}", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    _, ir_lbl = smart_scale(np.concatenate([I_sir, R_sir]))
    div = scale_div(ir_lbl)
    ax2 = ax1.twinx()
    ax2.plot(t, I_sir / div, color="tomato",        linewidth=2)
    ax2.plot(t, R_sir / div, color="mediumseagreen", linewidth=2)
    ax2.set_ylabel(f"I & R — {ir_lbl}", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    ax1.legend(handles=[
        Line2D([0],[0], color="steelblue",      linewidth=2, label=f"S — Susceptible ({s_lbl}, left)"),
        Line2D([0],[0], color="tomato",         linewidth=2, label=f"I — Infectious  ({ir_lbl}, right)"),
        Line2D([0],[0], color="mediumseagreen", linewidth=2, label=f"R — Recovered   ({ir_lbl}, right)"),
    ], loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "SIR_together_dualaxis.png")


def plot_seir_dual(t, S_seir, E_seir, I_seir, R_seir) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle("SEIR Model — S, E, I, R Together (Dual Axis)", fontsize=14, fontweight="bold")

    s_arr, s_lbl = smart_scale(S_seir)
    ax1.plot(t, s_arr, color="steelblue", linewidth=2)
    ax1.set_xlabel("Days")
    ax1.set_ylabel(f"S — {s_lbl}", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    _, eir_lbl = smart_scale(np.concatenate([E_seir, I_seir, R_seir]))
    div = scale_div(eir_lbl)
    ax2 = ax1.twinx()
    ax2.plot(t, E_seir / div, color="goldenrod",     linewidth=2)
    ax2.plot(t, I_seir / div, color="tomato",        linewidth=2)
    ax2.plot(t, R_seir / div, color="mediumseagreen", linewidth=2)
    ax2.set_ylabel(f"E, I & R — {eir_lbl}", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    ax1.legend(handles=[
        Line2D([0],[0], color="steelblue",      linewidth=2, label=f"S — Susceptible ({s_lbl}, left)"),
        Line2D([0],[0], color="goldenrod",      linewidth=2, label=f"E — Exposed     ({eir_lbl}, right)"),
        Line2D([0],[0], color="tomato",         linewidth=2, label=f"I — Infectious  ({eir_lbl}, right)"),
        Line2D([0],[0], color="mediumseagreen", linewidth=2, label=f"R — Recovered   ({eir_lbl}, right)"),
    ], loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "SEIR_together_dualaxis.png")


def plot_separate(t, compartments: list[tuple], title: str, filename: str) -> None:
    fig, axes = plt.subplots(1, len(compartments), figsize=(5.5 * len(compartments), 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (model_arr, actual_arr, color, label) in zip(axes, compartments):
        _, lbl = smart_scale(np.array([max(model_arr.max(), actual_arr.max())]))
        div = scale_div(lbl)
        ax.plot(t, model_arr  / div, color=color, linewidth=2,   label="Model")
        ax.plot(t, actual_arr / div, color=color, linewidth=1.5, linestyle="--", alpha=0.65, label="Actual")
        ax.set_title(label)
        ax.set_xlabel("Days")
        ax.set_ylabel(lbl)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Load & Prepare Data")
    print("=" * 60)
    t, S_data, I_data, R_data = load_data(DATA_IN)

    print("=" * 60)
    print("STEP 2: Parameter Estimation via curve_fit")
    print("=" * 60)
    beta, gamma = estimate_parameters(t, S_data, I_data, R_data)

    print("=" * 60)
    print("STEP 3: Back-calculate E")
    print("=" * 60)
    E_data, S_adj = back_calculate_E(I_data, R_data, gamma)

    print("=" * 60)
    print("STEP 4: Run SIR & SEIR Simulations")
    print("=" * 60)
    S_sir, I_sir, R_sir             = run_sir(t, S_data[0], I_data[0], R_data[0], beta, gamma)
    S_seir, E_seir, I_seir, R_seir  = run_seir(t, S_adj[0], E_data[0], I_data[0], R_data[0], beta, gamma)
    print(f"  SIR  peak I : {I_sir.max()/1e6:.2f}M  on day {t[I_sir.argmax()]:.0f}")
    print(f"  SEIR peak I : {I_seir.max()/1e6:.2f}M  on day {t[I_seir.argmax()]:.0f}\n")

    print("=" * 60)
    print("STEP 5: Save Plots & Data")
    print("=" * 60)

    plot_sir_dual(t, S_sir, I_sir, R_sir)
    plot_seir_dual(t, S_seir, E_seir, I_seir, R_seir)

    plot_separate(t, [
        (S_sir, S_data, "steelblue",     "S — Susceptible"),
        (I_sir, I_data, "tomato",         "I — Infectious"),
        (R_sir, R_data, "mediumseagreen", "R — Recovered"),
    ], "SIR — Each Compartment (Model vs Actual)", "SIR_separate.png")

    plot_separate(t, [
        (S_seir, S_adj,  "steelblue",     "S — Susceptible"),
        (E_seir, E_data, "goldenrod",     "E — Exposed"),
        (I_seir, I_data, "tomato",        "I — Infectious"),
        (R_seir, R_data, "mediumseagreen","R — Recovered"),
    ], "SEIR — Each Compartment (Model vs Actual)", "SEIR_separate.png")

    pd.DataFrame({
        "time": t.astype(int),
        "S":    S_adj.round().astype(int),
        "E":    E_data.round().astype(int),
        "I":    I_data.round().astype(int),
        "R":    R_data.round().astype(int),
    }).to_csv(DATA_OUT, index=False)
    print(f"  Saved: SEIR_data_from_SIR.csv")

    print()
    print("=" * 60)
    print(f"PIPELINE COMPLETE — outputs in {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
