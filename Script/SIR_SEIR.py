import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.lines import Line2D

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/piyushmaji/Desktop/Python/Computational_Project_2/Data/SEIR_data.csv")
df = df[df["time"] <= 250]       # only first 250 days

S_data = df["S"].values
E_data = df["E"].values
I_data = df["I"].values
R_data = df["R"].values
t      = df["time"].values

N = 7_900_000_000

# ─────────────────────────────────────────
# 2. SIR MODEL
# ─────────────────────────────────────────
beta  = 0.3
gamma = 0.1

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

sir_result          = odeint(sir_model, [S_data[0], I_data[0], R_data[0]], t, args=(N, beta, gamma))
S_sir, I_sir, R_sir = sir_result.T

# ─────────────────────────────────────────
# 3. SEIR MODEL
# ─────────────────────────────────────────
sigma = 1 / 5

def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]

seir_result                    = odeint(seir_model, [S_data[0], E_data[0], I_data[0], R_data[0]], t, args=(N, beta, sigma, gamma))
S_seir, E_seir, I_seir, R_seir = seir_result.T

# ═════════════════════════════════════════
# FIGURE 1 — SIR: all together (twin axis)
# ═════════════════════════════════════════
plt.figure(figsize=(10, 5))
plt.title("SIR Model — S, I, R Together (Dual Axis)", fontsize=14, fontweight="bold")

plt.plot(t, S_sir / 1e9, color="steelblue", linewidth=2)
plt.xlabel("Days")
plt.ylabel("S — Billions", color="steelblue")
plt.tick_params(axis="y", labelcolor="steelblue")

ax2 = plt.gca().twinx()
ax2.plot(t, I_sir / 1e6, color="tomato",         linewidth=2)
ax2.plot(t, R_sir / 1e6, color="mediumseagreen",  linewidth=2)
ax2.set_ylabel("I & R — Millions", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

legend_lines = [
    Line2D([0], [0], color="steelblue",      linewidth=2, label="S — Susceptible (Billions, left axis)"),
    Line2D([0], [0], color="tomato",         linewidth=2, label="I — Infectious  (Millions, right axis)"),
    Line2D([0], [0], color="mediumseagreen", linewidth=2, label="R — Recovered   (Millions, right axis)"),
]
plt.legend(handles=legend_lines, loc="upper left")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/Python/Computational_Project_2/plot/sir_seir/SIR_together_dualaxis.png", dpi=150)
plt.show()

# ═════════════════════════════════════════
# FIGURE 2 — SEIR: all together (twin axis)
# ═════════════════════════════════════════
plt.figure(figsize=(10, 5))
plt.title("SEIR Model — S, E, I, R Together (Dual Axis)", fontsize=14, fontweight="bold")

plt.plot(t, S_seir / 1e9, color="steelblue", linewidth=2)
plt.xlabel("Days")
plt.ylabel("S — Billions", color="steelblue")
plt.tick_params(axis="y", labelcolor="steelblue")

ax2 = plt.gca().twinx()
ax2.plot(t, E_seir / 1e6, color="goldenrod",      linewidth=2)
ax2.plot(t, I_seir / 1e6, color="tomato",         linewidth=2)
ax2.plot(t, R_seir / 1e6, color="mediumseagreen",  linewidth=2)
ax2.set_ylabel("E, I & R — Millions", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

legend_lines = [
    Line2D([0], [0], color="steelblue",      linewidth=2, label="S — Susceptible (Billions, left axis)"),
    Line2D([0], [0], color="goldenrod",      linewidth=2, label="E — Exposed     (Millions, right axis)"),
    Line2D([0], [0], color="tomato",         linewidth=2, label="I — Infectious  (Millions, right axis)"),
    Line2D([0], [0], color="mediumseagreen", linewidth=2, label="R — Recovered   (Millions, right axis)"),
]
plt.legend(handles=legend_lines, loc="upper left")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/Python/Computational_Project_2/plot/sir_seir/SEIR_together_dualaxis.png", dpi=150)
plt.show()

# ═════════════════════════════════════════
# FIGURE 3 — SIR: each compartment separately
# ═════════════════════════════════════════
plt.figure(figsize=(15, 4))
plt.suptitle("SIR — Each Compartment Separately", fontsize=14, fontweight="bold")

plt.subplot(1, 3, 1)
plt.title("S — Susceptible")
plt.plot(t, S_sir  / 1e9, color="steelblue", linewidth=2,   label="Model")
plt.plot(t, S_data / 1e9, color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Billions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.subplot(1, 3, 2)
plt.title("I — Infectious")
plt.plot(t, I_sir  / 1e6, color="tomato", linewidth=2,   label="Model")
plt.plot(t, I_data / 1e6, color="tomato", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Millions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.subplot(1, 3, 3)
plt.title("R — Recovered")
plt.plot(t, R_sir  / 1e6, color="mediumseagreen", linewidth=2,   label="Model")
plt.plot(t, R_data / 1e6, color="mediumseagreen", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Millions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/Python/Computational_Project_2/plot/sir_seir/SIR_separate.png", dpi=150)
plt.show()

# ═════════════════════════════════════════
# FIGURE 4 — SEIR: each compartment separately
# ═════════════════════════════════════════
plt.figure(figsize=(18, 4))
plt.suptitle("SEIR — Each Compartment Separately", fontsize=14, fontweight="bold")

plt.subplot(1, 4, 1)
plt.title("S — Susceptible")
plt.plot(t, S_seir / 1e9, color="steelblue", linewidth=2,   label="Model")
plt.plot(t, S_data / 1e9, color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Billions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.subplot(1, 4, 2)
plt.title("E — Exposed")
plt.plot(t, E_seir / 1e6, color="goldenrod", linewidth=2,   label="Model")
plt.plot(t, E_data / 1e6, color="goldenrod", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Millions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.subplot(1, 4, 3)
plt.title("I — Infectious")
plt.plot(t, I_seir / 1e6, color="tomato", linewidth=2,   label="Model")
plt.plot(t, I_data / 1e6, color="tomato", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Millions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.subplot(1, 4, 4)
plt.title("R — Recovered")
plt.plot(t, R_seir / 1e6, color="mediumseagreen", linewidth=2,   label="Model")
plt.plot(t, R_data / 1e6, color="mediumseagreen", linewidth=1.5, linestyle="--", alpha=0.7, label="Actual")
plt.xlabel("Days")
plt.ylabel("Millions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/Python/Computational_Project_2/plot/sir_seir/SEIR_separate.png", dpi=150)
plt.show()

print("All 4 plots saved!")