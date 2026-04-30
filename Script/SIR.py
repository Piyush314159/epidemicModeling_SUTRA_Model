import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# ── 1. Load and clean data ────────────────────────────────────────────────────
df = pd.read_csv("/Users/piyushmaji/Desktop/Python/Computational_Project_2/Data/national-history.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Drop rows where positive is NaN or 0 (pre-outbreak)
df = df.dropna(subset=["positive"])
df = df[df["positive"] > 0].reset_index(drop=True)

N = 331_000_000          # US population (2020 census approx.)
t_data = np.arange(len(df))

# Cumulative confirmed cases → proxy for I+R in SIR
# Recovered = cumulative cases - deaths (ignore active for simplicity)
cum_cases = df["positive"].values.astype(float)
deaths     = df["death"].fillna(0).values.astype(float)
recovered  = cum_cases - deaths      # crude R estimate

I_data = cum_cases - recovered       # currently infected (approx active)
R_data = recovered
S_data = N - cum_cases               # susceptible

# ── 2. SIR ODE ────────────────────────────────────────────────────────────────
def sir_ode(y, t, beta, gamma):
    S, I, R = y
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

# ── 3. Fit: match cumulative cases (I+R = N - S) ──────────────────────────────
def model_cum_cases(t, beta, gamma):
    y0 = [N - cum_cases[0], cum_cases[0], 0.0]
    sol = odeint(sir_ode, y0, t, args=(beta, gamma), mxstep=5000)
    return N - sol[:, 0]   # N - S = I + R = cumulative cases

p0     = [0.3, 0.05]
bounds = ([0.01, 0.001], [2.0, 0.5])

popt, pcov = curve_fit(model_cum_cases, t_data, cum_cases,
                       p0=p0, bounds=bounds, maxfev=10000)
beta_fit, gamma_fit = popt
R0 = beta_fit / gamma_fit

print(f"Fitted β  = {beta_fit:.4f}")
print(f"Fitted γ  = {gamma_fit:.4f}")
print(f"R₀        = {R0:.2f}")

# ── 4. Generate full SIR trajectory with fitted params ───────────────────────
y0  = [N - cum_cases[0], cum_cases[0], 0.0]
sol = odeint(sir_ode, y0, t_data, args=(beta_fit, gamma_fit), mxstep=5000)
S_pred = sol[:, 0]
I_pred = sol[:, 1]
R_pred = sol[:, 2]
cum_pred = N - S_pred

# ── 5. Plot ───────────────────────────────────────────────────────────────────
dates = df["date"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"SIR Model Fit — US COVID-19 National History\n"
             f"β = {beta_fit:.4f},  γ = {gamma_fit:.4f},  R₀ = {R0:.2f}",
             fontsize=13, fontweight="bold")

# --- subplot 1: cumulative cases ---
ax = axes[0, 0]
ax.plot(dates, cum_cases / 1e6,  color="steelblue",  lw=2, label="Actual cumulative cases")
ax.plot(dates, cum_pred  / 1e6,  color="tomato",     lw=2, linestyle="--", label="SIR predicted")
ax.set_title("Cumulative Confirmed Cases")
ax.set_ylabel("Millions")
ax.legend()
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

# --- subplot 2: active infected ---
ax = axes[0, 1]
ax.plot(dates, I_data / 1e6,  color="orange",  lw=2, label="Actual active (approx)")
ax.plot(dates, I_pred / 1e6,  color="crimson", lw=2, linestyle="--", label="SIR predicted I(t)")
ax.set_title("Active Infected I(t)")
ax.set_ylabel("Millions")
ax.legend()
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

# --- subplot 3: recovered ---
ax = axes[1, 0]
ax.plot(dates, R_data / 1e6,  color="seagreen",    lw=2, label="Actual recovered (approx)")
ax.plot(dates, R_pred / 1e6,  color="mediumorchid", lw=2, linestyle="--", label="SIR predicted R(t)")
ax.set_title("Recovered R(t)")
ax.set_ylabel("Millions")
ax.legend()
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

# --- subplot 4: susceptible ---
ax = axes[1, 1]
ax.plot(dates, S_data / 1e6,  color="royalblue",   lw=2, label="Actual susceptible (approx)")
ax.plot(dates, S_pred / 1e6,  color="darkorange",  lw=2, linestyle="--", label="SIR predicted S(t)")
ax.set_title("Susceptible S(t)")
ax.set_ylabel("Millions")
ax.legend()
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("/Users/piyushmaji/Desktop/Python/Computational_Project_2/Outputs/sir_fit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to sir_fit.png")