import pandas as pd
import numpy as np

DATA_PATH = "/Users/piyushmaji/Desktop/Python/Computational_Project_2/Data/global_covid_timeseries.csv"
INCUBATION_DAYS = 5


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date")


def build_compartments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["I"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
    df["R"] = df["Recovered"] + df["Deaths"]
    return df[df["I"] > 0].copy()


def estimate_parameters(df: pd.DataFrame) -> tuple[float, float, float]:
    dI = df["I"].diff()
    dR = df["R"].diff()

    gamma_series = (dR / df["I"]).replace([np.inf, -np.inf], np.nan)
    gamma = gamma_series.median()

    beta_series = (dI / df["I"] + gamma).replace([np.inf, -np.inf], np.nan)
    beta = beta_series.median()

    sigma = 1 / INCUBATION_DAYS
    return beta, gamma, sigma


def main():
    df = load_data(DATA_PATH)
    df = build_compartments(df)
    beta, gamma, sigma = estimate_parameters(df)

    R0 = beta / gamma
    print("Estimated Parameters:")
    print(f"  beta  (transmission rate): {beta:.4f}")
    print(f"  gamma (recovery rate):     {gamma:.4f}")
    print(f"  sigma (incubation rate):   {sigma:.4f}")
    print(f"  R0    (reproduction number): {R0:.2f}")


if __name__ == "__main__":
    main()
