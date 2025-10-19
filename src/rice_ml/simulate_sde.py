import numpy as np
import pandas as pd
import os

def sim_black_scholes(S0=100, mu=0.05, sigma=0.2, T=1.0, N=252):
    """Simulate one geometric Brownian motion path."""
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), N)
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t])
    return S

def generate_regime_dataset(n_paths=300):
    """Generate multiple simulated paths under calm, volatile, or stochastic regimes."""
    os.makedirs("data", exist_ok=True)
    data = []
    for _ in range(n_paths):
        regime = np.random.choice(["calm", "volatile", "stochastic"])
        if regime == "calm":
            sigma = 0.1
        elif regime == "volatile":
            sigma = 0.4
        else:
            sigma = np.random.uniform(0.2,0.6)
        S = sim_black_scholes(sigma=sigma)
        stats = {
            "regime": regime,
            "sigma": sigma,
            "mean": np.mean(S),
            "std": np.std(S),
            "skew": pd.Series(S).skew(),
            "kurt": pd.Series(S).kurt(),
        }
        data.append(stats)
    df = pd.DataFrame(data)
    df.to_csv("data/simulated_paths.csv", index=False)
    return df