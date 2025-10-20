import numpy as np
import pandas as pd
import os

def sim_black_scholes(S0=100, mu=0.05, sigma=0.2, T=1.0, N=252):
    """Simulate one geometric Brownian motion path."""
    dt = T / N # Time step, in years; e.g. 1 trading year with 252 steps
    dW = np.random.normal(0, np.sqrt(dt), N) # Generate Brownian motion increments
    S = np.zeros(N) # Preallocate price array
    S[0] = S0 # Set initial price
    
    # Iterate over time steps using the Black-Scholes exponential update rule
    for t in range(1, N):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t])
    return S

def generate_regime_dataset(n_paths=300):
    """Generate multiple simulated paths under calm, volatile, or stochastic regimes."""
    os.makedirs("data", exist_ok=True) # ensure output file exists
    data = []
    
    # Simulate n_paths independent scenarios
    for _ in range(n_paths):
        # Randomly assign each path to a regime type
        regime = np.random.choice(["calm", "volatile", "stochastic"])
        
        # Set volatility (sigma) based on regime
        if regime == "calm":
            sigma = 0.1
        elif regime == "volatile":
            sigma = 0.4
        else:
            sigma = np.random.uniform(0.2,0.6)
        
        # Simulate the price path under selected volatility
        S = sim_black_scholes(sigma=sigma)
        
        # Compute summary stats as simple features for classification tasks
        stats = {
            "regime": regime,
            "sigma": sigma,
            "mean": np.mean(S),
            "std": np.std(S),
            "skew": pd.Series(S).skew(),
            "kurt": pd.Series(S).kurt(),
        }
        data.append(stats)
    
    # Combine all simulated paths into a DataFrame and export to CSV
    df = pd.DataFrame(data)
    df.to_csv("data/simulated_paths.csv", index=False)
    return df