import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """Return Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calibrate_black_scholes(df, K=100, r=0.01, T=1.0):
    """Given simulated spot paths and sigmas, compute model prices and measure calibration RMSE"""
    results = []
    for _, row in df.iterrows():
        S = row["mean"] # use avg. price as spot proxy
        sigma = row["sigma"]
        model_price = black_scholes_price(S, K, r, sigma, T)
        # 'true' synthetic market price: small random offset
        market_price = model_price * np.random.normal(1, 0.05)
        rmse = np.sqrt((model_price - market_price)**2)
        results.append({
            "regime": row["regime"],
            "sigma": sigma,
            "model_price": model_price,
            "market_price": market_price,
            "rmse": rmse
        })
    out = pd.DataFrame(results)
    out.to_csv("data/model_calibration.csv", index=False)
    return out