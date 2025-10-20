import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """Return Black-Scholes price for a European call or put option."""
    # compute d1 and d2 terms for the closed-form solution
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # apply respective pricing formulas for call and put options
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calibrate_black_scholes(df, K=100, r=0.01, T=1.0):
    """
    Calibrate Black-Scholes model against simulated market data.
    
    For each simulated path, use its mean price as the spot proxy and compare the theoretical model price to the
    synethic 'market' price with random noise. 
    Returns a dataframe of regime-level RMSEs.
    """
    results = []
    
    # Iterate over each simulated regime entry
    for _, row in df.iterrows():
        S = row["mean"] # Use path mean as an approximate spot price
        sigma = row["sigma"] # Volatility parameter from the simulated regime
        
        # Compute theoretical option price
        model_price = black_scholes_price(S, K, r, sigma, T)
        # Generate noisy 'market' price to mimic imperfect calibration data
        market_price = model_price * np.random.normal(1, 0.05)
        # RMSE between model and synethic market price
        rmse = np.sqrt((model_price - market_price)**2)
        results.append({
            "regime": row["regime"],
            "sigma": sigma,
            "model_price": model_price,
            "market_price": market_price,
            "rmse": rmse
        })
    # Combine results and save for later analysis
    out = pd.DataFrame(results)
    out.to_csv("data/model_calibration.csv", index=False)
    return out