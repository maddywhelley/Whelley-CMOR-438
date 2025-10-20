from rice_ml.simulate_sde import sim_black_scholes
import numpy as np

def test_simulation_positive():
    """Verify that simulated price paths are positive and have correct lengths."""
    # Run one default Black-Scholes simulation (252 trading days)
    S = sim_black_scholes()
    
    # Check path length and enforce nonnegativity of all simulated prices
    assert len(S) == 252
    assert np.all(S > 0)