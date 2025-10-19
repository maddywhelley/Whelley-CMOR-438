from rice_ml.simulate_sde import sim_black_scholes
import numpy as np

def test_simulation_positive():
    S = sim_black_scholes()
    assert len(S) == 252
    assert np.all(S > 0)