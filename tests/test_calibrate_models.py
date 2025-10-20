from rice_ml.calibrate_models import black_scholes_price
import numpy as np

def test_black_scholes_symm():
    """
    Basic sanity test for the Black-Scholes pricing function.
    Ensures call and put prices are positive and finite under symmetric parameters.
    """
    # Compute call and put prices for the at-the-money options
    call = black_scholes_price(100, 100, 0.01, 0.2, 1, "call")
    put = black_scholes_price(100, 100, 0.01, 0.2, 1, "put")
    
    # Verify outputs are positive and finite
    assert call > 0 and put > 0
    assert np.isfinite(call) and np.isfinite(put)