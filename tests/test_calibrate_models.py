from rice_ml.calibrate_models import black_scholes_price
import numpy as np

def test_black_scholes_symm():
    call = black_scholes_price(100, 100, 0.01, 0.2, 1, "call")
    put = black_scholes_price(100, 100, 0.01, 0.2, 1, "put")
    assert call > 0 and put > 0
    assert np.isfinite(call) and np.isfinite(put)