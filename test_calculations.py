"""Small independent smoke tests for the app's core equations."""

import math


def close(actual, expected, rel=1e-9):
    assert math.isclose(actual, expected, rel_tol=rel), (actual, expected)


# Water at 20 C, V=1 m/s, D=0.05 m
rho = 998.2
mu = 1.002e-3
re = rho * 1.0 * 0.05 / mu
close(re, 49810.37924151697)

# 10 US gpm through a 2-inch pipe
q = 10.0 * (0.003785411784 / 60.0)
d = 2.0 * 0.0254
velocity = q / (math.pi * d**2 / 4.0)
close(velocity, 0.31127523769912896)

# Laminar flat-plate 99% thickness estimate, Re_x = 200,000 at x=1 m
delta_laminar = 5.0 * 1.0 / math.sqrt(200000.0)
close(delta_laminar, 0.011180339887498949)

# Fully turbulent reference thickness, Re_x = 5,000,000 at x=1 m
delta_turbulent = 0.37 * 1.0 / (5.0e6 ** 0.2)
close(delta_turbulent, 0.016920286921311072)

print("All calculation smoke tests passed.")
