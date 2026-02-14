
import numpy as np
import torch
from vehicle_pinn_pkg.vehicle import BicycleModel, LongitudinalModel, CombinedModel

def test_input_checks():
    print("=== Testing Input Dimension Checks ===")
    
    # 1. BicycleModel (state_dim=2, input_dim=1)
    print("\nTesting BicycleModel (2 states, 1 input):")
    model_lat = BicycleModel()
    
    # Correct
    try:
        model_lat.get_dynamics(0, [0.0, 0.0], 0.1)
        print("✓ Correct dimensions passed.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Wrong state
    try:
        model_lat.get_dynamics(0, [0.0], 0.1)
    except ValueError as e:
        print(f"✓ Caught wrong state: {e}")

    # Wrong input
    try:
        model_lat.get_dynamics(0, [0.0, 0.0], [0.1, 0.2])
    except ValueError as e:
        print(f"✓ Caught wrong input: {e}")

    # 2. CombinedModel (state_dim=3, input_dim=2)
    print("\nTesting CombinedModel (3 states, 2 inputs):")
    model_comb = CombinedModel()
    
    # Correct
    try:
        model_comb.get_dynamics(0, [10.0, 0.0, 0.0], [1000.0, 0.1])
        print("✓ Correct dimensions passed.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Wrong state
    try:
        model_comb.get_dynamics(0, [10.0, 0.0], [1000.0, 0.1])
    except ValueError as e:
        print(f"✓ Caught wrong state: {e}")

    # Wrong input
    try:
        model_comb.get_dynamics(0, [10.0, 0.0, 0.0], [1000.0])
    except ValueError as e:
        print(f"✓ Caught wrong input: {e}")

if __name__ == "__main__":
    test_input_checks()
