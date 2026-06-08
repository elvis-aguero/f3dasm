#!/usr/bin/env python3
"""
Replicate the 8D black-box optimization result.

Loads the canonical ExperimentData ledger and reproduces the headline finding:
the best minimum found and its coordinates.
"""

from f3dasm import ExperimentData

# Load the canonical ledger
project_dir = r"/Users/harrislab/Documents/GitHub/f3dasm/studies/agentic_black_box_8d/runs/20260608T192222/experiment_data"
data = ExperimentData.from_file(project_dir=project_dir)

# Extract the outputs (f values) and inputs (x1--x8)
f_values = data.output.values.flatten()  # Shape: (n_evals,)
x_values = data.input.values  # Shape: (n_evals, 8)

# Find the index of the minimum objective value
best_idx = f_values.argmin()
best_f = f_values[best_idx]
best_x = x_values[best_idx]

# Report the finding
print(f"Total evaluations: {len(f_values)}")
print(f"Best objective value: {best_f:.10f}")
print(f"Best point (x1-x8):")
for i, xi in enumerate(best_x, 1):
    print(f"  x{i} = {xi:.10f}")

# Verify against the expected headline result
expected_f = -1.7381308003291325
expected_x = [1.280077705, -0.673350521, -1.667181174, -1.301311749, 
              2.439143272, 1.623945636, -3.141398326, 3.181450491]

# Tolerance for floating-point comparison
tol_f = 1e-6
tol_x = 1e-4

assert abs(best_f - expected_f) < tol_f, \
    f"Objective mismatch: found {best_f}, expected ~{expected_f}"
for i, (xi_found, xi_expected) in enumerate(zip(best_x, expected_x)):
    assert abs(xi_found - xi_expected) < tol_x, \
        f"x{i+1} mismatch: found {xi_found}, expected ~{xi_expected}"

print(f"\n✓ REPLICATED: Best objective = {best_f:.10f}")
