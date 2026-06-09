#!/usr/bin/env python3
"""
Replicate the global optimum of resonance(k, m) from the canonical ExperimentData ledger.

The headline finding: The (k, m) pair that maximises resonance(k, m) = ord(k, m) / ln(m)
over k ∈ {2,...,50} and m ∈ {1000,...,100000} is (k=6, m=99991) with resonance ≈ 8685.089.

This script loads the complete evaluation ledger (all delegations) and reproduces the
headline by selecting the maximum resonance over all feasible points.
"""

from f3dasm import ExperimentData
import os

# Load the canonical ExperimentData ledger from the experiment_data directory
# The ledger is located at: runs/<timestamp>/experiment_data/
# We construct the path dynamically relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
study_dir = os.path.dirname(script_dir)  # goes up from runs/<timestamp>/ to study/
experiment_data_dir = os.path.join(study_dir, "experiment_data")

print(f"Loading canonical ExperimentData ledger from: {experiment_data_dir}")
data = ExperimentData.from_file(project_dir=experiment_data_dir)

# Extract input and output dataframes
df_in, df_out = data.to_pandas()

print(f"Loaded {len(df_out)} total evaluations from ledger.\n")

# The "resonance" column contains the objective values
# Filter to feasible points only (gcd(k, m) = 1 produces resonance > 0; 
# infeasible points get a large negative sentinel around -1e10)
feasible = df_out[df_out["resonance"] > -1e9]

print(f"Total evaluations in ledger: {len(df_out)}")
print(f"Feasible points (gcd(k,m)=1): {len(feasible)}")
print(f"Infeasible points: {len(df_out) - len(feasible)}\n")

# Find the row with maximum resonance
best_idx = feasible["resonance"].idxmax()
best_row_out = feasible.loc[best_idx]
best_row_in = df_in.loc[best_idx]

# Extract headline values
best_k = int(best_row_in["k"])
best_m = int(best_row_in["m"])
best_resonance = float(best_row_out["resonance"])

print("=" * 70)
print("HEADLINE RESULT (from canonical ledger):")
print("=" * 70)
print(f"Optimal (k, m) = ({best_k}, {best_m})")
print(f"Maximum resonance = {best_resonance:.10f}")
print("=" * 70)

# Verification: Check that (k=6, m=99991) is in the ledger and matches expected value
print("\nVerifying against known best point...")

# Find (k=6, m=99991) in the ledger
point_rows = df_in[(df_in["k"] == 6) & (df_in["m"] == 99991)]

if len(point_rows) > 0:
    point_idx = point_rows.index[0]
    point_resonance = df_out.loc[point_idx, "resonance"]
    print(f"✓ Found (k=6, m=99991) in ledger with resonance = {point_resonance:.10f}")
else:
    print("✗ WARNING: (k=6, m=99991) not found in ledger!")

# Final assertion
EXPECTED_K = 6
EXPECTED_M = 99991
EXPECTED_RESONANCE = 8685.089
TOLERANCE = 0.01  # Allow 0.01 units of numerical variation

assert best_k == EXPECTED_K, \
    f"Expected best k={EXPECTED_K}, but found k={best_k}"
assert best_m == EXPECTED_M, \
    f"Expected best m={EXPECTED_M}, but found m={best_m}"
assert abs(best_resonance - EXPECTED_RESONANCE) < TOLERANCE, \
    f"Expected resonance≈{EXPECTED_RESONANCE:.3f}, but found {best_resonance:.10f}; " \
    f"difference = {abs(best_resonance - EXPECTED_RESONANCE):.6f}"

print("\n" + "=" * 70)
print("✓ ASSERTION PASSED")
print("=" * 70)
print(f"The canonical ledger confirms the headline finding:")
print(f"  (k, m, resonance) = ({best_k}, {best_m}, {best_resonance:.6f})")
print("=" * 70)
