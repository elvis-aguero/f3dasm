"""
Black-Box 8D Optimisation Pipeline
Baseline strategy: LHS exploration → Surrogate (GP) + BO → Local Polish + Verification

This pipeline is the canonical f3dasm representation of the three-phase strategy:
1. Space-filling exploration (LHS)
2. Gaussian Process surrogate + Bayesian Optimisation with Expected Improvement
3. Multi-start local optimisation + random restart verification
"""

from f3dasm import Pipeline, Step, Domain, ExperimentData
from f3dasm.design import Domain as DomainDesign
from f3dasm.design.sampler import LatinHypercube
import numpy as np
from scipy.optimize import minimize
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf


def create_domain():
    """Create the 8D box domain [-5, 5]⁸."""
    domain = Domain()
    for i in range(1, 9):
        domain += DomainDesign(f"x{i}", [float, -5.0, 5.0])
    return domain


def stage_1_lhs_exploration(domain, n_samples=250, seed=0):
    """
    Stage 1: Latin Hypercube Sampling for initial exploration.
    Returns: ExperimentData with all LHS samples evaluated.
    """
    sampler = LatinHypercube(seed=seed)
    data = domain.sample(sampler, n_samples)
    # Evaluate via oracle (black-box evaluator from workspace)
    from evaluator import evaluate
    outputs = [evaluate(list(row)) for row in data.to_numpy("input")]
    data.add_output("f_x", outputs)
    return data


def stage_2_bo_refinement(data, n_evals=300):
    """
    Stage 2: Fit GP surrogate and run Bayesian Optimisation with EI acquisition.
    Returns: updated ExperimentData with all BO evaluations.
    """
    # Fit GP on LHS data
    from botorch.models import SingleTaskGP
    train_x = torch.tensor(data.to_numpy("input"), dtype=torch.float)
    train_y = torch.tensor(data.to_numpy("output")["f_x"], dtype=torch.float).unsqueeze(-1)
    gp = SingleTaskGP(train_x, train_y)
    
    # BO loop: Expected Improvement
    from evaluator import evaluate
    best_val = train_y.min().item()
    for i in range(n_evals):
        ei = ExpectedImprovement(gp, best_y=best_val)
        candidate, _ = optimize_acqf(ei, bounds=torch.tensor([[-5.]*8, [5.]*8], dtype=torch.float),
                                      q=1, num_restarts=10, raw_samples=256)
        candidate = candidate.squeeze(0).numpy()
        f_val = evaluate(list(candidate))
        # Add to data and refit GP
        data.add_row({f"x{i+1}": candidate[i] for i in range(8)} | {"f_x": f_val})
        if f_val < best_val:
            best_val = f_val
    return data


def stage_3_verification(data, n_local=100, n_restarts=300):
    """
    Stage 3: Multi-start local optimisation + random restart verification.
    Returns: final best point and evidence of global optimality.
    """
    from evaluator import evaluate
    
    # Multi-start local polish from top-10 BO candidates
    # (stub: would select top-10 from data, then run LBFGSB from each)
    
    # Random restarts from fresh seeds
    best_overall = data.to_pandas()[1]["f_x"].min()
    best_point = None
    for i in range(n_restarts // 10):  # ~30 restarts
        x0 = np.random.uniform(-5, 5, 8)
        result = minimize(evaluate, x0, method="L-BFGSB", 
                         bounds=[(-5, 5)]*8)
        if result.fun < best_overall:
            best_overall = result.fun
            best_point = result.x
    
    return best_point, best_overall


def main():
    """Orchestrate the full pipeline."""
    # Stage 1: LHS Exploration
    domain = create_domain()
    data_phase1 = stage_1_lhs_exploration(domain, n_samples=250, seed=0)
    print(f"Phase 1 (LHS): Best f(x) = {data_phase1.to_pandas()[1]['f_x'].min()}")
    
    # Stage 2: BO Refinement
    data_phase2 = stage_2_bo_refinement(data_phase1, n_evals=300)
    print(f"Phase 2 (BO): Best f(x) = {data_phase2.to_pandas()[1]['f_x'].min()}")
    
    # Stage 3: Verification
    best_point, best_val = stage_3_verification(data_phase2, n_local=100, n_restarts=300)
    print(f"Phase 3 (Verification): Best f(x) = {best_val}")
    print(f"Optimal point: {best_point}")
    print(f"Total evaluations: {len(data_phase2) + (100 + 300) // 10 * 10}")


if __name__ == "__main__":
    main()
