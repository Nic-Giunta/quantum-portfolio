import cvxpy as cp
from quantum_portfolio.utils.exceptions import SolverUnavailableError
DEFAULT_SOLVER_ORDER = ("CLARABEL", "ECOS", "OSQP", "SCS")
def installed_solvers() -> list[str]: return list(cp.installed_solvers())
def choose_solver(preferred: str | None=None) -> str:
    available = installed_solvers()
    if preferred:
        if preferred not in available: raise SolverUnavailableError(f"Requested solver {preferred} not installed. Available: {available}")
        return preferred
    for s in DEFAULT_SOLVER_ORDER:
        if s in available: return s
    raise SolverUnavailableError(f"No supported solver installed. Available: {available}")
