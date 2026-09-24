from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverResult:
    """Structured result from a solver run."""

    success: bool
    reward: float = 0.0
    elapsed_steps: int = 0
    info: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
    verification_status: Optional[str] = None


class BaseSolver(ABC):
    """Abstract base class for task solvers.

    Subclasses must implement ``solve()``. Env config requirements (control_mode,
    num_envs, etc.) are declared in the solver's Hydra config group YAML, not here.
    """

    requires_env = True

    @abstractmethod
    def solve(self, env, seed=None, cfg=None) -> SolverResult:
        """Run the solver on a single raw gym env.

        Args:
            env: A gymnasium env (num_envs=1, cpu backend).
            seed: Random seed for env reset.
            cfg: Optional resolved Hydra config (DictConfig). Solvers
                can store this alongside demos for reproducibility.

        Returns:
            SolverResult with success status, reward, and info dict.
        """
        ...


SOLVER_REGISTRY: dict[str, type] = {}


def register_solver(name: str):
    """Decorator to register a solver class.

    Usage::

        @register_solver("my_task")
        class MyTaskSolver(BaseSolver):
            def solve(self, env, seed=None) -> SolverResult:
                ...
    """

    def _register(cls):
        if not issubclass(cls, BaseSolver):
            raise TypeError(
                f"Solver {name!r} must inherit from BaseSolver, "
                f"got {cls.__name__}"
            )
        if name in SOLVER_REGISTRY:
            raise ValueError(f"Solver {name!r} already registered")
        SOLVER_REGISTRY[name] = cls
        return cls

    return _register


_discovered = False


def discover_solvers():
    """Auto-discover solver modules under ``taskbench.solvers``.

    Imports all Python modules in the ``taskbench/solvers/`` package,
    which triggers their ``@register_solver`` decorators.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    import importlib
    import pkgutil

    import taskbench.solvers as solvers_pkg

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        solvers_pkg.__path__, prefix="taskbench.solvers."
    ):
        importlib.import_module(modname)


def get_solver(name: str, **kwargs) -> BaseSolver:
    """Look up and instantiate a solver by registry name.

    Triggers auto-discovery on first call.
    Raises KeyError if the name is not registered.

    Any extra kwargs are forwarded to the solver's ``__init__``.
    """
    discover_solvers()

    if name not in SOLVER_REGISTRY:
        available = ", ".join(sorted(SOLVER_REGISTRY))
        raise KeyError(f"Unknown solver {name!r}. Available: {available}")

    return SOLVER_REGISTRY[name](**kwargs)
