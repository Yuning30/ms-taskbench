"""Hydra entry point for the current RoboVerify Stack pipeline."""

from taskbench.solver import BaseSolver, SolverResult, register_solver


@register_solver("program_synthesis")
@register_solver("roboverify_stack")
class ProgramSynthesisSolver(BaseSolver):
    """Run search and both proof stages; successful rollouts alone cannot pass."""

    requires_env = False

    def __init__(self, demos=None, **options):
        self.demos = demos
        self.options = options

    def solve(self, env=None, seed=None, cfg=None):
        if not self.demos:
            raise ValueError(
                "A current ManiSkill archive is required. Collect with "
                "uv run --extra roboverify python -m taskbench.roboverify.entry.collect_demos "
                "--program taskbench.roboverify.examples.stack:build_program; "
                "then set run.solver_kwargs.demos=<collection>/demonstrations.npz. "
                "Old HDF5 demos do not include replayable simulator state."
            )
        try:
            from taskbench.roboverify.entry.synthesize_cfg import main
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install verification dependencies: uv sync --extra roboverify"
            ) from exc

        options = dict(self.options)
        options.setdefault("seed", 0 if seed is None else int(seed))
        argv = ["--demos", str(self.demos)]
        for key, value in options.items():
            if value is None or value is False:
                continue
            argv.append("--" + key.replace("_", "-"))
            if value is not True:
                if isinstance(value, str) or not hasattr(value, "__iter__"):
                    argv.append(str(value))
                else:
                    argv.extend(map(str, value))
        code = main(argv)
        status = "verified_model" if code == 0 else "unverified"
        return SolverResult(
            success=code == 0,
            verification_status=status,
            info={"formal_verification": status},
            failure_reason=(
                None
                if code == 0
                else "See the RoboVerify run report for the failed stage"
            ),
        )
