"""Runtime motion selection shared by collection and synthesis."""

MOVE_CONTROLLERS = ("delta", "planner")
DEFAULT_PLANNER_STEP_LIMIT = 200


def add_controller_options(parser):
    parser.add_argument(
        "--move-controller",
        choices=MOVE_CONTROLLERS,
        default="delta",
        help="Cartesian delta feedback or mplib screw planning with joint control.",
    )
    parser.add_argument(
        "--planner-step-limit",
        type=int,
        default=DEFAULT_PLANNER_STEP_LIMIT,
        help="Maximum simulation steps per planner primitive, including gripper phases.",
    )
