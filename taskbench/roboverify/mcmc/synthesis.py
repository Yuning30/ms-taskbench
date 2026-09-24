"""Current RoboVerify mutation and RNG helpers; simulator construction is local."""

import contextlib
import random
from copy import deepcopy

import numpy as np

from taskbench.roboverify.api import program


def sample_proportional(values):
    """
    Given a list of positive integers, sample one index with probability
    proportional to its value.

    Parameters:
    - values: list of positive integers

    Returns:
    - An index sampled according to the proportional distribution
    """
    total = sum(values)
    probs = [v / total for v in values]
    return random.choices(range(len(values)), weights=probs, k=1)[0]


def mutate_program(
    current_program: program.Program,
    available_operands: dict,
    available_instructions: list,
) -> tuple[program.Program, bool, dict]:
    pc = 0  # opcode
    po = 1  # operand
    ps = 1  # swap
    pi = 1  # instruction
    sampled_mutation = sample_proportional([pc, po, ps, pi])
    mutate_type_name = {
        0: "opcode",
        1: "operand",
        2: "swap",
        3: "instruction",
    }[sampled_mutation]
    mutation_info = {
        "type": mutate_type_name,
        "changed": False,
        "details": [f"sampled mutation operator: {mutate_type_name}"],
    }

    new_program = deepcopy(current_program)
    if sampled_mutation == 0:
        mutation_info["details"].append(
            "opcode mutation is not implemented yet (no-op)"
        )
    elif sampled_mutation == 1:
        index = random.randint(0, len(new_program.instructions) - 1)
        target = new_program.instructions[index]
        old_operands = target.get_operand()
        mutation_info["details"].append(
            f"operand mutation at slot {index} on {target!s}"
        )
        if old_operands:
            operand_index = random.randint(0, len(old_operands) - 1)
            operand_spec = old_operands[operand_index]
            proposed_operand = random.choice(available_operands[operand_spec["type"]])
            mutation_info["details"].append(
                f"operand index {operand_index} ({operand_spec['type']}): "
                f"old={operand_spec.get('val', operand_spec)!r} "
                f"proposed={proposed_operand!r}"
            )
            if operand_spec.get("val", operand_spec) == proposed_operand:
                mutation_info["details"].append(
                    "proposed operand equals current value; no change"
                )
                return new_program, False, mutation_info
            operand_spec["val"] = proposed_operand
            target.set_operand(old_operands)
            mutation_info["changed"] = True
        else:
            mutation_info["details"].append(
                f"instruction at slot {index} has no mutable operands"
            )
    elif sampled_mutation == 2:
        i = random.randint(0, len(new_program.instructions) - 1)
        j = random.randint(0, len(new_program.instructions) - 1)
        before_i, before_j = (
            new_program.instructions[i],
            new_program.instructions[j],
        )
        new_program.instructions[i], new_program.instructions[j] = (
            new_program.instructions[j],
            new_program.instructions[i],
        )
        mutation_info["details"].append(
            f"swap slots {i} and {j}: " f"{before_i!s} <-> {before_j!s}"
        )
        if i == j:
            mutation_info["details"].append("same slot selected twice; swap is a no-op")
        else:
            mutation_info["changed"] = True
    elif sampled_mutation == 3:
        index = random.randint(0, len(new_program.instructions) - 1)
        old_instruction = new_program.instructions[index]
        pu = 0.25  # probability the SKIP token is proposed
        propose_skip = random.random() < pu
        mutation_info["details"].append(
            f"instruction mutation at slot {index}; "
            f"old={old_instruction!s}; propose_skip={propose_skip}"
        )
        if propose_skip:
            if type(old_instruction) == program.Skip:
                mutation_info["details"].append("Skip -> Skip; no change")
                return new_program, False, mutation_info
            new_program.instructions[index] = program.Skip()
            mutation_info["details"].append(
                f"replaced with Skip (was {old_instruction!s})"
            )
            mutation_info["changed"] = True
        else:
            new_instruction = random.choice(available_instructions)()
            operands = new_instruction.get_operand()
            operand_choices = []
            for operand_i in range(len(operands)):
                chosen = random.choice(available_operands[operands[operand_i]["type"]])
                operands[operand_i]["val"] = chosen
                operand_choices.append(f"{operands[operand_i]['type']}={chosen!r}")
            new_instruction.set_operand(operands)
            new_program.instructions[index] = new_instruction
            mutation_info["details"].append(
                f"replaced with {new_instruction!s} "
                f"(operands: {', '.join(operand_choices)})"
            )
            mutation_info["changed"] = True
    else:
        assert False, "unknown mutation"

    return new_program, mutation_info["changed"], mutation_info


def set_np_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


@contextlib.contextmanager
def preserved_global_rng():
    """Restore the global ``numpy``/``random`` state when the block exits.

    Rollouts must reseed the global RNG per demo: the environment draws its
    initial layout from ``numpy.random``, and a policy rollout is only
    comparable to its demonstration if both start from the same state. That
    property is load-bearing -- the KL/MMD objective is meaningless without it.

    The problem is that the seeding also clobbers the *caller's* stream, and
    the caller is the optimizer. ``cem_optimize`` draws its perturbations with
    ``np.random.randn(N, dim)`` in the parent process and then calls ``f(mu)``
    in the parent at the end of every iteration; ``f`` runs rollouts, which
    reset the global RNG to a state fixed by the last demo seed. Every
    iteration therefore began from the same state and drew the *identical*
    N x dim perturbation matrix, so CEM re-explored one frozen set of
    directions for the whole optimization instead of sampling fresh ones.

    Wrapping the rollout loops keeps both properties: each rollout still gets
    its demo seed, and whatever stream the caller was drawing from is handed
    back untouched.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def make_roboverify_env(task, num_blocks=3):
    if task != "stack":
        raise ValueError("The ManiSkill backend currently supports Stack only")
    from taskbench.roboverify.backend import StackBackend

    return StackBackend(num_blocks=num_blocks)
