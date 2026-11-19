"""Run the incompressible MHD examples in configs/parallel/validation.

    python -m scripts.validate_parallel_imhd configs/parallel/validation/abc-growing.yaml

The ABC and magnetic Taylor–Green initial conditions are registered here.
"""

import argparse
import os

import torch

from phrike.io import load_config
from phrike.parallel.comm import Comm
from phrike.parallel.driver import Simulation
from phrike.parallel.fields import random_vector_spectrum
from phrike.parallel.problems import register


@register("imhd_abc_kinematic", "imhd", "Kinematic ABC dynamo with A=B=C=1")
def abc(grid, initial, _gamma):
    x, y, z = grid.xyz_mesh()
    u = torch.stack((z.sin() + y.cos(), x.sin() + z.cos(), y.sin() + x.cos()))
    seed = random_vector_spectrum(grid, seed=initial["b_seed"], kmin=1.0, kmax=initial["b_kmax"],
                                  rms=initial["b_rms"], spectrum="flat")
    return "physical", torch.cat((u, grid.inverse(seed)))


@register("imhd_magnetic_taylor_green", "imhd", "Insulating magnetic Taylor–Green vortex I1")
def magnetic_taylor_green(grid, _initial, _gamma):
    x, y, z = grid.xyz_mesh()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    u = torch.stack((sx * cy * cz, -cx * sy * cz, torch.zeros_like(x)))
    b = torch.stack((cx * sy * sz, sx * cy * sz, -2.0 * sx * sy * cz)) / 3.0**0.5
    return "physical", torch.cat((u, b))


def hold_velocity(stepper):
    velocity = stepper.uhat[:3].clone()
    coupled_stage = stepper._stage

    def induction_stage(products, coefficient, parameters, first, final):
        coupled_stage(products, coefficient, parameters, first, final)
        state = stepper.uhat if final else stepper.shat
        state[:3].copy_(velocity)

    stepper._stage = induction_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outdir")
    parser.add_argument("--precision", choices=("single", "double"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-minutes", type=float)
    parser.add_argument("--restart")
    args = parser.parse_args()
    config = load_config(args.config)
    problem = config["problem"]
    parallel = config.get("parallel") or {}
    comm = Comm.auto(transpose=parallel.get("transpose"), gpu_aware=parallel.get("gpu_aware_mpi"))
    comm.install_abort_hook()
    expected_ranks = int(os.environ.get("SLURM_NTASKS", comm.size))
    if comm.size != expected_ranks:
        raise RuntimeError(f"PHRIKE saw {comm.size} ranks, expected {expected_ranks}")
    sim = Simulation(problem, config, comm=comm, backend=args.backend, outdir=args.outdir,
                     precision=args.precision, max_steps=args.max_steps, max_wall_minutes=args.max_wall_minutes,
                     restart=args.restart)
    if args.backend == "cuda":
        if sim.stepper._triton is None:
            raise RuntimeError("The CUDA validation requires the Triton kernels")
        sim.stepper._strict_kernels = True
    if problem == "imhd_abc_kinematic":
        hold_velocity(sim.stepper)
    sim.run()


if __name__ == "__main__":
    main()
