"""Замеры только итерационного цикла на системе 1000 x 500."""

import json
from pathlib import Path
import sys

from mpi4py import MPI
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cg_full import solve as full  # noqa: E402
from src.cg_simple import solve as simple  # noqa: E402
from src.common import metrics, read_data, reference_solution  # noqa: E402


def main():
    comm = MPI.COMM_WORLD
    A = b = reference = None
    if comm.rank == 0:
        A, b = read_data(Path(__file__).resolve().parents[1] / "data")
        reference = reference_solution(A, b)
    rows = []
    for name, solve in [("full", full), ("simple", simple)]:
        samples, iteration_counts, histories, errors = [], [], [], []
        for trial in range(6):
            result = solve(A, b, tol=1e-10, atol=1e-12, maxiter=2000, comm=comm)
            if not result.converged:
                raise RuntimeError("Benchmark solver did not converge")
            if comm.rank == 0:
                error = metrics(A, b, result.x, reference)
                np.testing.assert_allclose(result.x, reference, rtol=1e-7, atol=1e-8)
                if trial:
                    samples.append(result.kernel_seconds)
                    iteration_counts.append(result.iterations)
                    histories.append(result.history)
                    errors.append(error)
        if comm.rank == 0:
            rows.append(
                {
                    "version": name,
                    "processes": comm.size,
                    "median_kernel_s": float(np.median(samples)),
                    "samples": samples,
                    "iteration_counts": iteration_counts,
                    "history": histories[0],
                    "errors": errors,
                }
            )
    if comm.rank == 0:
        print(json.dumps({"mpi": MPI.Get_library_version().strip(), "rows": rows}))


if __name__ == "__main__":
    main()
