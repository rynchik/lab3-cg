"""Дополнительные опыты: регуляризация и влияние ранней остановки."""

import csv
import json
from pathlib import Path
import sys

from mpi4py import MPI
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cg_simple import solve  # noqa: E402
from src.common import metrics, read_data, reference_solution  # noqa: E402


def regularization():
    rng = np.random.default_rng(2026)
    M, N = 80, 40
    U, _ = np.linalg.qr(rng.normal(size=(M, N)))
    V, _ = np.linalg.qr(rng.normal(size=(N, N)))
    singular_values = np.geomspace(1.0, 1e-6, N)
    A = (U * singular_values) @ V.T
    truth = rng.normal(size=N)
    clean_b = A @ truth
    noise = rng.normal(size=M)
    noise *= 1e-4 * np.linalg.norm(clean_b) / np.linalg.norm(noise)
    b = clean_b + noise
    rows = []
    for alpha in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        result = solve(
            A, b, alpha=alpha, tol=1e-13, atol=0.0, maxiter=8000, comm=MPI.COMM_SELF
        )
        reference = reference_solution(A, b, alpha)
        row = {
            "alpha": alpha,
            "iterations": result.iterations,
            "converged": result.converged,
            "error_vs_truth": float(
                np.linalg.norm(result.x - truth) / np.linalg.norm(truth)
            ),
            "reference_error_vs_truth": float(
                np.linalg.norm(reference - truth) / np.linalg.norm(truth)
            ),
            **metrics(A, b, result.x, reference, alpha),
        }
        rows.append(row)
    return {
        "shape": [M, N],
        "condition_A": float(np.linalg.cond(A)),
        "relative_noise": 1e-4,
        "seed": 2026,
        "tol": 1e-13,
        "atol": 0.0,
        "rows": rows,
    }


def stopping():
    A, b = read_data(ROOT / "data")
    reference = reference_solution(A, b)
    rows = []
    for mode, tol, atol in [("early", 1e-10, 1e-12), ("limit_N", 0.0, 0.0)]:
        samples, last = [], None
        for trial in range(6):
            last = solve(
                A, b, tol=tol, atol=atol, maxiter=A.shape[1], comm=MPI.COMM_SELF
            )
            if trial:
                samples.append(last.kernel_seconds)
        rows.append(
            {
                "mode": mode,
                "tol": tol,
                "atol": atol,
                "maxiter": A.shape[1],
                "iterations": last.iterations,
                "stop_reason": last.stop_reason,
                "median_kernel_s": float(np.median(samples)),
                "samples": samples,
                **metrics(A, b, last.x, reference),
            }
        )
    return rows


def main():
    target = ROOT / "report"
    target.mkdir(exist_ok=True)
    data = {"regularization": regularization(), "early_stopping": stopping()}
    (target / "studies.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for name, rows in [
        ("regularization", data["regularization"]["rows"]),
        ("early_stopping", data["early_stopping"]),
    ]:
        keys = [key for key in rows[0] if key != "samples"]
        with (target / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    rows = data["regularization"]["rows"]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(
        positions, [r["error_vs_truth"] for r in rows], "o-", label="Метод CG"
    )
    axes[0].semilogy(
        positions,
        [r["reference_error_vs_truth"] for r in rows],
        "x--",
        label="NumPy lstsq",
    )
    axes[0].set_ylabel("Относительная ошибка к известному решению")
    axes[0].legend()
    axes[1].semilogy(positions, [r["relative_data_residual"] for r in rows], "o-")
    axes[1].set_ylabel("Относительная невязка исходной системы")
    for ax in axes:
        ax.set_xticks(
            positions, ["0" if r["alpha"] == 0 else f'{r["alpha"]:.0e}' for r in rows]
        )
        ax.set_xlabel("Параметр регуляризации alpha")
        ax.grid(alpha=0.25)
    fig.suptitle("Регуляризация: cond(A) ≈ 10⁶, относительный шум 10⁻⁴")
    fig.tight_layout()
    fig.savefig(target / "regularization.png", dpi=160)
    plt.close(fig)
    print("Regularization and stopping studies saved")


if __name__ == "__main__":
    main()
