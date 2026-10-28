"""Сравнение полной и упрощённой CG-версий на 1, 2, 4, 8 процессах."""

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def save_report(metadata):
    target = ROOT / "report"
    target.mkdir(exist_ok=True)
    (target / "raw_timings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    rows = metadata["rows"]
    columns = [
        "version",
        "processes",
        "median_kernel_s",
        "speedup",
        "efficiency",
        "iterations",
        "relative_solution_error",
        "relative_data_residual",
        "relative_normal_residual",
    ]
    with (target / "performance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            baseline = next(
                r["median_kernel_s"]
                for r in rows
                if r["version"] == row["version"] and r["processes"] == 1
            )
            speedup = baseline / row["median_kernel_s"]
            writer.writerow(
                {
                    "version": row["version"],
                    "processes": row["processes"],
                    "median_kernel_s": row["median_kernel_s"],
                    "speedup": speedup,
                    "efficiency": speedup / row["processes"],
                    "iterations": row["iteration_counts"][0],
                    **row["errors"][0],
                }
            )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, label in [("full", "Полная версия"), ("simple", "Упрощённая версия")]:
        selected = [r for r in rows if r["version"] == name]
        baseline = selected[0]["median_kernel_s"]
        processes = [r["processes"] for r in selected]
        speedups = [baseline / r["median_kernel_s"] for r in selected]
        axes[0].plot(processes, speedups, "o-", label=label)
        axes[1].plot(
            processes,
            [s / p * 100 for s, p in zip(speedups, processes)],
            "o-",
            label=label,
        )
    axes[0].plot(
        [1, 2, 4, 8], [1, 2, 4, 8], "--", color="gray", label="Идеальное ускорение"
    )
    axes[1].axhline(100, linestyle="--", color="gray", label="Идеальная эффективность")
    for ax in axes:
        ax.set_xlabel("Число MPI-процессов")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Ускорение T(1) / T(p)")
    axes[1].set_ylabel("Эффективность, %")
    fig.suptitle("Метод сопряжённых градиентов; матрица 1000 × 500")
    fig.tight_layout()
    fig.savefig(target / "scaling.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in [
        ("full", "Полная версия, 4 процесса"),
        ("simple", "Упрощённая версия, 4 процесса"),
    ]:
        history = next(
            r["history"] for r in rows if r["version"] == name and r["processes"] == 4
        )
        ax.semilogy(
            np.arange(len(history)), np.array(history) / history[0], label=label
        )
    ax.axhline(1e-10, color="gray", linestyle="--", label="Допуск 10⁻¹⁰")
    ax.set(
        xlabel="Итерация",
        ylabel="Относительная норма невязки",
        title="Сходимость CG для нормальных уравнений",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not launcher:
        raise SystemExit("Задайте MPIEXEC или добавьте mpiexec в PATH")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    rows, mpi = [], None
    for p in (1, 2, 4, 8):
        result = subprocess.run(
            [
                launcher,
                "-n",
                str(p),
                sys.executable,
                str(ROOT / "scripts/benchmark_worker.py"),
            ],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=240,
        )
        data = json.loads(result.stdout.strip().splitlines()[-1])
        rows.extend(data["rows"])
        mpi = data["mpi"]
        print(f"Completed: {p} processes", flush=True)
    save_report(
        {
            "shape": [1000, 500],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mpi": mpi,
            "logical_cpus": os.cpu_count(),
            "warmup": 1,
            "repeats": 5,
            "tol": 1e-10,
            "atol": 1e-12,
            "alpha": 0.0,
            "rows": rows,
        }
    )


if __name__ == "__main__":
    main()
