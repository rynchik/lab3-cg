"""Создать переопределённую систему с известным решением."""

import argparse
from pathlib import Path

import numpy as np


def generate(M, N, seed, output):
    if M < N or N <= 0:
        raise ValueError("Для тестового набора требуются M >= N > 0")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(M, N)) / np.sqrt(M)
    true_x = rng.normal(size=N)
    b = A @ true_x
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "in.dat").write_text(f"{N}\n{M}\n", encoding="utf-8")
    np.savetxt(output / "AData.dat", A, fmt="%.16e")
    np.savetxt(output / "bData.dat", b, fmt="%.16e")
    np.savetxt(output / "xTrue.dat", true_x, fmt="%.16e")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--cols", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    generate(args.rows, args.cols, args.seed, args.output)
