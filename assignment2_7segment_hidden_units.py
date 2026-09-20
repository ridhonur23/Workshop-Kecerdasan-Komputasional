"""Assignment 2: klasifikasi angka 0-9 dari tujuh segmen.

Input mengikuti urutan segmen [a, b, c, d, e, f, g].
Program membuat data training, menguji jumlah hidden unit, dan menyimpan grafik.
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


OUTPUT_DIR = Path("hasil_nn")
DATA_FILE = OUTPUT_DIR / "seven_segment_training.csv"
EPOCHS = 500
HIDDEN_UNITS = [3, 5, 10, 15, 20, 21, 22, 23, 24]


def create_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Membuat pola seven-segment standar untuk digit 0 sampai 9."""
    patterns = [
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1],
    ]
    return np.asarray(patterns, dtype=float), np.arange(10)


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))


def train_curve(data: np.ndarray, target: np.ndarray, hidden_units: int):
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="logistic",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.1,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(EPOCHS):
            model.fit(data, target)
            errors.append(mse(target, model.predict(data)))
    return np.asarray(errors), model


def main() -> None:
    # Membuat sepuluh pola seven-segment untuk digit 0 sampai 9.
    data, target = create_training_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Menyimpan data training agar dapat diperiksa dan digunakan dalam laporan.
    pd.DataFrame(data, columns=[f"segment_{name}" for name in "abcdefg"]).assign(
        digit=target
    ).to_csv(DATA_FILE, index=False)

    summary = []
    # Setiap subplot menunjukkan gradient descent untuk satu jumlah hidden unit.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for axis, hidden_units in zip(axes.flat, HIDDEN_UNITS):
        errors, model = train_curve(data, target, hidden_units)
        axis.plot(np.arange(1, EPOCHS + 1), errors, color="#0f766e")
        axis.set_title(f"Hidden unit = {hidden_units}")
        axis.grid(alpha=0.25)
        summary.append(
            {
                "hidden_units": hidden_units,
                "final_mse": errors[-1],
                "accuracy": model.score(data, target),
            }
        )
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("MSE")
    fig.suptitle("Assignment 2 - Pengaruh Hidden Unit pada 7-Segment")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "assignment2_7segment_hidden_units.png", dpi=150)
    plt.close(fig)

    result = pd.DataFrame(summary).sort_values("final_mse")
    result.to_csv(OUTPUT_DIR / "assignment2_7segment_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nData training: {DATA_FILE.resolve()}")


if __name__ == "__main__":
    main()