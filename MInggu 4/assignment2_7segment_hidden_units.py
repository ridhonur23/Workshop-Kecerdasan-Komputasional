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
EPOCHS = 500  # Jumlah epoch untuk setiap percobaan.
HIDDEN_UNITS = [3, 5, 10, 15, 20, 21, 22, 23, 24]  # Hidden unit yang diuji.


def create_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Membuat pola seven-segment standar untuk digit 0 sampai 9."""
    # Urutan setiap baris adalah [a, b, c, d, e, f, g].
    # Nilai 1 berarti segmen menyala dan 0 berarti segmen mati.
    patterns = [
        [1, 1, 1, 1, 1, 1, 0],  # Digit 0
        [0, 1, 1, 0, 0, 0, 0],  # Digit 1
        [1, 1, 0, 1, 1, 0, 1],  # Digit 2
        [1, 1, 1, 1, 0, 0, 1],  # Digit 3
        [0, 1, 1, 0, 0, 1, 1],  # Digit 4
        [1, 0, 1, 1, 0, 1, 1],  # Digit 5
        [1, 0, 1, 1, 1, 1, 1],  # Digit 6
        [1, 1, 1, 0, 0, 0, 0],  # Digit 7
        [1, 1, 1, 1, 1, 1, 1],  # Digit 8
        [1, 1, 1, 1, 0, 1, 1],  # Digit 9
    ]
    # Mengubah pola menjadi array input berukuran 10 x 7.
    data = np.asarray(patterns, dtype=float)
    # Label target mengikuti urutan baris: baris pertama = 0, terakhir = 9.
    target = np.arange(10)
    return data, target


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    # MSE adalah rata-rata kuadrat selisih target dan hasil prediksi.
    return float(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))


def train_curve(data: np.ndarray, target: np.ndarray, hidden_units: int):
    # Satu hidden layer digunakan; jumlah neuronnya diubah sesuai eksperimen.
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),  # Ukuran hidden layer.
        activation="logistic",  # Fungsi sigmoid sesuai ketentuan assignment.
        solver="sgd",  # Optimasi menggunakan Stochastic Gradient Descent.
        learning_rate="constant",  # Nilai learning rate tetap.
        learning_rate_init=0.1,  # mu ditetapkan 0.1 pada assignment 2.
        max_iter=1,  # Satu epoch setiap pemanggilan fit.
        warm_start=True,  # Bobot dilanjutkan dari epoch sebelumnya.
        random_state=42,  # Membuat hasil dapat direproduksi.
    )
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(EPOCHS):
            model.fit(data, target)  # Melatih model dan memperbarui bobot.
            prediction = model.predict(data)  # Memprediksi digit dari data training.
            errors.append(mse(target, prediction))  # Menyimpan error epoch ini.
    return np.asarray(errors), model


def main() -> None:
    # Membuat sepuluh pola seven-segment untuk digit 0 sampai 9.
    data, target = create_training_data()  # Menyiapkan input dan label digit.
    OUTPUT_DIR.mkdir(exist_ok=True)  # Membuat folder output jika belum tersedia.
    # Menyimpan data training agar dapat diperiksa dan digunakan dalam laporan.
    pd.DataFrame(data, columns=[f"segment_{name}" for name in "abcdefg"]).assign(
        digit=target
    ).to_csv(DATA_FILE, index=False)

    summary = []
    # Setiap subplot menunjukkan gradient descent untuk satu jumlah hidden unit.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for axis, hidden_units in zip(axes.flat, HIDDEN_UNITS):
        # Melatih model baru agar setiap ukuran hidden unit dibandingkan secara adil.
        errors, model = train_curve(data, target, hidden_units)
        axis.plot(np.arange(1, EPOCHS + 1), errors, color="#0f766e")  # Kurva MSE.
        axis.set_title(f"Hidden unit = {hidden_units}")
        axis.grid(alpha=0.25)
        summary.append(
            {
                "hidden_units": hidden_units,
                "final_mse": errors[-1],  # MSE pada epoch terakhir.
                "accuracy": model.score(data, target),  # Akurasi data training.
            }
        )
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("MSE")
    fig.suptitle("Assignment 2 - Pengaruh Hidden Unit pada 7-Segment")
    fig.tight_layout()
    # Grafik berisi sembilan subplot, satu subplot untuk setiap hidden unit.
    fig.savefig(OUTPUT_DIR / "assignment2_7segment_hidden_units.png", dpi=150)
    plt.close(fig)

    # Mengurutkan hasil berdasarkan MSE terkecil lalu menyimpannya ke CSV.
    result = pd.DataFrame(summary).sort_values("final_mse")
    result.to_csv(OUTPUT_DIR / "assignment2_7segment_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nData training: {DATA_FILE.resolve()}")


if __name__ == "__main__":
    main()