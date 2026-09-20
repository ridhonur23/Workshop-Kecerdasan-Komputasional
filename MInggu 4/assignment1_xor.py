"""Assignment 1: MLP untuk data XOR.

Program ini membandingkan fungsi aktivasi dan jumlah hidden unit.
Kurva yang dibuat adalah MSE pada data training setelah setiap epoch.
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
EPOCHS = 1000


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Menghitung mean squared error."""
    # MSE mengukur rata-rata kuadrat selisih antara target dan prediksi.
    return float(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))


def train_curve(
    data: np.ndarray,
    target: np.ndarray,
    hidden_units: int,
    activation: str,
    learning_rate: float = 0.5,
) -> tuple[np.ndarray, MLPClassifier]:
    """Melatih model satu epoch setiap pemanggilan dan mengembalikan kurva MSE."""
    # max_iter=1 membuat satu pemanggilan fit hanya menjalankan satu epoch.
    # warm_start=True menjaga bobot agar training berlanjut pada epoch berikutnya.
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),  # Satu hidden layer.
        activation=activation,  # Fungsi aktivasi yang sedang diuji.
        solver="sgd",  # Optimasi menggunakan Stochastic Gradient Descent.
        learning_rate="constant",  # Learning rate tetap selama training.
        learning_rate_init=learning_rate,  # Nilai awal learning rate/miu.
        max_iter=1,  # Satu epoch per pemanggilan fit.
        warm_start=True,  # Gunakan bobot dari epoch sebelumnya.
        random_state=42,  # Membuat hasil dapat direproduksi.
    )
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(EPOCHS):
            model.fit(data, target)  # Memperbarui bobot jaringan.
            prediction = model.predict(data)  # Menghasilkan prediksi terbaru.
            errors.append(mse(target, prediction))  # Menyimpan MSE epoch ini.
    return np.asarray(errors), model


def main() -> None:
    # Membaca dua kolom input XOR dan satu kolom target.
    dataset = pd.read_csv("XOR.csv")
    data = dataset.iloc[:, :-1].to_numpy()  # Kolom input X1 dan X2.
    target = dataset.iloc[:, -1].to_numpy()  # Kolom target Y.

    OUTPUT_DIR.mkdir(exist_ok=True)  # Membuat folder hasil jika belum ada.
    activations = ["logistic", "tanh", "relu"]
    hidden_units_list = [1, 2, 3, 5, 7]
    summary = []

    # Membandingkan tiga fungsi aktivasi pada beberapa ukuran hidden layer.
    fig, axes = plt.subplots(1, len(activations), figsize=(15, 4), sharey=True)
    for axis, activation in zip(axes, activations):
        for hidden_units in hidden_units_list:
            # Membuat dan melatih model baru untuk setiap kombinasi parameter.
            errors, model = train_curve(data, target, hidden_units, activation)
            axis.plot(np.arange(1, EPOCHS + 1), errors, label=f"hidden={hidden_units}")
            summary.append(
                {
                    "activation": activation,
                    "hidden_units": hidden_units,
                    "final_mse": errors[-1],  # Error pada epoch terakhir.
                    "accuracy": model.score(data, target),  # Akurasi training.
                }
            )
        axis.set_title(f"Aktivasi: {activation}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean Squared Error")
    fig.suptitle("Assignment 1 - Gradient Descent XOR")
    fig.tight_layout()
    # Menyimpan grafik
    fig.savefig(OUTPUT_DIR / "assignment1_xor_gradient_descent.png", dpi=150)
    plt.close(fig)

    # Mengurutkan hasil dari MSE terkecil dan menyimpannya ke file CSV.
    result = pd.DataFrame(summary).sort_values("final_mse")
    result.to_csv(OUTPUT_DIR / "assignment1_xor_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nHasil tersimpan di: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()