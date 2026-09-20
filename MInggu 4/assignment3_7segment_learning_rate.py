"""Assignment 3: pengaruh learning rate (mu) pada MLP 7-segment."""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from assignment2_7segment_hidden_units import create_training_data, mse


OUTPUT_DIR = Path("hasil_nn")
EPOCHS = 500  # Jumlah epoch untuk setiap nilai learning rate.
LEARNING_RATES = [0.1, 0.5, 5, 10, 11, 12]  # Nilai mu yang dibandingkan.


def train_curve(data: np.ndarray, target: np.ndarray, learning_rate: float):
    # Hidden unit dibuat tetap 10 agar hanya pengaruh mu yang diamati.
    model = MLPClassifier(
        hidden_layer_sizes=(10,),  # Satu hidden layer dengan 10 neuron.
        activation="logistic",  # Fungsi aktivasi sigmoid.
        solver="sgd",  # Optimasi menggunakan Stochastic Gradient Descent.
        learning_rate="constant",  # Nilai mu tetap selama satu percobaan.
        learning_rate_init=learning_rate,  # Besar langkah pembaruan bobot.
        max_iter=1,  # Satu epoch setiap pemanggilan fit.
        warm_start=True,  # Bobot dilanjutkan dari epoch sebelumnya.
        random_state=42,  # Membuat perbandingan dapat direproduksi.
    )
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(EPOCHS):
            model.fit(data, target)  # Memperbarui bobot dengan nilai mu terpilih.
            prediction = model.predict(data)  # Memprediksi digit setelah training.
            errors.append(mse(target, prediction))  # Menyimpan MSE epoch ini.
    return np.asarray(errors), model


def main() -> None:
    # Data dan target yang sama digunakan agar pengaruh mu dapat dibandingkan adil.
    data, target = create_training_data()
    OUTPUT_DIR.mkdir(exist_ok=True)  # Membuat folder output jika belum ada.
    summary = []
    # Setiap subplot menunjukkan pengaruh satu nilai learning rate terhadap MSE.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for axis, learning_rate in zip(axes.flat, LEARNING_RATES):
        # Model baru dibuat untuk setiap mu agar hasil percobaan tidak tercampur.
        errors, model = train_curve(data, target, learning_rate)
        axis.plot(np.arange(1, EPOCHS + 1), errors, color="#c2410c")  # Kurva MSE.
        axis.set_title(f"Learning rate (mu) = {learning_rate}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE")
        axis.grid(alpha=0.25)
        summary.append(
            {
                "learning_rate": learning_rate,
                "final_mse": errors[-1],  # Error pada epoch terakhir.
                "accuracy": model.score(data, target),  # Akurasi data training.
            }
        )
    fig.suptitle("Assignment 3 - Pengaruh Learning Rate pada 7-Segment")
    fig.tight_layout()
    # Menyimpan enam kurva, satu kurva untuk setiap nilai mu.
    fig.savefig(OUTPUT_DIR / "assignment3_7segment_learning_rate.png", dpi=150)
    plt.close(fig)

    # Mengurutkan hasil dari MSE terkecil dan menyimpannya untuk laporan.
    result = pd.DataFrame(summary).sort_values("final_mse")
    result.to_csv(OUTPUT_DIR / "assignment3_7segment_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nHasil tersimpan di: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()