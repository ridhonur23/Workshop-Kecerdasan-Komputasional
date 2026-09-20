from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_mlp"
RANDOM_STATE = 42


def load_dataset(filename):
    path = BASE_DIR / filename
    if filename == "iris.csv":
        dataset = pd.read_csv(path, header=None)
        dataset.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Class"]
    else:
        dataset = pd.read_csv(path)

    features = dataset.iloc[:, :-1].apply(pd.to_numeric)
    labels = dataset.iloc[:, -1]
    return dataset, features, labels


def make_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    solver="sgd",
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=80,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def plot_loss(model, name):
    loss = model.named_steps["mlp"].loss_curve_
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(loss) + 1), loss, color="#d95f02", linewidth=2)
    plt.xlabel("Iterasi")
    plt.ylabel("Loss")
    plt.title(f"Gradient Descent - {name.upper()}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"loss_{name}.png", dpi=150)
    plt.close()


def train_dataset(filename):
    name = Path(filename).stem
    dataset, features, labels = load_dataset(filename)
    class_counts = labels.value_counts()
    stratify = labels if class_counts.min() >= 2 else None
    test_size = 0.5 if len(dataset) <= 10 else 0.25

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = make_model()
    parameter_grid = {
        "mlp__hidden_layer_sizes": [(4,), (7,), (16,), (32,)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__learning_rate_init": [0.001, 0.01, 0.1],
        "mlp__alpha": [0.0001, 0.001],
    }

    if len(dataset) <= 10:
        # Cross-validation is unstable when a logical table has only four rows.
        search = GridSearchCV(model, parameter_grid, cv=2, scoring="accuracy", n_jobs=-1)
    else:
        cv = min(5, int(y_train.value_counts().min()))
        search = GridSearchCV(model, parameter_grid, cv=cv, scoring="accuracy", n_jobs=-1)

    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    train_predictions = search.predict(x_train)
    test_accuracy = accuracy_score(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)

    # Refit the selected configuration on the training data to expose its loss curve.
    best_model = search.best_estimator_
    plot_loss(best_model, name)

    result = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "prediction": predictions,
        }
    )
    result.to_csv(RESULT_DIR / f"prediksi_{name}.csv", index=False)

    print(f"\n===== {name.upper()} =====")
    print(f"Jumlah data: {len(dataset)} | fitur: {features.shape[1]}")
    print(f"Parameter terbaik: {search.best_params_}")
    print(f"Akurasi train: {accuracy_score(y_train, train_predictions):.4f}")
    print(f"Akurasi testing: {test_accuracy:.4f}")
    print(f"MSE testing: {test_mse:.4f}")
    print("Actual testing:", y_test.to_numpy())
    print("Prediction     :", predictions)


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    for filename in ["XOR.csv", "digit.csv", "iris.csv", "ruspini.csv"]:
        train_dataset(filename)
    print(f"\nHasil tersimpan di: {RESULT_DIR}")


if __name__ == "__main__":
    main()