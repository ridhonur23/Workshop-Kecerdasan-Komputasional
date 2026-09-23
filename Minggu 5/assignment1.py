import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
import warnings

warnings.filterwarnings('ignore')



# 1. Membaca dataset training
dataset = pd.read_csv('digit.csv')
data = dataset.drop(columns=['Class'])
label = dataset['Class'].astype(int).to_numpy()
label_one_hot = pd.DataFrame(
    np.eye(10, dtype=int)[label],
    columns=[f'class_{digit}' for digit in range(10)]
)
data.to_csv('digit_input.csv', index=False)
label_one_hot.to_csv('digit_label.csv', index=False)

print("Jumlah data training :", len(data))
print("Jumlah fitur         :", data.shape[1])
print("Kelas                 :", sorted(np.unique(label)))

print("\nData Input:")
print(data.head())

print("\nLabel:")
print(label_one_hot.head())

# 2. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=label
)

# 3. Percobaan jumlah neuron hidden layer
hidden_units = [3, 5, 10, 15, 20, 25]

hasil = []
print("HASIL PERCOBAAN MLP")
for unit in hidden_units:

    print("\nHidden Layer =", unit)

    model = MLP(
        hidden_layer_sizes=(unit,),
        activation='logistic',
        learning_rate_init=0.1,
        max_iter=5000,
        tol=0,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Akurasi training
    train_score = np.mean(
        model.predict(X_train).argmax(axis=1) == y_train.to_numpy().argmax(axis=1)
    )

    # Akurasi testing
    test_score = np.mean(
        model.predict(X_test).argmax(axis=1) == y_test.to_numpy().argmax(axis=1)
    )

    print("Akurasi Training :", train_score)
    print("Akurasi Testing  :", test_score)
    print("Jumlah Iterasi   :", model.n_iter_)
    print("Loss Akhir       :", model.loss_)

    hasil.append({
        'Hidden Unit': unit,
        'Akurasi Training': train_score,
        'Akurasi Testing': test_score,
        'Iterasi': model.n_iter_,
        'Loss': model.loss_
    })

# 4. Menampilkan hasil semua percobaan
hasil_df = pd.DataFrame(hasil)
print("TABEL HASIL")
print(hasil_df.to_string(index=False))

# 5. Membuat model final
# Gunakan 10 neuron hidden layer
model_final = MLP(
    hidden_layer_sizes=(10,),
    activation='logistic',
    learning_rate_init=0.1,
    max_iter=5000,
    tol=0,
    random_state=42
)

# Model final dilatih menggunakan seluruh data training
model_final.fit(data, label_one_hot)

# 6. Membuat dan membaca data input dari file lain
data.head(10).to_csv('digit_uji.csv', index=False)
label_one_hot.head(10).to_csv('digit_uji_label.csv', index=False)
data_uji = pd.read_csv('digit_uji.csv')
print("DATA UJI")
print(data_uji)

# 7. Melakukan prediksi
prediksi = model_final.predict(data_uji).argmax(axis=1)
print("HASIL PREDIKSI")

for i, hasil_prediksi in enumerate(prediksi):
    print(
        "Data ke-", i + 1,
        "diprediksi sebagai digit =",
        hasil_prediksi
    )

# 8. Probabilitas masing-masing kelas
probabilitas = model_final.predict_proba(data_uji)
print("PROBABILITAS PREDIKSI")
for i, prob in enumerate(probabilitas):

    print("\nData ke-", i + 1)

    for kelas, nilai in zip(range(10), prob):
        print(
            "Kelas", kelas,
            "=", round(nilai * 100, 2), "%"
        )

# 9. Grafik loss
plt.figure(figsize=(8, 5))

plt.plot(model_final.loss_curve_)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Gradient Descent - MLP')

plt.show()