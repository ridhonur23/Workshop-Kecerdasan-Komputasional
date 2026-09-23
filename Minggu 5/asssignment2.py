import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
import warnings

warnings.filterwarnings('ignore')

# Membaca dataset
dataset = pd.read_csv('digit.csv')
data = dataset.drop(columns=['Class'])
label = dataset['Class'].astype(int).to_numpy()
label_one_hot = pd.DataFrame(
    np.eye(10, dtype=int)[label],
    columns=[f'class_{digit}' for digit in range(10)]
)
data.to_csv('digit_input.csv', index=False)
label_one_hot.to_csv('digit_label.csv', index=False)

print("Jumlah data :", len(data))
print("Jumlah fitur :", data.shape[1])
print("Kelas :", sorted(np.unique(label)))

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=label
)

# Percobaan nilai miu
nilai_miu = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0
]

hasil = []

print("\n======================================")
print("HASIL PERCOBAAN MLP")
print("Hidden Layer = 10 Neuron")
print("======================================")

for miu in nilai_miu:

    model = MLP(
        hidden_layer_sizes=(10,),
        activation='logistic',
        learning_rate_init=miu,
        max_iter=5000,
        tol=0,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Akurasi
    akurasi_train = np.mean(
        model.predict(X_train).argmax(axis=1) == y_train.to_numpy().argmax(axis=1)
    )
    akurasi_test = np.mean(
        model.predict(X_test).argmax(axis=1) == y_test.to_numpy().argmax(axis=1)
    )

    hasil.append({
        'Miu': miu,
        'Akurasi Training': akurasi_train,
        'Akurasi Testing': akurasi_test,
        'Iterasi': model.n_iter_,
        'Loss Akhir': model.loss_
    })

    print("\nMiu =", miu)
    print("Akurasi Training :", akurasi_train)
    print("Akurasi Testing  :", akurasi_test)
    print("Jumlah Iterasi   :", model.n_iter_)
    print("Loss Akhir       :", model.loss_)

# Menampilkan tabel hasil
hasil_df = pd.DataFrame(hasil)

print("\n======================================")
print("TABEL HASIL")
print("======================================")
print(hasil_df.to_string(index=False))

# Grafik akurasi
plt.figure(figsize=(8, 5))

plt.plot(
    hasil_df['Miu'],
    hasil_df['Akurasi Training'],
    marker='o',
    label='Training'
)

plt.plot(
    hasil_df['Miu'],
    hasil_df['Akurasi Testing'],
    marker='o',
    label='Testing'
)

plt.xlabel('Miu (Learning Rate)')
plt.ylabel('Akurasi')
plt.title('Pengaruh Miu terhadap Akurasi MLP')
plt.legend()
plt.grid()
plt.show()