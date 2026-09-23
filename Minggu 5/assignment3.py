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

print("Jumlah data  :", len(data))
print("Jumlah fitur :", data.shape[1])
print("Kelas        :", sorted(np.unique(label)))

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=label
)

# Jumlah epoch yang diuji
jumlah_epoch = [100, 200, 300, 500, 1000]

hasil = []

print("HASIL PERCOBAAN MLP")
print("Hidden Layer = 10 Neuron")
print("Miu = 0.1")

for epoch in jumlah_epoch:

    model = MLP(
        hidden_layer_sizes=(10,),
        activation='logistic',
        learning_rate_init=0.1,
        max_iter=epoch,
        tol=0,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Menghitung akurasi
    akurasi_train = np.mean(
        model.predict(X_train).argmax(axis=1) == y_train.to_numpy().argmax(axis=1)
    )
    akurasi_test = np.mean(
        model.predict(X_test).argmax(axis=1) == y_test.to_numpy().argmax(axis=1)
    )

    hasil.append({
        'Epoch': epoch,
        'Akurasi Training': akurasi_train,
        'Akurasi Testing': akurasi_test,
        'Iterasi Aktual': model.n_iter_,
        'Loss Akhir': model.loss_
    })

    print("\nEpoch =", epoch)
    print("Akurasi Training :", akurasi_train)
    print("Akurasi Testing  :", akurasi_test)
    print("Iterasi Aktual   :", model.n_iter_)
    print("Loss Akhir       :", model.loss_)

# Tabel hasil
hasil_df = pd.DataFrame(hasil)

print("\nTABEL HASIL")

print(hasil_df.to_string(index=False))

# Grafik akurasi
plt.figure(figsize=(8, 5))

plt.plot(
    hasil_df['Epoch'],
    hasil_df['Akurasi Training'],
    marker='o',
    label='Training'
)

plt.plot(
    hasil_df['Epoch'],
    hasil_df['Akurasi Testing'],
    marker='o',
    label='Testing'
)

plt.xlabel('Jumlah Epoch')
plt.ylabel('Akurasi')
plt.title('Pengaruh Jumlah Epoch terhadap Akurasi MLP')
plt.legend()
plt.grid()
plt.show()

# Grafik loss
plt.figure(figsize=(8, 5))

plt.plot(
    hasil_df['Epoch'],
    hasil_df['Loss Akhir'],
    marker='o'
)

plt.xlabel('Jumlah Epoch')
plt.ylabel('Loss')
plt.title('Pengaruh Jumlah Epoch terhadap Loss MLP')
plt.grid()
plt.show()