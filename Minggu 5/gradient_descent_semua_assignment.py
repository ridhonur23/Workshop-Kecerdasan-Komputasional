import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP

warnings.filterwarnings('ignore')

# ==========================================
# 1. Membaca dataset dan membagi data
# ==========================================

dataset = pd.read_csv('digit.csv')
data = dataset.drop(columns=['Class'])
label = dataset['Class'].astype(int).to_numpy()
label_one_hot = pd.DataFrame(
    np.eye(10, dtype=int)[label],
    columns=[f'class_{digit}' for digit in range(10)]
)
data.to_csv('digit_input.csv', index=False)
label_one_hot.to_csv('digit_label.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=label
)

# ==========================================
# 2. Gradient Descent assignment 1
#    Miu = 0.1, hidden layer berubah
# ==========================================

hidden_units = [3, 5, 10, 15, 20, 25]
loss_assignment1 = {}

for unit in hidden_units:
    model = MLP(
        hidden_layer_sizes=(unit,),
        activation='logistic',
        learning_rate_init=0.1,
        max_iter=5000,
        tol=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    loss_assignment1[unit] = model.loss_curve_

# ==========================================
# 3. Gradient Descent assignment 2
#    Hidden layer = 10, Miu berubah
# ==========================================

nilai_miu = np.round(np.arange(0.1, 1.01, 0.1), 1)
loss_assignment2 = {}

for miu in nilai_miu:
    model = MLP(
        hidden_layer_sizes=(10,),
        activation='logistic',
        learning_rate_init=float(miu),
        max_iter=5000,
        tol=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    loss_assignment2[float(miu)] = model.loss_curve_

# ==========================================
# 4. Gradient Descent assignment 3
#    Hidden layer = 10, Miu = 0.1, epoch berubah
# ==========================================

jumlah_epoch = [100, 200, 300, 500, 1000]
loss_assignment3 = {}

for epoch in jumlah_epoch:
    model = MLP(
        hidden_layer_sizes=(10,),
        activation='logistic',
        learning_rate_init=0.1,
        max_iter=epoch,
        tol=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    loss_assignment3[epoch] = model.loss_curve_

# ==========================================
# 5. Menggambar Gradient Descent assignment 1-3
# ==========================================

figure, axes = plt.subplots(1, 3, figsize=(19, 5))

for unit, loss_curve in loss_assignment1.items():
    axes[0].plot(loss_curve, label=f'{unit} neuron')
axes[0].set_title('Assignment 1: Perubahan Layer')
axes[0].set_xlabel('Iterasi / Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)
axes[0].legend(ncol=2, fontsize=8)

for miu, loss_curve in loss_assignment2.items():
    axes[1].plot(loss_curve, label=f'miu={miu:.1f}')
axes[1].set_title('Assignment 2: Perubahan Miu')
axes[1].set_xlabel('Iterasi / Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2, fontsize=8)

for epoch, loss_curve in loss_assignment3.items():
    axes[2].plot(loss_curve, label=f'{epoch} epoch')
axes[2].set_title('Assignment 3: Perubahan Epoch')
axes[2].set_xlabel('Iterasi / Epoch')
axes[2].set_ylabel('Loss')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

figure.suptitle('Gradient Descent MLP Assignment 1-3', fontsize=15)
figure.tight_layout()
figure.savefig('gradient_descent_assignment1_3.png', dpi=150)
plt.show()

print('Grafik berhasil disimpan sebagai gradient_descent_assignment1_3.png')
