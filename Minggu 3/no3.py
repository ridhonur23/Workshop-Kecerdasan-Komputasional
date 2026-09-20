import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('XOR.csv')
data = dataset.iloc[:, 0:-1]
label = dataset.iloc[:, -1]

iter = np.array([10, 20, 30, 40, 50, 100, 150, 200, 250, 300])


def mse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual, pred)).mean()


# Parameter yang akan diuji
hidden_layer_sizes = [(2,), (4,), (7,), (10,)]
activation = ['logistic', 'tanh', 'relu']
learning_rate_init = [0.01, 0.1, 0.5, 1.0]

best_error = float('inf')
best_parameter = None

# Mencari parameter terbaik
for hidden in hidden_layer_sizes:
    for act in activation:
        for lr in learning_rate_init:

            model = MLP(
                hidden_layer_sizes=hidden,
                max_iter=1000,
                activation=act,
                learning_rate_init=lr,
                solver='sgd',
                random_state=42,
                momentum=0.9
            )

            model.fit(data, label)

            pred = model.predict(data)

            error = mse(label, pred)
            score = model.score(data, label)

            if error < best_error:
                best_error = error
                best_parameter = {
                    'hidden_layer_sizes': hidden,
                    'activation': act,
                    'learning_rate_init': lr,
                    'score': score
                }


# Menampilkan parameter terbaik
print('Parameter terbaik:')
print('Hidden Layer       :', best_parameter['hidden_layer_sizes'])
print('Activation         :', best_parameter['activation'])
print('Learning Rate      :', best_parameter['learning_rate_init'])
print('Score / Akurasi    :', best_parameter['score'])
print('Error minimum      :', best_error)


# Menguji pengaruh jumlah iterasi
error = []

for i in range(0, 10):

    model = MLP(
        hidden_layer_sizes=best_parameter['hidden_layer_sizes'],
        max_iter=iter[i],
        activation=best_parameter['activation'],
        learning_rate_init=best_parameter['learning_rate_init'],
        solver='sgd',
        random_state=42,
        momentum=0.9
    )

    model.fit(data, label)

    pred = model.predict(data)

    error.append(mse(label, pred))

    print(
        'Iterasi:', iter[i],
        '| Score:', model.score(data, label),
        '| MSE:', error[i]
    )


# Menampilkan grafik
plt.plot(iter, error, marker='o')
plt.xlabel('Iteration No.')
plt.ylabel('Mean Square Error')
plt.title('Gradient Descent XOR')
plt.show()