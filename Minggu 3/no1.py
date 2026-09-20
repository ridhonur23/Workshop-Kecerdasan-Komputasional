import pandas as pd
from sklearn.neural_network import MLPClassifier as MLP

dataset = pd.read_csv('OR.csv')
data = dataset.iloc[:,0:-1]
label = dataset.iloc[:,-1]
model = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
model.fit(data,label)

print('score:', model.score(data,label))
print('predictions:', model.predict(data))
print('expected:',label)