import pandas as pd
from sklearn.neural_network import MLPClassifier as MLP

# Dataset Digit
datasetDigit = pd.read_csv('digit.csv')
dataDigit = datasetDigit.iloc[:,0:-1]
labelDigit = datasetDigit.iloc[:,-1]
modelDigit = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
modelDigit.fit(dataDigit,labelDigit)

# Dataset Iris
datasetIris = pd.read_csv('iris.csv')
dataIris = datasetIris.iloc[:,0:-1]
labelIris = datasetIris.iloc[:,-1]
model = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
modelIris = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
modelIris.fit(dataIris,labelIris)

# Dataset ruspini
datasetRuspini = pd.read_csv('ruspini.csv')
dataRuspini = datasetRuspini.iloc[:,0:-1]
labelRuspini = datasetRuspini.iloc[:,-1]
model = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
modelRuspini = MLP(hidden_layer_sizes=(4), max_iter=100, activation = 'relu', learning_rate_init=0.1, solver='sgd')
modelRuspini.fit(dataRuspini,labelRuspini)

# mencetak hasil prediksi dan akurasi untuk dataset digit
print('score:', modelDigit.score(dataDigit,labelDigit))
print('predictions:', modelDigit.predict(dataDigit))
print('expected:',labelDigit)
print('========================')

# mencetak hasil prediksi dan akurasi untuk dataset iris
print('score:', modelIris.score(dataIris,labelIris))
print('predictions:', modelIris.predict(dataIris))
print('expected:',labelIris)
print('========================')

# mencetak hasil prediksi dan akurasi untuk dataset ruspini
print('score:', modelRuspini.score(dataRuspini,labelRuspini))
print('predictions:', modelRuspini.predict(dataRuspini))
print('expected:',labelRuspini)
print('========================')  