import pandas as pd
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Wczytanie danych notowań spółki
colnames=['data', 'otwarcie', 'najwyzszy', 'najnizszy', 'zamkniecie','wolumen']
data = pd.read_csv('cdr_d.csv',names=colnames)  # Załóżmy, że dane są w formacie CSV

data=data.drop(['najwyzszy','najnizszy','zamkniecie','wolumen'],axis=1)
print(data)

# normalizacja
Dmax = data["otwarcie"].max()
Dmin = data["otwarcie"].min()
data["otwarcie"]= (data["otwarcie"]-Dmin)/(Dmax-Dmin)
dataset = data["otwarcie"].to_numpy()

#Parametry
n = 20  # Liczba okresów do przodu (ile do przodu)
k = 50 # rząd predykcji (z ilu próbke do predykcji)
#data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def create_dataset(dataset, k, n):
    X, Y = [], []
    for i in range(len(dataset) - k - n + 1):
        a = dataset[i:(i + k)]
        X.append(a)
        Y.append(dataset[i + k : i + k + n ])
    return np.array(X), np.array(Y)

X, Y = create_dataset(dataset, k, n)
print("X:", X)
print("Y:", Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#sieć neuronowa

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(k,))) #warstwa 1 liczba wejsc
model.add(Dense(32, activation='relu')) #wartwa 2
model.add(Dense(n)) #liczba wyjść

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

test_loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss}')

# Przewidywania
to_predict = dataset[-k:].reshape(1,k)
predicted_prices = model.predict(to_predict)
print(predicted_prices)
#mse = tensorflow.keras.losses.mean_squared_error(Y_test,predicted_prices)
#print(f'MSE: {mse}')

to_predict=to_predict*(Dmax-Dmin)+Dmin
predicted_prices = predicted_prices*(Dmax-Dmin)+Dmin

to_predict_flat = to_predict.flatten()
predicted_prices_flat = predicted_prices.flatten()

# oś czasu
time_sequence = np.arange(len(to_predict_flat) + len(predicted_prices_flat))

# łączenie
combined_sequence = np.concatenate((to_predict_flat, predicted_prices_flat))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_sequence[:len(to_predict_flat)], to_predict_flat, 'b-', label='Ostatnie k wartości')
plt.plot(time_sequence[len(to_predict_flat):], predicted_prices_flat, 'r--', label='Następne n wartości')
plt.title('Akcje CD-Project')
plt.xlabel('Czas')
plt.ylabel('Cena')
plt.legend()
plt.show()

