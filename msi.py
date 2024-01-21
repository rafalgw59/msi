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

#Parametry
n = 20  # Liczba okresów do przodu (ile do przodu)
k = 50 # rząd predykcji (z ilu próbke do predykcji)


# Wczytanie danych notowań spółki
colnames=['data', 'otwarcie', 'najwyzszy', 'najnizszy', 'zamkniecie','wolumen']

def create_dataset(dataset, k, n):
    X, Y = [], []
    for i in range(len(dataset) - k - n + 1):
        a = dataset[i:(i + k)]
        X.append(a)
        Y.append(dataset[i + k : i + k + n ])
    return np.array(X), np.array(Y)

def a(l_neuro1,l_neuro2):
    df = pd.read_csv('cdr_d.csv', names=colnames)
    data = df.tail(1000)
    data = data.drop(['najwyzszy', 'najnizszy', 'zamkniecie', 'wolumen'], axis=1)

    data["otwarcie"] = pd.to_numeric(data["otwarcie"], errors="coerce")

    # normalizacja
    Dmax = data["otwarcie"].max()
    Dmin = data["otwarcie"].min()
    data["otwarcie"] = (data["otwarcie"] - Dmin) / (Dmax - Dmin)

    dataset = data["otwarcie"].to_numpy()

    X, Y = create_dataset(dataset, k, n)
    print("X:", X)
    print("Y:", Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #sieć neuronowa

    model = Sequential()
    model.add(Dense(l_neuro1, activation='relu', input_shape=(k,))) #warstwa 1 liczba wejsc
    model.add(Dense(l_neuro2, activation='relu')) #wartwa 2
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
    plt.plot(time_sequence[:len(to_predict_flat)], to_predict_flat, 'b-', label=f'Ostatnie {k} wartości')
    plt.plot(time_sequence[len(to_predict_flat):], predicted_prices_flat, 'r--', label=f'Następne {n} wartości')
    plt.title('Akcje CD-Project')
    plt.xlabel('Czas')
    plt.ylabel('Cena')
    plt.legend()
    plt.show()

def b(l_neuro1,l_neuro2):
    filenames = ['cdr_d.csv', '3rg_d.csv', '11b_d.csv', 'alg_d.csv', 'art_d.csv', 'bbt_d.csv']
    comp_names = ['CD-Project', '3R Games', '11 bit studios', 'All in! Games', ' Artifex Mundi', ' BoomBit']
    for i in range(len(filenames)):
        df = pd.read_csv(filenames[i], names=colnames)
        data = df.tail(1000)
        data = data.drop(['najwyzszy', 'najnizszy', 'zamkniecie', 'wolumen'], axis=1)
        data["otwarcie"] = pd.to_numeric(data["otwarcie"], errors="coerce")

        # normalizacja
        Dmax = data["otwarcie"].max()
        Dmin = data["otwarcie"].min()
        data["otwarcie"] = (data["otwarcie"] - Dmin) / (Dmax - Dmin)

        dataset = data["otwarcie"].to_numpy()

        X, Y = create_dataset(dataset, k, n)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # sieć neuronowa

        model = Sequential()
        model.add(Dense(l_neuro1, activation='relu', input_shape=(k,)))  # warstwa 1 liczba wejsc
        model.add(Dense(l_neuro2, activation='relu'))  # wartwa 2
        model.add(Dense(n))  # liczba wyjść

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

        test_loss = model.evaluate(X_test, Y_test)
        print(f'Test Loss: {test_loss}')

        # Przewidywania
        to_predict = dataset[-k:].reshape(1, k)
        predicted_prices = model.predict(to_predict)


        to_predict = to_predict * (Dmax - Dmin) + Dmin
        predicted_prices = predicted_prices * (Dmax - Dmin) + Dmin

        to_predict_flat = to_predict.flatten()
        predicted_prices_flat = predicted_prices.flatten()

        # oś czasu
        time_sequence = np.arange(len(to_predict_flat) + len(predicted_prices_flat))

        # łączenie
        combined_sequence = np.concatenate((to_predict_flat, predicted_prices_flat))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_sequence[:len(to_predict_flat)], to_predict_flat, 'b-', label=f'Ostatnie {k} wartości')
        plt.plot(time_sequence[len(to_predict_flat):], predicted_prices_flat, 'r--', label=f'Następne {n} wartości')
        plt.title(f'Akcje {comp_names[i]}')
        plt.xlabel('Czas')
        plt.ylabel('Cena')
        plt.legend()
        plt.savefig(f"ploty/{comp_names[i]}.jpg")
        #plt.show()

def start(podpunkt,l_neuro1,l_neuro2):
    if podpunkt == "a":
        a(l_neuro1,l_neuro2)
    elif podpunkt == "b":
        b(l_neuro1,l_neuro2)

start("b",64,32)