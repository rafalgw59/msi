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


# kolumny
colnames=['data', 'otwarcie', 'najwyzszy', 'najnizszy', 'zamkniecie','wolumen']

def create_dataset(dataset, k, n):
    X, Y = [], []
    for i in range(len(dataset) - k - n ):
        a = dataset[i:(i + k)]
        X.append(a)
        Y.append(dataset[i + k : i + k + n])
    return np.array(X), np.array(Y)

def generate_predictions(model,dataset,k,n,Dmax,Dmin):
    predictions = []
    for i in range(0, len(dataset) - k - n + 1, n):
        current_window = dataset[i:i + k].reshape(1, k)
        predicted = model.predict(current_window).flatten()
        predictions.extend(predicted)

    last_window_start = len(dataset) - k - n
    last_window = dataset[last_window_start:last_window_start + k].reshape(1, k)
    last_prediction = model.predict(last_window).flatten()
    predictions.extend(last_prediction[:len(dataset) - last_window_start - k])


    #sse
    pred_len = len(predictions)
    delta = dataset[-pred_len:]-predictions
    delta_shape = delta.shape
    delta = delta.flatten()
    suma=0
    sse_msi = 0
    for i in range(delta_shape[0]):
        suma = suma + delta[i]*delta[i]
    sse_msi = suma/delta_shape[0]

    print(f'SSE metoda z msi: {sse_msi}')

    dataset_scaled = (dataset * (Dmax - Dmin)) + Dmin
    predictions_scaled = (np.array(predictions) * (Dmax - Dmin)) + Dmin

    dataset_scaled_flatten = dataset_scaled.flatten()
    predictions_scaled_flatten = predictions_scaled.flatten()

    #cały wykres
    plt.figure(figsize=(17, 10))
    plt.plot(dataset_scaled_flatten, label='Rzeczywiste wartości', color='blue')
    plt.plot(range(k, k + len(predictions_scaled)), predictions_scaled_flatten, label='Predykcja', color='red', linestyle='--')
    plt.title(f'Predykcja kursu akcji, dla k={k}, SSE: {sse_msi}')
    plt.xlabel('Próbki')
    plt.ylabel('Cena')
    plt.legend()
#    plt.show()

    #przybliżenie (ostatnie 70 próbek)
    total_samples = k+n

    last_samples = dataset[-total_samples:]
    last_samples_scaled = (last_samples * (Dmax - Dmin)) + Dmin
    ts = np.arange(len(dataset) - total_samples, len(dataset))

    plt.figure(figsize=(17,10))
    plt.plot(ts[:k+n], last_samples_scaled[:k+n], label=f'Ostatnie {n+k} rzeczywistych wartości', color='blue')
    plt.plot(ts[k:], predictions_scaled_flatten[-n:], label=f'Predykcja kolejnych {n} wartości', color='red', linestyle='--')
    plt.title(f'Predykcja kursu akcji - ostatnie 70 próbke, dla k={k}, SSE:{sse_msi}')
    plt.xlabel('Próbki')
    plt.ylabel('Cena')
    plt.legend()
    plt.show()

def prediction(model,dataset,to_predict,k,n,Dmax,Dmin,target_Dmax,target_Dmin):
    predictions = []
    for i in range(0, len(dataset) - k - n + 1, n):
        current_window = to_predict[i:i + k].reshape(1, k)
        predicted = model.predict(current_window).flatten()
        predictions.extend(predicted)

    last_window_start = len(dataset) - k - n
    last_window = to_predict[last_window_start:last_window_start + k].reshape(1, k)
    last_prediction = model.predict(last_window).flatten()
    predictions.extend(last_prediction[:len(dataset) - last_window_start - k])

    # sse
    pred_len = len(predictions)
    delta = dataset[-pred_len:] - predictions
    delta_shape = delta.shape
    delta = delta.flatten()
    suma = 0
    sse_msi = 0
    for i in range(delta_shape[0]):
        suma = suma + delta[i] * delta[i]
    sse_msi = suma / delta_shape[0]

    print(f'SSE metoda z msi: {sse_msi}')

    dataset_scaled = (dataset * (Dmax - Dmin)) + Dmin
    to_predict_scaled = (to_predict * (target_Dmax-target_Dmin)) + target_Dmin
    predictions_scaled = (np.array(predictions) * (target_Dmax-target_Dmin)) + target_Dmin

    dataset_scaled_flatten = dataset_scaled.flatten()
    to_predict_scaled_flatten = to_predict_scaled.flatten()
    predictions_scaled_flatten = predictions_scaled.flatten()

    # cały wykres
    plt.figure(figsize=(17, 10))
    plt.plot(dataset_scaled_flatten, label='Średnie dane koszyka', color='blue')
    plt.plot(range(k, k + len(predictions_scaled)), predictions_scaled_flatten, label='Predykcje', color='red',
             linestyle='--')
    plt.plot(to_predict_scaled_flatten, label='Rzeczywiste dane szukanej spółki', color='green', linestyle='--')
    plt.title(f'Predykcja kursu akcji jednej spółki na podstawie średniej wartości danych koszyka spółek,dla k={k} SSE: {sse_msi}')
    plt.xlabel('Próbki')
    plt.ylabel('Cena')
    plt.legend()
    plt.show()

    # przybliżenie (ostatnie 70 próbek)
    total_samples = k + n

    last_samples = dataset[-total_samples:]
    last_samples_scaled = (last_samples * (Dmax - Dmin)) + Dmin
    ts = np.arange(len(dataset) - total_samples, len(dataset))

    plt.figure(figsize=(17, 10))
    plt.plot(ts[:k + n], last_samples_scaled[:k + n], label=f'Ostatnie {n+k} średnich wartości koszyka', color='blue')
    plt.plot(ts[k:], predictions_scaled_flatten[-n:], label=f'Predykcja {n} kolejnych wartości szukanej spółki', color='red', linestyle='--')
    plt.plot(ts[k:], to_predict_scaled_flatten[-n:], label =f'{k} kolejnych wartości rzeczywistych szukanej spółki', color='green', linestyle='--')
    plt.title(f'Predykcja kursu akcji jednej spółki na podstawie średniej wartości danych koszyka spółek - ostatnie {n+k} próbke, dla k={k}, SSE:{sse_msi}')
    plt.xlabel('Próbki')
    plt.ylabel('Cena')
    plt.legend()
    plt.show()

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

    #sieć neuronowa
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(l_neuro1, activation='relu', input_shape=(k,))) #warstwa 1 liczba wejsc
    model.add(Dense(l_neuro2, activation='relu')) #wartwa 2
    model.add(Dense(n)) #liczba wyjść

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')
    generate_predictions(model,dataset,k,n,Dmax,Dmin)
    #generate_last_n(model,dataset,n,k,Dmax,Dmin)



def b(l_neuro1, l_neuro2):
    filenames = ['cdr_d.csv', '3rg_d.csv', '11b_d.csv', 'alg_d.csv', 'art_d.csv', 'bbt_d.csv']
    comp_names = ['CD-Project', '3R Games', '11 bit studios', 'All in! Games', 'Artifex Mundi', 'BoomBit']

    aggregated_data = pd.DataFrame()

    for i, filename in enumerate(filenames):
        df = pd.read_csv(filename, names=colnames)
        df = df.tail(1000)  # Wybieramy ostatnie 1000 rekordów
        df = df.drop(['najwyzszy', 'najnizszy', 'zamkniecie', 'wolumen'], axis=1)
        df["otwarcie"] = pd.to_numeric(df["otwarcie"], errors="coerce")  # Konwersja na numeryczny typ danych

        # Zmiana nazwy kolumny 'otwarcie' aby uniknąć konfliktów przy scalaniu
        df.rename(columns={'otwarcie': f'otwarcie_{comp_names[i]}'}, inplace=True)

        # Scalenie danych
        if aggregated_data.empty:
            aggregated_data = df
        else:
            aggregated_data = aggregated_data.merge(df, on='data', how='inner')



    # Uśrednianie wartości otwarcia
    aggregated_data['otwarcie_avg'] = aggregated_data[[col for col in aggregated_data.columns if 'otwarcie_' in col]].mean(axis=1)

    dataset = aggregated_data['otwarcie_avg'].to_numpy()

    # Normalizacja
    Dmax = dataset.max()
    Dmin = dataset.min()
    dataset = (dataset - Dmin) / (Dmax - Dmin)

    # wczytanie danych szukanej spółki
    cdr = pd.read_csv('cdr_d.csv', names=colnames)
    cdr_data = cdr.tail(len(dataset))
    cdr_data = cdr_data.drop(['najwyzszy', 'najnizszy', 'zamkniecie', 'wolumen'], axis=1)
    cdr_data["otwarcie"] = pd.to_numeric(cdr_data["otwarcie"], errors="coerce")

    #normalizacja cdr
    cdr_Dmax = cdr_data["otwarcie"].max()
    cdr_Dmin = cdr_data["otwarcie"].min()
    cdr_data["otwarcie"] = (cdr_data["otwarcie"] - cdr_Dmin) / (cdr_Dmax - cdr_Dmin)
    cdr_dataset = cdr_data["otwarcie"].to_numpy()

    # Tworzenie zbioru danych
    X, Y = create_dataset(dataset, k, n)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Budowa i trening modelu
    model = Sequential()
    model.add(Dense(l_neuro1, activation='relu', input_shape=(k,)))
    model.add(Dense(l_neuro2, activation='relu'))
    model.add(Dense(n))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')

    # Przewidywania
    cdr_values_to_predict = cdr_dataset[-k:].reshape(1, k)
    cdr_real_values = cdr_dataset[-n:].reshape(1,n)

    koszyk_values = dataset[-k:].reshape(1, k)
    predicted_prices = model.predict(cdr_values_to_predict)

    # sse
    pred_len = len(predicted_prices)
    delta = cdr_real_values[-pred_len:] - predicted_prices
    delta_shape = delta.shape
    delta = delta.flatten()
    suma = 0
    sse_msi = 0
    for i in range(delta_shape[0]):
        suma = suma + delta[i] * delta[i]
    sse_msi = suma / delta_shape[0]

    print(f'SSE metoda z msi: {sse_msi}')
    prediction(model,dataset,cdr_dataset,k,n,Dmax,Dmin,cdr_Dmax,cdr_Dmin)
    # #denormalizacja
    # cdr_values_to_predict = cdr_values_to_predict * (cdr_Dmax - cdr_Dmin) + cdr_Dmin
    # cdr_real_values = cdr_real_values * (cdr_Dmax - cdr_Dmin) + cdr_Dmin
    # koszyk_values = koszyk_values * (Dmax - Dmin) + Dmin
    # predicted_prices = predicted_prices * (Dmax - Dmin) + Dmin
    #
    #
    # cdr_values_flat = cdr_values_to_predict.flatten()
    # cdr_real_values_flat = cdr_real_values.flatten()
    # koszyk_values_flat = koszyk_values.flatten()
    # predicted_prices_flat = predicted_prices.flatten()
    #
    # #wykres
    # plt.figure(figsize=(15,6))
    # plt.plot(dataset.flatten(), label='Średnie wartości koszyka', color='blue')
    # plt.plot(cdr_dataset.flatten(), label='Wartości CD-Project', color='green')
    # plt.plot()
    #
    #
    # # Wykres przyblizenie (70 wartosci)
    # time_sequence = np.arange(len(koszyk_values_flat) + len(predicted_prices_flat))
    # combined_sequence = np.concatenate((koszyk_values_flat, predicted_prices_flat))
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_sequence[:len(koszyk_values_flat)], koszyk_values_flat, 'b-', label=f'Ostatnie {k} średnich wartości koszyka')
    # plt.plot(time_sequence[len(koszyk_values_flat):], predicted_prices_flat, 'r--', label=f'Następne {n} wartości CD-Project (predykcja)')
    # plt.plot(time_sequence[len(koszyk_values_flat):], cdr_real_values_flat, 'g--', label=f'Następne {n} wartości CD-Project (rzeczywiste)')
    #
    # plt.title('Przewidywana średnia cena akcji')
    # plt.xlabel('Czas')
    # plt.ylabel('Cena')
    # plt.legend()
    # plt.show()




def start(podpunkt,l_neuro1,l_neuro2):
    if podpunkt == "a":
        a(l_neuro1,l_neuro2)
    elif podpunkt == "b":
        b(l_neuro1,l_neuro2)


start("a",64,32)