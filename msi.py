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
#k = 50 # rząd predykcji (z ilu próbke do predykcji)


# kolumny
colnames=['data', 'otwarcie', 'najwyzszy', 'najnizszy', 'zamkniecie','wolumen']

def create_dataset(dataset, k, n):
    X, Y = [], []
    for i in range(len(dataset) - k - n ):
        a = dataset[i:(i + k)]
        X.append(a)
        Y.append(dataset[i + k : i + k + n])
    return np.array(X), np.array(Y)

def predictions(model,dataset,k,n,Dmax,Dmin):
    #predykcja po okienkach
    predictions = []
    for i in range(0, len(dataset) - k - n + 1, n):
        current_window = dataset[i:i + k].reshape(1, k)
        predicted = model.predict(current_window).flatten()
        predictions.extend(predicted)
    #ostatnie okienko - może nie mieć wystarczającej liczby wartości
    last_window_start = len(dataset) - k - n
    last_window = dataset[last_window_start:last_window_start + k].reshape(1, k)
    last_prediction = model.predict(last_window).flatten()
    predictions.extend(last_prediction[:len(dataset) - last_window_start - k])


    #obliczanie sse metodą z zajęć
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

    #denormalizacja do rzeczywistych wartości
    dataset_scaled = (dataset * (Dmax - Dmin)) + Dmin
    predictions_scaled = (np.array(predictions) * (Dmax - Dmin)) + Dmin
    #spłaszczenie do wykresu
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
    plt.savefig(f"./ploty/CDR_predykcja_k_{k}_cały.png")
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
    plt.savefig(f"./ploty/CDR_predykcja_k_{k}_ostatnie_{k+n}_wartosci.png")
   #plt.show()

def prediction_koszyk(model,dataset,to_predict,k,n,Dmax,Dmin,target_Dmax,target_Dmin):
    #predykcja po okienkach
    predictions = []
    for i in range(0, len(dataset) - k - n + 1, n):
        current_window = to_predict[i:i + k].reshape(1, k)
        predicted = model.predict(current_window).flatten()
        predictions.extend(predicted)
    #ostatnie okienko - może nie mieć wystarczającej liczby wartości
    last_window_start = len(dataset) - k - n
    last_window = to_predict[last_window_start:last_window_start + k].reshape(1, k)
    last_prediction = model.predict(last_window).flatten()
    predictions.extend(last_prediction[:len(dataset) - last_window_start - k])

    # sse metodą z zajęć
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

    #denormalizacja - do rzeczywistych wartości
    dataset_scaled = (dataset * (Dmax - Dmin)) + Dmin
    to_predict_scaled = (to_predict * (target_Dmax-target_Dmin)) + target_Dmin
    predictions_scaled = (np.array(predictions) * (target_Dmax-target_Dmin)) + target_Dmin
    #spłaszczenie do wykresu
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
    #plt.show()
    plt.savefig(f"./ploty/CDR_predykcja_koszyk_k_{k}_cały.png")

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
    #plt.show()
    plt.savefig(f"./ploty/CDR_predykcja_koszyk_k_{k}_ostatnie_{n+k}_wartosci.png")

def a(l_neuro1,l_neuro2,k):
    #wczytywanie danych z csv
    df = pd.read_csv('cdr_d.csv', names=colnames)
    data = df.tail(1000) #ostatnie 1000 wartości
    data = data.drop(['najwyzszy', 'najnizszy', 'zamkniecie', 'wolumen'], axis=1) #usuwanie niepotrzebnych kolumn

    data["otwarcie"] = pd.to_numeric(data["otwarcie"], errors="coerce") #wszystkie wartości numeryczne

    # normalizacja
    Dmax = data["otwarcie"].max()
    Dmin = data["otwarcie"].min()
    data["otwarcie"] = (data["otwarcie"] - Dmin) / (Dmax - Dmin)

    dataset = data["otwarcie"].to_numpy()

    X, Y = create_dataset(dataset, k, n) #okienka
    # print("X:", X)
    # print("Y:", Y)

    #sieć neuronowa

    #split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #model
    model = Sequential()
    model.add(Dense(l_neuro1, activation='relu', input_shape=(k,))) #warstwa 1 liczba wejsc
    model.add(Dense(l_neuro2, activation='relu')) #wartwa 2
    model.add(Dense(n)) #liczba wyjść

    #trenowanie, wagi obliczane w procesie uczenia, optymalizator 'adam'
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    #spradzenie jakości
    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')
    predictions(model, dataset, k, n, Dmax, Dmin)




def b(l_neuro1, l_neuro2,k):
    #nazwy plików z danymi do koszyka
    filenames = ['cdr_d.csv', '3rg_d.csv', '11b_d.csv', 'alg_d.csv', 'art_d.csv', 'bbt_d.csv']
    #nazwy spółek
    comp_names = ['CD-Project', '3R Games', '11 bit studios', 'All in! Games', 'Artifex Mundi', 'BoomBit']

    #wczytywanie wartości
    aggregated_data = pd.DataFrame()

    for i, filename in enumerate(filenames):
        df = pd.read_csv(filename, names=colnames)
        df = df.tail(1000)  # ostatnie 1000 wartości
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

    # Budowa i trening modelu, wagi obliczane w procesie uczenia, optymalizator 'adam'
    model = Sequential()
    model.add(Dense(l_neuro1, activation='relu', input_shape=(k,)))
    model.add(Dense(l_neuro2, activation='relu'))
    model.add(Dense(n))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    test_loss = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')

    #predykcja wartości cdr na podstawie koszyka
    prediction_koszyk(model, dataset, cdr_dataset, k, n, Dmax, Dmin, cdr_Dmax, cdr_Dmin)



# automat
def start(podpunkt,l_neuro1,l_neuro2):
    if podpunkt == "a":
        k_values = [50,25,20,10,100]
        for k in k_values:
            a(l_neuro1,l_neuro2,k)
    elif podpunkt == "b":
        k_values = [50,25,20,10,100]
        for k in k_values:
            b(l_neuro1,l_neuro2,k)


start("a",64,32)

def zadania():
    start("a",64,32)
    start("b",64,32)

zadania()