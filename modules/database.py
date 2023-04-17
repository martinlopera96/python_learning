import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
import os


class DataBase:
    def __init__(self):
        pass

    def data_proc(file_name):

        # Va a quedar el directorio python_learning/data

        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

        # Añade el file_dame al path

        excel_path = os.path.join(data_dir, file_name)

        # Leer archivo de excel, pasarlo a un np.array y luego dividirlo por columnas

        data_raw = pd.read_excel(excel_path, sheet_name=0)
        data_array = np.array(data_raw)
        data_array = np.vstack((data_array))

        # Coger todos los datos de las columnas requeridas para entrenamiento (features) y predicción(labels)
        train_features = data_array[:, 1:-1]
        train_labels = data_array[:, -1]

        # Normalizar datos

        train_features_norm = ss.fit_transform(train_features)
        train_labels_norm = ss.fit_transform(train_labels)

        X_train, X_test, Y_train, Y_test = tts(train_features_norm, train_labels_norm, test_size=0.1)
        return X_train, X_test, Y_train, Y_test