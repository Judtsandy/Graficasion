import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, render_template
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Función para cargar y procesar el dataset
def load_and_process_data():
    # Leer el dataset
    df = pd.read_csv("datasets/reduced_dataset.csv")

    # Dividir el DataSet en entrenamiento, validación y test
    def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
        strat = df[stratify] if stratify else None
        train_set, test_set = train_test_split(
            df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
        strat = test_set[stratify] if stratify else None
        val_set, test_set = train_test_split(
            test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
        return train_set, val_set, test_set

    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = train_set.drop('calss', axis=1), train_set['calss']
    X_val, y_val = val_set.drop('calss', axis=1), val_set['calss']
    X_test, y_test = test_set.drop('calss', axis=1), test_set['calss']

    # Codificación de las etiquetas
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    # Escalado de los datos
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_encoded, y_val_encoded, y_test_encoded

# Función para entrenar el modelo y obtener las métricas
def train_model(X_train_scaled, X_val_scaled, X_test_scaled, y_train_encoded, y_val_encoded, y_test_encoded):
    clf_rnd_reg = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    clf_rnd_reg.fit(X_train_scaled, y_train_encoded)

    # Realizar las predicciones
    y_train_pred = clf_rnd_reg.predict(X_train_scaled)
    y_val_pred = clf_rnd_reg.predict(X_val_scaled)
    y_test_pred = clf_rnd_reg.predict(X_test_scaled)

    # Calcular las métricas
    metrics = {
        'mse_train': mean_squared_error(y_train_encoded, y_train_pred),
        'mse_val': mean_squared_error(y_val_encoded, y_val_pred),
        'mse_test': mean_squared_error(y_test_encoded, y_test_pred),
        'mae_train': mean_absolute_error(y_train_encoded, y_train_pred),
        'mae_val': mean_absolute_error(y_val_encoded, y_val_pred),
        'mae_test': mean_absolute_error(y_test_encoded, y_test_pred),
        'r2_train': r2_score(y_train_encoded, y_train_pred),
        'r2_val': r2_score(y_val_encoded, y_val_pred),
        'r2_test': r2_score(y_test_encoded, y_test_pred)
    }

    return metrics, y_train_pred, y_val_pred, y_test_pred

# Función para generar y codificar solo la gráfica de entrenamiento
def generate_train_plot(y_train_encoded, y_train_pred):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_train_encoded, y_train_pred, color='blue', alpha=0.5)
    plt.plot([y_train_encoded.min(), y_train_encoded.max()], [y_train_encoded.min(), y_train_encoded.max()], color='red', linestyle='--')
    plt.title("Entrenamiento: Real vs Predicción")
    plt.xlabel("Real")
    plt.ylabel("Predicción")

    # Guardar la imagen en un buffer
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Codificar la imagen en base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return img_base64

@app.route('/')
def home():
    # Cargar y procesar los datos
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_encoded, y_val_encoded, y_test_encoded = load_and_process_data()

    # Entrenar el modelo y obtener las métricas y predicciones
    metrics, y_train_pred, y_val_pred, y_test_pred = train_model(X_train_scaled, X_val_scaled, X_test_scaled, y_train_encoded, y_val_encoded, y_test_encoded)

    # Generar la gráfica de entrenamiento
    plot_base64 = generate_train_plot(y_train_encoded, y_train_pred)

    # Pasar los resultados a la plantilla
    return render_template('index.html', metrics=metrics, plot=plot_base64)

if __name__ == '__main__':
    app.run(debug=True)
