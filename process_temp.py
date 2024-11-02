import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

columns = ["CodigoEstacion", "Latitud", "Longitud", "Altitud", "Fecha", "Valor"]

def write_to_file(file_path: str, message: str):
    with open(file_path, "a") as file:
        file.write(message + "\n")

def processing_temp_data(data_path: str, columns: list) -> pd.DataFrame:
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    log_path = f"logs/{file_name}.txt"
    os.makedirs("logs", exist_ok=True)
    
    write_to_file(log_path, f"Loading data from {data_path}")
    
    data = pd.read_excel(data_path)
    write_to_file(log_path, f"Original dataset size: {data.shape}")
    
    data = data[columns]
    max_value = data["Valor"].max()
    min_value = data["Valor"].min()

    write_to_file(log_path, f"Max value: {max_value}")
    write_to_file(log_path, f"Min value: {min_value}")
    write_to_file(log_path, f"Number of NaN values: {data['Valor'].isna().sum()}")
    write_to_file(log_path, "Deleting NaN values ...")
    data = data.dropna(subset=["Valor"])
    write_to_file(log_path, f"New dataset size: {data.shape}")

    write_to_file(log_path, "Descriptive statistics:")
    write_to_file(log_path, str(data["Valor"].describe()))
    
    data["Fecha"] = pd.to_datetime(data["Fecha"], format="%m/%d/%Y %H:%M")
    return data

def filter_temp_data(tmax: pd.DataFrame, tmin: pd.DataFrame) -> pd.DataFrame:
    log_path = "logs/temperature.txt"
    os.makedirs("logs", exist_ok=True)

    tmin_min = tmin["Valor"].min()
    tmax = tmax.drop(tmax[tmax["Valor"] < tmin_min].index)
    write_to_file(log_path, f"New dataset size with filter: {tmax.shape}")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].hist(tmax["Valor"], color="blue")
    ax[0].set_title("Histograma temperatura máxima")
    ax[0].set_xlabel("Temperatura °C")
    ax[1].hist(tmin["Valor"], color="red")
    ax[1].set_title("Histograma temperatura mínima")
    ax[1].set_xlabel("Temperatura °C")
    plt.savefig(f"logs/histogram.png")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_title("Boxplot temperatura máxima")
    ax[0].set_ylabel("Temperatura °C")
    ax[0].boxplot(tmax["Valor"], patch_artist=True)
    ax[1].set_title("Boxplot temperatura mínima")
    ax[1].set_ylabel("Temperatura °C")
    ax[1].boxplot(tmin["Valor"], patch_artist=True)
    plt.savefig(f"logs/boxplot.png")
    plt.show()

    return tmax

tmax_data = processing_temp_data("data/tmax.xlsx", columns)
tmin_data = processing_temp_data("data/tmin.xlsx", columns)

filtered_data = filter_temp_data(tmax_data, tmin_data)
