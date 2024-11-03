import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

columns = ["CodigoEstacion", "Latitud", "Longitud", "Altitud", "Fecha", "Valor"]


def write_to_file(file_path: str, message: str, mode: str = "a"):
    with open(file_path, mode) as file:
        file.write(message + "\n")


def processing_temp_data(data_path: str, columns: list) -> pd.DataFrame:
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    log_path = f"logs/{file_name}.txt"
    os.makedirs("logs", exist_ok=True)

    write_to_file(log_path, f"Loading data from {data_path}", mode="w")

    data = pd.read_excel(data_path)
    write_to_file(log_path, f"Original dataset size: {data.shape}")

    data = data[columns]
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

    write_to_file(log_path, "Starting temperature data filtering", mode="w")

    tmin_min = tmin["Valor"].min()
    tmax = tmax.drop(tmax[tmax["Valor"] < tmin_min].index)
    write_to_file(log_path, f"New dataset size with filtering tmax < tmin: {tmax.shape}")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].hist(tmax["Valor"], color="blue")
    ax[0].set_title("Histograma temperatura máxima")
    ax[0].set_xlabel("Temperatura °C")
    ax[1].hist(tmin["Valor"], color="red")
    ax[1].set_title("Histograma temperatura mínima")
    ax[1].set_xlabel("Temperatura °C")
    plt.savefig(f"logs/histogram.png")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_title("Boxplot temperatura máxima")
    ax[0].set_ylabel("Temperatura °C")
    ax[0].boxplot(tmax["Valor"], patch_artist=True)
    ax[1].set_title("Boxplot temperatura mínima")
    ax[1].set_ylabel("Temperatura °C")
    ax[1].boxplot(tmin["Valor"], patch_artist=True)
    plt.savefig(f"logs/boxplot.png")

    return tmax, tmin


def compute_mean_temperature(tmax: pd.DataFrame, tmin: pd.DataFrame) -> pd.DataFrame:
    log_path = "logs/temperature.txt"
    os.makedirs("logs", exist_ok=True)

    merged_data = pd.merge(
        tmax,
        tmin,
        on=["CodigoEstacion", "Latitud", "Longitud", "Altitud", "Fecha"],
        suffixes=("_max", "_min"),
    )

    write_to_file(log_path, f"Merged dataset size: {merged_data.shape}", mode="a")

    merged_data["Valor_medio"] = (merged_data["Valor_max"] + merged_data["Valor_min"]) / 2
    write_to_file(log_path, "Computed Valor_medio for each matching row", mode="a")

    result = merged_data

    write_to_file(log_path, f"Result dataset size: {result.shape}", mode="a")
    write_to_file(log_path, "\nDescriptive statistics valor min:")
    write_to_file(log_path, str(result["Valor_min"].describe()))
    write_to_file(log_path, "\nDescriptive statistics valor medio:")
    write_to_file(log_path, str(result["Valor_medio"].describe()))
    write_to_file(log_path, "\nDescriptive statistics valor max:")
    write_to_file(log_path, str(result["Valor_max"].describe()))

    return result


def process_temp_data():
    tmax_data = processing_temp_data("data/tmax.xlsx", columns)
    tmin_data = processing_temp_data("data/tmin.xlsx", columns)

    filtered_tmax, filtered_tmin = filter_temp_data(tmax_data, tmin_data)
    mean_temperature = compute_mean_temperature(filtered_tmax, filtered_tmin)
    return mean_temperature
