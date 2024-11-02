import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

columns = ["CodigoEstacion", "Latitud", "Longitud", "Altitud", "Fecha", "Valor"]


def processing_temp_data(data_path: str, columns: list) -> pd.DataFrame:

    file_name = data_path.split("/")[-1]
    file_name = file_name.split(".")[0]

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        filename=f"logs/{file_name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info(f"loading data from {data_path}")

    data = pd.read_excel(data_path)
    logging.info(f"orginal dataset size: {data.shape}")

    data = data[columns]
    max_value = data["Valor"].max()
    min_value = data["Valor"].min()

    logging.info(f"max value: {max_value}")
    logging.info(f"min value: {min_value}")
    logging.info(f"Number of NaN values: {data['Valor'].isna().sum()}")
    logging.info("Deleting NaN values ...")
    data = data.dropna(subset=["Valor"])
    logging.info(f"new dataset size: {data.shape}")

    logging.info("Descriptive statistics ...")
    logging.info(data["Valor"].describe())

    data["Fecha"] = pd.to_datetime(data["Fecha"], format="%m/%d/%Y %H:%M")
    return data


def filter_temp_data(tmax: pd.DataFrame, tmin: pd.DataFrame) -> pd.DataFrame:

    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        filename="logs/temperature.log",
        level=logging.INFO,
        format="%(message)s",
    )

    tmin_min = tmin["Valor"].min()
    tmax = tmax.drop(tmax[tmax["Valor"] < tmin_min].index)
    logging.info(f"new dataset size with filter: {tmax.shape}")

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
    bplot = ax[0].boxplot(tmax["Valor"], patch_artist=True)  # fill with color
    ax[1].set_title("Boxplot temperatura mínima")
    ax[1].set_ylabel("Temperatura °C")
    bplot = ax[1].boxplot(tmin["Valor"], patch_artist=True)  # fill with color
    plt.savefig(f"logs/boxplot.png")
    plt.show()

    return tmax

tmax_data = processing_temp_data("data/tmax.xlsx", columns)
tmin_data = processing_temp_data("data/tmin.xlsx", columns)

filtered_data = filter_temp_data(tmax_data, tmin_data)
