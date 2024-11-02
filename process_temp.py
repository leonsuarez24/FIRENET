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

    write_to_file(log_path, "Starting temperature data filtering", mode="w")

    tmin_min = tmin["Valor"].min()
    tmax = tmax.drop(tmax[tmax["Valor"] < tmin_min].index)
    write_to_file(log_path, f"New dataset size with filtering tmax < tmin: {tmax.shape}")

    unique_stations_tmax = tmax["CodigoEstacion"].unique()
    unique_stations_tmin = tmin["CodigoEstacion"].unique()

    write_to_file(log_path, f"Number of unique stations in tmax: {len(unique_stations_tmax)}")
    write_to_file(log_path, f"Number of unique stations in tmin: {len(unique_stations_tmin)}")

    common_stations = np.intersect1d(unique_stations_tmax, unique_stations_tmin)
    write_to_file(log_path, f"Number of common stations: {len(common_stations)}")

    tmax = tmax[tmax["CodigoEstacion"].isin(common_stations)]
    tmin = tmin[tmin["CodigoEstacion"].isin(common_stations)]

    write_to_file(log_path, f"New dataset size after filtering common stations: tmax->{tmax.shape}, tmin->{tmin.shape}")
    
    unique_dates_tmax = tmax["Fecha"].dt.date.unique()
    unique_dates_tmin = tmin["Fecha"].dt.date.unique()

    write_to_file(log_path, f"Number of unique dates in tmax: {len(unique_dates_tmax)}")
    write_to_file(log_path, f"Number of unique dates in tmin: {len(unique_dates_tmin)}")

    common_dates = np.intersect1d(unique_dates_tmax, unique_dates_tmin)
    write_to_file(log_path, f"Number of common dates: {len(common_dates)}")

    tmax = tmax[tmax["Fecha"].dt.date.isin(common_dates)]
    tmin = tmin[tmin["Fecha"].dt.date.isin(common_dates)]

    write_to_file(log_path, f"New dataset size after filtering common dates: tmax->{tmax.shape}, tmin->{tmin.shape}")    

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

    return tmax, tmin


def compute_mean_temperature(tmax: pd.DataFrame, tmin: pd.DataFrame) -> pd.DataFrame:
    log_path = "logs/temperature.txt"
    os.makedirs("logs", exist_ok=True)

    merged_data = pd.merge(
        tmax, tmin, 
        on=["CodigoEstacion", "Fecha"], 
        suffixes=("_max", "_min")
    )
    
    write_to_file(log_path, f"Merged dataset size: {merged_data.shape}", mode="a")
    
    merged_data["Taverage"] = (merged_data["Valor_max"] + merged_data["Valor_min"]) / 2
    write_to_file(log_path, "Computed Taverage for each matching row", mode="a")
    
    result = merged_data[[
        "CodigoEstacion", "Latitud_max", "Longitud_max", "Altitud_max", "Fecha", "Taverage"
    ]]
    result.rename(columns={
        "Latitud_max": "Latitud", 
        "Longitud_max": "Longitud", 
        "Altitud_max": "Altitud"
    }, inplace=True)
    
    return result

    

tmax_data = processing_temp_data("data/tmax.xlsx", columns)
tmin_data = processing_temp_data("data/tmin.xlsx", columns)

filtered_tmax, filtered_tmin = filter_temp_data(tmax_data, tmin_data)





