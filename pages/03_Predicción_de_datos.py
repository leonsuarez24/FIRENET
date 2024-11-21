import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import geopandas as gpd
import verde as vd


APP_TITLE = "FireNet"


def main():
    st.set_page_config(layout="wide",
                       page_icon="🔥",
    )
    
    st.image(
        "https://raw.githubusercontent.com/leonsuarez24/FIRENET/refs/heads/main/figs/portada.png",
        caption=None,
        use_column_width=True,
    )
    st.title(APP_TITLE)

    st.markdown("## Predicción de datos")

    st.markdown(
        "En esta sección se presenta una predicción de los datos de temperatura y precipitación mediante un modelo de aprendizaje profundo basado en redes neuronales recurrentes (RNN) y redes neuronales convolucionales (CNN) para los siguientes 10 meses en la región de Santander, Colombia."
    )

    with st.container():

        prediction_time = st.slider(
            "Seleccionar fecha para la predicción",
            min_value=datetime(2024, 11, 1),
            max_value=datetime(2025, 8, 1),
            value=datetime(2024, 11, 1),
            format="YYYY-MM",
            key="date_slider_pred",
        )

        col5, col6 = st.columns([1, 1])

        with col5:

            st.markdown("## Predicción de temperatura")
            plot_prediction(
                prediction_time,
                "temperatura",
            )

        with col6:

            st.markdown("## Predicción de precipitación")

            plot_prediction(
                prediction_time,
                "precipitacion",
            )


def plot_prediction(start_time, data: str):
    region = ((-74.6), (-72.4), (5.5), (8.2))
    spacing = 0.01
    grid = vd.grid_coordinates(region, spacing=spacing)

    # data/precipitacion_prediction\precipitacion_2024-11-01.npy
    path = (
        f"data/precipitacion_prediction/precipitacion_{start_time.strftime('%Y-%m')}-01.npy"
        if data == "precipitacion"
        else f"data/temperature_prediction/temperatura_{start_time.strftime('%Y-%m')}-01.npy"
    )
    grid_temperatura = np.load(path)

    gdf = gpd.read_file("data/aoi/Departamento.shp")
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    santander_gdf = gdf[gdf["DeNombre"] == "Santander"]
    mask = np.load("data/mask.npy")

    fig, _ = plt.subplots(figsize=(10, 10))
    colormap = "coolwarm" if data == "temperatura" else "Blues"
    plt.pcolormesh(grid[0], grid[1], grid_temperatura * mask, cmap=colormap, shading="auto")
    label = "Temperatura (°C)" if data == "temperatura" else "Precipitación (mm)"
    plt.colorbar(label=label)
    santander_gdf.boundary.plot(
        ax=plt.gca(), linewidth=1, edgecolor="black", label="Límites de Santander"
    )
    valor = "Valor_medio" if data == "temperatura" else "Valor"

    buf = BytesIO()
    fig.savefig(buf, format="png",  bbox_inches="tight")
    st.image(buf, caption=None, use_column_width=True)


def plot_maps(start_time, data: str):
    region = ((-74.6), (-72.4), (5.5), (8.2))
    spacing = 0.01
    grid = vd.grid_coordinates(region, spacing=spacing)
    path = (
        f"data/precipitacion_interp_final/npy/precipitacion_{start_time.strftime('%Y-%m')}-01.npy"
        if data == "precipitacion"
        else f"data/tmean_interp_final/npy/temperatura_{start_time.strftime('%Y-%m')}-01.npy"
    )
    grid_temperatura = np.load(path)
    dataset_path = (
        "data/precipitacion_filtrado.xlsx" if data == "precipitacion" else "data/tmean.xlsx"
    )
    dataset = pd.read_excel(dataset_path)
    gdf = gpd.read_file("data/aoi/Departamento.shp")
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    santander_gdf = gdf[gdf["DeNombre"] == "Santander"]
    mask = np.load("data/mask.npy")

    fig, _ = plt.subplots(figsize=(10, 10))
    colormap = "coolwarm" if data == "temperatura" else "Blues"
    plt.pcolormesh(grid[0], grid[1], grid_temperatura * mask, cmap=colormap, shading="auto")
    label = "Temperatura (°C)" if data == "temperatura" else "Precipitación (mm)"
    plt.colorbar(label=label)
    santander_gdf.boundary.plot(
        ax=plt.gca(), linewidth=1, edgecolor="black", label="Límites de Santander"
    )
    valor = "Valor_medio" if data == "temperatura" else "Valor"
    plt.scatter(
        dataset["Longitud"], dataset["Latitud"], c=dataset[valor], cmap=colormap, edgecolor="k", s=5
    )

    buf = BytesIO()
    fig.savefig(buf, format="png",  bbox_inches="tight")
    st.image(buf, caption=None, use_column_width=True)


if __name__ == "__main__":
    main()
