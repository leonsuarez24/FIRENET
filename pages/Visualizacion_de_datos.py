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
    st.set_page_config(layout="wide")
    st.image(
        "https://raw.githubusercontent.com/leonsuarez24/FIRENET/refs/heads/main/figs/portada.png",
        caption=None,
        use_column_width=True,
    )
    st.title(APP_TITLE)

    st.markdown(
        "En esta sección se presentan los datos de temperatura y precipitación de la región de Santander, Colombia. Adicionalmente, se presenta una interpolación espacial de los datos de las estaciones meteorológicas y una predicción de los datos de temperatura y precipitación mediante un modelo de aprendizaje profundo basado en redes neuronales recurrentes (RNN) y redes neuronales convolucionales (CNN) para los siguientes 10 meses."
    )
    st.markdown(
        "Adicionalmente, se presenta una visualización de los delitos ambientales en la región de Santander y como se relacionan con los datos de temperatura y precipitación."
    )

    boundaries = "data/santander_boundary.json"

    with open(boundaries, "r") as f:
        region_geojson = json.load(f)

    df_t = pd.read_excel("data/tmean.xlsx")
    df_t["Latitud"] = df_t["Latitud"].round(2)
    df_t["Longitud"] = df_t["Longitud"].round(2)

    df_p = pd.read_excel("data/precipitacion_filtrado.xlsx")
    df_p["Latitud"] = df_p["Latitud"].round(2)
    df_p["Longitud"] = df_p["Longitud"].round(2)

    df_t.columns = df_t.columns.str.strip()
    df_t["Latitud"] = pd.to_numeric(df_t["Latitud"], errors="coerce")
    df_t["Longitud"] = pd.to_numeric(df_t["Longitud"], errors="coerce")
    df_t["Fecha"] = pd.to_datetime(df_t["Fecha"])

    df_p.columns = df_p.columns.str.strip()
    df_p["Latitud"] = pd.to_numeric(df_p["Latitud"], errors="coerce")
    df_p["Longitud"] = pd.to_numeric(df_p["Longitud"], errors="coerce")
    df_p["Fecha"] = pd.to_datetime(df_p["Fecha"])

    min_date = df_t["Fecha"].min()
    max_date = df_t["Fecha"].max()

    start_time = st.slider(
        "Seleccionar fecha",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=datetime(2000, 1, 1),
        format="YYYY-MM",
        key="date_slider",
    )

    st.markdown(f"## Mapas de temperatura y precipitación para {start_time.strftime('%Y-%m')}")

    filtered_df_t = df_t[df_t["Fecha"].dt.strftime("%Y-%m") == start_time.strftime("%Y-%m")]
    filtered_df_p = df_p[df_p["Fecha"].dt.strftime("%Y-%m") == start_time.strftime("%Y-%m")]

    if filtered_df_t.empty and filtered_df_p.empty:
        st.write("No data available for the selected date.")
    else:

        with st.container():
            st.markdown("## Temperatura")
            col1, col2 = st.columns([1, 1])

            with col1:

                st.markdown("### Estaciones de temperatura")

                filtered_df_t["Valor_medio"] = filtered_df_t["Valor_medio"].round(2)
                filtered_df_p["Valor"] = filtered_df_p["Valor"].round(2)

                fig_temp = px.scatter_mapbox(
                    filtered_df_t,
                    lat="Latitud",
                    lon="Longitud",
                    color="Valor_medio",
                    size="Valor_medio",
                    hover_name="Valor_medio",
                    size_max=15,
                    zoom=7,
                    color_continuous_scale="Reds",
                    mapbox_style="carto-positron",
                )

                fig_temp.update_layout(
                    coloraxis_colorbar_title="Temperatura media (°C)", height=750
                )

                st.plotly_chart(fig_temp, key="temperatura_media")

            with col2:
                st.markdown("### Interpolación de temperatura")
                plot_maps(
                    start_time,
                    "temperatura",
                )

        with st.container():
            st.markdown("## Precipitación")
            col3, col4 = st.columns([1, 1])
            with col3:
                st.markdown("### Estaciones de precipitación")
                fig_prep = px.scatter_mapbox(
                    filtered_df_p,
                    lat="Latitud",
                    lon="Longitud",
                    color="Valor",
                    size="Valor",
                    hover_name="Valor",
                    size_max=15,
                    zoom=7,
                    color_continuous_scale="Blues",
                    mapbox_style="carto-positron",
                )

                fig_prep.update_layout(coloraxis_colorbar_title="Precipitación (mm)", height=750)
                st.plotly_chart(fig_prep, key="precipitacion")

            with col4:
                st.markdown("### Interpolación de precipitación")


                plot_maps(
                    start_time,
                    "precipitacion",
                )

        st.markdown("## Visualización de delitos ambientales")

        with st.container():
            col7, col8, col9 = st.columns([1, 1, 1])

            with col7:

                st.image(
                    "data/delitos/delitos_prec_santander.png",
                    caption=None,
                    use_column_width=True,
                )

            with col8:

                st.image(
                    "data/delitos/delitos_temp_santander.png",
                    caption=None,
                    use_column_width=True,
                )

            with col9:
                delitos_df = pd.read_csv("data/delitos/delitos_santander_total.csv")
                st.write(delitos_df)

        st.markdown(
            "## Predicción de datos de temperatura y precipitación para los siguientes 10 meses"
        )
        st.markdown(
            "Esta predicción se basa en los datos históricos de temperatura y precipitación de la región de Santander de los últimos 24 años. Se utilizó un modelo de aprendizaje profundo basado en redes neuronales recurrentes (RNN) y redes neuronales convolucionales (CNN)."
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
