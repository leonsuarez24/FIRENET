import folium
from streamlit_folium import st_folium
from scripts.process_temp import process_temp_data
import folium
import streamlit as st
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
from folium.raster_layers import ImageOverlay
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def get_santander_boundaries():
    shapefile = "data/aoi/Departamento.shp"
    df = gpd.read_file(shapefile)
    santander = df[df["DeNombre"] == "Santander"]
    santander = santander.to_crs("EPSG:4326")
    boundary = santander.boundary
    boundary_geo_json = boundary.to_json()

    with open("data/santander_boundary.json", "w") as f:
        f.write(boundary_geo_json)

    shapefile_2 = "data/aoi/aoi.shp"
    df_2 = gpd.read_file(shapefile_2)
    df_2 = df_2.to_crs("EPSG:4326")
    boundary_2 = df_2.boundary
    boundary_geo_json_2 = boundary_2.to_json()

    with open("data/aoi/aoi_boundary.json", "w") as f:
        f.write(boundary_geo_json_2)


def display_map():

    df = pd.read_excel("data/tmean.xlsx")

    min_date = df["Fecha"].min()
    max_date = df["Fecha"].max()

    start_time = st.slider(
        "Seleccionar fecha",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=datetime(2000, 1, 1),
        format="YYYY-MM",
    )

    st.write(f"Fecha seleccionada: {start_time.strftime('%Y-%m')}")

    selected_period = start_time.strftime("%Y-%m")
    df["YearMonth"] = df["Fecha"].dt.to_period("M").astype(str)
    df = df[df["YearMonth"] == selected_period]

    map = folium.Map(location=[7, -73.6536], zoom_start=8, tiles="CartoDB positron")

    folium.GeoJson(
        "data/santander_boundary.json",
        style_function=lambda feature: {
            "weight": 2,
        },
    ).add_to(map)

    folium.GeoJson(
        "data/aoi/aoi_boundary.json",
        style_function=lambda feature: {
            "weight": 2,
        },
    ).add_to(map)

    boundaries = gpd.read_file("data/aoi/aoi_boundary.json")
    minx, miny, maxx, maxy = boundaries.total_bounds

    image_data = np.load(f"data/tmean_interp/npy/{selected_period}-01.npy")
    norm = mcolors.Normalize(vmin=image_data.min(), vmax=image_data.max())
    colormap = plt.cm.coolwarm
    rgba_img = colormap(norm(image_data))

    img = ImageOverlay(
        image=rgba_img,
        bounds=[[miny, minx], [maxy, maxx]],
        colormap=lambda x: colormap,
        origin="lower",
        opacity=0.5,
    ).add_to(map)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row["Latitud"], row["Longitud"]),
            radius=1,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.6,
            popup=f"Fecha: {row['Fecha'].strftime('%Y-%m-%d')}\nTemperatura media: {np.round(row['Valor_medio'], 2)} °C",
        ).add_to(map)

    st_map = st_folium(map, width=700, height=400)
