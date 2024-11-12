import folium
import folium.raster_layers
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
from PIL import Image
from io import BytesIO


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


# def display_map(file_name: str, start_time: datetime, map_width: int, map_height: int, key: str):

#     df = pd.read_excel(file_name)

#     min_date = df["Fecha"].min()
#     max_date = df["Fecha"].max()

#     selected_period = start_time.strftime("%Y-%m")
#     df["YearMonth"] = df["Fecha"].dt.to_period("M").astype(str)
#     df = df[df["YearMonth"] == selected_period]

#     map = folium.Map(location=[7, -73.6536], zoom_start=8, tiles="CartoDB positron")

#     folium.GeoJson(
#         "data/santander_boundary.json",
#         style_function=lambda feature: {"weight": 2, "color": "black"},
#     ).add_to(map)

#     folium.GeoJson(
#         "data/aoi/aoi_boundary.json",
#         style_function=lambda feature: {"weight": 2, "color": "black"},
#     ).add_to(map)

#     boundaries = gpd.read_file("data/aoi/aoi_boundary.json")
#     minx, miny, maxx, maxy = boundaries.total_bounds

#     path = (
#         f"data/tmean_interp_final/npy/temperatura_{selected_period}-01.npy"
#         if key == "tmean"
#         else f"data/precipitacion_interp_final/npy/precipitacion_{selected_period}-01.npy"
#     )
#     image_data = np.load(path)
#     norm = mcolors.Normalize(vmin=image_data.min(), vmax=image_data.max())
#     colormap = plt.cm.coolwarm if key == "tmean" else plt.cm.Blues
#     rgba_img = colormap(norm(image_data))

#     img = ImageOverlay(
#         image=rgba_img,
#         bounds=[[miny, minx], [maxy, maxx]],
#         colormap=lambda x: colormap,
#         origin="lower",
#         opacity=0.5,
#     ).add_to(map)

#     valor = "Valor_medio" if key == "tmean" else "Valor"

#     for _, row in df.iterrows():
#         folium.CircleMarker(
#             location=(row["Latitud"], row["Longitud"]),
#             radius=1,
#             color="black",
#             fill=True,
#             fill_color="black",
#             fill_opacity=0.6,
#             popup=f"Fecha: {row['Fecha'].strftime('%Y-%m-%d')}\nTemperatura media: {np.round(row[valor], 2)} °C",
#         ).add_to(map)

#     fig, ax = plt.subplots(figsize=(0.8, 5))
#     fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

#     cbar = plt.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, aspect=20, shrink=0.5
#     )

#     label = "Temperatura media (°C)" if key == "tmean" else "Precipitación (mm)"
#     cbar.set_label(label)

#     buf = BytesIO()
#     plt.savefig(buf, format="png", transparent=True, bbox_inches="tight")
#     buf.seek(0)

#     col1, col2 = st.columns([map_width / 1000, 0.2])

#     with col1:
#         st_map = st_folium(map, width=map_width, height=map_height, key=key)

#     with col2:
#         st.image(buf, use_column_width=True)
