import folium
from streamlit_folium import st_folium
from scripts.process_temp import process_temp_data
import folium
import streamlit as st
from datetime import datetime
import pandas as pd


def krigin_interpolation(data: pd.DataFrame):
    df = pd.read_excel("data/tmean.xlsx")


def display_map():

    df = pd.read_excel("data/tmean.xlsx")

    min_date = df['Fecha'].min()
    max_date = df['Fecha'].max()

    start_time = st.slider(
        "Seleccionar fecha",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=datetime(2020, 1, 1),
        format="YYYY-MM",
    )

    st.write(f"Fecha seleccionada: {start_time.strftime('%Y-%m')}")

    selected_period = start_time.strftime('%Y-%m')
    df['YearMonth'] = df['Fecha'].dt.to_period('M').astype(str)
    df = df[df['YearMonth'] == selected_period]

    map = folium.Map(location=[7, -73.6536], zoom_start=8, tiles="CartoDB positron")
    folium.GeoJson(
        "data/santander.geojson",
        style_function=lambda feature: {
            "weight": 2,
        },
    ).add_to(map)

    
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['Latitud'], row['Longitud']),
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Date: {row['Fecha'].strftime('%Y-%m-%d')}, Valor_medio: {row['Valor_medio']}"
        ).add_to(map)






    st_map = st_folium(map, width=700, height=400)
