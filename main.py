import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk


st.set_page_config(
    page_title="GEOHIDRO: Plataforma de Diagnóstico y Visualización del Estado de Recursos Hídricos de Santander",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.sidebar.title("Panel de navegación")


st.title(
    "GEOHIDRO: Plataforma de Diagnóstico y Visualización del Estado de Recursos Hídricos de Santander"
)

st.divider()

st.markdown(
    "GEOHIDRO es una plataforma de diagnóstico y visualización del estado de los recursos hídricos en Santander. Permite a los usuarios visualizar datos espacio-temporales sobre las precipitaciones, la temperatura y la vulnerabilidad de los recursos hídricos en el departamento de Santander. GEOHIDRO utiliza modelos de inteligencia artificial para interpolar de manera espacio-temporal la información de las estaciones meteorológicas."
)

chart_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=["lat", "lon"],
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=chart_data,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=chart_data,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)
