import streamlit as st
from scripts.utils import display_map
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime
import os

APP_TITLE = "FireNet:Plataforma Integrada para la Gestión de Incendios Forestales Utilizando Tecnologías Geoespaciales e Inteligencia Artificial"


def main():

    st.title(APP_TITLE)

    df = pd.read_excel("data/tmean.xlsx")
    min_date = df["Fecha"].min()
    max_date = df["Fecha"].max()

    start_time = st.slider(
        "Seleccionar fecha",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=datetime(2000, 1, 1),
        format="YYYY-MM",
        key="date_slider",
    )

    st.write(f"Fecha seleccionada: {start_time.strftime('%Y-%m')}")

    col1, col2 = st.columns(2)

    with col1:
        display_map("data/tmean.xlsx", start_time=start_time, map_width=600, map_height=400, key="tmean")

    with col2:
        display_map("data/tmean.xlsx", start_time=start_time, map_width=600, map_height=400, key="pmean")


if __name__ == "__main__":
    main()
