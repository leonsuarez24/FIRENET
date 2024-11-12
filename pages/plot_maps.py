import streamlit as st
from scripts.utils import display_map_temp_precip
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime
import os

APP_TITLE = "FireNet:Plataforma Integrada para la Gestión de Incendios Forestales Utilizando Tecnologías Geoespaciales e Inteligencia Artificial"


def main():

    st.title(APP_TITLE)

    data_temp = pd.read_excel("data/tmean.xlsx")

    st.markdown("### Datos de temperatura")
    st.write(data_temp)

    data_prep = pd.read_excel("data/precipitacion_filtrado.xlsx")
    st.markdown("### Datos de precipitación")
    st.write(data_prep)

    display_map_temp_precip()




if __name__ == "__main__":
    main()
