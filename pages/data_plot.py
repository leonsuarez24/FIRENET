import streamlit as st
from scripts.utils import display_map
import pandas as pd

APP_TITLE = "GEOHIDRO: Plataforma de Diagnóstico y Visualización del Estado de Recursos Hídricos de Santander"


def main():

    st.title(APP_TITLE)

    data_temp = pd.read_excel("data/tmean.xlsx")

    st.markdown("### Datos de temperatura")
    st.write(data_temp)

    display_map()


if __name__ == "__main__":
    main()
