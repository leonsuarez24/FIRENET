import streamlit as st
from scripts.utils import display_map
import pandas as pd
from pykrige.ok import OrdinaryKriging

APP_TITLE = "FireNet:Plataforma Integrada para la Gestión de Incendios Forestales Utilizando Tecnologías Geoespaciales e Inteligencia Artificial"


def main():

    st.title(APP_TITLE)

    data_temp = pd.read_excel("data/tmean.xlsx")

    st.markdown("### Datos de temperatura")
    st.write(data_temp)

    display_map()


if __name__ == "__main__":
    main()
