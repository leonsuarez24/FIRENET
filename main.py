import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk


APP_TITLE = "GEOHIDRO: Plataforma de Diagnóstico y Visualización del Estado de Recursos Hídricos de Santander"


def main():

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🌧️",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title(APP_TITLE)
    st.divider()

    st.markdown(
        "GEOHIDRO es una plataforma de diagnóstico y visualización del estado de los recursos hídricos en Santander. Permite a los usuarios visualizar datos espacio-temporales sobre las precipitaciones, la temperatura y la vulnerabilidad de los recursos hídricos en el departamento de Santander. GEOHIDRO utiliza modelos de inteligencia artificial para interpolar de manera espacio-temporal la información de las estaciones meteorológicas."
    )


if __name__ == "__main__":
    main()
