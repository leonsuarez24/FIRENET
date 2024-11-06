import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk


APP_TITLE = "FireNet:Plataforma Integrada para la Gestión de Incendios Forestales Utilizando Tecnologías Geoespaciales e Inteligencia Artificial"


def main():

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🌧️",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title(APP_TITLE)
    st.divider()

    st.markdown("FireNet es una plataforma de diagnóstico y visualización de...")


if __name__ == "__main__":
    main()
