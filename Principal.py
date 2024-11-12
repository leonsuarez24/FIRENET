import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk


APP_TITLE = "Bienvenido a FireNet"


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

    st.markdown("## **Información adicional**")

    st.markdown(
        "* **Autores**: Ana Mantilla, León Suárez y Luis Rodríguez \n* **Contacto**: [Correo electrónico] \n* **Repositorio**: https://github.com/leonsuarez24/FIRENET"
    )

    st.markdown("## **Visualización de datos**")

    st.markdown(
        "Para visualizar los datos, rediríjase a la pestaña visualizacion de datos en el menú lateral."
    )


if __name__ == "__main__":
    main()
