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

    st.markdown("FireNet es una plataforma cuya principal función es contribuir a la gestión del riesgo de incendios forestales con tecnologías geoespaciales y de inteligencia artifical el cual incorpora datos abiertos meteorológicos y delitos ambientales en el departamento de Santander.")

    st.markdown("## **Información adicional**")

    st.markdown(
        "* **Autores**: Ana Mantilla, León Suárez y Luis Rodríguez \n* **Contacto**: ana.mantilla@correo.uis.edu.co \n* **Repositorio**: https://github.com/leonsuarez24/FIRENET"
    )

    st.markdown("## **Visualización de datos**")

    st.markdown(
        "Para visualizar los datos, rediríjase a la pestaña visualizacion de datos en el menú lateral."
    )


if __name__ == "__main__":
    main()
