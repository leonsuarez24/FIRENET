import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk


APP_TITLE = "Bienvenido a FireNet"


def main():

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔥",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.image(
        "https://raw.githubusercontent.com/leonsuarez24/FIRENET/refs/heads/main/figs/portada.png",
        caption=None,
        use_column_width=True,
    )

    st.title(APP_TITLE)
    st.divider()

    st.markdown(
        "FireNet es una plataforma cuya principal función es contribuir a la gestión del riesgo de incendios forestales con tecnologías geoespaciales y de inteligencia artifical el cual incorpora datos abiertos meteorológicos y delitos ambientales en el departamento de Santander."
    )

    st.markdown("## **Datos empleados**")

    st.markdown(
        """
        * [Delitos contra el medio ambiente](https://www.datos.gov.co/Seguridad-y-Defensa/DELITOS-CONTRA-EL-MEDIO-AMBIENTE/9zck-qfvc/about_data)
        * [Datos hidrometeorológicos (IDEAM) - Temperatura y precipitación](http://dhime.ideam.gov.co/atencionciudadano/)
        """
    )

    st.image(
        "https://raw.githubusercontent.com/leonsuarez24/FIRENET/refs/heads/main/figs/DatosUsados.jpg",
        caption=None,
        use_column_width=True,
    )

    st.markdown("## **Metodos de Inteligencia Artificial empleados**")

    st.markdown(
        """
        * Regresión mediante splines
        * [Red neuronal convolucional recurrente ConvLSTM](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)
        """
    )

    st.markdown("## **Información adicional**")

    st.markdown(
        "* **Autores**: Ana Mantilla, León Suárez y Luis Rodríguez \n* **Contacto**: ana.mantilla@correo.uis.edu.co \n* **Repositorio**: https://github.com/leonsuarez24/FIRENET"
    )

    st.markdown("## **Visualización de la información**")

    st.markdown(
        "Para visualizar los datos de temperatura y precipitación, rediríjase a la pestaña visualizacion de datos en el menú lateral."
    )

    st.markdown(
        "Para visualizar los datos sobre delitos ambientales, rediríjase a la pestaña delitos ambientales en el menú lateral."
    )

    st.markdown(
        "Para visualizar las predicciones de temperatura y precipitación, rediríjase a la pestaña prendicción de datos en el menú lateral."
    )


if __name__ == "__main__":
    main()
