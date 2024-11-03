import streamlit as st
from scripts.process_temp import process_temp_data
from scripts.utils import display_map

APP_TITLE = "GEOHIDRO: Plataforma de Diagnóstico y Visualización del Estado de Recursos Hídricos de Santander"


def main():

    st.title(APP_TITLE)

    data_temp = process_temp_data()

    st.markdown("### Datos de temperatura")
    st.write(data_temp)

    display_map()


if __name__ == "__main__":
    main()
