import streamlit as st
#from scripts.utils import display_map_temp_precip
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime
import os

APP_TITLE = "FireNet:Plataforma Integrada para la Gestión de Incendios Forestales Utilizando Tecnologías Geoespaciales e Inteligencia Artificial"

# Definir la ruta donde están los archivos .npy
data_folder = 'data/tmean_interp_final/npy'

def main():

    st.title(APP_TITLE)

    #data_temp = pd.read_excel("data/tmean.xlsx")

    #st.markdown("### Datos de temperatura")
    #st.write(data_temp)

    #data_prep = pd.read_excel("data/precipitacion_filtrado.xlsx")
    #st.markdown("### Datos de precipitación")
    #st.write(data_prep)

    #display_map_temp_precip()

        # Listar los archivos y extraer fechas a partir del nombre
    file_dates = []
    file_paths = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.npy'):
            # Suponiendo que el nombre del archivo es "data_YYYY-MM.npy"
            date_str = filename.split('_')[1].replace('.npy', '')
            date_obj = datetime.strptime(date_str, '%Y-%m')
            file_dates.append(date_obj)
            file_paths.append(os.path.join(data_folder, filename))

    # Crear un DataFrame para almacenar las rutas y fechas de los archivos
    df_files = pd.DataFrame({'date': file_dates, 'file_path': file_paths})
    df_files['date_str'] = df_files['date'].dt.strftime('%Y-%m')

    # Entrada de fecha en formato de año y mes
    selected_date_str = st.selectbox("Seleccione una fecha (mes y año):", df_files['date_str'].unique())

    # Filtrar el archivo correspondiente a la fecha seleccionada
    selected_file_path = df_files[df_files['date_str'] == selected_date_str]['file_path'].values[0]

    # Cargar la matriz desde el archivo .npy
    selected_matrix = np.load(selected_file_path)

    # Mostrar la matriz y la fecha seleccionada
    st.write(f"Matriz de datos para {selected_date_str}")

    # Graficar la matriz
    fig, ax = plt.subplots()
    cax = ax.matshow(selected_matrix, cmap='viridis')  # Ajusta el colormap si lo deseas
    fig.colorbar(cax)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


if __name__ == "__main__":
    main()
