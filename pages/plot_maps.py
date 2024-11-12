import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO


APP_TITLE = "FireNet"


def main():
    st.set_page_config(layout="wide")
    st.title(APP_TITLE)

    boundaries = "data/santander_boundary.json"

    with open(boundaries, "r") as f:
        region_geojson = json.load(f)

    df_t = pd.read_excel("data/tmean.xlsx")
    df_t["Latitud"] = df_t["Latitud"].round(2)
    df_t["Longitud"] = df_t["Longitud"].round(2)

    df_p = pd.read_excel("data/precipitacion_filtrado.xlsx")
    df_p["Latitud"] = df_p["Latitud"].round(2)
    df_p["Longitud"] = df_p["Longitud"].round(2)

    df_t.columns = df_t.columns.str.strip()
    df_t["Latitud"] = pd.to_numeric(df_t["Latitud"], errors="coerce")
    df_t["Longitud"] = pd.to_numeric(df_t["Longitud"], errors="coerce")
    df_t["Fecha"] = pd.to_datetime(df_t["Fecha"])

    df_p.columns = df_p.columns.str.strip()
    df_p["Latitud"] = pd.to_numeric(df_p["Latitud"], errors="coerce")
    df_p["Longitud"] = pd.to_numeric(df_p["Longitud"], errors="coerce")
    df_p["Fecha"] = pd.to_datetime(df_p["Fecha"])

    min_date = df_t["Fecha"].min()
    max_date = df_t["Fecha"].max()

    start_time = st.slider(
        "Seleccionar fecha",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=datetime(2000, 1, 1),
        format="YYYY-MM",
        key="date_slider",
    )

    st.write(f"Fecha seleccionada: {start_time.strftime('%Y-%m')}")

    filtered_df_t = df_t[df_t["Fecha"].dt.strftime("%Y-%m") == start_time.strftime("%Y-%m")]
    filtered_df_p = df_p[df_p["Fecha"].dt.strftime("%Y-%m") == start_time.strftime("%Y-%m")]

    if filtered_df_t.empty and filtered_df_p.empty:
        st.write("No data available for the selected date.")
    else:

        with st.container():
            col1, col2 = st.columns([1, 1])

            with col1:

                filtered_df_t["Valor_medio"] = filtered_df_t["Valor_medio"].round(2)
                filtered_df_p["Valor"] = filtered_df_p["Valor"].round(2)

                fig_temp = px.scatter_mapbox(
                    filtered_df_t,
                    lat="Latitud",
                    lon="Longitud",
                    color="Valor_medio",
                    size="Valor_medio",
                    hover_name="Valor_medio",
                    size_max=15,
                    zoom=7,
                    color_continuous_scale="Reds",
                    mapbox_style="carto-positron",
                    title=f"Temperatura media para {start_time.strftime('%Y-%m')} (°C)",
                )

                fig_temp.update_layout(coloraxis_colorbar_title="Temperatura media (°C)")

                st.plotly_chart(fig_temp, key="temperatura_media")

            with col2:
                inter_temp = np.load(
                    f"data/tmean_interp_final/npy/temperatura_{start_time.strftime('%Y-%m')}-01.npy"
                )
                fig_t, ax = plt.subplots()
                plt.imshow(inter_temp, cmap="coolwarm")
                plt.colorbar()

                buf = BytesIO()
                fig_t.savefig(buf, format="png")
                st.image(buf)


        with st.container():
            col3, col4 = st.columns([1, 1])
            with col3:

                fig_prep = px.scatter_mapbox(
                    filtered_df_p,
                    lat="Latitud",
                    lon="Longitud",
                    color="Valor",
                    size="Valor",
                    hover_name="Valor",
                    size_max=15,
                    zoom=7,
                    color_continuous_scale="Blues",
                    mapbox_style="carto-positron",
                    title=f"Precipitación para {start_time.strftime('%Y-%m')} (mm)",
                )

                fig_prep.update_layout(coloraxis_colorbar_title="Precipitación (mm)")
                st.plotly_chart(fig_prep, key="precipitacion")

            with col4:
                inter_temp = np.load(
                    f"data/precipitacion_interp_final/npy/precipitacion_{start_time.strftime('%Y-%m')}-01.npy"
                )
                fig_p, ax = plt.subplots()
                plt.imshow(inter_temp, cmap="Blues")
                plt.colorbar()

                buf = BytesIO()
                fig_p.savefig(buf, format="png")
                st.image(buf)



if __name__ == "__main__":
    main()
