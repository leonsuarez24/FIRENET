import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

APP_TITLE = "FireNet"


def main():
    st.title(APP_TITLE)

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

    filtered_df = df_t[df_t["Fecha"].dt.strftime("%Y-%m") == start_time.strftime("%Y-%m")]

    if filtered_df.empty:
        st.write("No data available for the selected date.")
    else:
        filtered_df["Valor_medio"] = filtered_df["Valor_medio"].round(2)

        fig = px.scatter_mapbox(
            filtered_df,
            lat="Latitud",
            lon="Longitud",
            color="Valor_medio",
            size="Valor_medio",
            hover_name="Valor_medio",
            size_max=15,
            zoom=7,
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            title=f"Temperatura media para {start_time.strftime('%Y-%m')} (°C)",
        )

        fig.update_layout(coloraxis_colorbar_title="Temperatura media (°C)")

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
