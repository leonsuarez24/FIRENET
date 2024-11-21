import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import geopandas as gpd
import verde as vd


APP_TITLE = "FireNet"


def main():
    st.set_page_config(layout="wide",
                       page_icon="🔥",
    )
    
    st.image(
        "https://raw.githubusercontent.com/leonsuarez24/FIRENET/refs/heads/main/figs/portada.png",
        caption=None,
        use_column_width=True,
    )
    st.title(APP_TITLE)

    st.markdown("## Visualización de delitos ambientales")

    st.markdown(
        "En esta sección se presenta una visualización de los delitos ambientales en la región de Santander, Colombia, y como estos relacionan con los datos de temperatura y precipitación"
    )

    with st.container():
        col7, col8, col9 = st.columns([1, 1, 1])

        with col7:

            st.image(
                "data/delitos/delitos_temp_santander.png",
                caption=None,
                use_column_width=True,
            )

        with col8:

            st.image(
                "data/delitos/delitos_prec_santander.png",
                caption=None,
                use_column_width=True,
            )

        with col9:
            delitos_df = pd.read_csv("data/delitos/delitos_santander_total.csv")
            st.write(delitos_df)        



if __name__ == "__main__":
    main()
