import folium
from streamlit_folium import st_folium
from scripts.process_temp import process_temp_data
import folium


def display_map():
    map = folium.Map(location=[7, -73.6536], zoom_start=8, tiles="CartoDB positron")
    folium.GeoJson(
        "data/santander.geojson",
        style_function=lambda feature: {
            "weight": 2,
        },
    ).add_to(map)

    df = process_temp_data()

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['Latitud'], row['Longitud']),
            radius=5,  # Adjust size based on `Valor_medio`, if desired
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Date: {row['Fecha'].strftime('%Y-%m-%d')}, Valor_medio: {row['Valor_medio']}"
        ).add_to(map)






    st_map = st_folium(map, width=700, height=400)
