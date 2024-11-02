# GEOHIDRO: PLATAFORMA DE DIAGNÓSTICO  Y VISUALIZACIÓN DEL ESTADO DE RECURSOS HÍDRICOS DE SANTANDER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tmax = pd.read_excel("data/tmax.xlsx")
tmin = pd.read_excel("data/tmin.xlsx")

print("tmax dataset size: ", tmax.shape)
print("tmin dataset size: ", tmin.shape)

columns = ["CodigoEstacion", "Latitud", "Longitud", "Altitud", "Fecha", "Valor"]
tmax = tmax[columns]

print("Tmax min value: ", tmax["Valor"].min())
print("Tmax max value: ", tmax["Valor"].max())
print("Number of NaN values: ", tmax["Valor"].isna().sum())

print("eliminando valores nulos ...")
tmax = tmax.dropna(subset=["Valor"])
print("nuevo tamaño de los datos: ", tmax.shape)

# eliminar temperaturas máximas menores que temperaturas mínimas
tmin_min = tmin["Valor"].min()
print("valor mínimo del dataset de tmin: ", tmin_min)
tmax = tmax.drop(tmax[tmax["Valor"] < tmin_min].index)
print("nuevo tamaño de los datos con filtro: ", tmax.shape)

# plotear histograma
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].hist(tmax["Valor"], color="blue")
ax[0].set_title("Histograma temperatura máxima")
ax[0].set_xlabel("Temperatura °C")
ax[1].hist(tmin["Valor"], color="red")
ax[1].set_title("Histograma temperatura mínima")
ax[1].set_xlabel("Temperatura °C")
plt.show()

# plotear boxplot
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].set_title("Boxplot temperatura máxima")
ax[0].set_ylabel("Temperatura °C")
bplot = ax[0].boxplot(tmax["Valor"], patch_artist=True)  # fill with color
ax[1].set_title("Boxplot temperatura mínima")
ax[1].set_ylabel("Temperatura °C")
bplot = ax[1].boxplot(tmin["Valor"], patch_artist=True)  # fill with color
plt.show()

# estadística descriptiva
print("cargando estadística descriptiva...")
tmax["Valor"].describe()

# plotear una gráfica de la variación de la temperatura en cada año en cada mes

# convertir la columna de fecha a tipo datetime (ajusta el nombre de la columna si es diferente)
tmax["Fecha"] = pd.to_datetime(tmax["Fecha"], format="%m/%d/%Y %H:%M")

# Extraer los meses y los años como columnas separadas
tmax["Mes"] = tmax["Fecha"].dt.month
tmax["Año"] = tmax["Fecha"].dt.year

# graficar enero primer año

year1 = tmax["Año"].min()
month1_year1 = tmax[(tmax["Mes"] == 1) & (tmax["Año"] == year1)]

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    month1_year1["Longitud"], month1_year1["Latitud"], c=month1_year1["Valor"], cmap="jet"
)
plt.colorbar(scatter, label="Valor de Temperatura")
plt.title(f"Temperatura en Enero del Año {year1}")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.grid(True)
plt.show()
