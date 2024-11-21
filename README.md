# FireNet 

[![Python 3.11](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF0000)

FireNet es una plataforma cuya principal función es contribuir a la gestión del riesgo de incendios forestales con tecnologías geoespaciales y de inteligencia artifical el cual incorpora datos abiertos meteorológicos y delitos ambientales en el departamento de Santander.

<p align="center">
<img src="https://github.com/leonsuarez24/FIRENET/blob/main/figs/portada.png" width="900">
</p>

## **Datos usados** 📋
* [Delitos contra el medio ambiente](https://www.datos.gov.co/Seguridad-y-Defensa/DELITOS-CONTRA-EL-MEDIO-AMBIENTE/9zck-qfvc/about_data)
* [Datos hidrometeorológicos (IDEAM) - Temperatura y precipitación](http://dhime.ideam.gov.co/atencionciudadano/)

<p align="center">
<img src="https://github.com/leonsuarez24/FIRENET/blob/main/figs/DatosUsados.jpg" width="900">
</p>

## **Limpieza de datos** 📋

El script "process_data_main.py" tiene las técnicas de filtrado para los datos de anomalías que no corresponden con eventos de temperatura y/o precipitación, y el análisis estadístico de ambos conjuntos de datos. Para ejecutar el archivo debes tener la carpeta "script" que se encuentra en el repositorio.

## **Interpolación de datos** 📋

El notebook "interpolate.ipynb" tiene el procedimiento para interpolar los datos de temperatura y precipitación media mensual de cada estación en todo el departamento de Santander. Este archivo se encuentra en la carpeta "notebooks".

## **Metodos de Inteligencia artificial usados**

* Regresión mediante splines
* [Red neuronal ConvLSTM](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)

Los pesos de la red neuronal entrenada se encuentran en la ruta "model/weights" tanto para la precipitación como temperatura. Estos modelos se usaron para predecir los mapas mensuales desde noviembre-2024 hasta octubre-2025. Los resultados se guardaron como matrices numpy en la carpeta "data". 

## **Información adicional**

    
**Autores**: 
* Ana Mantilla
* León Suárez 
* Luis Rodríguez 

**Contacto**: ana.mantilla@correo.uis.edu.co 

**Repositorio**: https://github.com/leonsuarez24/FIRENET

**Pagina web**: https://leonsuarez24-geohidro-principal-op9byl.streamlit.app

    
