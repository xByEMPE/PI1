# **PROYECTO INDIVIDUAL 1**

**`Machine Learning Operations (MLOps)`**

---

¡Hola! Este es mi proyecto individual No. 1 para el curso de Data Science de la carrera de Data Science dentro del Bootcamp de Henry.

En esta ocasión me tocó realizar el rol de un **MLOps Engineer**. Para ello, se me asignó realizar lo siguiente:

- Realizar un modelo de Machine Learning para predecir el retorno de una película.
- Esto conlleva hacer **ETL, EDA, Modelamiento y Deployment**.
- Todo esto con ayuda de la ciencia de datos y usando herramientas de **Python** y sus bibliotecas.
- Usamos **FastAPI** para el deployment y **RENDER** para el hosting.

¡Así que empecemos!

---

## Tabla de contenido:

1. [Introducción](#introducción)
2. [Instalación y requerimientos](#instalación-y-requerimientos)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Uso de los archivos](#uso-de-los-archivos)
5. [Datos](#datos)
6. [Resultados](#resultados)
7. [Deployment](#deployment)
8. [Conclusiones](#conclusiones)

---

## Introducción
Este proyecto busca predecir el retorno de una película utilizando un enfoque de MLOps. El pipeline incluye ETL, análisis exploratorio de datos (EDA), modelado de Machine Learning y despliegue de la API.

## Instalación y requerimientos
Requisitos para poder hacer uso de este proyecto:

- **Python 3.12.6**
- **FastAPI**
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **unidecode**
- **scikit-learn**
- **uvicorn**
- **wordcloud**

### Pasos para configurar el entorno:
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/xByEMPE/PI1.git

2. Crear un entorno virtual:
   ```bash
   python -m venv .venv

3. Activar el entorno virtual:
   ```bash
   .\.venv\Scripts\activate

4. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt

5. Ejecutar el archivo main.py:
   ```bash
   uvicorn main:app --reload

## estructura del proyecto

```plaintext
PI1/
├── data/
│   ├── raw/         # Datos en formato original
│   ├── cleaned/     # Datos limpios
│   └── processed/   # Datos procesados durante el ETL
├── notebooks/
│   └── etl.ipynb    # Notebook del proceso ETL
├── reports/         # Reportes generados en el EDA
    └── eda.ipynb    # Notebook de EDA
├── main.py          # Código para el uso de la API
└── requirements.txt # Archivo con las dependencias necesarias
```
## Uso de los archivos
Para ejeccutar la API se debe seguir los siguientes pasos:

1. Clonar el repositorio
2. Crear un entorno virtual
3. Activar el entorno virtual
4. Instalar las dependencias con el comando: pip install -r requirements.txt.
5. Ejecutar el archivo main.py:
   ```bash
   uvicorn main:app --reload

Para ejecutar el proceso de ETL se debe seguir los siguientes pasos:

1. Abrir el archivo etl.ipynb en Jupyter Notebook
2. Ejecutar el notebook completo para que se realice el proceso de ETL

Para ejecutar el proceso de EDA se debe seguir los siguientes pasos:

1. Abrir el archivo eda.ipynb en Jupyter Notebook
2. Ejecutar el notebook completo para que se realice el proceso de EDA y visualizar los graficos generados.

## Datos
Se tienen dos archivos de datos:

movies_dataset.csv: Contiene información detallada de las películas.
credits.csv: Contiene datos sobre el elenco y equipo de producción.
Estos archivos fueron proporcionados por el equipo de Henry y se encuentran en la carpeta data/raw/.

## Resultados
Algunos resultados destacados del análisis exploratorio de datos (EDA):

- **Existe una correlación positiva entre budget e revenue, lo que indica que las películas con mayor presupuesto tienden a generar más ingresos.**
- **La nube de palabras muestra que términos como "Love", "War" y "Adventure" son comunes en los títulos de películas, indicando temas recurrentes.**
- **El modelo de recomendación de películas puede recomendar 5 películas similares a una película dada mediante la API.**

## Deployment

Deployment
El hosting se realizó en RENDER, y la API está disponible en el siguiente enlace:

- **https://pi1-s1up.onrender.com/**

Esto permite acceder a la API desde cualquier parte del mundo sin necesidad de instalar Python ni sus dependencias.

## Conclusiones

Este proyecto me permitió:

- **Aprender a hacer uso de FastAPI para el deployment de un modelo de Machine Learning.**
- **Aprender a usar RENDER para el hosting de la API.**
- **Aprender a hacer uso de Jupyter Notebook para la realizacion de ETL y EDA.**
- **Aprender a usar GitHub para el control de versiones y organizar el proyecto.**

Espero que les haya gustado el proyecto y les haya servido para aprender un poco mas sobre el mundo de la ciencia de datos y el desarrollo de modelos de Machine Learning.

Saludos.!!