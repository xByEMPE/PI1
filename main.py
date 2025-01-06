from fastapi import FastAPI
import pandas as pd
import unidecode
import numpy as np
from fastapi.responses import JSONResponse                                      
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(content={
        "message": "Hola, bienvenido, no te preocupes si no observas nada, solo agrega /docs al final del link que tienes en tu navegador"
    })

# Load datasets
credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies_cleaned_fixed.csv")


# Preprocess datasets
movies_df = movies_df.fillna(0)
movies_df.replace([float('inf'), float('-inf')], 0, inplace=True)
movies_df["budget"] = movies_df["budget"].astype(float)
movies_df["revenue"] = movies_df["revenue"].astype(float)
movies_df["return"] = np.where(movies_df["budget"] > 0, movies_df["revenue"] / movies_df["budget"], 0)
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors='coerce')
movies_df["normalized_title"] = movies_df["title"].apply(lambda x: unidecode.unidecode(x.lower()) if isinstance(x, str) else "")

# Helper functions
SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}

def normalize_string(s):
    return unidecode.unidecode(s.lower()) if isinstance(s, str) else ""

def get_movie_by_title(titulo):
    titulo = normalize_string(titulo)
    return movies_df[movies_df["normalized_title"] == titulo]

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes = normalize_string(mes)
    if mes not in SPANISH_MONTHS:
        return {"error": "Mes inválido"}
    month_number = SPANISH_MONTHS[mes]
    count = movies_df[movies_df["release_date"].dt.month == month_number].shape[0]
    return {"message": f"{count} cantidad de películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: int):
    if dia < 1 or dia > 31:
        return {"error": "Día inválido"}
    count = movies_df[movies_df["release_date"].dt.day == dia].shape[0]
    return {"message": f"{count} cantidad de películas fueron estrenadas el día {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    movie = get_movie_by_title(titulo)
    if movie.empty:
        return {"error": "Título no encontrado"}
    movie = movie.iloc[0]
    return {
        "titulo": movie["title"],
        "año": int(movie["release_year"]),
        "score": movie["popularity"]
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    movie = get_movie_by_title(titulo)
    if movie.empty:
        return {"error": "Título no encontrado"}
    movie = movie.iloc[0]
    if movie["vote_count"] < 2000:
        return {"message": "La película no cumple con el mínimo de 2000 valoraciones"}
    return {
        "titulo": movie["title"],
        "votos_totales": int(movie["vote_count"]),
        "promedio_votos": movie["vote_average"]
    }

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    nombre_actor = normalize_string(nombre_actor)
    def is_actor_in_cast(cast_row):
        cast = ast.literal_eval(cast_row)
        return any(normalize_string(actor.get('name', '')) == nombre_actor for actor in cast)

    relevant_movies = credits_df[credits_df["cast"].apply(is_actor_in_cast)]
    if relevant_movies.empty:
        return {"error": "Actor no encontrado"}

    relevant_movies = relevant_movies.merge(movies_df, on="id", how="left")
    relevant_movies["return"] = np.where(relevant_movies["budget"] > 0, relevant_movies["revenue"] / relevant_movies["budget"], 0)

    total_return = relevant_movies["return"].sum()
    count = relevant_movies.shape[0]

    return {
        "actor": nombre_actor,
        "cantidad_peliculas": count,
        "retorno_total": total_return,
        "promedio_retorno": total_return / count if count > 0 else 0
    }

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    nombre_director = normalize_string(nombre_director)

    def is_director_in_crew(crew_row):
        try:
            crew = ast.literal_eval(crew_row)
            return any(member.get("job", "").lower() == "director" and normalize_string(member.get("name", "")) == nombre_director for member in crew)
        except (ValueError, SyntaxError):
            return False

    relevant_movies = credits_df[credits_df["crew"].apply(is_director_in_crew)]
    if relevant_movies.empty:
        return JSONResponse(content={"error": "Director no encontrado"}, status_code=404)

    relevant_movies = relevant_movies.merge(movies_df, on="id", how="left").fillna(0)
    relevant_movies["budget"] = relevant_movies["budget"].replace([np.inf, -np.inf], 0)
    relevant_movies["revenue"] = relevant_movies["revenue"].replace([np.inf, -np.inf], 0)

    movie_details = []
    for _, movie in relevant_movies.iterrows():
        title = movie.get("title", "Desconocido")
        release_date = str(movie.get("release_date", ""))
        retorno_individual = max(movie["revenue"] - movie["budget"], 0)
        cost = movie.get("budget", 0)
        revenue = movie.get("revenue", 0)

        if pd.isna(title) or title == 0:
            title = "Desconocido"
        if pd.isna(release_date) or release_date == "0":
            release_date = "Fecha desconocida"
        if pd.isna(retorno_individual) or np.isinf(retorno_individual):
            retorno_individual = 0
        if pd.isna(cost) or np.isinf(cost):
            cost = 0
        if pd.isna(revenue) or np.isinf(revenue):
            revenue = 0

        movie_details.append({
            "titulo": title,
            "fecha_lanzamiento": release_date,
            "retorno_individual": retorno_individual,
            "costo": cost,
            "ganancia": revenue,
        })

    return {
        "director": nombre_director,
        "peliculas": movie_details
    }

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    titulo = normalize_string(titulo)
    
    # Verificar si el título existe en el dataset
    if titulo not in movies_df["normalized_title"].values:
        return JSONResponse(content={"error": "Película no encontrada"}, status_code=404)
    
    # Asegurar que todos los valores en la columna "overview" sean cadenas de texto
    movies_df["overview"] = movies_df["overview"].astype(str).fillna("")
    
    # Crear un vectorizador TF-IDF para calcular la similitud del texto
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
    
    # Obtener el índice de la película ingresada
    idx = movies_df[movies_df["normalized_title"] == titulo].index[0]
    
    # Calcular la similitud del coseno entre la película ingresada y el resto
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Obtener los índices de las películas más similares
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]  # Excluir la película original
    
    # Obtener los títulos de las películas similares
    similar_movies = movies_df.iloc[similar_indices]["title"].tolist()
    
    return {
        "pelicula": titulo,
        "recomendaciones": similar_movies
    }
