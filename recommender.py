import os
import csv
import numpy as np
from typing import List, Dict

# Ruta al archivo de datos
CSV_PATH = "playlist.csv"

def normalize(v):
    """Asegura que el vector sume 1.0 para comparaciones probabilísticas."""
    a = np.asarray(v, dtype=float)
    sum_val = a.sum()
    return a / sum_val if sum_val > 0 else a

def recommend_top_n(current: List[float], target: List[float], n=3) -> List[Dict]:
    """
    Calcula las canciones que mejor ayudan a transicionar del estado 
    visual (current) al estado deseado/expresado por voz (target).
    """
    songs = []
    
    # 1. Verificar si el archivo existe
    if not os.path.exists(CSV_PATH):
        print(f"⚠️ Error: No se encontró el archivo {CSV_PATH}")
        return []

    # 2. Cargar canciones del CSV
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extraer vectores emocionales del CSV
                vec = normalize([
                    row['emo_happy'], 
                    row['emo_sad'], 
                    row['emo_angry'], 
                    row['emo_calm']
                ])
                
                songs.append({
                    "title": row['title'], 
                    "artist": row['artist'], 
                    "url": row.get('url', ''), 
                    "vec": vec
                })
            except (ValueError, KeyError) as e:
                # Ignorar filas mal formadas o con datos no numéricos
                continue

    if not songs:
        print("⚠️ No hay canciones válidas en la playlist.")
        return []

    # 3. Preparar vectores de comparación
    c = normalize(current)
    t = normalize(target)
    
    scored = []
    for s in songs:
        # LÓGICA DE TRANSICIÓN:
        # Calculamos un estado intermedio (mezcla de lo que ve la cámara y la canción)
        # y vemos qué tan cerca queda del objetivo (lo que dice la voz/Gemini).
        # El 0.5 es el peso de influencia de la canción.
        next_state = 0.5 * c + 0.5 * s["vec"]
        
        # Distancia Euclidiana (entre más pequeña, mejor es la canción)
        score = np.linalg.norm(next_state - t)
        scored.append((score, s))
    
    # 4. Ordenar por el mejor score (menor distancia)
    scored.sort(key=lambda x: x[0])
    
    # 5. --- LIMPIEZA DE DATOS PARA FASTAPI ---
    # Es VITAL convertir a tipos de datos estándar de Python (str) 
    # y eliminar los arrays de NumPy para evitar errores de serialización JSON.
    top_recommendations = []
    for _, song in scored[:n]:
        clean_song = {
            "title": str(song["title"]),
            "artist": str(song["artist"]),
            "url": str(song["url"])
        }
        top_recommendations.append(clean_song)
        
    return top_recommendations