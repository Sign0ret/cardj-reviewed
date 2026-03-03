import os
import cv2
import asyncio
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict

# Importamos tus módulos personalizados
from image_emotion_detector import ImageEmotionDetector
from gemini_emotion_detection import FurhatDrivingAssistant
from recommender import recommend_top_n

app = FastAPI(title="AI Driver Assistant")

# --- CONFIGURACIÓN E INICIALIZACIÓN ---

# 1. Cargamos el detector visual (ResNet local)
detector = ImageEmotionDetector(model_path='resnet18_emotion.pth')

# 2. Cargamos el asistente de voz (Gemini 2.5/3.1)
# Asegúrate de que GEMINI_API_KEY esté en tu launch.sh
api_key = os.getenv("GEMINI_API_KEY")
assistant = FurhatDrivingAssistant(api_key=api_key)

# 3. Estado global del sistema
state = {
    "camera_probs": [0.25, 0.25, 0.25, 0.25], # [happy, sad, angry, calm]
    "last_recs": []
}

# 4. Montar archivos estáticos para el Dashboard
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- LÓGICA DE FONDO (BACKGROUND TASKS) ---

def camera_worker():
    """Hilo independiente que procesa la cámara con ResNet a 10 FPS"""
    cap = cv2.VideoCapture(0)
    print("Cámara: Hilo de visión iniciado.")
    while True:
        ret, frame = cap.read()
        if ret:
            # Obtenemos las probabilidades del modelo local
            probs = detector.get_4d_probs(frame)
            state["camera_probs"] = probs.tolist()
        cv2.waitKey(100) # Pausa de 100ms para estabilidad

def refresh_recommendations():
    """Se ejecuta cada vez que Gemini detecta un cambio en la voz"""
    recs = recommend_top_n(
        current=state["camera_probs"],        # Lo que ve la cámara
        target=assistant.emotion_probs.tolist(), # Lo que siente por la voz
        n=3
    )
    state["last_recs"] = recs
    print(f"\n🎵 Nueva Playlist: {[r['title'] for r in recs]}")

# Conectamos el callback del asistente a nuestra función de actualización
assistant.on_update = refresh_recommendations

# --- RUTAS DE LA API Y DASHBOARD ---

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar el servidor uvicorn"""
    # Iniciamos el hilo de la cámara (Vision)
    threading.Thread(target=camera_worker, daemon=True).start()
    
    # Iniciamos la escucha continua de Furhat (Voice)
    asyncio.create_task(assistant.continuous_listen())
    
    # Saludo inicial del robot
    assistant.furhat.say(text="System online. I am watching and listening.", blocking=False)
    print("Sistema: Flujo continuo activado.")

@app.get("/")
async def get_dashboard():
    """Sirve la interfaz web principal"""
    return FileResponse('static/index.html')

@app.get("/status")
async def get_status():
    """Endpoint que el HTML consulta cada segundo para actualizarse"""
    # Obtenemos la etiqueta de la emoción predominante en cámara
    cam_idx = np.argmax(state["camera_probs"])
    cam_label = assistant.labels[cam_idx]
    
    return {
        "camera": cam_label,
        "voice": assistant.main_emotion,
        "recommendations": state["last_recs"]
    }

# --- ENDPOINT OPCIONAL PARA TEST MANUAL ---
@app.post("/force-sync")
async def force_sync():
    """Por si quieres disparar una recomendación manualmente desde Swagger"""
    refresh_recommendations()
    return {"status": "Updated", "current_recs": state["last_recs"]}