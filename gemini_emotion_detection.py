import json
import numpy as np
import re
import asyncio
from google import genai
from furhat_remote_api import FurhatRemoteAPI

class FurhatDrivingAssistant:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.furhat = FurhatRemoteAPI("localhost")
        self.emotion_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.labels = ["happy", "sad", "angry", "calm"]
        self.main_emotion = "calm"
        self.on_update = None # Callback para notificar a app.py

    def sync_gesture(self, emotion):
        gestures = {"happy": "BigSmile", "sad": "Sad", "angry": "ExpressFear", "calm": "Wink"}
        if emotion in gestures:
            self.furhat.gesture(name=gestures[emotion])

    async def continuous_listen(self):
        """Bucle infinito que analiza todo lo que el usuario dice"""
        print("Furhat: Escucha activa iniciada...")
        while True:
            # Escucha de forma no bloqueante
            response = self.furhat.listen()
            if response.message:
                print(f"Usuario: {response.message}")
                try:
                    # Usamos el modelo detectado en tu lista
                    prompt = f"Analyze sentiment: '{response.message}'. Return JSON with probabilities for: happy, sad, angry, calm"
                    res = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    
                    data = json.loads(re.search(r"\{.*\}", res.text, re.DOTALL).group(0))
                    self.emotion_probs = np.array([data.get('happy',0.25), data.get('sad',0.25), data.get('angry',0.25), data.get('calm',0.25)])
                    self.main_emotion = self.labels[np.argmax(self.emotion_probs)]
                    
                    self.sync_gesture(self.main_emotion)
                    if self.on_update: self.on_update() # Disparar recomendación
                    
                except Exception as e:
                    print(f"Gemini Error: {e}")
            await asyncio.sleep(0.5)