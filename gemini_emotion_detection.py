import json
import numpy as np
import re
import asyncio
import random
import time
from google import genai
from google.genai import types
from furhat_remote_api import FurhatRemoteAPI

class FurhatDrivingAssistant:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.furhat = FurhatRemoteAPI("localhost")
        self.emotion_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.labels = ["happy", "sad", "angry", "calm"]
        self.main_emotion = "calm"
        self.last_transcript = ""
        self.is_performing = False # Lock to prevent overlapping animations
        self.on_update = None 

    async def perform_happy_concert(self):
        """HAPPY: 5-second High-Energy Concert Mode"""
        print("Performance: HAPPY CONCERT")
        self.furhat.say(text="Yeah! I love this track! Let's go!", blocking=False)
        start_time = time.time()
        while time.time() - start_time < 5:
            # Random head bobbing + random smiles/winks
            self.furhat.attend(location=f"{random.randint(-15, 15)}, {random.randint(-10, 10)}, 2")
            self.furhat.gesture(name=random.choice(["BigSmile", "Wink", "Surprise"]), blocking=False)
            await asyncio.sleep(0.4)
        self.furhat.attend(location="0, 0, 1") # Return to front

    async def perform_sad_melancholy(self):
        """SAD: 5-second Slow Tilted Head Sway"""
        print("Performance: SAD MELANCHOLY")
        self.furhat.say(text="I'm here with you... let's just listen.", blocking=False)
        self.furhat.gesture(name="Sad", blocking=False)
        # Slow head tilt and look down
        self.furhat.attend(location="2, -5, 2")
        await asyncio.sleep(2)
        self.furhat.attend(location="-2, -8, 2")
        await asyncio.sleep(2)
        self.furhat.attend(location="0, 0, 1")

    async def perform_angry_intensity(self):
        """ANGRY: Intense Forward Stare with Quick Head Shakes"""
        print("Performance: ANGRY INTENSITY")
        self.furhat.say(text="I feel that fire too. Don't hold back.", blocking=False)
        start_time = time.time()
        while time.time() - start_time < 4:
            self.furhat.attend(location="0, 0, 1") # Locked stare
            self.furhat.gesture(name="BrowFrown", blocking=False)
            self.furhat.gesture(name="ExpressFear", blocking=False)
            # Quick aggressive micro-shakes
            self.furhat.attend(location=f"{random.randint(-2, 2)}, 0, 1")
            await asyncio.sleep(0.2)

    async def perform_calm_vibe(self):
        """CALM: 5-second Smooth Head Vibe"""
        print("Performance: CALM VIBE")
        start_time = time.time()
        while time.time() - start_time < 5:
            # Smooth circular head movement
            self.furhat.attend(location="5, 2, 2")
            await asyncio.sleep(1.2)
            self.furhat.attend(location="-5, 2, 2")
            await asyncio.sleep(1.2)
        self.furhat.attend(location="0, 0, 1")

    def sync_gesture(self, emotion):
        """Triggers the long-form performances asynchronously"""
        # Create a task for the specific performance so it doesn't block Gemini
        if emotion == "happy":
            asyncio.create_task(self.perform_happy_concert())
        elif emotion == "sad":
            asyncio.create_task(self.perform_sad_melancholy())
        elif emotion == "angry":
            asyncio.create_task(self.perform_angry_intensity())
        else:
            asyncio.create_task(self.perform_calm_vibe())

    async def emotional_pulse_loop(self):
        """Background heartbeat for when no new voice is detected"""
        while True:
            # Only do idle micro-gestures if not currently in a 'Big Performance'
            self.furhat.attend(location="0, 0, 1")
            if self.main_emotion == "happy":
                self.furhat.gesture(name="Smile", blocking=False)
            elif self.main_emotion == "angry":
                self.furhat.gesture(name="BrowFrown", blocking=False)
            
            await asyncio.sleep(3.0) # Slower idle pulse to let performances shine

    async def continuous_listen(self):
        print("System: Listening for voice input...")
        while True:
            response = self.furhat.listen()
            if response.message:
                self.last_transcript = response.message
                try:
                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                        response_mime_type="application/json"
                    )
                    prompt = f"Analyze sentiment: '{response.message}'. Return probabilities for happy, sad, angry, calm in JSON."
                    res = self.client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview", 
                        contents=prompt,
                        config=config
                    )
                    
                    json_str = re.search(r"\{.*\}", res.text, re.DOTALL).group(0)
                    data = json.loads(json_str)
                    
                    self.emotion_probs = np.array([data.get('happy',0.25), data.get('sad',0.25), data.get('angry',0.25), data.get('calm',0.25)])
                    self.main_emotion = self.labels[np.argmax(self.emotion_probs)]
                    
                    self.sync_gesture(self.main_emotion)
                    if self.on_update: self.on_update()
                    
                except Exception as e:
                    print(f"Gemini 3.1 Error: {e}")
            await asyncio.sleep(0.4)