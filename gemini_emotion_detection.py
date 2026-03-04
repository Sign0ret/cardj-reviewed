import json
import numpy as np
import re
import asyncio
from google import genai
from google.genai import types  # Needed for Gemini 3 configuration
from furhat_remote_api import FurhatRemoteAPI

class FurhatDrivingAssistant:
    def __init__(self, api_key):
        # Initializing with the modern SDK client
        self.client = genai.Client(api_key=api_key)
        self.furhat = FurhatRemoteAPI("localhost")
        self.emotion_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.labels = ["happy", "sad", "angry", "calm"]
        self.main_emotion = "calm"
        self.last_transcript = ""
        self.on_update = None 

    def sync_gesture(self, emotion):
        """Intense, heavy gestures while maintaining forward gaze"""
        self.furhat.attend(location="0, 0, 1") # Lock gaze to the driver
        
        if emotion == "happy":
            # Chained gestures for 'Heavy' emotional feedback
            self.furhat.gesture(name="BigSmile")
            self.furhat.gesture(name="Surprise")
            self.furhat.say(text="I love this energy! Let's crank it up!", blocking=False)
        elif emotion == "angry":
            self.furhat.gesture(name="ExpressFear") # High intensity stare
            self.furhat.gesture(name="BrowFrown")
            self.furhat.say(text="I can feel the tension. Let's vent through the music.", blocking=False)
        elif emotion == "sad":
            self.furhat.gesture(name="Sad")
            self.furhat.gesture(name="Thoughtful")
            self.furhat.say(text="It's okay to feel this way. I'm right here with you.", blocking=False)
        else:
            self.furhat.gesture(name="Smile")

    async def emotional_pulse_loop(self):
        """Background heartbeat to keep the robot 'alive' and staring"""
        while True:
            self.furhat.attend(location="0, 0, 1") # Ensure gaze is always front
            # Micro-gestures to reinforce the 'Heavy' emotional state
            if self.main_emotion == "happy":
                self.furhat.gesture(name="Smile", blocking=False)
            elif self.main_emotion == "angry":
                self.furhat.gesture(name="BrowFrown", blocking=False)
            elif self.main_emotion == "sad":
                self.furhat.gesture(name="Sad", blocking=False)
            
            # Fast pulse for happy/angry, slow for sad
            wait = 0.5 if self.main_emotion in ["happy", "angry"] else 2.5
            await asyncio.sleep(wait)

    async def continuous_listen(self):
        print("System: Listening for voice input...")
        while True:
            response = self.furhat.listen()
            if response.message:
                self.last_transcript = response.message
                print(f"Driver: {self.last_transcript}")
                
                try:
                    # CONFIG: We use 'minimal' thinking for the fastest possible response
                    config = types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                        response_mime_type="application/json"
                    )
                    
                    prompt = f"Analyze driver sentiment: '{response.message}'. Return probabilities for happy, sad, angry, calm in valid JSON."
                    
                    # Call to Gemini 3.1 Flash-Lite
                    res = self.client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview", 
                        contents=prompt,
                        config=config
                    )
                    
                    # Clean the JSON from potential thinking blocks or markdown
                    json_str = re.search(r"\{.*\}", res.text, re.DOTALL).group(0)
                    data = json.loads(json_str)
                    
                    self.emotion_probs = np.array([
                        data.get('happy', 0.25), 
                        data.get('sad', 0.25), 
                        data.get('angry', 0.25), 
                        data.get('calm', 0.25)
                    ])
                    self.main_emotion = self.labels[np.argmax(self.emotion_probs)]
                    
                    self.sync_gesture(self.main_emotion)
                    if self.on_update: self.on_update()
                    
                except Exception as e:
                    print(f"Gemini 3.1 Error: {e}")
            await asyncio.sleep(0.4)