import os
import cv2
import asyncio
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict

# Custom Modules
from image_emotion_detector import ImageEmotionDetector
from gemini_emotion_detection import FurhatDrivingAssistant
from recommender import recommend_top_n

app = FastAPI(title="AI Driver Assistant Pro")

# --- INITIALIZATION ---
detector = ImageEmotionDetector(model_path='resnet18_emotion.pth')
api_key = os.getenv("GEMINI_API_KEY")
assistant = FurhatDrivingAssistant(api_key=api_key)

# Global State
state = {
    "camera_probs": [0.25, 0.25, 0.25, 0.25],
    "last_recs": []
}

# Video Feed Shared Variables
output_frame = None
lock = threading.Lock()

# Directory Setup
if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- BACKGROUND WORKERS ---

def camera_worker():
    """High-speed vision thread with video overlay."""
    global output_frame
    cap = cv2.VideoCapture(0)
    # Optimization: smaller buffer for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Camera: Visual Debugger Active.")
    while True:
        ret, frame = cap.read()
        if ret:
            # 1. AI Inference
            probs = detector.get_4d_probs(frame)
            state["camera_probs"] = probs.tolist()
            
            # 2. Visual Overlay for the HTML feed
            current_emo = assistant.labels[np.argmax(probs)]
            color = (0, 255, 0) # Green
            cv2.putText(frame, f"AI VISION: {current_emo.upper()}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw probability bars (mini-graph)
            for i, (label, prob) in enumerate(zip(assistant.labels, probs)):
                width = int(prob * 150)
                cv2.rectangle(frame, (20, 80 + (i*30)), (20 + width, 100 + (i*30)), color, -1)
                cv2.putText(frame, label, (180, 95 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 3. Encoding for Streaming
            with lock:
                _, buffer = cv2.imencode('.jpg', frame)
                output_frame = buffer.tobytes()
        
        cv2.waitKey(10) # ~60 FPS capability

def refresh_recommendations():
    recs = recommend_top_n(
        current=state["camera_probs"],
        target=assistant.emotion_probs.tolist(),
        n=3
    )
    state["last_recs"] = recs

assistant.on_update = refresh_recommendations

# --- ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=camera_worker, daemon=True).start()
    asyncio.create_task(assistant.continuous_listen())
    assistant.furhat.say(text="Visual feed and voice detection online.", blocking=False)

def gen_frames():
    """Video streaming generator."""
    while True:
        with lock:
            if output_frame is None: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        import time
        time.sleep(0.03) # Match camera FPS

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def get_dashboard():
    return FileResponse('static/index.html')

@app.get("/status")
async def get_status():
    cam_idx = np.argmax(state["camera_probs"])
    return {
        "camera": assistant.labels[cam_idx],
        "voice": assistant.main_emotion,
        "recommendations": state["last_recs"]
    }