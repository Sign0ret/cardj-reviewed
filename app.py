import os
import cv2
import asyncio
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from image_emotion_detector import ImageEmotionDetector
from gemini_emotion_detection import FurhatDrivingAssistant
from recommender import recommend_top_n

app = FastAPI()

# Init
detector = ImageEmotionDetector(model_path='resnet18_emotion.pth')
api_key = os.getenv("GEMINI_API_KEY")
assistant = FurhatDrivingAssistant(api_key=api_key)

state = {"camera_probs": [0.25, 0.25, 0.25, 0.25], "last_recs": []}
output_frame = None
lock = threading.Lock()

if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def camera_worker():
    global output_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if ret:
            probs = detector.get_4d_probs(frame)
            state["camera_probs"] = probs.tolist()
            
            # Visual Feedback on Video
            current_emo = assistant.labels[np.argmax(probs)]
            cv2.putText(frame, f"AI VISION: {current_emo.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            with lock:
                _, buffer = cv2.imencode('.jpg', frame)
                output_frame = buffer.tobytes()
        cv2.waitKey(10)

def update_playlist():
    state["last_recs"] = recommend_top_n(state["camera_probs"], assistant.emotion_probs.tolist())

assistant.on_update = update_playlist

@app.on_event("startup")
async def startup():
    threading.Thread(target=camera_worker, daemon=True).start()
    asyncio.create_task(assistant.continuous_listen())
    asyncio.create_task(assistant.emotional_pulse_loop()) # This now runs the Heavy Gestures
    print("Full System Started in Intensive Mode!")

def gen_frames():
    while True:
        with lock:
            if output_frame is None: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        import time
        time.sleep(0.03)

@app.get("/video_feed")
async def video_feed(): return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index(): return FileResponse('static/index.html')

# Inside app.py, update the status function:

@app.get("/status")
async def get_status():
    cam_idx = np.argmax(state["camera_probs"])
    return {
        "camera": assistant.labels[cam_idx],
        "voice": assistant.main_emotion,
        "transcript": assistant.last_transcript, # NEW: Send text to HTML
        "recommendations": state["last_recs"]
    }