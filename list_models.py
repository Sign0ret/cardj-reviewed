import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing available models for your API key...")
try:
    for model in client.models.list():
        # Only show the name to avoid attribute errors
        print(f"Model ID: {model.name}")
except Exception as e:
    print(f"An error occurred: {e}")