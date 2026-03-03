import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

class ImageEmotionDetector:
    def __init__(self, model_path='resnet18_emotion.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Arquitectura idéntica a tu entrenamiento en Colab (7 clases)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("ResNet: Modelo visual cargado correctamente.")
        except Exception as e:
            print(f"Error cargando ResNet: {e}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_4d_probs(self, frame):
        """Convierte las 7 clases de FER2013 a 4 para el recomendador: [Happy, Sad, Angry, Calm]"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Mapeo: [0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral]
        happy = probs[3] + (probs[5] * 0.5) 
        sad = probs[4] + (probs[2] * 0.3)
        angry = probs[0] + probs[1]
        calm = probs[6]
        
        vec = np.array([happy, sad, angry, calm])
        return vec / vec.sum()