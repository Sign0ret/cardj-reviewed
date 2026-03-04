import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

class ImageEmotionDetector:
    def __init__(self, model_path='resnet18_emotion.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_4d_probs(self, frame):
        img_tensor = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(img_tensor), dim=1).cpu().numpy()[0]
        
        # Mapping with higher sensitivity for movement
        happy = (probs[3] * 1.5) + (probs[5] * 0.5)
        sad = probs[4] + (probs[2] * 0.4)
        angry = (probs[0] * 1.3) + probs[1]
        calm = probs[6] * 0.7
        
        v = np.array([happy, sad, angry, calm])
        return v / v.sum()