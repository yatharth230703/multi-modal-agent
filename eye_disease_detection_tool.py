from crewai_tools import BaseTool
from fastai.vision.all import *
import matplotlib.pyplot as plt
import torch
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class EyeDiseaseDetectionTool(BaseTool):
    name: str = "Eye Disease Detection Tool"
    description: str = "This tool detects eye diseases from an input image and generates a prompt based on the detection results."

    def __init__(self, model_path: str):
        super().__init__()
        self._learn = load_learner(model_path, cpu=True)

    def _run(self, image_path: str) -> str:
        pred, prob = self.predict_eye_disease(image_path)
        prompt = self.generate_prompt(pred, prob)
        return prompt

    def predict_eye_disease(self, image_path: str):
        img = PILImage.create(image_path)
        pred, pred_idx, probs = self._learn.predict(img)
        return pred, probs[pred_idx].item()

    def generate_prompt(self, prediction: str, probability: float) -> str:
        if probability > 0.5:
            prompt = f"Detected disease: {prediction} with a probability of {probability:.2f}. Please consult a specialist for further diagnosis and treatment."
        else:
            prompt = "No eye diseases detected. The eye appears healthy."
        return prompt
