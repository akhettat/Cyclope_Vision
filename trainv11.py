import torch
from ultralytics import YOLO  # Ultralytics YOLO
import os

# Test de la disponibilité du GPU
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

model = YOLO("yolo11m.pt")
data_yaml = "C:/Users/esteb/Desktop/Dataset/data.yaml"

results = model.train(
    data= data_yaml,         
    epochs=400,                
    batch=16,                 
    imgsz=640,               
    patience= 5,             
    save = True,
    device = 0,
    workers = 0,
    freeze = 10,
    dropout = 0.3,
    project="runs/train",     
    name="custom_yolov11_training_v3",  
)

model_path = "runs/train/custom_yolov11_training_v3/model_complete.pth"  # Chemin pour sauvegarder le modèle complet
torch.save(model.model, model_path)