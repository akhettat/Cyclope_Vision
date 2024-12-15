import torch
from ultralytics import YOLO  # Ultralytics YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os

# Test de la disponibilité du GPU
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

model = YOLO("yolo11n.pt")
data_yaml = "C:/Users/esteb/Desktop/Dataset/data.yaml"

results = model.train(
    data= data_yaml,         
    epochs=50,                
    batch=16,                 
    imgsz=640,               
    patience=10,             
    save = True,
    device = 0,
    workers = 0,
    freeze = 10,
    dropout = 0.3,
    project="runs/train",     
    name="custom_yolov11_training",  
)

# Résultats d'entraînement
results_path = "runs/train/custom_yolov8_training/results.csv" 
results_df = pd.read_csv(results_path)

# Tracer les courbes de loss
plt.figure(figsize=(10, 5))
plt.plot(results_df['epoch'], results_df['box_loss'], label='Box Loss')
plt.plot(results_df['epoch'], results_df['obj_loss'], label='Objectness Loss')
plt.plot(results_df['epoch'], results_df['cls_loss'], label='Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.grid()
plt.show()

# Tracer la précision mAP
plt.figure(figsize=(10, 5))
plt.plot(results_df['epoch'], results_df['metrics/mAP_50'], label='mAP@0.5')
plt.plot(results_df['epoch'], results_df['metrics/mAP_50_95'], label='mAP@0.5:0.95')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.title('Validation mAP')
plt.grid()
plt.show()