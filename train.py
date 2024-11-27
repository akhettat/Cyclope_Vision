# Importation des librairies pour le modèle et l'affichage de l'entrainement

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from yolov5 import train
from yolov5.utils.metrics import fitness  
from yolov5.utils.torch_utils import EarlyStopping 

# Test de la disponibilité du GPU
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# On clone le yolov5 si on ne l'a pas fait
if not Path("yolov5").exists():
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    os.system("pip install -r yolov5/requirements.txt")


data_yaml = "path/to/your/dataset.yaml"  # Chemein vers notre dataset (format .yaml avec annotations dans un fichier .txt)  https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
weights = "yolov5s.pt" # Matrice des poids si préentrainé (à voir) 
epochs = 50  # Nombre d'époques
batch_size = 16  # Taille du lot
img_size = 640  # Taille des images 

# Importation du modèle sur PyTorch pour cahnger le modèle (rajout dropout)
from yolov5.models.yolo import Model

class CustomModel(Model):
    def __init__(self, cfg='yolov5/models/yolov5s.yaml', ch=3, nc=None):  # Initialisation
        super().__init__(cfg, ch, nc)
        # Ajout de Dropout dans la tête de détection
        for m in self.model:
            if isinstance(m, torch.nn.Conv2d):
                self.dropout = torch.nn.Dropout(p=0.3)

early_stopping = EarlyStopping(patience=10)

# Lancer l'entraînement
results = train.run(
    data=data_yaml,  # Données
    weights=weights,  # Poids pré-entraînés
    epochs=epochs,  # Nombre d'époques
    batch_size=batch_size,  # Taille des lots
    imgsz=img_size,  # Taille des images
    patience=10,  # Early stopping (patience)
    save_period=1,  # Sauvegarder les poids chaque époque
    project="runs/train",  # Dossier de sortie
    name="custom_yolov5_training",  # Nom du run
)

#Evaluation du modèle

results_path = 'runs/train/exp/results.csv'  # Remplacez par le chemin de votre dossier d'entraînement
results = pd.read_csv(results_path)

# Tracer les courbes de loss
plt.figure(figsize=(10, 5))
plt.plot(results['epoch'], results['train/box_loss'], label='Box Loss')
plt.plot(results['epoch'], results['train/obj_loss'], label='Objectness Loss')
plt.plot(results['epoch'], results['train/cls_loss'], label='Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.grid()
plt.show()

# Tracer la précision mAP
plt.figure(figsize=(10, 5))
plt.plot(results['epoch'], results['metrics/mAP_0.5'], label='mAP@0.5')
plt.plot(results['epoch'], results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.title('Validation mAP')
plt.grid()
plt.show()